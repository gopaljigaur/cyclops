"""Soak tests: many agent runs against mocked completions, nothing may pile up."""

import asyncio
import itertools
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple

import litellm
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cyclops import Agent, AgentConfig, InMemorySessionStore, Session
from cyclops.core.events import TextDelta, ToolCallStarted
from cyclops.observability import TelemetryHooks
from cyclops.toolkit import tool
from tests.soak.harness import WARMUP, soak

pytestmark = pytest.mark.soak

MODEL = "gpt-4o-mini"
LONG = {"tools": [], "completion_kwargs": {"mock_response": " ".join(["word"] * 1000)}}


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
async def wait() -> str:
    """Wait until cancelled."""
    await asyncio.Event().wait()
    return "never"


def agent(**config) -> Agent:
    tools = config.pop("tools", [add])
    return Agent(AgentConfig(model=MODEL, **config), tools=tools)


async def plain():
    await agent(tools=[]).arun("hi")


async def tools():
    result = await agent().arun_with_response("hi")
    assert result.stop_reason == "completed" and result.tool_calls


async def streaming():
    async for _ in agent().astream_events("hi"):
        pass


async def early_close():
    events = agent().astream_events("hi")
    async for event in events:
        if isinstance(event, TextDelta):
            break
    await events.aclose()


async def abandoned_generator():
    events = agent().astream_events("hi")
    async for event in events:
        if isinstance(event, TextDelta):
            break
    del events


async def cancel_mid_tool():
    handle = agent(tools=[wait]).start("hi")
    async for event in handle:
        if isinstance(event, ToolCallStarted):
            await asyncio.sleep(0)  # Let the tool start.
            break
    handle.cancel()
    assert (await handle).stop_reason == "cancelled"


async def failure():
    result = await agent(
        num_retries=0,
        completion_kwargs={
            "mock_response": litellm.APIError(500, "down", "openai", MODEL)
        },
    ).arun_with_response("hi")
    assert result.stop_reason == "error"


_store = InMemorySessionStore()
# Saves each session to the store as it runs, by default in the background.
_session_agent = Agent(AgentConfig(model=MODEL), tools=[add], session_store=_store)
_turns = itertools.count()


async def sessions():
    # One long lived agent; the caller keeps 50 conversations and trims them.
    session_id = f"s{next(_turns) % 50}"
    session = await _store.load(session_id) or Session(id=session_id)
    session.messages = session.messages[-20:]
    await _session_agent.arun("hi", session=session)


_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
_telemetry = TelemetryHooks(tracer_provider=_provider)


async def telemetry():
    async for _ in agent(hooks=_telemetry).astream_events("hi"):
        pass
    _exporter.clear()


_worker = ThreadPoolExecutor(max_workers=1)


async def sync_api():
    # From a worker thread, as a sync web handler would call it. One worker,
    # so a pool adding threads doesn't show as growth.
    await asyncio.get_running_loop().run_in_executor(_worker, agent().run, "hi")


SUCCEEDING = [
    plain,
    tools,
    streaming,
    early_close,
    abandoned_generator,
    cancel_mid_tool,
    sessions,
    telemetry,
    sync_api,
]


@pytest.mark.parametrize("scenario", SUCCEEDING, ids=lambda f: f.__name__)
async def test_runs_leave_nothing_behind(scenario):
    result = await soak(scenario)
    result.assert_steady()
    assert result.cyclic_garbage == 0, result


async def test_failed_runs_leave_nothing_behind():
    # LiteLLM's error path takes longer to fill its caches.
    (await soak(failure, warmup=2 * WARMUP)).assert_steady()


@pytest.mark.xfail(strict=True, reason="#87: the traceback keeps the failed run")
async def test_failed_runs_leave_no_cyclic_garbage():
    assert (await soak(failure, runs=100)).cyclic_garbage == 0


async def test_long_lived_agent_without_session():
    shared = agent()
    (await soak(lambda: shared.arun("hi"))).assert_steady()


async def _loop_iterations(consume) -> Tuple[int, int]:
    """Event loop iterations and text deltas while consume() runs."""
    loop = asyncio.get_running_loop()
    iterations = 0
    run_once = loop._run_once  # type: ignore[attr-defined]

    def counting():
        nonlocal iterations
        iterations += 1
        run_once()

    loop._run_once = counting  # type: ignore[attr-defined]
    try:
        deltas = await consume()
    finally:
        del loop._run_once  # type: ignore[attr-defined]
    return iterations, deltas


async def test_iterated_handle_takes_about_one_loop_turn_per_token():
    async def consume():
        return sum([isinstance(e, TextDelta) async for e in agent(**LONG).start("hi")])

    iterations, deltas = await _loop_iterations(consume)
    assert iterations <= deltas + 50


async def test_event_stream_takes_about_one_loop_turn_per_token():
    async def consume():
        return sum(
            [isinstance(e, TextDelta) async for e in agent(**LONG).astream_events("hi")]
        )

    iterations, deltas = await _loop_iterations(consume)
    assert iterations <= deltas + 50
