"""Tests for TelemetryHooks"""

import asyncio
import threading
from contextlib import aclosing
from unittest.mock import AsyncMock, MagicMock, patch

import litellm

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)
from opentelemetry.trace import StatusCode

from cyclops.core.agent import Agent
from cyclops.core.hooks import current_tool_call_id
from cyclops.core.types import AgentConfig
from cyclops.observability import TelemetryHooks
from cyclops.observability.otel import _frame
from cyclops.toolkit.tool import BaseTool, Tool


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


@pytest.fixture
def hooks(provider):
    return TelemetryHooks(tracer_provider=provider)


def _response(content=None, tool_calls=None):
    msg = MagicMock(reasoning_content=None, thinking_blocks=None)
    msg.content = content
    msg.tool_calls = tool_calls or []
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "tool_calls" if tool_calls else "stop"
    response = MagicMock()
    response.choices = [choice]
    response.model = "gpt-4o-mini"
    response.id = "resp-1"
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 20
    return response


def _tool_call(id, name, arguments):
    tc = MagicMock()
    tc.id = id
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def _children(spans, parent):
    children = [
        s for s in spans if s.parent and s.parent.span_id == parent.context.span_id
    ]
    return sorted(children, key=lambda s: s.start_time)


def _roots(spans):
    return [s for s in spans if s.name == "invoke_agent"]


def _echo_tool():
    async def echo(text: str, delay: float) -> str:
        await asyncio.sleep(delay)
        return text

    return Tool(name="echo", description="Echo text after a delay", func=echo)


def _scripted_acompletion(calls):
    """First completion asks for `calls`, the second answers."""

    async def acompletion(**kwargs):
        await asyncio.sleep(0)
        if kwargs["messages"][-1]["role"] == "tool":
            return _response(content="done")
        user = kwargs["messages"][-1]["content"]
        return _response(tool_calls=calls(user))

    return acompletion


async def test_concurrent_runs_keep_their_own_spans(hooks, exporter):
    def calls(user):
        return [
            _tool_call(f"call-{user}", "echo", f'{{"text": "{user}", "delay": 0.01}}')
        ]

    inputs = ["a", "bb", "ccc", "dddd"]
    with patch("litellm.acompletion", new=_scripted_acompletion(calls)):
        await asyncio.gather(
            *[
                Agent(
                    AgentConfig(model="gpt-4o-mini", hooks=hooks), tools=[_echo_tool()]
                ).arun(text)
                for text in inputs
            ]
        )

    spans = exporter.get_finished_spans()
    roots = _roots(spans)
    assert len(roots) == len(inputs)
    for root in roots:
        assert root.parent is None
        children = _children(spans, root)
        assert [s.name for s in children].count("chat") == 2
        (tool,) = [s for s in children if s.name == "execute_tool echo"]
        # The tool echoes the run's input, so result length ties tool span to its run.
        assert (
            tool.attributes["cyclops.tool.result.length"]
            == root.attributes["cyclops.input.length"]
        )
    assert len(spans) == len(inputs) * 4


async def test_parallel_tool_calls_pair_by_call(hooks, exporter):
    # The first call finishes first, so popping the latest span by name would swap them.
    def calls(user):
        return [
            _tool_call("call-1", "echo", '{"text": "x", "delay": 0}'),
            _tool_call("call-2", "echo", '{"text": "slow call", "delay": 0.05}'),
        ]

    agent = Agent(AgentConfig(model="gpt-4o-mini", hooks=hooks), tools=[_echo_tool()])
    with patch("litellm.acompletion", new=_scripted_acompletion(calls)):
        await agent.arun("go")

    spans = exporter.get_finished_spans()
    (root,) = _roots(spans)
    tools = sorted(
        (s for s in _children(spans, root) if s.name == "execute_tool echo"),
        key=lambda s: s.start_time,
    )
    assert [s.attributes["cyclops.tool.result.length"] for s in tools] == [1, 9]
    assert [s.attributes["gen_ai.tool.call.id"] for s in tools] == ["call-1", "call-2"]


def test_concurrent_threads_keep_their_own_spans(hooks, exporter):
    barrier = threading.Barrier(4)

    def run(i):
        hooks.on_run_start("x" * i)
        barrier.wait()
        hooks.on_llm_start([{"role": "user", "content": "hi"}])
        barrier.wait()
        hooks.on_llm_end(_response(content="ok"))
        hooks.on_tool_start("t", {"n": i})
        barrier.wait()
        hooks.on_tool_end("t", {"n": i}, "r" * i)
        hooks.on_run_end("done")

    threads = [threading.Thread(target=run, args=(i,)) for i in range(1, 5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    spans = exporter.get_finished_spans()
    for root in _roots(spans):
        chat, tool = sorted(_children(spans, root), key=lambda s: s.start_time)
        assert chat.name == "chat"
        assert (
            tool.attributes["cyclops.tool.result.length"]
            == root.attributes["cyclops.input.length"]
        )
    assert len(spans) == 12


def test_llm_attributes_follow_genai_conventions(hooks, exporter):
    hooks.on_run_start("hi")
    hooks.on_llm_start([{"role": "user", "content": "hi"}])
    hooks.on_llm_end(_response(content="ok"))
    hooks.on_run_end("ok")

    chat = next(s for s in exporter.get_finished_spans() if s.name == "chat")
    assert chat.kind == trace.SpanKind.CLIENT
    assert chat.attributes["gen_ai.operation.name"] == "chat"
    assert chat.attributes["gen_ai.response.model"] == "gpt-4o-mini"
    assert chat.attributes["gen_ai.response.id"] == "resp-1"
    assert chat.attributes["gen_ai.response.finish_reasons"] == ("stop",)
    assert chat.attributes["gen_ai.usage.input_tokens"] == 10
    assert chat.attributes["gen_ai.usage.output_tokens"] == 20


def _assert_context_clean():
    assert _frame.get() is None
    assert trace.get_current_span() is trace.INVALID_SPAN


async def test_llm_error_fails_chat_and_run(hooks, exporter):
    agent = Agent(AgentConfig(model="gpt-4o-mini", hooks=hooks))
    with patch("litellm.acompletion", new=AsyncMock(side_effect=ValueError("boom"))):
        with pytest.raises(ValueError):
            await agent.arun("hi")

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["chat", "invoke_agent"]
    for span in spans:
        assert span.status.status_code == StatusCode.ERROR
        assert span.attributes["error.type"] == "ValueError"
    _assert_context_clean()


def _unsupported_tools():
    return litellm.UnsupportedParamsError(
        message="does not support parameters: ['tools']",
        llm_provider="ollama",
        model="llama2",
    )


async def test_naive_fallback_inside_a_tool_keeps_the_trace(hooks, exporter):
    # The inner model rejects native tools; its run recovers in naive mode.
    inner = Agent(AgentConfig(model="ollama/llama2", hooks=hooks), tools=[_echo_tool()])

    ids_after_inner_run = []

    async def ask(question: str) -> str:
        answer = await inner.arun(question)
        ids_after_inner_run.append(current_tool_call_id())
        return answer

    outer = Agent(
        AgentConfig(model="gpt-4o-mini", hooks=hooks),
        tools=[Tool(name="ask", description="Ask the inner agent", func=ask)],
    )
    responses = [
        _response(tool_calls=[_tool_call("call-1", "ask", '{"question": "q"}')]),
        _unsupported_tools(),
        _response(content='{"tool": "echo", "args": {"text": "hey", "delay": 0}}'),
        _response(content="inner answer"),
        _response(content="outer answer"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        assert await outer.arun("go") == "outer answer"

    spans = exporter.get_finished_spans()
    outer_root, inner_root = sorted(_roots(spans), key=lambda s: s.start_time)
    assert outer_root.parent is None
    assert outer_root.status.status_code == StatusCode.UNSET
    (ask_span,) = [
        s for s in _children(spans, outer_root) if s.name == "execute_tool ask"
    ]
    assert ask_span.attributes["cyclops.tool.result.length"] == len("inner answer")
    assert inner_root.parent.span_id == ask_span.context.span_id
    assert inner_root.status.status_code == StatusCode.UNSET
    assert inner_root.attributes["cyclops.output.length"] == len("inner answer")
    inner_children = [s.name for s in _children(spans, inner_root)]
    assert inner_children == ["chat", "chat", "execute_tool echo", "chat"]
    rejected = _children(spans, inner_root)[0]
    assert rejected.attributes["error.type"] == "UnsupportedParamsError"
    # The inner naive call has no id of its own and does not inherit the outer one.
    (echo,) = [s for s in spans if s.name == "execute_tool echo"]
    assert "gen_ai.tool.call.id" not in echo.attributes
    assert ask_span.attributes["gen_ai.tool.call.id"] == "call-1"
    assert ids_after_inner_run == ["call-1"]
    _assert_context_clean()


async def test_cancelled_tool_ends_its_span(hooks, exporter):
    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(10)
        return "late"

    agent = Agent(
        AgentConfig(model="gpt-4o-mini", hooks=hooks),
        tools=[Tool(name="slow", description="Sleeps", func=slow)],
    )
    first = _response(tool_calls=[_tool_call("call-1", "slow", "{}")])
    with patch("litellm.acompletion", new=AsyncMock(return_value=first)):
        task = asyncio.ensure_future(agent.arun("go"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    spans = {s.name: s for s in exporter.get_finished_spans()}
    for name in ("execute_tool slow", "invoke_agent"):
        assert spans[name].status.status_code == StatusCode.ERROR
        assert spans[name].attributes["error.type"] == "CancelledError"


async def test_tool_error_fails_its_span_and_restores_context(hooks, exporter):
    def broken() -> str:
        raise RuntimeError("nope")

    # Naive mode runs the tool in the run's own context.
    agent = Agent(
        AgentConfig(model="gpt-4o-mini", hooks=hooks, tool_mode="naive"),
        tools=[Tool(name="broken", description="Always fails", func=broken)],
    )
    responses = [
        _response(content='{"tool": "broken", "args": {}}'),
        _response(content="it failed"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        assert await agent.arun("go") == "it failed"

    spans = exporter.get_finished_spans()
    (tool,) = [s for s in spans if s.name == "execute_tool broken"]
    assert tool.status.status_code == StatusCode.ERROR
    assert tool.attributes["error.type"] == "RuntimeError"
    (root,) = _roots(spans)
    assert [s.name for s in _children(spans, root)] == [
        "chat",
        "execute_tool broken",
        "chat",
    ]
    _assert_context_clean()


def _text_chunk(text):
    return ModelResponseStream(
        model="gpt-4o-mini", choices=[StreamingChoices(delta=Delta(content=text))]
    )


async def _chunks(*texts):
    for text in texts:
        yield _text_chunk(text)


def test_sync_stream_closes_its_spans(hooks, exporter):
    agent = Agent(AgentConfig(model="gpt-4o-mini", hooks=hooks))
    with patch("litellm.acompletion", new=AsyncMock(return_value=_chunks("a", "b"))):
        assert list(agent.stream("hi")) == ["a", "b"]

    spans = exporter.get_finished_spans()
    (root,) = _roots(spans)
    (chat,) = _children(spans, root)
    assert chat.attributes["gen_ai.response.model"] == "gpt-4o-mini"
    assert root.attributes["cyclops.output.length"] == 2
    _assert_context_clean()


async def test_astream_closed_early_ends_the_run_without_error(hooks, exporter):
    async def stalled():
        yield _text_chunk("a")
        # The run is suspended mid-stream when the consumer closes it.
        await asyncio.sleep(10)
        yield _text_chunk("b")

    agent = Agent(AgentConfig(model="gpt-4o-mini", hooks=hooks))
    with patch("litellm.acompletion", new=AsyncMock(return_value=stalled())):
        async with aclosing(agent.astream("hi")) as stream:
            async for _ in stream:
                break

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["chat", "invoke_agent"]
    for span in spans:
        assert span.status.status_code == StatusCode.UNSET
        assert "error.type" not in span.attributes
    _assert_context_clean()

    # Closed while a tool runs: the tool stops for the same reason.
    exporter.clear()
    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(10)
        return "late"

    agent = Agent(
        AgentConfig(model="gpt-4o-mini", hooks=hooks),
        tools=[Tool(name="slow", description="Sleeps", func=slow)],
    )
    call = ChatCompletionDeltaToolCall(
        index=0,
        id="call-1",
        type="function",
        function=Function(name="slow", arguments="{}"),
    )
    chunk = ModelResponseStream(
        model="gpt-4o-mini",
        choices=[StreamingChoices(delta=Delta(tool_calls=[call]))],
    )

    async def tool_stream():
        yield chunk

    with patch("litellm.acompletion", new=AsyncMock(return_value=tool_stream())):
        async with aclosing(agent.astream_events("go")) as events:
            async for _ in events:
                break
            await started.wait()

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["chat", "execute_tool slow", "invoke_agent"]
    for span in spans:
        assert span.status.status_code == StatusCode.UNSET
        assert "error.type" not in span.attributes
    _assert_context_clean()


async def test_break_out_of_astream_leaves_later_runs_alone(hooks, exporter):
    agent = Agent(AgentConfig(model="gpt-4o-mini", hooks=hooks))
    with patch("litellm.acompletion", new=AsyncMock(return_value=_chunks("a", "b"))):
        # A plain break leaves the generator open until it is garbage collected.
        stream = agent.astream("hi")
        async for _ in stream:
            break
    assert trace.get_current_span() is trace.INVALID_SPAN

    with patch("litellm.acompletion", new=AsyncMock(return_value=_response("ok"))):
        await agent.arun("again")
    await stream.aclose()

    roots = _roots(exporter.get_finished_spans())
    assert len(roots) == 2
    assert all(root.parent is None for root in roots)


def test_sync_run_nests_under_caller_and_tool_spans(hooks, provider, exporter):
    tracer = provider.get_tracer("test")

    class Lookup(BaseTool):
        def execute(self) -> str:  # type: ignore[override]
            with tracer.start_as_current_span("inside"):
                return "found"

    agent = Agent(
        AgentConfig(model="gpt-4o-mini", hooks=hooks),
        tools=[Lookup("lookup", "Look up")],
    )
    responses = [
        _response(tool_calls=[_tool_call("call-7", "lookup", "{}")]),
        _response(content="done"),
    ]

    async def caller():
        # A sync run from code that already runs an event loop.
        with tracer.start_as_current_span("caller"):
            return agent.run("go")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        assert asyncio.run(caller()) == "done"

    spans = {s.name: s for s in exporter.get_finished_spans()}
    tool = spans["execute_tool lookup"]
    assert spans["invoke_agent"].parent.span_id == spans["caller"].context.span_id
    assert tool.attributes["gen_ai.tool.call.id"] == "call-7"
    assert spans["inside"].parent.span_id == tool.context.span_id


def test_factories_leave_the_global_provider_alone():
    hooks = TelemetryHooks.console()
    assert trace.get_tracer_provider() is not hooks._provider
