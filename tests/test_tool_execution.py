"""Tests for how a step's tool calls run: rendering, limits and timeouts"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from cyclops.core.agent import Agent
from cyclops.core.events import ToolCallFinished, ToolCallStarted
from cyclops.core.hooks import AgentHooks
from cyclops.core.middleware import ToolMiddleware
from cyclops.toolkit import ImageContent, TextContent, ToolResult, tool
from cyclops.toolkit.tool import Tool
from tests.test_agent import (
    READ_ONLY,
    _make_completion_response,
    _make_config,
    _make_tool_call,
)


def _run_round(agent: Agent, *names: str):
    """Run one step that calls each named tool, then a final answer."""
    calls = [_make_tool_call(f"tc_{i}", n, "{}") for i, n in enumerate(names)]
    responses = [
        _make_completion_response(content=None, tool_calls=calls),
        _make_completion_response(content="done"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        return agent.run_with_response("go")


def _tool_messages(agent: Agent):
    return [m["content"] for m in agent.messages if m["role"] == "tool"]


class _Ends(AgentHooks):
    def __init__(self):
        self.ended = []
        self.errors = []

    def on_tool_end(self, tool_name, args, result):
        self.ended.append(result)

    def on_tool_error(self, tool_name, args, error):
        self.errors.append(error)


def test_message_text_drops_the_images():
    image = ImageContent(data="aGk=", mime_type="image/png")
    content = ToolResult(content=[TextContent(text="a"), image]).to_message_content()

    assert ToolResult.message_text(content) == "a"
    assert ToolResult.message_text("plain") == "plain"


@pytest.mark.parametrize("with_hooks", [False, True])
def test_a_result_is_rendered_once(with_hooks):
    @tool
    def data() -> dict:
        """Return data."""
        return {"rows": 2}

    hooks = _Ends() if with_hooks else None
    agent = Agent(config=_make_config(hooks=hooks), tools=[data])
    render = ToolResult.to_message_content
    with patch.object(
        ToolResult, "to_message_content", autospec=True, side_effect=render
    ) as spy:
        result = _run_round(agent, "data")

    assert spy.call_count == 1
    assert _tool_messages(agent) == ['{"rows": 2}']
    assert result.tool_calls[0].result == '{"rows": 2}'
    if hooks:
        assert hooks.ended == ['{"rows": 2}']


def test_middleware_sees_its_change_after_the_hooks_rendered():
    """A result changed in place after on_tool_end is rendered again."""

    @tool
    def data() -> str:
        """Return data."""
        return "secret"

    class Redact(ToolMiddleware):
        async def after_tool(self, call, result, error):
            result.result = "[redacted]"

    hooks = _Ends()
    agent = Agent(config=_make_config(hooks=hooks, middleware=[Redact()]), tools=[data])
    _run_round(agent, "data")

    assert hooks.ended == ["secret"]
    assert _tool_messages(agent) == ["[redacted]"]


def test_naive_result_is_rendered_once():
    @tool
    def data() -> str:
        """Return data."""
        return "rows"

    agent = Agent(config=_make_config(tool_mode="naive"), tools=[data])
    responses = [
        _make_completion_response(json.dumps({"tool": "data", "args": {}})),
        _make_completion_response("done"),
    ]
    render = ToolResult.to_message_content
    with (
        patch.object(
            ToolResult, "to_message_content", autospec=True, side_effect=render
        ) as spy,
        patch("litellm.acompletion", new=AsyncMock(side_effect=responses)),
    ):
        assert agent.run("go") == "done"

    assert spy.call_count == 1
    assert agent.messages[-2]["content"] == "Tool result: rows"


def _tracked(names, running, peak, log, read_only=True, delay=0.02):
    """Tools that record their start and end and the most running at once."""

    def make(name, delay):
        async def run() -> str:
            log.append(("start", name))
            running.append(name)
            peak[0] = max(peak[0], len(running))
            await asyncio.sleep(delay)
            running.remove(name)
            log.append(("end", name))
            return name

        return Tool(name, name, run, READ_ONLY if read_only else None)

    # Later calls finish sooner, so running them out of order would show.
    return [make(n, delay * (len(names) - i)) for i, n in enumerate(names)]


def test_parallel_tool_calls_are_capped():
    running, peak, log = [], [0], []
    names = ["a", "b", "c", "d", "e"]
    tools = _tracked(names, running, peak, log)
    agent = Agent(config=_make_config(max_parallel_tool_calls=2), tools=tools)
    _run_round(agent, *names)

    assert peak[0] == 2
    assert _tool_messages(agent) == names


def test_read_only_tools_run_concurrently_by_default():
    running, peak, log = [], [0], []
    names = ["a", "b", "c"]
    agent = Agent(config=_make_config(), tools=_tracked(names, running, peak, log))
    _run_round(agent, *names)

    assert peak[0] == 3


def test_tools_not_read_only_run_one_at_a_time_in_order():
    running, peak, log = [], [0], []
    names = ["a", "b", "c"]
    tools = _tracked(names, running, peak, log, read_only=False)
    agent = Agent(config=_make_config(), tools=tools)
    _run_round(agent, *names)

    assert log == [(e, n) for n in names for e in ("start", "end")]


def test_read_only_tools_run_alongside_the_ordered_ones():
    running, peak, log = [], [0], []
    writes = _tracked(["w1", "w2"], running, peak, log, read_only=False)
    reads = _tracked(["r1"], running, peak, log)
    agent = Agent(config=_make_config(), tools=writes + reads)
    _run_round(agent, "w1", "r1", "w2")

    assert peak[0] == 2
    assert log.index(("end", "w1")) < log.index(("start", "w2"))


class _AfterTool(ToolMiddleware):
    def __init__(self):
        self.errors = []

    async def after_tool(self, call, result, error):
        self.errors.append(error)


@pytest.mark.asyncio
async def test_a_call_over_the_timeout_is_cancelled_and_fails():
    cleaned = []

    async def slow() -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cleaned.append("slow")
            raise
        return "done"

    @tool
    def quick() -> str:
        """Answer at once."""
        return "quick"

    hooks = _Ends()
    after = _AfterTool()
    agent = Agent(
        config=_make_config(tool_timeout=0.05, hooks=hooks, middleware=[after]),
        tools=[Tool("slow", "Slow", slow), quick],
    )
    calls = [
        _make_tool_call("tc_1", "slow", "{}"),
        _make_tool_call("tc_2", "quick", "{}"),
    ]
    responses = [
        _make_completion_response(content=None, tool_calls=calls),
        _make_completion_response(content="done"),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        events = [e async for e in agent.start("go", stream=False)]

    assert cleaned == ["slow"]
    assert _tool_messages(agent) == [
        "Error: slow timed out after 0.05 seconds",
        "quick",
    ]
    assert isinstance(after.errors[0], asyncio.TimeoutError)
    assert isinstance(hooks.errors[0], asyncio.TimeoutError)
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert (finished[0].result, finished[0].is_error) == (
        "Error: slow timed out after 0.05 seconds",
        True,
    )


def test_a_sync_tool_over_the_timeout_fails():
    @tool(timeout=0.01)
    def stuck() -> str:
        """Block for a while."""
        time.sleep(0.2)
        return "done"

    agent = Agent(config=_make_config(), tools=[stuck])
    _run_round(agent, "stuck")

    assert _tool_messages(agent) == ["Error: stuck timed out after 0.01 seconds"]


def test_a_tool_timeout_wins_over_the_default():
    async def nap() -> str:
        await asyncio.sleep(0.05)
        return "rested"

    agent = Agent(
        config=_make_config(tool_timeout=0.01),
        tools=[Tool("nap", "Nap", nap, timeout=1)],
    )
    _run_round(agent, "nap")

    assert _tool_messages(agent) == ["rested"]


def test_a_tools_own_timeout_error_is_a_failure():
    async def flaky() -> str:
        raise asyncio.TimeoutError()

    agent = Agent(
        config=_make_config(tool_timeout=5), tools=[Tool("flaky", "Flaky", flaky)]
    )
    _run_round(agent, "flaky")

    assert _tool_messages(agent) == ["Error: flaky failed: "]


@pytest.mark.asyncio
async def test_a_cancel_as_the_tool_returns_stops_the_run():
    async def finish() -> str:
        # Lands while the run waits on this call, as the call returns.
        asyncio.get_running_loop().call_soon(handle.cancel)
        await asyncio.sleep(0)
        return "done"

    # A naive call runs in the run's own task, so nothing else sees the cancel.
    agent = Agent(
        config=_make_config(tool_mode="naive", tool_timeout=5),
        tools=[Tool("finish", "Finish", finish)],
    )
    responses = [
        _make_completion_response(json.dumps({"tool": "finish", "args": {}})),
        _make_completion_response("done"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)) as call:
        handle = agent.start("go", stream=False)
        result = await handle

    assert result.stop_reason == "cancelled"
    assert call.await_count == 1


@pytest.mark.asyncio
async def test_calls_waiting_their_turn_never_start_once_the_step_fails():
    ran = []

    def make(name):
        async def run() -> str:
            ran.append(name)
            return name

        return Tool(name, name, run)

    class Broken(ToolMiddleware):
        async def after_tool(self, call, result, error):
            raise RuntimeError("broke")

    agent = Agent(
        config=_make_config(middleware=[Broken()]), tools=[make("a"), make("b")]
    )
    calls = [_make_tool_call("tc_1", "a", "{}"), _make_tool_call("tc_2", "b", "{}")]
    response = _make_completion_response(content=None, tool_calls=calls)
    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        events = [e async for e in agent.start("go", stream=False)]

    assert ran == ["a"]
    assert [e.call_id for e in events if isinstance(e, ToolCallStarted)] == ["tc_1"]
    assert _tool_messages(agent) == ["Error: broke", "Tool call cancelled."]


def test_a_tool_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout of tool 'nap' must be positive"):

        @tool(timeout=0)
        def nap() -> str:
            """Nap."""
            return "rested"
