"""Tests for async hooks, hook timeouts, input rewriting and subagent hooks"""

import asyncio
import threading
import time
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from cyclops.core.agent import Agent
from cyclops.core.hooks import AgentHooks
from cyclops.core.session import Session
from cyclops.core.types import RunResult
from cyclops.toolkit.tool import Tool
from tests.test_agent import (
    _make_completion_response,
    _make_config,
    _make_simple_tool,
    _make_tool_call,
)
from tests.test_subagents import _agents, _asks, _call, _make_completion, _scripted


def _sent_input(mock: AsyncMock):
    """The user message of the first completion."""
    return mock.call_args_list[0].kwargs["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_on_run_start_replaces_the_input():
    class Rewrite(AgentHooks):
        def on_run_start(self, input_message):
            return input_message.upper()

    session = Session()
    agent = Agent(_make_config(hooks=Rewrite()))
    mock = AsyncMock(return_value=_make_completion_response("ok"))
    with patch("litellm.acompletion", new=mock):
        assert await agent.arun("hello", session=session) == "ok"

    assert _sent_input(mock) == "HELLO"
    assert session.messages[0] == {"role": "user", "content": "HELLO"}


@pytest.mark.asyncio
async def test_async_on_run_start_adds_context_as_parts():
    class AddContext(AgentHooks):
        async def on_run_start(self, input_message):
            return [
                {"type": "text", "text": "The user is in Paris."},
                {"type": "text", "text": input_message},
            ]

    agent = Agent(_make_config(hooks=AddContext()))
    mock = AsyncMock(return_value=_make_completion_response("ok"))
    with patch("litellm.acompletion", new=mock):
        await agent.arun("weather?")

    assert _sent_input(mock) == [
        {"type": "text", "text": "The user is in Paris."},
        {"type": "text", "text": "weather?"},
    ]


@pytest.mark.asyncio
async def test_replacement_input_is_checked_as_run_input_is():
    errors = []

    class Empty(AgentHooks):
        def on_run_start(self, input_message):
            return []

        def on_run_error(self, error):
            errors.append(error)

    agent = Agent(_make_config(hooks=Empty()))
    mock = AsyncMock(return_value=_make_completion_response("ok"))
    with patch("litellm.acompletion", new=mock):
        result = await agent.arun_with_response("hi")

    assert isinstance(result.error, ValueError)
    assert errors == [result.error]
    mock.assert_not_called()


@pytest.mark.asyncio
async def test_on_run_start_past_its_timeout_fires_on_run_error():
    errors = []

    class Slow(AgentHooks):
        async def on_run_start(self, input_message):
            await asyncio.sleep(10)

        def on_run_error(self, error):
            errors.append(error)

    agent = Agent(_make_config(hooks=Slow(), hook_timeout=0.01))
    mock = AsyncMock(return_value=_make_completion_response("ok"))
    with patch("litellm.acompletion", new=mock):
        result = await agent.arun_with_response("hi")

    assert isinstance(result.error, asyncio.TimeoutError)
    assert errors == [result.error]
    mock.assert_not_called()


@pytest.mark.asyncio
async def test_async_hooks_are_awaited():
    seen = []

    class Async(AgentHooks):
        async def on_llm_end(self, response):
            await asyncio.sleep(0)
            seen.append("llm_end")

        async def on_tool_start(self, tool_name, args):
            seen.append(f"tool_start {tool_name}")
            return "deny"

        async def on_run_end(self, content):
            seen.append(f"run_end {content}")

    agent = Agent(_make_config(hooks=Async()), tools=[_make_simple_tool()])
    asks = _make_completion_response(
        None, tool_calls=[_make_tool_call("tc_1", "add", '{"a": 1, "b": 2}')]
    )
    mock = AsyncMock(side_effect=[asks, _make_completion_response("done")])
    with patch("litellm.acompletion", new=mock):
        result = await agent.arun_with_response("go")

    assert seen == [
        "llm_end",
        "tool_start add",
        "llm_end",
        "run_end done",
    ]
    assert [d.name for d in result.denied_tool_calls] == ["add"]


@pytest.mark.asyncio
async def test_async_hook_past_its_timeout_fails_the_run():
    class Slow(AgentHooks):
        async def on_llm_end(self, response):
            await asyncio.sleep(10)

    agent = Agent(_make_config(hooks=Slow(), hook_timeout=0.01))
    mock = AsyncMock(return_value=_make_completion_response("ok"))
    with patch("litellm.acompletion", new=mock):
        result = await agent.arun_with_response("hi")

    assert result.stop_reason == "error"
    assert isinstance(result.error, asyncio.TimeoutError)


@pytest.mark.asyncio
async def test_async_on_tool_start_past_its_timeout_fails_closed():
    ran = []

    def add(a: int, b: int) -> int:
        ran.append((a, b))
        return a + b

    class Slow(AgentHooks):
        async def on_tool_start(self, tool_name, args):
            await asyncio.sleep(10)

    agent = Agent(
        _make_config(hooks=Slow(), hook_timeout=0.01),
        tools=[Tool("add", "Add two numbers", add)],
    )
    asks = _make_completion_response(
        None, tool_calls=[_make_tool_call("tc_1", "add", '{"a": 1, "b": 2}')]
    )
    mock = AsyncMock(side_effect=[asks, _make_completion_response("done")])
    with patch("litellm.acompletion", new=mock):
        result = await agent.arun_with_response("hi")

    assert isinstance(result.error, asyncio.TimeoutError)
    assert ran == []


@pytest.mark.asyncio
async def test_plain_hooks_run_inline_without_a_timeout():
    """hook_timeout bounds async hooks only; a plain hook runs on the loop's thread."""
    threads = []

    class Plain(AgentHooks):
        def on_run_start(self, input_message):
            time.sleep(0.05)
            threads.append(threading.get_ident())

    agent = Agent(_make_config(hooks=Plain(), hook_timeout=0.01))
    mock = AsyncMock(return_value=_make_completion_response("ok"))
    with patch("litellm.acompletion", new=mock):
        result = await agent.arun_with_response("hi")

    assert result.content == "ok"
    assert threads == [threading.get_ident()]


@pytest.mark.asyncio
async def test_cancelling_a_run_during_a_hook():
    started = asyncio.Event()
    errors = []

    class Waits(AgentHooks):
        async def on_tool_start(self, tool_name, args):
            started.set()
            await asyncio.sleep(10)

        def on_run_error(self, error):
            errors.append(error)

    agent = Agent(
        _make_config(hooks=Waits(), hook_timeout=10), tools=[_make_simple_tool()]
    )
    asks = _make_completion_response(
        None, tool_calls=[_make_tool_call("tc_1", "add", '{"a": 1, "b": 2}')]
    )
    mock = AsyncMock(side_effect=[asks, _make_completion_response("done")])
    with patch("litellm.acompletion", new=mock):
        handle = agent.start("hi", stream=False)
        await started.wait()
        handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"
    assert result.error is None
    assert [type(e) for e in errors] == [asyncio.CancelledError]


@pytest.mark.asyncio
async def test_hooks_within_their_timeout_run_as_without_one():
    class Hooks(AgentHooks):
        def on_run_start(self, input_message):
            return "rewritten"

        async def on_tool_start(self, tool_name, args):
            return "deny"

    agent = Agent(
        _make_config(hooks=Hooks(), hook_timeout=5), tools=[_make_simple_tool()]
    )
    asks = _make_completion_response(
        None, tool_calls=[_make_tool_call("tc_1", "add", '{"a": 1, "b": 2}')]
    )
    mock = AsyncMock(side_effect=[asks, _make_completion_response("done")])
    with patch("litellm.acompletion", new=mock):
        result = await agent.arun_with_response("hi")

    assert result.content == "done"
    assert _sent_input(mock) == "rewritten"
    assert [d.name for d in result.denied_tool_calls] == ["add"]


@pytest.mark.asyncio
async def test_base_hooks_change_nothing():
    """A run with AgentHooks' defaults sends what a run without hooks sends."""
    sent = []
    for hooks in (None, AgentHooks()):
        agent = Agent(_make_config(hooks=hooks), tools=[_make_simple_tool()])
        asks = _make_completion_response(
            None, tool_calls=[_make_tool_call("tc_1", "add", '{"a": 1, "b": 2}')]
        )
        mock = AsyncMock(side_effect=[asks, _make_completion_response("done")])
        with patch("litellm.acompletion", new=mock):
            result = await agent.arun_with_response("hi")
        assert result.content == "done"
        sent.append([c.kwargs["messages"] for c in mock.call_args_list])

    assert sent[0] == sent[1]


class _Watch(AgentHooks):
    def __init__(self):
        self.seen = []

    def on_subagent_start(self, agent_path, input_message):
        self.seen.append(("start", agent_path, input_message))

    async def on_subagent_end(self, agent_path, result):
        self.seen.append(("end", agent_path, result))


def test_parent_hooks_see_a_subagent_run():
    lead, _ = _agents()
    watch = _Watch()
    lead.config.hooks = watch
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("x is 5")],
            "child": [_make_completion("found: 5")],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        assert lead.run("what is x?") == "x is 5"

    assert [s[:2] for s in watch.seen] == [
        ("start", ("researcher",)),
        ("end", ("researcher",)),
    ]
    assert watch.seen[0][2] == "find x"
    result = watch.seen[1][2]
    assert isinstance(result, RunResult)
    assert result.content == "found: 5"
    assert result.stop_reason == "completed"


def test_parent_hooks_see_a_failed_subagent_run():
    lead, _ = _agents(num_retries=0)
    watch = _Watch()
    lead.config.hooks = watch
    error = litellm.APIConnectionError(message="down", llm_provider="openai", model="m")
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("sorry")],
            "child": [error],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        assert lead.run("what is x?") == "sorry"

    result = watch.seen[-1][2]
    assert result.stop_reason == "error"
    assert result.error is error


def test_subagent_hooks_go_to_the_direct_parent_only():
    """A subagent's own hooks see its subagents; its parent's hooks do not."""
    grandchild = Agent(_make_config(system_prompt="grandchild"))
    child_watch, lead_watch = _Watch(), _Watch()
    child = Agent(
        _make_config(system_prompt="child", hooks=child_watch),
        tools=[grandchild.as_tool("digger", "Digs")],
    )
    lead = Agent(
        _make_config(system_prompt="lead", hooks=lead_watch),
        tools=[child.as_tool("researcher", "Researches")],
    )
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("x is 5")],
            "child": [_asks(_call("digger", "dig")), _make_completion("found: 5")],
            "grandchild": [_make_completion("5")],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        assert lead.run("what is x?") == "x is 5"

    assert [s[:2] for s in lead_watch.seen] == [
        ("start", ("researcher",)),
        ("end", ("researcher",)),
    ]
    assert [s[:2] for s in child_watch.seen] == [
        ("start", ("researcher", "digger")),
        ("end", ("researcher", "digger")),
    ]


@pytest.mark.asyncio
async def test_parent_hooks_see_a_cancelled_subagent_run():
    started = asyncio.Event()

    class Waits(AgentHooks):
        async def on_llm_start(self, messages):
            started.set()
            await asyncio.sleep(10)

    lead, child = _agents()
    child.config.hooks = Waits()
    watch = _Watch()
    lead.config.hooks = watch
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x"))],
            "child": [_make_completion("found: 5")],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        handle = lead.start("what is x?", stream=False)
        await started.wait()
        handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"
    assert [(s[0], s[1], s[2].stop_reason) for s in watch.seen[1:]] == [
        ("end", ("researcher",), "cancelled")
    ]
