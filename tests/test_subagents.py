"""Tests for agents used as tools by other agents"""

import asyncio
from typing import Dict, List
from unittest.mock import patch

import litellm
import pytest

from cyclops.core.agent import Agent
from cyclops.core.events import (
    RunFinished,
    TextDelta,
    ToolCallFinished,
)
from cyclops.core.middleware import ToolMiddleware
from cyclops.toolkit.tool import Tool
from tests.test_agent import (
    _make_completion_response,
    _make_config,
    _make_simple_tool,
    _make_tool_call,
    _stream,
    _text_chunk,
    _tool_call_chunk,
)


def _call(name: str, task: str, id: str = "tc_1"):
    return _make_tool_call(id, name, f'{{"input": "{task}"}}')


def _asks(*calls):
    return _make_completion_response(content=None, tool_calls=list(calls))


def _scripted(scripts: Dict[str, List]):
    """acompletion that answers each agent from its own list, by system prompt."""
    seen: Dict[str, List] = {name: [] for name in scripts}

    async def acompletion(**kwargs):
        name = kwargs["messages"][0]["content"]
        seen[name].append(kwargs["messages"][1:])
        response = scripts[name].pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    return acompletion, seen


def _make_completion(content: str):
    return _make_completion_response(content=content)


def _agents(lead_tools=(), **child_config):
    child = Agent(
        _make_config(system_prompt="child", **child_config),
        tools=[_make_simple_tool()],
    )
    lead = Agent(
        _make_config(system_prompt="lead"),
        tools=[child.as_tool("researcher", "Researches a question"), *lead_tools],
    )
    return lead, child


def test_answer_is_the_tool_result():
    lead, child = _agents()
    acompletion, seen = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("x is 5")],
            "child": [_make_completion("found: 5")],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        result = lead.run_with_response("what is x?")

    assert result.content == "x is 5"
    assert result.tool_calls[0].result == "found: 5"
    assert seen["child"] == [[{"role": "user", "content": "find x"}]]
    # Each call has a transcript of its own, kept out of the agent's history.
    assert child.messages == []


def test_failed_run_gives_an_error_result():
    lead, _ = _agents()
    error = litellm.BadRequestError(
        message="down", llm_provider="openai", model="gpt-3.5-turbo"
    )
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("sorry")],
            "child": [error],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        result = lead.run_with_response("what is x?")

    assert result.stop_reason == "completed"
    assert result.tool_calls[0].result.startswith("Error: researcher failed: ")
    assert "down" in result.tool_calls[0].result


def test_run_without_an_answer_gives_an_error_result():
    lead, _ = _agents(max_steps=1)
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("sorry")],
            "child": [_asks(_make_tool_call("c_1", "add", '{"a": 1, "b": 2}'))],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        result = lead.run_with_response("what is x?")

    assert result.tool_calls[0].result == (
        "Error: researcher stopped without an answer: max_steps"
    )


async def test_events_carry_the_agent_path():
    checker = Agent(_make_config(system_prompt="checker"))
    child = Agent(
        _make_config(system_prompt="child"),
        tools=[checker.as_tool("checker", "Checks a claim")],
    )
    lead = Agent(
        _make_config(system_prompt="lead"),
        tools=[child.as_tool("researcher", "Researches a question")],
    )
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("5")],
            "child": [_asks(_call("checker", "is x 5?")), _make_completion("x=5")],
            "checker": [_make_completion("yes")],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        handle = lead.start("what is x?", stream=False)
        events = [e async for e in handle]

    summary = [(type(e).__name__, e.agent_path) for e in events]
    lead_path, child_path, checker_path = (), ("researcher",), ("researcher", "checker")
    assert summary == [
        ("ToolCallStarted", lead_path),
        ("ToolCallStarted", child_path),
        ("TextDelta", checker_path),
        ("StepFinished", checker_path),
        ("RunFinished", checker_path),
        ("ToolCallFinished", child_path),
        ("StepFinished", child_path),
        ("TextDelta", child_path),
        ("StepFinished", child_path),
        ("RunFinished", child_path),
        ("ToolCallFinished", lead_path),
        ("StepFinished", lead_path),
        ("TextDelta", lead_path),
        ("StepFinished", lead_path),
        ("RunFinished", lead_path),
    ]
    nested = [e for e in events if isinstance(e, RunFinished) and e.agent_path]
    assert [e.result.content for e in nested] == ["yes", "x=5"]


async def test_astream_yields_only_the_outer_answer():
    lead, _ = _agents()
    lead_first = [
        _tool_call_chunk(0, '{"input": "find x"}', id="tc_1", name="researcher")
    ]
    scripts = {
        "lead": [_stream(lead_first), _stream([_text_chunk("x is 5")])],
        "child": [_stream([_text_chunk("found "), _text_chunk("5")])],
    }
    acompletion, _ = _scripted(scripts)
    with patch("litellm.acompletion", new=acompletion):
        chunks = [c async for c in lead.astream("what is x?")]
        assert scripts == {"lead": [], "child": []}

    assert chunks == ["x is 5"]


async def test_nested_run_streams_when_the_outer_run_streams():
    lead, _ = _agents()
    lead_first = [
        _tool_call_chunk(0, '{"input": "find x"}', id="tc_1", name="researcher")
    ]
    acompletion, _ = _scripted(
        {
            "lead": [_stream(lead_first), _stream([_text_chunk("x is 5")])],
            "child": [_stream([_text_chunk("found "), _text_chunk("5")])],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        events = [e async for e in lead.astream_events("what is x?")]

    deltas = [e.text for e in events if isinstance(e, TextDelta) and e.agent_path]
    assert deltas == ["found ", "5"]


def test_usage_rolls_up_into_the_outer_run():
    lead, _ = _agents()
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("5")],
            "child": [
                _asks(_make_tool_call("c_1", "add", '{"a": 2, "b": 3}')),
                _make_completion("5"),
            ],
        }
    )
    with (
        patch("litellm.acompletion", new=acompletion),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        result = lead.run_with_response("what is x?")

    # Two completions of the lead and two of the researcher, 30 tokens each.
    assert result.steps == 2
    assert result.tokens_used == 120
    assert result.cost == pytest.approx(0.004)


def test_nested_usage_counts_in_the_outer_budget():
    lead, _ = _agents()
    lead.config.max_total_tokens = 70
    acompletion, seen = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x"))],
            "child": [
                _asks(_make_tool_call("c_1", "add", '{"a": 2, "b": 3}')),
                _make_completion("5"),
            ],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        result = lead.run_with_response("what is x?")

    # 60 tokens leave the researcher room to answer, which brings the lead to
    # 90, so the lead stops before its next completion.
    assert result.tool_calls[0].result == "5"
    assert len(seen["lead"]) == 1
    assert result.stop_reason == "budget"
    assert result.tokens_used == 90


def test_outer_budget_stops_the_nested_run():
    lead, _ = _agents()
    lead.config.max_total_tokens = 50
    acompletion, seen = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x"))],
            "child": [
                _asks(_make_tool_call("c_1", "add", '{"a": 2, "b": 3}')),
                _make_completion("never asked for"),
            ],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        result = lead.run_with_response("what is x?")

    assert len(seen["child"]) == 1
    assert result.tool_calls[0].result == (
        "Error: researcher stopped without an answer: budget"
    )
    assert result.stop_reason == "budget"


async def test_cancelling_the_outer_run_cancels_the_nested_run():
    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(10)
        return "late"

    child = Agent(
        _make_config(system_prompt="child"), tools=[Tool("slow", "Sleeps", slow)]
    )
    lead = Agent(
        _make_config(system_prompt="lead"),
        tools=[child.as_tool("researcher", "Researches a question")],
    )
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x"))],
            "child": [_asks(_make_tool_call("c_1", "slow", "{}"))],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        handle = lead.start("what is x?", stream=False)
        events = []

        async def collect():
            events.extend([e async for e in handle])

        collecting = asyncio.ensure_future(collect())
        await started.wait()
        handle.cancel()
        result = await handle
        await collecting

    assert result.stop_reason == "cancelled"
    finished = [
        (e.agent_path, e.name, e.result)
        for e in events
        if isinstance(e, ToolCallFinished)
    ]
    assert finished == [
        (("researcher",), "slow", "Tool call cancelled."),
        ((), "researcher", "Tool call cancelled."),
    ]
    nested = [e for e in events if isinstance(e, RunFinished) and e.agent_path]
    assert nested[0].result.stop_reason == "cancelled"


async def test_calls_in_one_step_run_concurrently():
    lead, _ = _agents()
    both_asked = asyncio.Event()
    asked: List[str] = []

    async def acompletion(**kwargs):
        messages = kwargs["messages"]
        if messages[0]["content"] == "lead":
            if messages[-1]["role"] == "tool":
                return _make_completion("done")
            return _asks(
                _call("researcher", "find x", id="tc_1"),
                _call("researcher", "find y", id="tc_2"),
            )
        asked.append(messages[1]["content"])
        if len(asked) == 2:
            both_asked.set()
        # Each answers only once the other has started.
        await asyncio.wait_for(both_asked.wait(), 1)
        return _make_completion(f"answer to {messages[1]['content']}")

    with patch("litellm.acompletion", new=acompletion):
        result = await lead.arun_with_response("x and y?")

    assert sorted(asked) == ["find x", "find y"]
    assert [c.result for c in result.tool_calls] == [
        "answer to find x",
        "answer to find y",
    ]


def test_middleware_sees_the_agent_path():
    class Record(ToolMiddleware):
        def __init__(self):
            self.calls = []

        async def before_tool(self, call):
            self.calls.append((call.tool_name, call.agent_path))
            return None

    record = Record()
    child = Agent(
        _make_config(system_prompt="child", middleware=[record]),
        tools=[_make_simple_tool()],
    )
    lead = Agent(
        _make_config(system_prompt="lead", middleware=[record]),
        tools=[child.as_tool("researcher", "Researches a question")],
    )
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("5")],
            "child": [
                _asks(_make_tool_call("c_1", "add", '{"a": 2, "b": 3}')),
                _make_completion("5"),
            ],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        lead.run("what is x?")

    assert record.calls == [("researcher", ()), ("add", ("researcher",))]


def test_guards_see_the_agent_path():
    paths = []

    async def guard(context):
        paths.append(context.agent_path)
        return None

    child = Agent(_make_config(system_prompt="child", guards=[guard]))
    lead = Agent(
        _make_config(system_prompt="lead", guards=[guard]),
        tools=[child.as_tool("researcher", "Researches a question")],
    )
    acompletion, _ = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("5")],
            "child": [_make_completion("5")],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        lead.run("what is x?")

    assert paths == [("researcher",), ()]


async def test_called_outside_a_run_runs_on_its_own():
    child = Agent(_make_config(system_prompt="child"))
    tool = child.as_tool("researcher", "Researches a question")
    acompletion, _ = _scripted({"child": [_make_completion("found")]})
    with patch("litellm.acompletion", new=acompletion):
        assert await tool.execute(input="find x") == "found"


def test_nesting_stops_at_the_depth_limit():
    checker = Agent(_make_config(system_prompt="checker"))
    child = Agent(
        _make_config(system_prompt="child"),
        tools=[checker.as_tool("checker", "Checks a claim")],
    )
    lead = Agent(
        _make_config(system_prompt="lead", max_subagent_depth=1),
        tools=[child.as_tool("researcher", "Researches a question")],
    )
    acompletion, seen = _scripted(
        {
            "lead": [_asks(_call("researcher", "find x")), _make_completion("5")],
            "child": [_asks(_call("checker", "is x 5?")), _make_completion("x=5")],
            "checker": [],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        lead.run("what is x?")

    # The lead allows one level below it, so the researcher can't call the checker.
    assert seen["checker"] == []
    tool_result = seen["child"][1][-1]
    assert tool_result["content"] == (
        "Error: checker failed: "
        "subagents are nested as deep as max_subagent_depth allows"
    )


async def test_concurrent_runs_wait_for_a_slot():
    lead, _ = _agents()
    lead.config.max_concurrent_subagents = 1
    running = 0
    most = 0

    async def acompletion(**kwargs):
        nonlocal running, most
        messages = kwargs["messages"]
        if messages[0]["content"] == "lead":
            if messages[-1]["role"] == "tool":
                return _make_completion("done")
            return _asks(
                _call("researcher", "find x", id="tc_1"),
                _call("researcher", "find y", id="tc_2"),
            )
        running += 1
        most = max(most, running)
        await asyncio.sleep(0.01)
        running -= 1
        return _make_completion(f"answer to {messages[1]['content']}")

    with patch("litellm.acompletion", new=acompletion):
        result = await lead.arun_with_response("x and y?")

    assert most == 1
    assert [c.result for c in result.tool_calls] == [
        "answer to find x",
        "answer to find y",
    ]


def test_calls_waiting_for_a_slot_stop_once_the_budget_is_spent():
    lead, _ = _agents()
    lead.config.max_total_tokens = 50
    lead.config.max_concurrent_subagents = 1
    acompletion, seen = _scripted(
        {
            "lead": [
                _asks(
                    _call("researcher", "find x", id="tc_1"),
                    _call("researcher", "find y", id="tc_2"),
                )
            ],
            "child": [_make_completion("x is 5")],
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        result = lead.run_with_response("x and y?")

    # The first call leaves 60 tokens spent, so the second makes no completion.
    assert len(seen["child"]) == 1
    assert [c.result for c in result.tool_calls] == [
        "x is 5",
        "Error: researcher stopped without an answer: budget",
    ]
    assert result.stop_reason == "budget"


async def test_cancelling_cancels_calls_waiting_for_a_slot():
    lead, _ = _agents()
    lead.config.max_concurrent_subagents = 1
    started = asyncio.Event()
    children = 0

    async def acompletion(**kwargs):
        nonlocal children
        if kwargs["messages"][0]["content"] == "lead":
            return _asks(
                _call("researcher", "find x", id="tc_1"),
                _call("researcher", "find y", id="tc_2"),
            )
        children += 1
        started.set()
        await asyncio.sleep(10)

    with patch("litellm.acompletion", new=acompletion):
        handle = lead.start("x and y?", stream=False)
        await started.wait()
        handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"
    assert children == 1
    tool_messages = [m["content"] for m in lead.messages if m["role"] == "tool"]
    assert tool_messages == ["Tool call cancelled.", "Tool call cancelled."]


def test_depth_limit_stops_an_agent_that_calls_itself():
    config = _make_config(system_prompt="self", max_subagent_depth=2)
    delegate = Agent(config).as_tool("delegate", "Hands the task on")
    agent = Agent(config, tools=[delegate])
    delegate.agent = agent
    acompletion, seen = _scripted(
        {
            "self": [
                _asks(_call("delegate", "go on")),
                _asks(_call("delegate", "go on")),
                _asks(_call("delegate", "go on")),
                _make_completion("level 2"),
                _make_completion("level 1"),
                _make_completion("level 0"),
            ]
        }
    )
    with patch("litellm.acompletion", new=acompletion):
        assert agent.run("start") == "level 0"

    # The run two levels down can't nest a third.
    refused = seen["self"][3][-1]
    assert refused["content"] == (
        "Error: delegate failed: "
        "subagents are nested as deep as max_subagent_depth allows"
    )
