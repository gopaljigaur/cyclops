"""Tests for declarative tool permissions"""

import asyncio
from typing import List
from unittest.mock import AsyncMock, patch

import pytest

from cyclops.core.agent import Agent
from cyclops.core.middleware import Deny, ToolCallContext
from cyclops.core.permissions import ToolPermissions
from cyclops.core.session import Session
from cyclops.core.types import DeniedToolCall
from cyclops.toolkit.tool import Tool
from tests.test_agent import (
    READ_ONLY,
    _make_completion_response,
    _make_config,
    _make_simple_tool,
    _make_tool_call,
)
from tests.test_middleware import _run, _tool_message, _tool_round


def _agent(permissions: ToolPermissions, name: str = "add") -> Agent:
    return Agent(
        config=_make_config(middleware=[permissions]), tools=[_make_simple_tool(name)]
    )


def _approver(answer, seen: List[ToolCallContext]):
    async def approve(call):
        seen.append(call)
        return answer

    return approve


def test_allowed_call_runs():
    agent = _agent(ToolPermissions(allow=["add"], default="deny"))
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert _tool_message(session) == "5"
    assert result.denied_tool_calls == []


def test_denied_call_gives_the_model_an_error_and_is_on_the_result():
    agent = _agent(ToolPermissions(deny=["add"]))
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert _tool_message(session) == "Error: Tool 'add' is not permitted"
    assert result.denied_tool_calls == [
        DeniedToolCall(
            id="tc_1",
            name="add",
            arguments={"a": 2, "b": 3},
            reason="Tool 'add' is not permitted",
        )
    ]


def test_unmatched_call_gets_the_default():
    agent = _agent(ToolPermissions(allow=["search_*"], default="deny"))
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert _tool_message(session) == "Error: Tool 'add' is not permitted"
    assert [d.name for d in result.denied_tool_calls] == ["add"]


def test_ask_approved_runs_the_call():
    seen: List[ToolCallContext] = []
    agent = _agent(ToolPermissions(ask=["add"], approve=_approver(True, seen)))
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert _tool_message(session) == "5"
    assert result.denied_tool_calls == []
    [call] = seen
    assert (call.tool_name, call.arguments, call.call_id) == (
        "add",
        {"a": 2, "b": 3},
        "tc_1",
    )


def test_ask_refused_denies_the_call():
    agent = _agent(ToolPermissions(ask=["add"], approve=_approver(False, [])))
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert _tool_message(session) == "Error: Tool 'add' was not approved"
    assert [d.reason for d in result.denied_tool_calls] == [
        "Tool 'add' was not approved"
    ]


def test_ask_denied_with_a_reason():
    approve = _approver(Deny("needs a ticket number"), [])
    agent = _agent(ToolPermissions(ask=["add"], approve=approve))
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert _tool_message(session) == "Error: needs a ticket number"
    assert [d.reason for d in result.denied_tool_calls] == ["needs a ticket number"]


def test_default_ask_asks_about_unmatched_calls():
    seen: List[ToolCallContext] = []
    approve = _approver(True, seen)
    agent = _agent(ToolPermissions(allow=["search_*"], default="ask", approve=approve))
    _run(agent, _tool_round())

    assert [c.tool_name for c in seen] == ["add"]


@pytest.mark.parametrize(
    "name, expected",
    [
        ("github_delete_repo", "deny"),
        ("github_create_issue", "ask"),
        ("github_get_issue", "allow"),
        ("GITHUB_get_issue", "ask"),
        ("search", "ask"),
    ],
)
def test_patterns_and_precedence(name, expected):
    async def approve(call):
        return True

    permissions = ToolPermissions(
        allow=["github_*"],
        ask=["github_create_*", "github_delete_*"],
        deny=["*_delete_*"],
        approve=approve,
        default="ask",
    )

    assert permissions.permission(name) == expected


def test_deny_wins_over_ask_and_allow():
    seen: List[ToolCallContext] = []
    permissions = ToolPermissions(
        allow=["*"], ask=["add"], deny=["a?d"], approve=_approver(True, seen)
    )
    agent = _agent(permissions)
    session = Session()
    _run(agent, _tool_round(), session=session)

    assert seen == []
    assert _tool_message(session) == "Error: Tool 'add' is not permitted"


def test_raising_approval_callback_fails_the_run_without_the_call():
    async def approve(call):
        raise RuntimeError("approval service down")

    agent = _agent(ToolPermissions(ask=["add"], approve=approve))
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert result.stop_reason == "error"
    assert str(result.error) == "approval service down"
    assert _tool_message(session) == "Error: approval service down"
    assert result.denied_tool_calls == []


def test_denied_list_is_per_run():
    permissions = ToolPermissions(deny=["add"])
    agent = _agent(permissions)
    first, _ = _run(agent, _tool_round())
    second, _ = _run(agent, [_make_completion_response("hi")])

    assert len(first.denied_tool_calls) == 1
    assert second.denied_tool_calls == []


def test_naive_call_is_denied_without_an_id():
    agent = Agent(
        config=_make_config(
            tool_mode="naive", middleware=[ToolPermissions(deny=["add"])]
        ),
        tools=[_make_simple_tool()],
    )
    responses = [
        _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 2}}'),
        _make_completion_response("ok"),
    ]
    result, _ = _run(agent, responses)

    [denied] = result.denied_tool_calls
    assert (denied.id, denied.name) == (None, "add")


@pytest.mark.parametrize("kwargs", [{"ask": ["add"]}, {"default": "ask"}])
def test_asking_needs_an_approve_callback(kwargs):
    with pytest.raises(ValueError, match="approve callback"):
        ToolPermissions(**kwargs)


def _tools(names, read_only):
    async def run() -> str:
        return "ran"

    return [Tool(n, n, run, READ_ONLY if read_only else None) for n in names]


def _parallel_round(*names):
    calls = [_make_tool_call(f"tc_{n}", n, "{}") for n in names]
    return [
        _make_completion_response(content=None, tool_calls=calls),
        _make_completion_response(content="done"),
    ]


def _tracking_approver(log, running, peak, delays):
    async def approve(call):
        log.append(("ask", call.tool_name))
        running.append(call.tool_name)
        peak[0] = max(peak[0], len(running))
        await asyncio.sleep(delays[call.tool_name])
        running.remove(call.tool_name)
        log.append(("answer", call.tool_name))
        return False

    return approve


def test_read_only_calls_are_asked_concurrently_and_recorded_in_model_order():
    log, running, peak = [], [], [0]
    # The first call's approval takes longer, so it is answered last.
    approve = _tracking_approver(log, running, peak, {"a": 0.05, "b": 0.01})
    agent = Agent(
        config=_make_config(middleware=[ToolPermissions(ask=["*"], approve=approve)]),
        tools=_tools(["a", "b"], read_only=True),
    )
    result, _ = _run(agent, _parallel_round("a", "b"))

    assert peak[0] == 2
    assert log[-2:] == [("answer", "b"), ("answer", "a")]
    assert [d.id for d in result.denied_tool_calls] == ["tc_a", "tc_b"]
    assert [c.id for c in result.tool_calls] == ["tc_a", "tc_b"]


def test_other_calls_are_asked_one_at_a_time_in_model_order():
    log, running, peak = [], [], [0]
    approve = _tracking_approver(log, running, peak, {"a": 0.05, "b": 0.01})
    agent = Agent(
        config=_make_config(middleware=[ToolPermissions(ask=["*"], approve=approve)]),
        tools=_tools(["a", "b"], read_only=False),
    )
    result, _ = _run(agent, _parallel_round("a", "b"))

    assert log == [("ask", "a"), ("answer", "a"), ("ask", "b"), ("answer", "b")]
    assert [d.name for d in result.denied_tool_calls] == ["a", "b"]


def test_failed_step_records_no_denial():
    async def approve(call):
        await asyncio.sleep(0.02)
        raise RuntimeError("approval service down")

    permissions = ToolPermissions(deny=["b"], ask=["a"], approve=approve)
    agent = Agent(
        config=_make_config(middleware=[permissions]),
        tools=_tools(["a", "b"], read_only=True),
    )
    result, _ = _run(agent, _parallel_round("a", "b"))

    assert result.stop_reason == "error"
    assert result.tool_calls == []
    assert result.denied_tool_calls == []


@pytest.mark.asyncio
async def test_cancelling_during_an_approval_runs_nothing_and_denies_nothing():
    asking = asyncio.Event()
    ran = []

    async def approve(call):
        asking.set()
        await asyncio.sleep(5)
        return True

    async def add(a: int, b: int) -> int:
        ran.append(True)
        return a + b

    agent = Agent(
        config=_make_config(middleware=[ToolPermissions(ask=["add"], approve=approve)]),
        tools=[Tool("add", "Add", add)],
    )
    session = Session()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        handle = agent.start("go", stream=False, session=session)
        await asking.wait()
        handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"
    assert ran == []
    assert result.denied_tool_calls == []
    assert _tool_message(session) == "Tool call cancelled."
