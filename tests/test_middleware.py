"""Tests for tool-call middleware and end-of-run guards"""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from cyclops.core.agent import Agent
from cyclops.core.events import ToolCallFinished, ToolCallStarted
from cyclops.core.hooks import AgentHooks, current_tool_call_id
from cyclops.core.middleware import Deny, ToolMiddleware
from cyclops.core.session import Session
from cyclops.toolkit import ToolAnnotations, ToolResult, tool
from tests.test_agent import (
    _make_completion_response,
    _make_config,
    _make_simple_tool,
    _make_tool_call,
    _stream,
    _text_chunk,
    _tool_call_chunk,
)


def _tool_round(arguments: str = '{"a": 2, "b": 3}', final: str = "done"):
    return [
        _make_completion_response(
            content=None, tool_calls=[_make_tool_call("tc_1", "add", arguments)]
        ),
        _make_completion_response(content=final),
    ]


def _run(agent: Agent, responses, message: str = "go", session=None):
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)) as mock:
        result = agent.run_with_response(message, session=session)
    return result, mock


class _EndHooks(AgentHooks):
    def __init__(self):
        self.ended = []

    def on_run_end(self, content):
        self.ended.append(content)


def _tool_message(session: Session) -> str:
    return next(m["content"] for m in session.messages if m["role"] == "tool")


def test_deny_sends_the_reason_and_skips_the_tool():
    ran = []

    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        ran.append((a, b))
        return a + b

    class Block(ToolMiddleware):
        async def before_tool(self, call):
            return Deny("adding is disabled today")

    agent = Agent(config=_make_config(middleware=[Block()]), tools=[add])
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert ran == []
    assert _tool_message(session) == "Error: adding is disabled today"
    assert result.tool_calls[0].result == "Error: adding is disabled today"
    assert [d.reason for d in result.denied_tool_calls] == ["adding is disabled today"]


def test_before_tool_can_replace_the_arguments():
    class Double(ToolMiddleware):
        async def before_tool(self, call):
            return {k: v * 2 for k, v in call.arguments.items()}

    agent = Agent(
        config=_make_config(middleware=[Double()]), tools=[_make_simple_tool()]
    )
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert _tool_message(session) == "10"
    assert result.tool_calls[0].arguments == {"a": 4, "b": 6}
    # The transcript keeps what the model sent.
    assert session.messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"a": 2, "b": 3}'
    )


def test_after_tool_can_replace_the_result():
    class Redact(ToolMiddleware):
        async def after_tool(self, call, result, error):
            return ToolResult.success_result("[redacted]")

    agent = Agent(
        config=_make_config(middleware=[Redact()]), tools=[_make_simple_tool()]
    )
    session = Session()
    _run(agent, _tool_round(), session=session)

    assert _tool_message(session) == "[redacted]"


def test_after_tool_sees_the_tool_error():
    seen = []

    @tool
    def boom() -> str:
        """Fail."""
        raise RuntimeError("disk full")

    class Watch(ToolMiddleware):
        async def after_tool(self, call, result, error):
            seen.append((result.success, error))
            return None

    agent = Agent(config=_make_config(middleware=[Watch()]), tools=[boom])
    responses = [
        _make_completion_response(
            content=None, tool_calls=[_make_tool_call("tc_1", "boom", "{}")]
        ),
        _make_completion_response(content="done"),
    ]
    session = Session()
    _run(agent, responses, session=session)

    [(success, error)] = seen
    assert success is False
    assert isinstance(error, RuntimeError)
    assert _tool_message(session) == "Error: boom failed: disk full"


def test_middleware_run_in_order_and_unwind_in_reverse():
    events = []

    class Record(ToolMiddleware):
        def __init__(self, name):
            self.name = name

        async def before_tool(self, call):
            events.append(f"before {self.name}")
            return None

        async def after_tool(self, call, result, error):
            events.append(f"after {self.name}")
            return None

    class Block(ToolMiddleware):
        async def before_tool(self, call):
            events.append("before block")
            return Deny("no")

        async def after_tool(self, call, result, error):
            events.append("after block")
            return None

    agent = Agent(
        config=_make_config(middleware=[Record("a"), Record("b")]),
        tools=[_make_simple_tool()],
    )
    _run(agent, _tool_round())
    assert events == ["before a", "before b", "after b", "after a"]

    events.clear()
    agent = Agent(
        config=_make_config(middleware=[Record("a"), Block(), Record("c")]),
        tools=[_make_simple_tool()],
    )
    _run(agent, _tool_round())
    # A denied call skips the rest; only the middleware that allowed it unwind.
    assert events == ["before a", "before block", "after a"]


def test_call_context_carries_the_call():
    calls = []

    @tool(annotations=ToolAnnotations(read_only=True))
    def lookup(key: str) -> str:
        """Look up a key."""
        return key

    class Capture(ToolMiddleware):
        async def before_tool(self, call):
            calls.append(call)
            return None

    agent = Agent(config=_make_config(middleware=[Capture()]), tools=[lookup])
    responses = [
        _make_completion_response(
            content=None,
            tool_calls=[_make_tool_call("tc_7", "lookup", '{"key": "k"}')],
        ),
        _make_completion_response(content="done"),
    ]
    result, _ = _run(agent, responses)
    _run(agent, responses)

    first, second = calls
    assert first.run_id == result.run_id
    assert (first.tool_name, first.arguments, first.call_id) == (
        "lookup",
        {"key": "k"},
        "tc_7",
    )
    assert first.annotations.read_only is True
    assert first.agent_path == ()
    assert first.run_id != second.run_id


def _parallel_round():
    return [
        _make_completion_response(
            content=None,
            tool_calls=[
                _make_tool_call("tc_1", "add", '{"a": 1, "b": 2}'),
                _make_tool_call("tc_2", "slow", "{}"),
            ],
        ),
        _make_completion_response(content="done"),
    ]


def test_middleware_on_parallel_calls():
    @tool
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(0.01)
        return "ok"

    class Tag(ToolMiddleware):
        async def after_tool(self, call, result, error):
            return ToolResult.success_result(
                f"{call.call_id}: {result.to_message_content()}"
            )

    agent = Agent(
        config=_make_config(middleware=[Tag()]), tools=[_make_simple_tool(), slow]
    )
    session = Session()
    result, _ = _run(agent, _parallel_round(), session=session)

    tool_messages = [m["content"] for m in session.messages if m["role"] == "tool"]
    assert tool_messages == ["tc_1: 3", "tc_2: ok"]
    assert [c.id for c in result.tool_calls] == ["tc_1", "tc_2"]


def test_raising_middleware_answers_every_call():
    finished = []

    @tool
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(5)
        finished.append(True)
        return "ok"

    class Broken(ToolMiddleware):
        async def before_tool(self, call):
            if call.tool_name == "add":
                raise RuntimeError("policy store down")
            return None

    agent = Agent(
        config=_make_config(middleware=[Broken()]), tools=[_make_simple_tool(), slow]
    )
    session = Session()
    result, _ = _run(agent, _parallel_round(), session=session)

    assert result.stop_reason == "error"
    assert str(result.error) == "policy store down"
    assert finished == []
    assert [(m["role"], m.get("content")) for m in session.messages[2:]] == [
        ("tool", "Error: policy store down"),
        ("tool", "Tool call cancelled."),
    ]


@pytest.mark.asyncio
async def test_cancelled_call_runs_after_tool():
    seen = []

    @tool
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(5)
        return "ok"

    class Watch(ToolMiddleware):
        async def after_tool(self, call, result, error):
            seen.append(type(error))
            return None

    class Hooks(AgentHooks):
        def on_tool_error(self, tool_name, args, error):
            seen.append(("hook", type(error)))

    agent = Agent(
        config=_make_config(middleware=[Watch()], hooks=Hooks()), tools=[slow]
    )
    responses = [
        _make_completion_response(
            content=None, tool_calls=[_make_tool_call("tc_1", "slow", "{}")]
        )
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        task = asyncio.ensure_future(agent.arun("go"))
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert seen == [("hook", asyncio.CancelledError), asyncio.CancelledError]


def test_tool_call_id_is_set_for_middleware():
    seen = []

    class Watch(ToolMiddleware):
        async def before_tool(self, call):
            seen.append(current_tool_call_id())
            return None

    agent = Agent(
        config=_make_config(middleware=[Watch()]), tools=[_make_simple_tool()]
    )
    _run(agent, _tool_round())

    assert seen == ["tc_1"]
    assert current_tool_call_id() is None


def test_legacy_deny_hook_still_blocks():
    class DenyAll(AgentHooks):
        def __init__(self):
            self.ended = []

        def on_tool_start(self, tool_name, args):
            return "deny"

        def on_tool_end(self, tool_name, args, result):
            self.ended.append(tool_name)

    hooks = DenyAll()
    agent = Agent(config=_make_config(hooks=hooks), tools=[_make_simple_tool()])
    session = Session()
    result, _ = _run(agent, _tool_round(), session=session)

    assert _tool_message(session) == "Error: Tool 'add' was not approved"
    assert hooks.ended == []
    assert [(d.id, d.reason) for d in result.denied_tool_calls] == [
        ("tc_1", "Tool 'add' was not approved")
    ]


def test_hooks_see_the_arguments_middleware_chose():
    class Recording(AgentHooks):
        def __init__(self):
            self.events = []

        def on_tool_start(self, tool_name, args):
            self.events.append(("start", args))

        def on_tool_end(self, tool_name, args, result):
            self.events.append(("end", result))

    class Double(ToolMiddleware):
        async def before_tool(self, call):
            return {k: v * 2 for k, v in call.arguments.items()}

    hooks = Recording()
    agent = Agent(
        config=_make_config(hooks=hooks, middleware=[Double()]),
        tools=[_make_simple_tool()],
    )
    _run(agent, _tool_round())

    assert hooks.events == [("start", {"a": 4, "b": 6}), ("end", "10")]


def test_middleware_runs_for_naive_tool_calls():
    calls = []

    class Capture(ToolMiddleware):
        async def before_tool(self, call):
            calls.append(call)
            return None

    agent = Agent(
        config=_make_config(tool_mode="naive", middleware=[Capture()]),
        tools=[_make_simple_tool()],
    )
    responses = [
        _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 2}}'),
        _make_completion_response("3"),
    ]
    _run(agent, responses)

    [call] = calls
    assert call.call_id is None
    assert call.arguments == {"a": 1, "b": 2}


def test_guard_sends_the_model_back():
    async def needs_citation(context):
        if "[1]" not in context.content:
            return "Cite a source."
        return None

    agent = Agent(config=_make_config(guards=[needs_citation]))
    responses = [
        _make_completion_response("Paris."),
        _make_completion_response("Paris [1]."),
    ]
    session = Session()
    result, mock = _run(agent, responses, session=session)

    assert result.content == "Paris [1]."
    assert mock.call_count == 2
    assert [(m["role"], m["content"]) for m in session.messages] == [
        ("user", "go"),
        ("assistant", "Paris."),
        ("user", "Cite a source."),
        ("assistant", "Paris [1]."),
    ]


def test_guard_cap_ends_the_run_with_stop_reason_guard():
    contexts = []

    async def never_happy(context):
        contexts.append(context)
        return "Try again."

    hooks = _EndHooks()
    agent = Agent(
        config=_make_config(guards=[never_happy], max_guard_rounds=2, hooks=hooks)
    )
    responses = [_make_completion_response(f"answer {i}") for i in range(3)]
    result, mock = _run(agent, list(responses))

    assert mock.call_count == 3
    assert result.stop_reason == "guard"
    assert result.content == ""
    assert hooks.ended == [""]
    assert len(contexts) == 3
    assert contexts[0].messages[-1] == {"role": "assistant", "content": "answer 0"}

    # The cap is per run.
    _run(agent, list(responses))
    assert len(contexts) == 6


def test_guard_rounds_count_as_steps():
    async def never_happy(context):
        return "Try again."

    agent = Agent(
        config=_make_config(guards=[never_happy], max_steps=2, max_guard_rounds=5)
    )
    responses = [_make_completion_response(f"answer {i}") for i in range(2)]
    session = Session()
    result, mock = _run(agent, responses, session=session)

    assert mock.call_count == 2
    assert result.steps == 2
    assert result.stop_reason == "max_steps"
    # A rejected answer is never the result.
    assert result.content == ""
    # Feedback no step answers is not left in the history.
    assert session.messages[-1] == {"role": "assistant", "content": "answer 1"}


def test_guard_rounds_are_under_the_budget():
    async def never_happy(context):
        return "Try again."

    agent = Agent(config=_make_config(guards=[never_happy], max_total_tokens=30))
    responses = [_make_completion_response(f"answer {i}") for i in range(2)]
    session = Session()
    result, mock = _run(agent, responses, session=session)

    assert mock.call_count == 1
    assert result.stop_reason == "budget"
    assert result.content == ""
    assert session.messages[-1] == {"role": "assistant", "content": "answer 0"}


def test_first_guard_with_feedback_wins():
    called = []

    async def first(context):
        called.append("first")
        return None if called.count("first") > 1 else "Shorter."

    async def second(context):
        called.append("second")
        return None

    agent = Agent(config=_make_config(guards=[first, second]))
    _run(agent, [_make_completion_response("a"), _make_completion_response("b")])

    assert called == ["first", "first", "second"]


@pytest.mark.asyncio
async def test_guard_in_a_streamed_run():
    async def needs_total(context):
        return None if "total" in context.content else "Give the total."

    agent = Agent(
        config=_make_config(guards=[needs_total]), tools=[_make_simple_tool()]
    )
    steps = [
        [_tool_call_chunk(0, '{"a": 2, "b": 3}', id="tc_1", name="add")],
        [_text_chunk("Five.")],
        [_text_chunk("The total "), _text_chunk("is 5.")],
    ]
    mock = AsyncMock(side_effect=[_stream(s) for s in steps])
    session = Session()

    with patch("litellm.acompletion", new=mock):
        chunks = [c async for c in agent.astream("2 + 3?", session=session)]

    assert chunks == ["Five.", "The total ", "is 5."]
    assert [m["role"] for m in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]
    assert session.messages[4]["content"] == "Give the total."


@pytest.mark.asyncio
async def test_cancel_during_before_tool_unwinds_the_allowed_middleware():
    seen = []
    waiting = asyncio.Event()

    class Allow(ToolMiddleware):
        async def after_tool(self, call, result, error):
            seen.append(type(error))
            return None

    class Slow(ToolMiddleware):
        async def before_tool(self, call):
            waiting.set()
            await asyncio.sleep(5)
            return None

    agent = Agent(
        config=_make_config(middleware=[Allow(), Slow()]),
        tools=[_make_simple_tool()],
    )
    session = Session()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        task = asyncio.ensure_future(agent.arun("go", session=session))
        await waiting.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert seen == [asyncio.CancelledError]
    assert session.messages[-1]["content"] == "Tool call cancelled."


def _events(agent: Agent, steps):
    mock = AsyncMock(side_effect=[_stream(chunks) for chunks in steps])
    with patch("litellm.acompletion", new=mock):
        return list(agent.stream_events("go"))


_ADD_STEP = [_tool_call_chunk(0, '{"a": 2, "b": 3}', id="tc_1", name="add")]


def test_guard_rejection_is_an_event_after_the_rejected_text():
    async def needs_citation(context):
        return None if "[1]" in context.content else "Cite a source."

    agent = Agent(config=_make_config(guards=[needs_citation]))
    events = _events(
        agent,
        [[_text_chunk("Paris.")], [_text_chunk("Paris [1].")]],
    )

    assert [type(e).__name__ for e in events] == [
        "TextDelta",
        "GuardRejected",
        "StepFinished",
        "TextDelta",
        "StepFinished",
        "RunFinished",
    ]
    assert events[1].feedback == "Cite a source."
    assert events[-1].result.content == "Paris [1]."


def test_tool_events_show_the_model_arguments_and_the_result_it_saw():
    class Double(ToolMiddleware):
        async def before_tool(self, call):
            return {k: v * 2 for k, v in call.arguments.items()}

        async def after_tool(self, call, result, error):
            return ToolResult.success_result(f"sum {result.result}")

    agent = Agent(
        config=_make_config(middleware=[Double()]), tools=[_make_simple_tool()]
    )
    events = _events(agent, [_ADD_STEP, [_text_chunk("done")]])

    [started] = [e for e in events if isinstance(e, ToolCallStarted)]
    [finished] = [e for e in events if isinstance(e, ToolCallFinished)]
    assert started.arguments == {"a": 2, "b": 3}
    assert (finished.result, finished.is_error) == ("sum 10", False)


def test_denied_call_still_starts_and_finishes():
    class Block(ToolMiddleware):
        async def before_tool(self, call):
            return Deny("not now")

    agent = Agent(
        config=_make_config(middleware=[Block()]), tools=[_make_simple_tool()]
    )
    events = _events(agent, [_ADD_STEP, [_text_chunk("done")]])

    tool_events = [
        e for e in events if isinstance(e, (ToolCallStarted, ToolCallFinished))
    ]
    assert [type(e) for e in tool_events] == [ToolCallStarted, ToolCallFinished]
    assert (tool_events[1].result, tool_events[1].is_error) == ("Error: not now", True)


def test_every_started_call_finishes_when_middleware_raises():
    @tool(annotations=ToolAnnotations(read_only=True))
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(5)
        return "ok"

    class Broken(ToolMiddleware):
        async def before_tool(self, call):
            if call.tool_name == "add":
                await asyncio.sleep(0.01)
                raise RuntimeError("policy store down")
            return None

    agent = Agent(
        config=_make_config(middleware=[Broken()]), tools=[_make_simple_tool(), slow]
    )
    parallel = [
        _tool_call_chunk(0, '{"a": 1, "b": 2}', id="tc_1", name="add"),
        _tool_call_chunk(1, "{}", id="tc_2", name="slow"),
    ]
    events = _events(agent, [parallel])

    started = {e.call_id for e in events if isinstance(e, ToolCallStarted)}
    finished = {e.call_id: e.result for e in events if isinstance(e, ToolCallFinished)}
    assert started == {"tc_1", "tc_2"}
    assert finished == {
        "tc_1": "Error: policy store down",
        "tc_2": "Tool call cancelled.",
    }
    assert events[-1].result.stop_reason == "error"
