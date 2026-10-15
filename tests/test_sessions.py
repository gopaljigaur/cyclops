"""Tests for sessions and session stores"""

import asyncio
import json
import logging
import threading
import time
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from cyclops import (
    Agent,
    FileSessionStore,
    InMemorySessionStore,
    Session,
)
from tests.test_agent import (
    _make_completion_response,
    _unsupported_params_error,
    _make_config,
    _make_simple_tool,
    _make_tool_call,
    _stream,
    _text_chunk,
)


def _tool_round(answer="5"):
    tc = _make_tool_call("tc_1", "add", '{"a": 2, "b": 3}')
    first = _make_completion_response(content=None, tool_calls=[tc])
    first.choices[0].message.reasoning_content = "Add them."
    return [first, _make_completion_response(answer)]


def test_session_keeps_the_transcript_and_usage():
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    session = Session()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        assert agent.run("2 + 3?", session=session) == "5"

    assert [m["role"] for m in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert session.messages[1]["reasoning_content"] == "Add them."
    assert session.messages[1]["tool_calls"][0]["id"] == "tc_1"
    assert session.messages[2]["content"] == "5"
    assert session.usage.total_tokens == 60

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("10")),
    ) as mock:
        agent.run("Double it.", session=session)

    sent = mock.call_args.kwargs["messages"]
    assert [m["role"] for m in sent] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assert session.usage.total_tokens == 90


def test_a_run_without_a_session_starts_a_fresh_transcript():
    agent = Agent(config=_make_config())
    responses = [_make_completion_response("one"), _make_completion_response("two")]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)) as mock:
        agent.run("first")
        agent.run("second")

    sent = mock.call_args.kwargs["messages"]
    assert [m["content"] for m in sent] == ["second"]


def test_session_usage_is_unknown_once_a_run_reports_none():
    agent = Agent(config=_make_config())
    session = Session()
    response = _make_completion_response("hi")
    response.usage = None

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        agent.run_with_response("hello", session=session)

    assert session.usage is None


@pytest.mark.asyncio
async def test_one_agent_serves_two_sessions_at_once():
    agent = Agent(config=_make_config())
    first, second = Session(), Session()

    async def complete(**kwargs):
        await asyncio.sleep(0)
        return _make_completion_response(f"echo {kwargs['messages'][-1]['content']}")

    with patch("litellm.acompletion", new=complete):
        await asyncio.gather(
            agent.arun("one", session=first), agent.arun("two", session=second)
        )

    assert [m["content"] for m in first.messages] == ["one", "echo one"]
    assert [m["content"] for m in second.messages] == ["two", "echo two"]


@pytest.mark.asyncio
async def test_streamed_runs_use_the_session():
    agent = Agent(config=_make_config())
    session = Session()

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_stream([_text_chunk("Hi"), _text_chunk("!")])),
    ):
        text = [t async for t in agent.astream("hello", session=session)]

    assert text == ["Hi", "!"]
    assert session.messages[-1] == {"role": "assistant", "content": "Hi!"}


def test_session_json_round_trips_with_a_version():
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    session = Session(id="chat-1")
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        agent.run("2 + 3?", session=session)

    data = session.model_dump_json()
    assert json.loads(data)["version"] == 1
    assert Session.model_validate_json(data) == session

    with pytest.raises(ValidationError):
        Session.model_validate({**json.loads(data), "version": 2})


@pytest.mark.asyncio
async def test_in_memory_store_saves_copies():
    store = InMemorySessionStore()
    session = Session(messages=[{"role": "user", "content": "hi"}])

    await store.save(session)
    saved = session.model_copy(deep=True)
    session.messages.append({"role": "assistant", "content": "hello"})
    loaded = await store.load(session.id)
    assert loaded == saved
    loaded.messages.clear()
    assert await store.load(session.id) == saved

    await store.delete(session.id)
    assert await store.load(session.id) is None
    await store.delete(session.id)


@pytest.mark.asyncio
async def test_file_store_saves_loads_and_deletes(tmp_path):
    store = FileSessionStore(tmp_path / "sessions")
    session = Session(id="chat-1", messages=[{"role": "user", "content": "hi"}])

    assert await store.load("chat-1") is None
    await store.save(session)
    path = tmp_path / "sessions" / "chat-1.jsonl"
    assert json.loads(path.read_text())["id"] == "chat-1"
    assert await store.load("chat-1") == session

    await store.delete("chat-1")
    assert not path.exists()
    await store.delete("chat-1")


@pytest.mark.asyncio
async def test_file_store_rejects_ids_that_leave_its_directory(tmp_path):
    store = FileSessionStore(tmp_path)
    with pytest.raises(ValueError):
        await store.load("../outside")
    with pytest.raises(ValueError):
        await store.save(Session(id="a/b"))


@pytest.mark.asyncio
async def test_file_store_takes_concurrent_saves(tmp_path):
    store = FileSessionStore(tmp_path)
    sessions = [
        Session(id="chat", messages=[{"role": "user", "content": str(i)}])
        for i in range(20)
    ]

    await asyncio.gather(*(store.save(s) for s in sessions))

    assert await store.load("chat") in sessions
    assert [p.name for p in tmp_path.iterdir()] == ["chat.jsonl"]


def test_raising_middleware_leaves_every_call_answered_in_the_session():
    from cyclops.core.middleware import ToolMiddleware

    class Broken(ToolMiddleware):
        async def before_tool(self, call):
            raise RuntimeError("policy store down")

    agent = Agent(
        config=_make_config(middleware=[Broken()]), tools=[_make_simple_tool()]
    )
    session = Session()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        result = agent.run_with_response("2 + 3?", session=session)

    assert result.stop_reason == "error"
    calls = [tc["id"] for m in session.messages for tc in m.get("tool_calls", [])]
    answered = [m["tool_call_id"] for m in session.messages if m["role"] == "tool"]
    assert calls == answered == ["tc_1"]
    assert session.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_cancelled_run_counts_in_the_session_usage():
    from cyclops.toolkit.tool import Tool

    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(10)
        return "done"

    agent = Agent(config=_make_config(), tools=[Tool("slow", "Slow tool", slow)])
    session = Session()
    tc = _make_tool_call("tc_1", "slow", "{}")
    response = _make_completion_response(content=None, tool_calls=[tc])

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        handle = agent.start("go", stream=False, session=session)
        await started.wait()
        handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"
    assert session.usage.total_tokens == 30
    assert session.messages[-1]["content"] == "Tool call cancelled."


def test_naive_fallback_keeps_one_attempt_in_the_session():
    agent = Agent(config=_make_config("ollama/llama2"), tools=[_make_simple_tool()])
    earlier = [
        {"role": "user", "content": "Hi."},
        {"role": "assistant", "content": "Hello."},
    ]
    session = Session(messages=list(earlier))
    responses = [
        _unsupported_params_error(),
        _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 2}}'),
        _make_completion_response("It is 3."),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        assert agent.run("1 + 2?", session=session) == "It is 3."

    assert [m["content"] for m in session.messages] == [
        "Hi.",
        "Hello.",
        "1 + 2?",
        "[Used tool: add]",
        "Tool result: 3",
        "It is 3.",
    ]


def test_guards_see_the_session_transcript():
    seen = []

    async def guard(context):
        seen.append([m["content"] for m in context.messages])
        return None

    agent = Agent(config=_make_config(guards=[guard]))
    session = Session(
        messages=[
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
        ]
    )

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("Fine.")),
    ):
        agent.run("How are you?", session=session)

    assert seen == [["Hi.", "Hello.", "How are you?", "Fine."]]


class _RecordingStore(InMemorySessionStore):
    """Records each save and addition as (method, message count)."""

    def __init__(self):
        super().__init__()
        self.calls = []

    async def save(self, session):
        self.calls.append(("save", len(session.messages)))
        await super().save(session)

    async def add_messages(self, session, messages):
        self.calls.append(("add_messages", len(messages)))
        await super().add_messages(session, messages)


@pytest.mark.asyncio
async def test_exit_durability_saves_once_when_the_run_ends():
    store = _RecordingStore()
    agent = Agent(
        config=_make_config(),
        tools=[_make_simple_tool()],
        session_store=store,
        durability="exit",
    )
    session = Session()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        await agent.arun("2 + 3?", session=session)

    assert store.calls == [("save", 4)]
    assert await store.load(session.id) == session


@pytest.mark.asyncio
async def test_step_durability_saves_each_step_and_then_adds_messages():
    store = _RecordingStore()
    agent = Agent(
        config=_make_config(), tools=[_make_simple_tool()], session_store=store
    )
    session = Session()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        await agent.arun("2 + 3?", session=session)
    # The run's first save stores the whole session, later ones only what is new.
    assert store.calls == [("save", 3), ("add_messages", 1)]
    assert await store.load(session.id) == session

    store.calls.clear()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        await agent.arun("And again?", session=session)
    assert store.calls == [("save", 7), ("add_messages", 1)]
    assert await store.load(session.id) == session
    assert session.usage.total_tokens == 120


@pytest.mark.asyncio
async def test_async_durability_saves_while_the_next_step_runs():
    release = asyncio.Event()
    completions = []

    class SlowStore(InMemorySessionStore):
        async def save(self, session):
            await release.wait()
            await super().save(session)

    responses = iter(_tool_round())

    async def complete(**kwargs):
        completions.append(len(kwargs["messages"]))
        if len(completions) == 2:
            # The first step's save is still waiting.
            release.set()
        return next(responses)

    store = SlowStore()
    agent = Agent(
        config=_make_config(), tools=[_make_simple_tool()], session_store=store
    )
    session = Session()

    with patch("litellm.acompletion", new=complete):
        assert await agent.arun("2 + 3?", session=session) == "5"

    assert completions == [1, 3]
    assert await store.load(session.id) == session


@pytest.mark.asyncio
async def test_sync_durability_saves_before_the_next_step():
    store = InMemorySessionStore()
    stored = []
    responses = iter(_tool_round())

    async def complete(**kwargs):
        stored.append(await store.load(session.id))
        return next(responses)

    agent = Agent(
        config=_make_config(),
        tools=[_make_simple_tool()],
        session_store=store,
        durability="sync",
    )
    session = Session()

    with patch("litellm.acompletion", new=complete):
        await agent.arun("2 + 3?", session=session)

    assert stored[0] is None
    assert [m["role"] for m in stored[1].messages] == ["user", "assistant", "tool"]


@pytest.mark.parametrize("durability", ["sync", "async", "exit"])
def test_a_failed_save_fails_the_run(durability):
    class BrokenStore(InMemorySessionStore):
        async def save(self, session):
            raise OSError("disk full")

    agent = Agent(
        config=_make_config(),
        tools=[_make_simple_tool()],
        session_store=BrokenStore(),
        durability=durability,
    )
    session = Session()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        result = agent.run_with_response("2 + 3?", session=session)

    assert result.stop_reason == "error"
    assert isinstance(result.error, OSError)
    # A sync save fails the run before its next step.
    assert len(session.messages) == (3 if durability == "sync" else 4)
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        with pytest.raises(OSError, match="disk full"):
            agent.run("2 + 3?", session=Session())


@pytest.mark.asyncio
async def test_a_cancelled_run_is_saved():
    from cyclops.toolkit.tool import Tool

    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(10)
        return "done"

    store = InMemorySessionStore()
    agent = Agent(
        config=_make_config(),
        tools=[Tool("slow", "Slow tool", slow)],
        session_store=store,
    )
    session = Session()
    tc = _make_tool_call("tc_1", "slow", "{}")
    response = _make_completion_response(content=None, tool_calls=[tc])

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        handle = agent.start("go", stream=False, session=session)
        await started.wait()
        handle.cancel()
        assert (await handle).stop_reason == "cancelled"

    assert await store.load(session.id) == session


def test_a_run_without_a_session_saves_nothing():
    store = _RecordingStore()
    agent = Agent(config=_make_config(), session_store=store)

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("hi")),
    ):
        agent.run("hello")

    assert store.calls == []


def test_fork_copies_the_session_under_a_new_id():
    session = Session(messages=[{"role": "user", "content": "hi"}])
    fork = session.fork()

    assert fork.id != session.id
    assert fork.messages == session.messages
    assert fork.usage == session.usage
    fork.messages[0]["content"] = "changed"
    fork.messages.append({"role": "assistant", "content": "hello"})
    assert session.messages == [{"role": "user", "content": "hi"}]
    assert session.fork().id != fork.id


def test_truncate_keeps_the_messages_before_the_index():
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    session = Session()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        agent.run("2 + 3?", session=session)
    messages = list(session.messages)

    for index in (2, -2):
        with pytest.raises(ValueError, match="tool result"):
            session.truncate(index)
        assert session.messages == messages

    session.truncate(-1)
    assert session.messages == messages[:3]
    session.truncate(1)
    assert session.messages == messages[:1]
    session.truncate(5)
    assert session.messages == messages[:1]

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("Five.")),
    ) as mock:
        agent.run("Try again.", session=session)
    sent = mock.call_args.kwargs["messages"]
    assert [m["content"] for m in sent] == ["2 + 3?", "Try again."]


@pytest.mark.asyncio
async def test_autosave_after_truncate_stores_the_whole_session_again():
    store = _RecordingStore()
    agent = Agent(
        config=_make_config(), tools=[_make_simple_tool()], session_store=store
    )
    session = Session()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        await agent.arun("2 + 3?", session=session)

    session.truncate(1)
    store.calls.clear()
    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("Five.")),
    ):
        await agent.arun("Try again.", session=session)

    assert store.calls == [("save", 3)]
    assert await store.load(session.id) == session


@pytest.mark.asyncio
async def test_file_store_round_trips_added_messages(tmp_path):
    store = FileSessionStore(tmp_path)
    agent = Agent(
        config=_make_config(), tools=[_make_simple_tool()], session_store=store
    )
    session = Session(id="chat-1")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_tool_round())):
        await agent.arun("2 + 3?", session=session)

    path = tmp_path / "chat-1.jsonl"
    assert len(path.read_text().splitlines()) == 2
    assert await store.load("chat-1") == session

    # A crash during an addition leaves a line without its newline.
    with path.open("a") as f:
        f.write('{"messages": [{"role": "us')
    assert await store.load("chat-1") == session

    await store.save(session)
    assert len(path.read_text().splitlines()) == 1
    assert await store.load("chat-1") == session


@pytest.mark.asyncio
async def test_file_store_keeps_other_line_breaks_in_messages(tmp_path):
    store = FileSessionStore(tmp_path)
    session = Session(messages=[{"role": "user", "content": "a\u2028b\x85c\u2029d\re"}])

    await store.save(session)
    await store.add_messages(session, [{"role": "assistant", "content": "f\u2028g"}])

    loaded = await store.load(session.id)
    assert [m["content"] for m in loaded.messages] == [
        "a\u2028b\x85c\u2029d\re",
        "f\u2028g",
    ]


def _three_steps():
    first = _make_tool_call("tc_1", "add", '{"a": 2, "b": 3}')
    second = _make_tool_call("tc_2", "add", '{"a": 5, "b": 5}')
    return [
        _make_completion_response(content=None, tool_calls=[first]),
        _make_completion_response(content=None, tool_calls=[second]),
        _make_completion_response("10"),
    ]


@pytest.mark.asyncio
async def test_a_save_in_progress_finishes_once_when_the_run_is_cancelled(tmp_path):
    started = threading.Event()

    class SlowStore(FileSessionStore):
        @staticmethod
        def _append(path, line):
            started.set()
            time.sleep(0.2)
            FileSessionStore._append(path, line)

    store = SlowStore(tmp_path)
    agent = Agent(
        config=_make_config(),
        tools=[_make_simple_tool()],
        session_store=store,
        durability="sync",
    )
    session = Session()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_three_steps())):
        handle = agent.start("go", stream=False, session=session)
        while not started.is_set():
            await asyncio.sleep(0.01)
        handle.cancel()
        assert (await handle).stop_reason == "cancelled"

    # Long enough for a write the run left behind to land.
    await asyncio.sleep(0.3)
    assert await store.load(session.id) == session


@pytest.mark.asyncio
async def test_the_save_after_a_failed_one_stores_the_whole_session():
    class FlakyStore(_RecordingStore):
        async def add_messages(self, session, messages):
            self.calls.append(("add_messages", len(messages)))
            raise OSError("disk full")

    store = FlakyStore()
    agent = Agent(
        config=_make_config(),
        tools=[_make_simple_tool()],
        session_store=store,
        durability="sync",
    )
    session = Session()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=_three_steps())):
        result = await agent.arun_with_response("go", session=session)

    assert isinstance(result.error, OSError)
    assert store.calls == [("save", 3), ("add_messages", 2), ("save", 5)]
    assert await store.load(session.id) == session


def test_a_failed_run_keeps_its_error_and_logs_a_failed_save(caplog):
    class BrokenStore(InMemorySessionStore):
        async def save(self, session):
            raise OSError("disk full")

    agent = Agent(config=_make_config(), session_store=BrokenStore(), durability="exit")
    session = Session(id="chat-1")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("down"))):
        with caplog.at_level(logging.ERROR, logger="cyclops.core.session"):
            with pytest.raises(RuntimeError, match="down"):
                agent.run("hi", session=session)

    [record] = caplog.records
    assert "chat-1" in record.getMessage()
    assert isinstance(record.exc_info[1], OSError)


def test_a_failed_end_save_fails_a_run_that_succeeded():
    from cyclops.core.hooks import AgentHooks

    class BrokenStore(InMemorySessionStore):
        async def save(self, session):
            raise OSError("disk full")

    seen = []

    class Hooks(AgentHooks):
        def on_run_end(self, content):
            seen.append("end")

        def on_run_error(self, error):
            seen.append(error)

    agent = Agent(
        config=_make_config(hooks=Hooks()),
        session_store=BrokenStore(),
        durability="exit",
    )

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("hi")),
    ):
        result = agent.run_with_response("hello", session=Session())

    assert result.stop_reason == "error"
    assert isinstance(result.error, OSError)
    assert seen == [result.error]


@pytest.mark.asyncio
async def test_a_naive_fallback_stores_the_whole_session_again():
    store = _RecordingStore()
    agent = Agent(
        config=_make_config("ollama/llama2"),
        tools=[_make_simple_tool()],
        session_store=store,
    )
    session = Session()
    saver_calls = []
    responses = [
        _unsupported_params_error(),
        _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 2}}'),
        _make_completion_response("It is 3."),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        with patch(
            "cyclops.core.session._SessionSaver.rewritten",
            autospec=True,
            side_effect=lambda saver: saver_calls.append(saver),
        ):
            await agent.arun("1 + 2?", session=session)

    assert len(saver_calls) == 1
    assert await store.load(session.id) == session


@pytest.mark.asyncio
async def test_a_save_after_rewritten_stores_the_whole_session():
    from cyclops.core.session import _SessionSaver

    store = _RecordingStore()
    session = Session(messages=[{"role": "user", "content": str(i)} for i in range(3)])
    saver = _SessionSaver(store, session, "sync")

    await saver.step()
    session.messages[:2] = [{"role": "user", "content": "summary"}]
    saver.rewritten()
    await saver.step()

    assert store.calls == [("save", 3), ("save", 2)]
    assert await store.load(session.id) == session
