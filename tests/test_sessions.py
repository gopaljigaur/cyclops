"""Tests for sessions and session stores"""

import asyncio
import json
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
    path = tmp_path / "sessions" / "chat-1.json"
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
    assert [p.name for p in tmp_path.iterdir()] == ["chat.json"]


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
