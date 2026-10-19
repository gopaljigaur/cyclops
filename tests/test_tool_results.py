"""Tests for tool results past their limit: stored, previewed, read in pages"""

import asyncio
import inspect
import json
import re
from contextlib import aclosing
from typing import List
from unittest.mock import AsyncMock, patch

import litellm
import pytest

from cyclops import (
    Agent,
    AgentConfig,
    AgentHooks,
    Compaction,
    InMemorySessionStore,
    InMemoryToolResultStore,
    Session,
    ToolMiddleware,
    ToolPermissions,
    ToolResult,
    tool,
)
from cyclops.core.events import ToolCallFinished
from cyclops.core.tool_results import _ToolResults
from tests.test_agent import _stream, _tool_call_chunk
from tests.test_fallback import _overloaded
from cyclops.toolkit.types import ImageContent, TextContent
from tests.test_compaction import MODEL

TEXT = "0123456789" * 5


@tool
def dump() -> str:
    """Dump a log."""
    return TEXT


def _call(name: str, **args):
    return {
        "mock_response": "",
        "mock_tool_calls": [
            {
                "id": f"call_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _read(offset: int, **args):
    """A reply that reads the result stored last, from offset."""

    def reply(kwargs):
        sent = json.dumps(kwargs["messages"])
        handle = re.findall(r'handle=\\"(\w+)\\"', sent)[-1]
        return _call("read_tool_result", handle=handle, offset=offset, **args)

    return reply


def _replies(*replies):
    """Patch litellm.acompletion to answer each call with the next reply.

    A reply is the text of a mock_response, a dict of mock arguments, an
    exception to raise, or a function of the call's arguments to one of
    those or to an awaitable of one.
    """
    real = litellm.acompletion
    queue = list(replies)

    async def complete(**kwargs):
        reply = queue.pop(0)
        if callable(reply):
            reply = reply(kwargs)
        if inspect.isawaitable(reply):
            reply = await reply
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            reply = {"mock_response": reply}
        return await real(**kwargs, **reply)

    return patch("litellm.acompletion", new=AsyncMock(side_effect=complete))


def _sent_tools(mock) -> List[List[str]]:
    return [
        [t["function"]["name"] for t in call.kwargs.get("tools", [])]
        for call in mock.call_args_list
    ]


def _tool_messages(session: Session) -> List[str]:
    return [m["content"] for m in session.messages if m["role"] == "tool"]


def _preview(limit: int) -> str:
    """How the preview of TEXT starts, up to the note's handle, for limit."""
    head = limit // 2
    tail = 50 - (limit - head)
    return f"{TEXT[:head]}\n[Characters {head} to {tail} of 50 omitted."


def _agent(*tools, **config) -> Agent:
    return Agent(AgentConfig(model=MODEL, **config), tools=list(tools or [dump]))


def test_a_long_result_is_stored_and_read_to_the_end():
    agent = _agent(max_tool_output_chars=20)
    session = Session()

    with _replies(_call("dump"), _read(10), _read(30), "Done.") as mock:
        result = agent.run_with_response("Dump it.", session=session)

    preview, middle, last = _tool_messages(session)
    handle = re.findall(r'handle="(\w+)"', preview)[0]
    # The start and the end, around a note of what is left out.
    assert preview == (
        f"{TEXT[:10]}\n[Characters 10 to 40 of 50 omitted. Read them with "
        f'read_tool_result(handle="{handle}", offset=10).]\n{TEXT[40:]}'
    )
    assert middle == (
        f"{TEXT[10:30]}\n[Characters 10 to 30 of 50. Read on with "
        f'read_tool_result(handle="{handle}", offset=30).]'
    )
    # Reading on stops where the preview's end part starts.
    assert last == (
        f"{TEXT[30:40]}\n[Characters 30 to 40 of 50. The rest, from 40, "
        "was in the preview.]"
    )
    # The reader is sent from the first step, so the tools never change.
    assert _sent_tools(mock) == [["dump", "read_tool_result"]] * 4
    assert result.tool_calls[0].result == preview


@pytest.mark.parametrize("size, limit", [(50, 20), (21, 20), (1000, 100)])
async def test_reading_on_from_the_preview_covers_the_omitted_range_once(size, limit):
    text = "".join(chr(ord("a") + i % 26) for i in range(size))
    results = _ToolResults(None)
    preview = await results.preview(text, limit)
    handle = re.findall(r'handle="(\w+)"', preview)[0]
    offset = int(re.findall(r"offset=(\d+)", preview)[0])

    read = []
    while True:
        page = await results.read(handle, offset)
        body, note = page.rsplit("\n", 1)
        read.append(body)
        found = re.findall(r"offset=(\d+)", note)
        if not found:
            break
        offset = int(found[0])

    # The preview's start, the pages and the preview's end make the text once.
    head, tail = limit // 2, size - (limit - limit // 2)
    assert text[:head] + "".join(read) + text[tail:] == text


def test_a_page_is_at_most_the_limit_of_the_stored_result():
    agent = _agent(max_tool_output_chars=20)
    session = Session()

    with _replies(_call("dump"), _read(5, limit=10), _read(15, limit=500), "Done."):
        agent.run("Dump it.", session=session)

    _, short, capped = _tool_messages(session)
    assert short.startswith(f"{TEXT[5:15]}\n[Characters 5 to 15 of 50.")
    assert capped.startswith(f"{TEXT[15:35]}\n[Characters 15 to 35 of 50.")


def test_reading_an_unknown_handle_or_past_the_end_is_an_error():
    agent = _agent(max_tool_output_chars=20)
    session = Session()
    unknown = _call("read_tool_result", handle="nope")

    with _replies(_call("dump"), unknown, _read(50), "Done."):
        agent.run("Dump it.", session=session)

    _, missing, past = _tool_messages(session)
    assert missing == (
        "Error: read_tool_result failed: No stored tool result has the handle 'nope'"
    )
    assert past == "Error: read_tool_result failed: offset 50 is past the end, at 50"


def test_short_results_are_sent_whole():
    @tool
    def short() -> str:
        """Say little."""
        return "short"

    agent = _agent(short, max_tool_output_chars=20)
    session = Session()

    with _replies(_call("short"), _call("short"), "Done.") as mock:
        agent.run("Go.", session=session)

    assert _tool_messages(session) == ["short", "short"]
    assert _sent_tools(mock) == [["short", "read_tool_result"]] * 3


def test_no_reader_without_a_limit():
    agent = _agent()

    with _replies(_call("dump"), "Done.") as mock:
        agent.run("Dump it.")

    assert _sent_tools(mock) == [["dump"]] * 2


def test_a_tool_limit_alone_sends_the_reader():
    @tool(max_output_chars=20)
    def strict() -> str:
        """Dump strictly."""
        return TEXT

    with _replies("Hi.") as mock:
        _agent(strict, dump).run("Hi.")

    assert _sent_tools(mock) == [["strict", "dump", "read_tool_result"]]


def test_a_tool_limit_overrides_the_agent_limit():
    @tool(max_output_chars=20)
    def strict() -> str:
        """Dump strictly."""
        return TEXT

    @tool(max_output_chars=100)
    def loose() -> str:
        """Dump loosely."""
        return TEXT

    agent = _agent(strict, loose, dump, max_tool_output_chars=40)
    session = Session()

    with _replies(_call("strict"), _call("loose"), _call("dump"), "Done."):
        agent.run("Go.", session=session)

    strict_page, loose_whole, dump_page = _tool_messages(session)
    assert strict_page.startswith(_preview(20))
    assert loose_whole == TEXT
    assert dump_page.startswith(_preview(40))


def test_a_tool_limit_works_without_an_agent_limit():
    @tool(max_output_chars=20)
    def strict() -> str:
        """Dump strictly."""
        return TEXT

    agent = _agent(strict, dump)
    session = Session()

    with _replies(_call("strict"), _call("dump"), "Done."):
        agent.run("Go.", session=session)

    preview, whole = _tool_messages(session)
    assert preview.startswith(_preview(20))
    assert whole == TEXT


def test_a_tool_limit_must_be_positive():
    with pytest.raises(ValueError, match="max_output_chars"):
        tool(max_output_chars=0)(lambda: "")


def test_the_reader_name_is_reserved_when_a_limit_is_set():
    @tool
    def read_tool_result() -> str:
        """My own reader."""
        return ""

    with pytest.raises(ValueError, match="read_tool_result"):
        _agent(read_tool_result, max_tool_output_chars=20)
    # Without a limit nothing is stored, so the name is free.
    _agent(read_tool_result)


class _RecordingStore(InMemoryToolResultStore):
    def __init__(self) -> None:
        super().__init__()
        self.stored: List[str] = []

    async def put(self, handle: str, text: str) -> None:
        self.stored.append(text)
        await super().put(handle, text)


def test_the_store_is_emptied_when_the_run_ends():
    store = _RecordingStore()
    agent = Agent(
        AgentConfig(model=MODEL, max_tool_output_chars=20),
        tools=[dump],
        tool_result_store=store,
    )

    with _replies(_call("dump"), _call("dump"), "Done."):
        agent.run("Go.")

    assert store.stored == [TEXT, TEXT]
    assert store._results == {}


def test_the_store_is_emptied_when_the_run_fails():
    store = _RecordingStore()
    agent = Agent(
        AgentConfig(model=MODEL, max_tool_output_chars=20, num_retries=0),
        tools=[dump],
        tool_result_store=store,
    )

    with _replies(_call("dump"), RuntimeError("down")):
        result = agent.run_with_response("Go.")

    assert result.stop_reason == "error"
    assert store.stored == [TEXT]
    assert store._results == {}


async def test_the_store_is_emptied_when_the_run_is_cancelled():
    store = _RecordingStore()
    agent = Agent(
        AgentConfig(model=MODEL, max_tool_output_chars=20),
        tools=[dump],
        tool_result_store=store,
    )
    stored = asyncio.Event()

    async def stall(kwargs):
        stored.set()
        await asyncio.Event().wait()

    with _replies(_call("dump"), stall):
        handle = agent.start("Go.", stream=False)
        await stored.wait()
        handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"
    assert store.stored == [TEXT]
    assert store._results == {}


def test_a_session_keeps_the_preview_and_the_next_run_cannot_read_on():
    sessions = InMemorySessionStore()
    agent = Agent(
        AgentConfig(model=MODEL, max_tool_output_chars=20),
        tools=[dump],
        session_store=sessions,
    )
    session = Session()

    with _replies(_call("dump"), "Done."):
        agent.run("Dump it.", session=session)
    with _replies(_read(10), "Gone."):
        agent.run("Read the middle.", session=session)

    saved = asyncio.run(sessions.load(session.id))
    preview, later = _tool_messages(saved)
    assert preview.startswith(_preview(20))
    assert preview.endswith(TEXT[40:])
    assert TEXT not in json.dumps(saved.messages)
    # The result was freed with the run that stored it.
    assert later.startswith(
        "Error: read_tool_result failed: No stored tool result has the handle"
    )


def test_compaction_counts_the_preview_not_the_whole_result():
    @tool
    def huge() -> str:
        """Dump a huge log."""
        return "word " * 2000

    agent = _agent(huge, compaction=Compaction(), max_tool_output_chars=100)

    # Whole, the result would pass the 400 token limit and be summarized.
    with _replies(_call("huge"), "Done.") as mock:
        result = agent.run_with_response("Go.")

    assert result.error is None
    assert mock.call_count == 2


def test_permissions_apply_to_the_reader():
    denied = _agent(
        max_tool_output_chars=20,
        middleware=[ToolPermissions(allow=["dump"], default="deny")],
    )
    session = Session()
    with _replies(_call("dump"), _read(10), "Done."):
        result = denied.run_with_response("Dump it.", session=session)

    assert (
        _tool_messages(session)[1] == "Error: Tool 'read_tool_result' is not permitted"
    )
    assert result.denied_tool_calls[0].name == "read_tool_result"

    allowed = _agent(
        max_tool_output_chars=20,
        middleware=[
            ToolPermissions(allow=["dump", "read_tool_result"], default="deny")
        ],
    )
    session = Session()
    with _replies(_call("dump"), _read(10), "Done."):
        allowed.run("Dump it.", session=session)

    assert _tool_messages(session)[1].startswith(TEXT[10:30])


def test_a_deferred_tool_result_is_stored_and_read():
    @tool(defer_loading=True, max_output_chars=20)
    def logs() -> str:
        """Fetch the server logs."""
        return TEXT

    agent = _agent(dump, logs)
    session = Session()
    replies = [_call("search_tools", query="server logs"), _call("logs"), _read(10)]

    with _replies(*replies, "Done.") as mock:
        agent.run("Read the logs.", session=session)

    assert _tool_messages(session)[2].startswith(TEXT[10:30])
    assert _sent_tools(mock) == [
        ["dump", "read_tool_result", "search_tools"],
        *[["dump", "read_tool_result", "search_tools", "logs"]] * 3,
    ]


def test_naive_mode_lists_the_reader():
    agent = _agent(max_tool_output_chars=20, tool_mode="naive")
    dump_call = json.dumps({"tool": "dump", "args": {}})
    session = Session()

    with _replies(dump_call, "Done.") as mock:
        agent.run("Dump it.", session=session)

    assert (
        "- read_tool_result:" in mock.call_args_list[0].kwargs["messages"][0]["content"]
    )
    assert _preview(20) in session.messages[2]["content"]


async def test_events_carry_the_preview_and_hooks_see_the_whole_result():
    class Hooks(AgentHooks):
        def __init__(self) -> None:
            self.results: List[str] = []

        def on_tool_end(self, tool_name, args, result):
            self.results.append(result)

    class Seen(ToolMiddleware):
        def __init__(self) -> None:
            self.results: List[str] = []

        async def after_tool(self, call, result, error):
            self.results.append(ToolResult.message_text(result.to_message_content()))

    hooks, seen = Hooks(), Seen()
    agent = _agent(max_tool_output_chars=20, hooks=hooks, middleware=[seen])

    with _replies(_call("dump"), "Done."):
        events = [e async for e in agent.start("Dump it.", stream=False)]

    assert hooks.results == [TEXT]
    assert seen.results == [TEXT]
    (finished,) = [e for e in events if isinstance(e, ToolCallFinished)]
    assert finished.result.startswith(_preview(20))


def test_images_stay_with_the_preview():
    @tool
    def shot() -> ToolResult:
        """Take a screenshot."""
        image = ImageContent(data="aGk=", mime_type="image/png")
        return ToolResult(content=[TextContent(text=TEXT), image])

    agent = _agent(shot, max_tool_output_chars=20)
    session = Session()

    with _replies(_call("shot"), "Done."):
        agent.run("Look.", session=session)

    assert session.messages[2]["content"].startswith(_preview(20))
    # The model takes no images in a tool message, so they follow in a user one.
    images = session.messages[3]["content"]
    assert images[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aGk="},
    }


async def test_a_run_cannot_read_another_runs_results_in_a_shared_store():
    store = InMemoryToolResultStore()
    first, second = (
        Agent(
            AgentConfig(model=MODEL, max_tool_output_chars=20),
            tools=[dump],
            tool_result_store=store,
        )
        for _ in range(2)
    )
    stored, release = asyncio.Event(), asyncio.Event()
    handles: List[str] = []

    async def wait(kwargs):
        handles.extend(re.findall(r'handle=\\"(\w+)\\"', json.dumps(kwargs)))
        stored.set()
        await release.wait()
        return "Done."

    def read_first(kwargs):
        return _call("read_tool_result", handle=handles[0], offset=10)

    replies = {"A": [_call("dump"), wait], "B": [read_first, "Done."]}
    real = litellm.acompletion

    async def complete(**kwargs):
        reply = replies[kwargs["messages"][0]["content"]].pop(0)
        if callable(reply):
            reply = reply(kwargs)
        if inspect.isawaitable(reply):
            reply = await reply
        if isinstance(reply, str):
            reply = {"mock_response": reply}
        return await real(**kwargs, **reply)

    session = Session()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=complete)):
        running = first.start("A", stream=False)
        await stored.wait()
        assert len(store._results) == 1
        await second.arun("B", session=session)
        release.set()
        await running

    assert _tool_messages(session) == [
        "Error: read_tool_result failed: "
        f"No stored tool result has the handle '{handles[0]}'"
    ]
    assert store._results == {}


async def test_an_abandoned_event_stream_empties_the_store():
    store = _RecordingStore()
    agent = Agent(
        AgentConfig(model="gpt-4o-mini", max_tool_output_chars=20),
        tools=[dump],
        tool_result_store=store,
    )

    async def complete(**kwargs):
        if len(kwargs["messages"]) == 1:
            return _stream([_tool_call_chunk(0, "{}", id="tc_1", name="dump")])
        await asyncio.Event().wait()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=complete)):
        async with aclosing(agent.astream_events("Dump it.")) as events:
            async for event in events:
                if isinstance(event, ToolCallFinished):
                    break

    assert store.stored == [TEXT]
    assert store._results == {}


def test_a_fallback_model_gets_the_reader():
    agent = _agent(max_tool_output_chars=20, fallback_models=["gpt-4o"], num_retries=0)
    session = Session()

    with _replies(_call("dump"), _overloaded(), _read(10), "Done.") as mock:
        agent.run("Dump it.", session=session)

    fallback = mock.call_args_list[2].kwargs
    assert fallback["model"] == "gpt-4o"
    assert "read_tool_result" in _sent_tools(mock)[2]
    assert _tool_messages(session)[1].startswith(TEXT[10:30])
