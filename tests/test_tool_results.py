"""Tests for tool results past their limit: stored, sent as a first page, read on"""

import asyncio
import inspect
import json
import re
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


def _agent(*tools, **config) -> Agent:
    return Agent(AgentConfig(model=MODEL, **config), tools=list(tools or [dump]))


def test_a_long_result_is_stored_and_read_to_the_end():
    agent = _agent(max_tool_output_chars=20)
    session = Session()

    with _replies(_call("dump"), _read(20), _read(40), "Done.") as mock:
        result = agent.run_with_response("Dump it.", session=session)

    first, second, last = _tool_messages(session)
    handle = re.findall(r'handle="(\w+)"', first)[0]
    assert first == (
        f"{TEXT[:20]}\n[Characters 0 to 20 of 50. Read on with "
        f'read_tool_result(handle="{handle}", offset=20).]'
    )
    assert second == (
        f"{TEXT[20:40]}\n[Characters 20 to 40 of 50. Read on with "
        f'read_tool_result(handle="{handle}", offset=40).]'
    )
    assert last == f"{TEXT[40:]}\n[Characters 40 to 50 of 50, the end of the result.]"
    # The reader is sent once a result is stored.
    assert _sent_tools(mock) == [["dump"]] + [["dump", "read_tool_result"]] * 3
    assert result.tool_calls[0].result == first


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


def test_short_results_are_sent_whole_and_add_no_reader():
    @tool
    def short() -> str:
        """Say little."""
        return "short"

    agent = _agent(short, max_tool_output_chars=20)
    session = Session()

    with _replies(_call("short"), _call("short"), "Done.") as mock:
        agent.run("Go.", session=session)

    assert _tool_messages(session) == ["short", "short"]
    assert _sent_tools(mock) == [["short"]] * 3


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
    assert strict_page.startswith(f"{TEXT[:20]}\n[Characters 0 to 20 of 50.")
    assert loose_whole == TEXT
    assert dump_page.startswith(f"{TEXT[:40]}\n[Characters 0 to 40 of 50.")


def test_a_tool_limit_works_without_an_agent_limit():
    @tool(max_output_chars=20)
    def strict() -> str:
        """Dump strictly."""
        return TEXT

    agent = _agent(strict, dump)
    session = Session()

    with _replies(_call("strict"), _call("dump"), "Done."):
        agent.run("Go.", session=session)

    page, whole = _tool_messages(session)
    assert page.startswith(f"{TEXT[:20]}\n[Characters 0 to 20 of 50.")
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


def test_a_session_keeps_the_first_page_and_the_next_run_has_no_reader():
    sessions = InMemorySessionStore()
    agent = Agent(
        AgentConfig(model=MODEL, max_tool_output_chars=20),
        tools=[dump],
        session_store=sessions,
    )
    session = Session()

    with _replies(_call("dump"), "Done."):
        agent.run("Dump it.", session=session)
    with _replies("Still done.") as mock:
        agent.run("And now?", session=session)

    saved = asyncio.run(sessions.load(session.id))
    (page,) = _tool_messages(saved)
    assert page.startswith(f"{TEXT[:20]}\n[Characters 0 to 20 of 50.")
    assert TEXT not in json.dumps(saved.messages)
    # The result was freed with the run that stored it.
    assert _sent_tools(mock) == [["dump"]]


def test_compaction_counts_the_first_page_not_the_whole_result():
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
    with _replies(_call("dump"), _read(20), "Done."):
        result = denied.run_with_response("Dump it.", session=session)

    assert _tool_messages(session)[1] == "Error: Tool 'read_tool_result' is not permitted"
    assert result.denied_tool_calls[0].name == "read_tool_result"

    allowed = _agent(
        max_tool_output_chars=20,
        middleware=[
            ToolPermissions(allow=["dump", "read_tool_result"], default="deny")
        ],
    )
    session = Session()
    with _replies(_call("dump"), _read(20), "Done."):
        allowed.run("Dump it.", session=session)

    assert _tool_messages(session)[1].startswith(TEXT[20:40])


def test_a_deferred_tool_result_is_stored_and_read():
    @tool(defer_loading=True, max_output_chars=20)
    def logs() -> str:
        """Fetch the server logs."""
        return TEXT

    agent = _agent(dump, logs)
    session = Session()
    replies = [_call("search_tools", query="server logs"), _call("logs"), _read(20)]

    with _replies(*replies, "Done.") as mock:
        agent.run("Read the logs.", session=session)

    assert _tool_messages(session)[2].startswith(TEXT[20:40])
    assert _sent_tools(mock) == [
        ["dump", "search_tools"],
        ["dump", "search_tools", "logs"],
        ["dump", "search_tools", "logs", "read_tool_result"],
        ["dump", "search_tools", "logs", "read_tool_result"],
    ]


def test_naive_mode_lists_the_reader_once_a_result_is_stored():
    agent = _agent(max_tool_output_chars=20, tool_mode="naive")
    dump_call = json.dumps({"tool": "dump", "args": {}})

    with _replies(dump_call, "Done.") as mock:
        agent.run("Dump it.")

    first, second = (c.kwargs["messages"][0]["content"] for c in mock.call_args_list)
    assert "read_tool_result" not in first
    assert "- read_tool_result:" in second


async def test_events_carry_the_first_page_and_hooks_see_the_whole_result():
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
    assert finished.result.startswith(f"{TEXT[:20]}\n[Characters 0 to 20 of 50.")


def test_images_stay_with_the_first_page():
    @tool
    def shot() -> ToolResult:
        """Take a screenshot."""
        image = ImageContent(data="aGk=", mime_type="image/png")
        return ToolResult(content=[TextContent(text=TEXT), image])

    agent = _agent(shot, max_tool_output_chars=20)
    session = Session()

    with _replies(_call("shot"), "Done."):
        agent.run("Look.", session=session)

    assert session.messages[2]["content"].startswith(f"{TEXT[:20]}\n[Characters")
    # The model takes no images in a tool message, so they follow in a user one.
    images = session.messages[3]["content"]
    assert images[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aGk="},
    }
