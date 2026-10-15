"""Tests for deferred tools and the search tool that loads them into a run"""

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from cyclops.core.agent import Agent
from cyclops.core.session import Session
from cyclops.core.tool_search import _ToolSearch
from cyclops.mcp import MCPBridge, MCPClient
from cyclops.toolkit import tool
from cyclops.toolkit.tool import Tool
from tests.test_agent import _make_completion_response, _make_config, _make_tool_call

COMMAND = [
    sys.executable,
    str(Path(__file__).parent / "fixtures" / "mcp_demo_server.py"),
]


@tool
def clock() -> str:
    """Tell the time."""
    return "noon"


read_file = Tool(
    "read_file", "Read a text file.", lambda path: f"text of {path}", defer_loading=True
)
send_email = Tool(
    "send_email",
    "Send an email message.",
    lambda to: f"sent to {to}",
    defer_loading=True,
)


def _call(name, **args):
    return _make_tool_call(f"tc_{name}", name, json.dumps(args))


def _sent_tools(mock):
    """The tool names each completion was sent."""
    return [
        [t["function"]["name"] for t in call.kwargs.get("tools", [])]
        for call in mock.call_args_list
    ]


def _run(agent, *steps, session=None):
    """Run with each step's tool calls, then a final answer."""
    responses = [_make_completion_response(None, tool_calls=list(s)) for s in steps]
    responses.append(_make_completion_response("done"))
    mock = AsyncMock(side_effect=responses)
    with patch("litellm.acompletion", new=mock):
        result = agent.run_with_response("go", session=session)
    assert result.error is None
    return mock


def test_without_deferred_tools_every_tool_is_sent():
    mock = _run(Agent(_make_config(), tools=[clock]))

    assert _sent_tools(mock) == [["clock"]]


def test_deferred_tools_are_sent_only_once_found():
    agent = Agent(_make_config(), tools=[clock, read_file, send_email])

    mock = _run(
        agent,
        [_call("search_tools", query="read a file")],
        [_call("read_file", path="a.txt")],
    )

    assert _sent_tools(mock) == [
        ["clock", "search_tools"],
        ["clock", "search_tools", "read_file"],
        ["clock", "search_tools", "read_file"],
    ]
    results = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert results == [
        "These tools can now be called:\n- read_file: Read a text file.",
        "text of a.txt",
    ]


def test_found_tools_last_for_the_run_only():
    agent = Agent(_make_config(), tools=[read_file, send_email])
    session = Session()

    _run(agent, [_call("search_tools", query="email")], session=session)
    mock = _run(agent, session=session)

    assert _sent_tools(mock) == [["search_tools"]]


def test_naive_mode_lists_found_tools_in_the_system_prompt():
    agent = Agent(_make_config(tool_mode="naive"), tools=[clock, read_file])
    responses = [
        _make_completion_response(
            '{"tool": "search_tools", "args": {"query": "file"}}'
        ),
        _make_completion_response("done"),
    ]
    mock = AsyncMock(side_effect=responses)
    with patch("litellm.acompletion", new=mock):
        assert agent.run("go") == "done"

    first, second = (c.kwargs["messages"][0]["content"] for c in mock.call_args_list)
    assert "- search_tools:" in first and "read_file" not in first
    assert "- read_file: Read a text file." in second
    assert "path (string)" in second


async def test_search_ranks_by_keywords_and_ignores_the_rest():
    tools = _ToolSearch([clock, read_file, send_email], {})

    assert await tools.search("weather") == "No tools match 'weather'."
    assert tools.loaded[-1].name == "search_tools"
    found = await tools.search("send email message, or file")
    assert found.splitlines()[1:] == [
        "- send_email: Send an email message.",
        "- read_file: Read a text file.",
    ]
    assert [t.name for t in tools.loaded] == [
        "clock",
        "search_tools",
        "send_email",
        "read_file",
    ]


def test_no_search_tool_without_deferred_tools():
    tools = _ToolSearch([clock], {"clock": clock})

    assert tools.loaded == [clock]
    assert tools.tools_by_name == {"clock": clock}


def test_search_tool_name_is_reserved_with_deferred_tools():
    @tool
    def search_tools(query: str) -> str:
        """My own search."""
        return ""

    @tool(defer_loading=True)
    def lookup(key: str) -> str:
        """Look up a key."""
        return key

    Agent(_make_config(), tools=[search_tools])
    with pytest.raises(ValueError, match="search_tools"):
        Agent(_make_config(), tools=[search_tools, lookup])


async def test_mcp_tools_can_be_deferred():
    async with MCPClient() as client:
        tools = await client.attach("demo", COMMAND, defer_loading=True)
        agent = Agent(_make_config(), tools=[clock, *tools])
        responses = [
            _make_completion_response(
                None, tool_calls=[_call("search_tools", query="numbers")]
            ),
            _make_completion_response(None, tool_calls=[_call("demo__add", a=2, b=3)]),
            _make_completion_response("done"),
        ]
        mock = AsyncMock(side_effect=responses)
        with patch("litellm.acompletion", new=mock):
            assert await agent.arun("go") == "done"

    assert all(t.defer_loading for t in tools)
    assert _sent_tools(mock) == [
        ["clock", "search_tools"],
        ["clock", "search_tools", "demo__add"],
        ["clock", "search_tools", "demo__add"],
    ]
    assert agent.messages[-2]["content"] == "5"


def test_bridge_keeps_deferred_mcp_tools_deferred():
    bridge = MCPBridge()
    try:
        assert all(
            t.defer_loading for t in bridge.attach("demo", COMMAND, defer_loading=True)
        )
        assert all(t.defer_loading for t in bridge.tools())
    finally:
        bridge.stop()


def test_found_tools_reach_anthropic_through_litellm():
    replies = [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "search_tools",
            "input": {"query": "file"},
        },
        {"type": "text", "text": "done"},
    ]
    bodies = []

    async def post(self, url, data=None, **kwargs):
        bodies.append(json.loads(data) if data else kwargs["json"])
        block = replies[len(bodies) - 1]
        reply = {
            "id": f"msg_{len(bodies)}",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "stop_reason": "tool_use" if block["type"] == "tool_use" else "end_turn",
            "content": [block],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        return httpx.Response(200, json=reply, request=httpx.Request("POST", url))

    config = _make_config(
        "anthropic/claude-sonnet-4-5", completion_kwargs={"api_key": "sk-test"}
    )
    with patch.object(AsyncHTTPHandler, "post", post):
        assert Agent(config, tools=[clock, read_file]).run("go") == "done"

    sent = [[t["name"] for t in body["tools"]] for body in bodies]
    assert sent == [["clock", "search_tools"], ["clock", "search_tools", "read_file"]]
    assert bodies[1]["tools"][2]["input_schema"]["required"] == ["path"]
