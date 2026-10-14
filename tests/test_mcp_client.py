"""Tests for the MCP client, tools, bridge and server against real MCP servers."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from mcp import types as mt

from cyclops.mcp import MCPBridge, MCPClient, MCPServer, to_tool_result
from cyclops.mcp.tools import _mcp_schema_to_params
from cyclops.toolkit.types import ImageContent, TextContent, ToolResult

SERVER = str(Path(__file__).parent / "fixtures" / "mcp_demo_server.py")
COMMAND = [sys.executable, SERVER]


@pytest.fixture
async def mcp():
    async with MCPClient() as client:
        await client.attach("demo", COMMAND)
        yield client


def _tool(client, name):
    return next(t for t in client.tools() if t.name == name)


def _live_sessions():
    return [
        t
        for t in asyncio.all_tasks()
        if t.get_coro().__qualname__ == "_Connection._run" and not t.done()
    ]


class TestStdio:
    async def test_tools_are_prefixed_with_server_name(self, mcp):
        names = {t.name for t in mcp.tools()}
        assert {"demo__add", "demo__fail", "demo__picture"} <= names

    async def test_call_returns_success_result(self, mcp):
        result = await _tool(mcp, "demo__add").execute(a=2, b=3)
        assert result.success
        assert result.content == [TextContent(text="5")]
        assert result.result == {"result": 5}

    async def test_annotations_are_mapped(self, mcp):
        assert _tool(mcp, "demo__add").definition.annotations.read_only is True

    async def test_input_schema_is_passed_through(self, mcp):
        tool = _tool(mcp, "demo__distance")
        schema = tool.definition.to_json_schema()
        assert schema == tool.tool.inputSchema
        assert "$defs" in schema
        result = await tool.execute(points=[{"x": 0, "y": 0}, {"x": 3, "y": 4}])
        assert result.to_message_content() == "5.0"

    async def test_is_error_maps_to_error_result(self, mcp):
        result = await _tool(mcp, "demo__fail").execute()
        assert result.success is False
        assert "boom" in result.error

    async def test_image_maps_to_content_parts(self, mcp):
        result = await _tool(mcp, "demo__picture").execute()
        assert result.success
        [block] = result.content
        assert isinstance(block, ImageContent)
        assert block.mime_type == "image/png"

    async def test_call_timeout_gives_error_result(self):
        async with MCPClient() as client:
            await client.attach("demo", COMMAND, call_timeout=0.5)
            result = await _tool(client, "demo__slow").execute(seconds=5)
        assert result.success is False
        assert "Timed out" in result.error

    async def test_reconnects_after_server_exits_mid_call(self, mcp):
        first = (await _tool(mcp, "demo__pid").execute()).to_message_content()
        crashed = await _tool(mcp, "demo__crash").execute()
        assert crashed.error == "Connection closed"
        second = (await _tool(mcp, "demo__pid").execute()).to_message_content()
        assert second != first

    async def test_reconnects_after_server_is_killed(self, mcp):
        first = (await _tool(mcp, "demo__pid").execute()).to_message_content()
        os.kill(int(first), signal.SIGKILL)
        await asyncio.sleep(0.5)
        second = await _tool(mcp, "demo__pid").execute()
        assert second.success and second.to_message_content() != first

    async def test_concurrent_calls_share_one_reconnect(self, mcp):
        first = (await _tool(mcp, "demo__pid").execute()).to_message_content()
        os.kill(int(first), signal.SIGKILL)
        await asyncio.sleep(0.5)
        pid = _tool(mcp, "demo__pid")
        results = await asyncio.gather(*[pid.execute() for _ in range(3)])
        assert len({r.to_message_content() for r in results}) == 1
        assert len(_live_sessions()) == 1

    async def test_tools_must_run_on_the_attach_loop(self, mcp):
        add = _tool(mcp, "demo__add")
        result = await asyncio.to_thread(asyncio.run, add.execute(a=1, b=1))
        assert result.success is False
        assert "another event loop" in result.error


class TestAttachDetach:
    async def test_custom_and_no_prefix(self):
        async with MCPClient() as client:
            custom = await client.attach("a", COMMAND, prefix="calc")
            plain = await client.attach("b", COMMAND, prefix=False)
            assert "calc__add" in {t.name for t in custom}
            assert "add" in {t.name for t in plain}

    async def test_tools_per_server_and_detach(self):
        async with MCPClient() as client:
            await client.attach("a", COMMAND)
            await client.attach("b", COMMAND)
            assert {t.server for t in client.tools(["b"])} == {"b"}
            await client.detach("a")
            assert client.servers == ["b"]
            assert {t.server for t in client.tools()} == {"b"}
        assert client.servers == []

    async def test_attach_and_detach_from_different_tasks(self):
        client = MCPClient()
        await asyncio.create_task(client.attach("demo", COMMAND))
        await asyncio.create_task(client.aclose())
        assert client.servers == []

    async def test_concurrent_attach_of_one_name_keeps_one_session(self):
        client = MCPClient()
        await asyncio.gather(client.attach("d", COMMAND), client.attach("d", COMMAND))
        assert client.servers == ["d"]
        assert len(_live_sessions()) == 1
        await client.aclose()
        assert _live_sessions() == []

    async def test_detach_waits_for_attach(self):
        client = MCPClient()
        attach = asyncio.create_task(client.attach("d", COMMAND))
        await asyncio.sleep(0)
        await client.detach("d")
        await attach
        await client.detach("d")
        assert client.servers == []
        assert _live_sessions() == []

    async def test_cancelled_detach_leaves_nothing_running(self):
        client = MCPClient()
        await client.attach("d", COMMAND)
        detach = asyncio.create_task(client.detach("d"))
        await asyncio.sleep(0.01)
        detach.cancel()
        await asyncio.gather(detach, return_exceptions=True)
        assert client.servers == ["d"]
        assert _live_sessions() == []
        await client.aclose()
        assert client.servers == []

    async def test_aclose_waits_for_pending_attach(self):
        client = MCPClient()
        attach = asyncio.create_task(client.attach("d", COMMAND))
        await asyncio.sleep(0)
        await client.aclose()
        await attach
        assert client.servers == []
        assert _live_sessions() == []

    async def test_tool_name_clash_across_servers_raises(self):
        async with MCPClient() as client:
            await client.attach("a", COMMAND, prefix=False)
            with pytest.raises(ValueError, match="add"):
                await client.attach("b", COMMAND, prefix=False)
            assert client.servers == ["a"]
            assert len(_live_sessions()) == 1

    async def test_detached_tool_returns_error(self):
        async with MCPClient() as client:
            await client.attach("d", COMMAND)
            add = _tool(client, "d__add")
            await client.detach("d")
            result = await add.execute(a=1, b=1)
        assert "not attached" in result.error

    async def test_connect_failure_raises(self):
        with pytest.raises(Exception):
            async with MCPClient() as client:
                await client.attach(
                    "bad", [sys.executable, "-c", "pass"], connect_timeout=5
                )

    async def test_requires_command_or_url(self):
        with pytest.raises(ValueError):
            await MCPClient().attach("x")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(params=["streamable-http", "sse"])
def http_server(request):
    port = _free_port()
    proc = subprocess.Popen(
        [*COMMAND, request.param, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                break
        time.sleep(0.1)
    else:
        proc.kill()
        pytest.fail(f"{request.param} server did not start")
    path, transport = ("/mcp", "http") if request.param != "sse" else ("/sse", "sse")
    yield f"http://127.0.0.1:{port}{path}", transport
    proc.terminate()
    proc.wait()


async def test_http_transports_send_headers(http_server):
    url, transport = http_server
    async with MCPClient() as client:
        await client.attach(
            "remote", url=url, transport=transport, headers={"X-Demo": "hi"}
        )
        add = await _tool(client, "remote__add").execute(a=1, b=1)
        header = await _tool(client, "remote__header").execute(name="x-demo")
    assert add.to_message_content() == "2"
    assert header.to_message_content() == "hi"


async def test_http_call_may_outlast_connect_timeout(http_server):
    url, transport = http_server
    async with MCPClient() as client:
        await client.attach(
            "remote", url=url, transport=transport, connect_timeout=1, call_timeout=10
        )
        result = await _tool(client, "remote__slow").execute(seconds=2)
    assert result.to_message_content() == "done"


def test_bridge_tools_work_from_another_loop():
    bridge = MCPBridge()
    try:
        tools = bridge.attach("demo", COMMAND)
        add = next(t for t in tools if t.name == "demo__add")
        assert asyncio.run(add.execute(a=4, b=5)).to_message_content() == "9"
        assert bridge.connected_names == ["demo"]
        assert [t.name for t in bridge.tools(["demo"])] == [t.name for t in tools]
    finally:
        bridge.stop()
    assert asyncio.run(add.execute(a=1, b=1)).error == "MCPBridge is stopped"


def test_to_tool_result_mixed_content():
    result = to_tool_result(
        mt.CallToolResult(
            content=[
                mt.TextContent(type="text", text="see"),
                mt.ImageContent(type="image", data="AAAA", mimeType="image/png"),
            ]
        )
    )
    assert result.content == [
        TextContent(text="see"),
        ImageContent(data="AAAA", mime_type="image/png"),
    ]


def test_to_tool_result_error_keeps_text():
    result = to_tool_result(
        mt.CallToolResult(
            content=[mt.TextContent(type="text", text="bad input")], isError=True
        )
    )
    assert result == ToolResult.error_result("bad input")


class TestMCPSchemaToParams:
    def test_empty_schema_returns_empty_params(self):
        assert _mcp_schema_to_params({}) == {}

    def test_basic_string_property(self):
        schema = {"properties": {"name": {"type": "string", "description": "A name"}}}
        params = _mcp_schema_to_params(schema)
        assert "name" in params
        assert params["name"].type == "string"
        assert params["name"].description == "A name"

    def test_required_fields_marked_correctly(self):
        schema = {
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        params = _mcp_schema_to_params(schema)
        assert params["a"].required is True
        assert params["b"].required is False

    def test_all_json_types_preserved(self):
        schema = {
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "arr": {"type": "array"},
                "obj": {"type": "object"},
            }
        }
        params = _mcp_schema_to_params(schema)
        assert params["s"].type == "string"
        assert params["i"].type == "integer"
        assert params["n"].type == "number"
        assert params["b"].type == "boolean"
        assert params["arr"].type == "array"
        assert params["obj"].type == "object"

    def test_unknown_type_falls_back_to_string(self):
        schema = {"properties": {"x": {"type": "exotic_type"}}}
        params = _mcp_schema_to_params(schema)
        assert params["x"].type == "string"

    def test_missing_type_falls_back_to_string(self):
        schema = {"properties": {"x": {"description": "no type field"}}}
        params = _mcp_schema_to_params(schema)
        assert params["x"].type == "string"

    def test_union_type_uses_first_non_null(self):
        schema = {"properties": {"x": {"type": ["null", "integer"]}}}
        assert _mcp_schema_to_params(schema)["x"].type == "integer"


def test_to_tool_result_structured_only():
    result = to_tool_result(mt.CallToolResult(content=[], structuredContent={"a": 1}))
    assert result.result == {"a": 1}
    assert result.to_message_content() == '{"a": 1}'


async def test_server_serves_tool_results():
    server = MCPServer(load_plugins=False)

    def explode() -> str:
        raise RuntimeError("nope")

    server.add_function_tool("explode", "Fails", explode)
    server.add_function_tool("echo", "Echo", lambda text: text)
    server.add_function_tool(
        "refuse", "Fails as a result", lambda: ToolResult.error_result("denied")
    )
    server.add_function_tool(
        "photo",
        "An image",
        lambda: ToolResult(
            content=[
                TextContent(text="a pixel"),
                ImageContent(data="AAAA", mime_type="image/png"),
            ]
        ),
    )
    server.add_function_tool("data", "A dict", lambda: {"a": 1})

    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server.server) as session:
        failed = await session.call_tool("explode", {})
        ok = await session.call_tool("echo", {"text": "hi"})
        refused = await session.call_tool("refuse", {})
        photo = await session.call_tool("photo", {})
        data = await session.call_tool("data", {})
    assert failed.isError is True
    assert "nope" in failed.content[0].text
    assert ok.isError is False
    assert ok.content[0].text == "hi"
    assert refused.isError is True
    assert refused.content[0].text == "denied"
    assert [c.type for c in photo.content] == ["text", "image"]
    assert photo.content[1].mimeType == "image/png"
    assert data.content[0].text == '{"a": 1}'
