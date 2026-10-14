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
from mcp.types import CallToolResult, ImageContent, TextContent

from cyclops.mcp import MCPBridge, MCPClient, MCPServer, to_tool_result
from cyclops.toolkit.types import ToolResult

SERVER = str(Path(__file__).parent / "fixtures" / "mcp_demo_server.py")
COMMAND = [sys.executable, SERVER]


@pytest.fixture
async def mcp():
    async with MCPClient() as client:
        await client.attach("demo", COMMAND)
        yield client


def _tool(client, name):
    return next(t for t in client.tools() if t.name == name)


class TestStdio:
    async def test_tools_are_prefixed_with_server_name(self, mcp):
        names = {t.name for t in mcp.tools()}
        assert {"demo__add", "demo__fail", "demo__picture"} <= names

    async def test_call_returns_success_result(self, mcp):
        result = await _tool(mcp, "demo__add").execute(a=2, b=3)
        assert result == ToolResult.success_result("5")

    async def test_input_schema_is_passed_through(self, mcp):
        tool = _tool(mcp, "demo__distance")
        schema = tool.definition.to_json_schema()
        assert schema == tool.tool.inputSchema
        assert "$defs" in schema
        result = await tool.execute(points=[{"x": 0, "y": 0}, {"x": 3, "y": 4}])
        assert result.result == "5.0"

    async def test_is_error_maps_to_error_result(self, mcp):
        result = await _tool(mcp, "demo__fail").execute()
        assert result.success is False
        assert "boom" in result.error

    async def test_image_maps_to_content_parts(self, mcp):
        result = await _tool(mcp, "demo__picture").execute()
        assert result.success
        [part] = result.result
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/png;base64,")

    async def test_call_timeout_gives_error_result(self):
        async with MCPClient() as client:
            await client.attach("demo", COMMAND, call_timeout=0.5)
            result = await _tool(client, "demo__slow").execute(seconds=5)
        assert result.success is False
        assert "Timed out" in result.error

    async def test_reconnects_after_server_exits_mid_call(self, mcp):
        first = (await _tool(mcp, "demo__pid").execute()).result
        crashed = await _tool(mcp, "demo__crash").execute()
        assert crashed.error == "Connection closed"
        second = (await _tool(mcp, "demo__pid").execute()).result
        assert second != first

    async def test_reconnects_after_server_is_killed(self, mcp):
        first = (await _tool(mcp, "demo__pid").execute()).result
        os.kill(int(first), signal.SIGKILL)
        await asyncio.sleep(0.5)
        second = await _tool(mcp, "demo__pid").execute()
        assert second.success and second.result != first


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
    assert add.result == "2"
    assert header.result == "hi"


def test_bridge_tools_work_from_another_loop():
    bridge = MCPBridge()
    try:
        tools = bridge.attach("demo", COMMAND)
        add = next(t for t in tools if t.name == "demo__add")
        assert asyncio.run(add.execute(a=4, b=5)).result == "9"
        assert bridge.connected_names == ["demo"]
    finally:
        bridge.stop()


def test_to_tool_result_mixed_content():
    result = to_tool_result(
        CallToolResult(
            content=[
                TextContent(type="text", text="see"),
                ImageContent(type="image", data="AAAA", mimeType="image/png"),
            ]
        )
    )
    assert result.result == [
        {"type": "text", "text": "see"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
    ]


async def test_server_returns_is_error_for_failures():
    server = MCPServer(load_plugins=False)

    def explode() -> str:
        raise RuntimeError("nope")

    server.add_function_tool("explode", "Fails", explode)
    server.add_function_tool("echo", "Echo", lambda text: text)

    from mcp.shared.memory import create_connected_server_and_client_session

    async with create_connected_server_and_client_session(server.server) as session:
        failed = await session.call_tool("explode", {})
        ok = await session.call_tool("echo", {"text": "hi"})
    assert failed.isError is True
    assert "nope" in failed.content[0].text
    assert ok.isError is False
    assert ok.content[0].text == "hi"
