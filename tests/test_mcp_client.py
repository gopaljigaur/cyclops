"""Tests for the MCP client, tools, bridge and server against real MCP servers."""

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from mcp import types as mt

from cyclops.mcp import (
    MCPBridge,
    MCPClient,
    MCPServer,
    ReconnectionOptions,
    ServerStatus,
    to_tool_result,
)
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


async def _until(condition, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        await asyncio.sleep(0.05)


def _live_tasks(qualname):
    return [
        t
        for t in asyncio.all_tasks()
        if t.get_coro().__qualname__ == qualname and not t.done()
    ]


def _live_sessions():
    return _live_tasks("_Connection._run")


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


class TestResilience:
    async def test_timed_out_call_leaves_session_usable(self):
        async with MCPClient() as client:
            await client.attach("demo", COMMAND, call_timeout=2)
            slow = await _tool(client, "demo__slow").execute(seconds=10)
            add = await _tool(client, "demo__add").execute(a=1, b=2)
            assert "Timed out" in slow.error
            assert add.to_message_content() == "3"
            assert client.status("demo").state == "connected"

    async def test_idle_session_closes_and_reopens_on_next_call(self):
        async with MCPClient() as client:
            await client.attach("demo", COMMAND, idle_timeout=0.3)
            pid = _tool(client, "demo__pid")
            first = (await pid.execute()).to_message_content()
            await _until(lambda: client.status("demo").state == "disconnected")
            assert client.status("demo").error is None
            await _until(lambda: _live_sessions() == [])
            second = (await pid.execute()).to_message_content()
            assert second != first
            assert client.status("demo").state == "connected"

    async def test_busy_session_is_not_idle(self):
        async with MCPClient() as client:
            await client.attach("demo", COMMAND, idle_timeout=0.3)
            result = await _tool(client, "demo__slow").execute(seconds=1)
            assert result.to_message_content() == "done"
            assert client.status("demo").state == "connected"

    async def test_tools_follow_list_changed(self, mcp):
        added = await _tool(mcp, "demo__grow").execute(name="extra")
        assert added.success
        await _until(lambda: "demo__extra" in {t.name for t in mcp.tools()})
        result = await _tool(mcp, "demo__extra").execute()
        assert result.to_message_content() == "extra"

    async def test_clashing_refresh_keeps_the_old_tools(self, caplog):
        async with MCPClient() as client:
            await client.attach("a", COMMAND, prefix="p")
            await client.attach("b", COMMAND, prefix=False)
            before = [t.name for t in client.tools(["b"])]
            await _tool(client, "grow").execute(name="p__add")
            await _until(lambda: "Could not refresh tools" in caplog.text)
            assert [t.name for t in client.tools(["b"])] == before
            assert [t.server for t in client.tools() if t.name == "p__add"] == ["a"]

    async def test_dropped_stdio_session_reports_disconnected(self, mcp):
        crashed = await _tool(mcp, "demo__crash").execute()
        assert crashed.error == "Connection closed"
        assert mcp.status("demo") == ServerStatus("disconnected", "Connection closed")
        assert (await _tool(mcp, "demo__add").execute(a=1, b=1)).success
        assert mcp.status("demo") == ServerStatus("connected")


FAST = ReconnectionOptions(
    initial_reconnection_delay=0.05, max_reconnection_delay=0.5, max_retries=100
)


async def test_remote_server_reconnects_in_background_after_kill():
    port = _free_port()
    proc = _start("streamable-http", port)
    try:
        async with MCPClient() as client:
            await client.attach(
                "remote", url=_url("streamable-http", port), reconnection=FAST
            )
            assert client.status("remote").state == "connected"
            proc.kill()
            proc.wait()
            failed = await _tool(client, "remote__add").execute(a=1, b=1)
            assert failed.error == "Connection closed"
            await _until(lambda: client.status("remote").error != "Connection closed")
            assert client.status("remote").state == "connecting"
            proc = await asyncio.to_thread(_start, "streamable-http", port)
            await _until(lambda: client.status("remote").state == "connected")
            add = await _tool(client, "remote__add").execute(a=1, b=1)
            assert add.to_message_content() == "2"
    finally:
        proc.kill()
        proc.wait()


async def test_remote_server_fails_after_bounded_retries():
    port = _free_port()
    proc = _start("streamable-http", port)
    options = ReconnectionOptions(initial_reconnection_delay=0.01, max_retries=2)
    async with MCPClient() as client:
        await client.attach(
            "remote", url=_url("streamable-http", port), reconnection=options
        )
        proc.kill()
        proc.wait()
        await _tool(client, "remote__add").execute(a=1, b=1)
        await _until(lambda: client.status("remote").state == "failed")
        assert "TaskGroup" not in client.status("remote").error
        assert _live_tasks("_Connection._reconnect") == []


async def test_restarted_http_server_gets_a_new_session():
    port = _free_port()
    proc = _start("streamable-http", port)
    try:
        async with MCPClient() as client:
            await client.attach("remote", url=_url("streamable-http", port))
            proc.kill()
            proc.wait()
            proc = await asyncio.to_thread(_start, "streamable-http", port)
            add = await _tool(client, "remote__add").execute(a=1, b=1)
            assert add.to_message_content() == "2"
            assert client.status("remote") == ServerStatus("connected")
    finally:
        proc.kill()
        proc.wait()


async def test_reconnect_wait_is_bounded_by_call_timeout():
    port = _free_port()
    proc = _start("streamable-http", port)
    slow = ReconnectionOptions(initial_reconnection_delay=60)
    async with MCPClient() as client:
        await client.attach(
            "remote",
            url=_url("streamable-http", port),
            call_timeout=0.5,
            reconnection=slow,
        )
        proc.kill()
        proc.wait()
        add = _tool(client, "remote__add")
        await add.execute(a=1, b=1)
        await _until(lambda: client.status("remote").error != "Connection closed")
        started = time.monotonic()
        result = await add.execute(a=1, b=1)
        assert time.monotonic() - started < 5
        assert "still reconnecting" in result.error
        assert len(_live_tasks("_Connection._reconnect")) == 1


async def test_cancelled_caller_leaves_shared_reconnect_running():
    port = _free_port()
    proc = _start("streamable-http", port)
    try:
        async with MCPClient() as client:
            await client.attach(
                "remote", url=_url("streamable-http", port), reconnection=FAST
            )
            proc.kill()
            proc.wait()
            add = _tool(client, "remote__add")
            await add.execute(a=1, b=1)
            waiting = asyncio.create_task(add.execute(a=1, b=1))
            await asyncio.sleep(0.2)
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
            assert len(_live_tasks("_Connection._reconnect")) == 1
            proc = await asyncio.to_thread(_start, "streamable-http", port)
            await _until(lambda: client.status("remote").state == "connected")
    finally:
        proc.kill()
        proc.wait()


async def test_detach_cancels_background_reconnect():
    port = _free_port()
    proc = _start("streamable-http", port)
    slow = ReconnectionOptions(initial_reconnection_delay=60)
    client = MCPClient()
    await client.attach("remote", url=_url("streamable-http", port), reconnection=slow)
    proc.kill()
    proc.wait()
    await _tool(client, "remote__add").execute(a=1, b=1)
    await _until(lambda: client.status("remote").error != "Connection closed")
    assert len(_live_tasks("_Connection._reconnect")) == 1
    await asyncio.wait_for(client.aclose(), 5)
    assert _live_tasks("_Connection._reconnect") == []
    assert _live_sessions() == []


def test_reconnection_delay_grows_with_jitter_up_to_the_cap():
    options = ReconnectionOptions(
        initial_reconnection_delay=1,
        max_reconnection_delay=5,
        reconnection_delay_grow_factor=2,
    )
    for retry, cap in [(0, 1), (1, 2), (2, 4), (3, 5), (10, 5)]:
        delays = [options.delay(retry) for _ in range(200)]
        assert all(0 <= d <= cap for d in delays)
        assert max(delays) > cap / 2


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
        assert len(client._locks) == 0
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
        assert len(client._locks) == 0

    async def test_unknown_detaches_keep_no_locks(self):
        client = MCPClient()
        for i in range(10_000):
            await client.detach(f"unknown{i}")
        assert len(client._locks) == 0

    async def test_late_caller_waits_for_a_woken_waiter(self):
        client = MCPClient()
        inside = []

        async def critical(gate):
            async with client._locks.setdefault("n", asyncio.Lock()):
                inside.append(gate)
                assert len(inside) == 1
                await gate.wait()
                inside.remove(gate)

        first, second, third = asyncio.Event(), asyncio.Event(), asyncio.Event()
        a = asyncio.create_task(critical(first))
        await asyncio.sleep(0)
        b = asyncio.create_task(critical(second))
        await asyncio.sleep(0)
        first.set()
        await a
        c = asyncio.create_task(critical(third))
        await asyncio.sleep(0)
        second.set()
        third.set()
        await asyncio.gather(b, c)
        assert len(client._locks) == 0

    async def test_attach_detach_cycles_keep_no_idle_locks(self):
        client = MCPClient()
        await client.attach("kept", COMMAND)
        for i in range(3):
            await client.attach(f"s{i}", COMMAND)
            await asyncio.gather(
                client.attach(f"s{i}", COMMAND), client.detach(f"s{i}")
            )
            await client.detach(f"s{i}")
        assert client.servers == ["kept"]
        assert len(client._locks) == 0
        assert len(_live_sessions()) == 1
        await client.aclose()
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


def _start(kind: str, port: int) -> subprocess.Popen:
    proc = subprocess.Popen(
        [*COMMAND, kind, str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return proc
        time.sleep(0.1)
    proc.kill()
    raise RuntimeError(f"{kind} server did not start")


def _url(kind: str, port: int) -> str:
    return f"http://127.0.0.1:{port}{'/sse' if kind == 'sse' else '/mcp'}"


@pytest.fixture(params=["streamable-http", "sse"])
def http_server(request):
    port = _free_port()
    proc = _start(request.param, port)
    yield _url(request.param, port), "sse" if request.param == "sse" else "http"
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


class _Bearer(httpx.Auth):
    def auth_flow(self, request):
        request.headers["Authorization"] = "Bearer token"
        yield request


async def test_http_transports_use_auth(http_server):
    url, transport = http_server
    async with MCPClient() as client:
        await client.attach("remote", url=url, transport=transport, auth=_Bearer())
        header = await _tool(client, "remote__header").execute(name="authorization")
    assert header.to_message_content() == "Bearer token"


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
