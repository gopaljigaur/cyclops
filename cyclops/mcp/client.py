"""Async MCP client that manages connections to several MCP servers."""

import asyncio
import contextlib
import logging
from datetime import timedelta
from typing import Any, AsyncIterator, Dict, List, Literal, Optional, Sequence, Union

import anyio
import httpx
from mcp import McpError
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import CONNECTION_CLOSED, CallToolResult, Tool

from cyclops.mcp.tools import MCPClientTool

logger = logging.getLogger(__name__)

Transport = Literal["stdio", "http", "sse"]


class _Connection:
    """One server session, owned by a background task on the client's loop.

    The transport is opened and closed inside that task, so connect and close
    may be awaited from any task on the loop.
    """

    def __init__(
        self,
        name: str,
        transport: Transport,
        command: Optional[List[str]],
        url: Optional[str],
        env: Optional[Dict[str, str]],
        headers: Optional[Dict[str, str]],
        connect_timeout: float,
        call_timeout: Optional[float],
    ) -> None:
        self.name = name
        self.transport = transport
        self.command = command
        self.url = url
        self.env = env
        self.headers = headers
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout
        self.session: Optional[ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._closing = asyncio.Event()

    @contextlib.asynccontextmanager
    async def _streams(self) -> AsyncIterator[Any]:
        if self.transport == "stdio":
            assert self.command
            params = StdioServerParameters(
                command=self.command[0], args=self.command[1:], env=self.env
            )
            async with stdio_client(params) as (read, write):
                yield read, write
        elif self.transport == "http":
            assert self.url
            http = create_mcp_http_client(
                headers=self.headers, timeout=httpx.Timeout(self.connect_timeout)
            )
            async with (
                http,
                streamable_http_client(self.url, http_client=http) as (
                    read,
                    write,
                    _,
                ),
            ):
                yield read, write
        else:
            assert self.url
            async with sse_client(
                self.url, headers=self.headers, timeout=self.connect_timeout
            ) as (read, write):
                yield read, write

    async def _run(self, ready: "asyncio.Future[ClientSession]") -> None:
        try:
            async with self._streams() as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    if not ready.done():
                        ready.set_result(session)
                    await self._closing.wait()
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            else:
                logger.warning("MCP server %r disconnected: %s", self.name, e)

    async def connect(self) -> None:
        self._closing = asyncio.Event()
        ready: asyncio.Future[ClientSession] = (
            asyncio.get_running_loop().create_future()
        )
        self._task = asyncio.create_task(self._run(ready))
        try:
            self.session = await asyncio.wait_for(ready, self.connect_timeout)
        except BaseException:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
            raise

    async def close(self) -> None:
        self._closing.set()
        task, self._task, self.session = self._task, None, None
        if task is None:
            return
        _, pending = await asyncio.wait({task}, timeout=5)
        if pending:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def list_tools(self) -> List[Tool]:
        assert self.session
        tools: List[Tool] = []
        cursor: Optional[str] = None
        while True:
            page = await self.session.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                return tools

    async def call_tool(self, tool: str, arguments: Dict[str, Any]) -> CallToolResult:
        if self._task is None or self._task.done():
            await self.reconnect()
        try:
            return await self._call(tool, arguments)
        except McpError as e:
            # Dropped mid call: the tool may have run, so reconnect on the next call.
            if e.error.code == CONNECTION_CLOSED:
                await self.close()
            raise
        except (anyio.ClosedResourceError, anyio.BrokenResourceError):
            # Dropped before sending: safe to retry once.
            await self.reconnect()
            return await self._call(tool, arguments)

    async def _call(self, tool: str, arguments: Dict[str, Any]) -> CallToolResult:
        assert self.session
        timeout = timedelta(seconds=self.call_timeout) if self.call_timeout else None
        return await self.session.call_tool(
            tool, arguments, read_timeout_seconds=timeout
        )

    async def reconnect(self) -> None:
        logger.info("Reconnecting to MCP server %r", self.name)
        await self.close()
        await self.connect()


class MCPClient:
    """Async client for MCP servers, running on the caller's event loop.

    Attach servers by name, hand their tools to an agent, detach when done::

        async with MCPClient() as mcp:
            await mcp.attach("files", command=["mcp-server-files", "/tmp"])
            await mcp.attach("docs", url="https://example.com/mcp")
            agent = Agent(config, tools=mcp.tools())

    Tools are named ``<server>__<tool>`` unless another prefix is given.
    """

    def __init__(self) -> None:
        self._connections: Dict[str, _Connection] = {}
        self._tools: Dict[str, List[MCPClientTool]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def __aenter__(self) -> "MCPClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def attach(
        self,
        name: str,
        command: Optional[List[str]] = None,
        *,
        url: Optional[str] = None,
        transport: Optional[Transport] = None,
        env: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        prefix: Union[str, bool] = True,
        connect_timeout: float = 30.0,
        call_timeout: Optional[float] = None,
    ) -> List[MCPClientTool]:
        """Connect to a server and return its tools.

        Pass ``command`` for a stdio server or ``url`` for a remote one.
        ``transport`` defaults to ``"stdio"`` for a command and ``"http"``
        (streamable HTTP) for a url; use ``"sse"`` for older servers.
        ``prefix`` is ``True`` for the server name, a string, or ``False``
        for none. Attaching an existing name replaces that server.
        """
        if (command is None) == (url is None):
            raise ValueError("Pass exactly one of command or url")
        self._loop = asyncio.get_running_loop()
        await self.detach(name)
        conn = _Connection(
            name=name,
            transport=transport or ("stdio" if command else "http"),
            command=command,
            url=url,
            env=env,
            headers=headers,
            connect_timeout=connect_timeout,
            call_timeout=call_timeout,
        )
        await conn.connect()
        try:
            remote_tools = await conn.list_tools()
        except BaseException:
            await conn.close()
            raise
        label = name if prefix is True else prefix
        tools = [
            MCPClientTool(
                client=self,
                server=name,
                tool=t,
                name=f"{label}__{t.name}" if label else t.name,
            )
            for t in remote_tools
        ]
        self._connections[name] = conn
        self._tools[name] = tools
        return tools

    async def detach(self, name: str) -> None:
        """Disconnect a server and drop its tools. Unknown names are ignored."""
        conn = self._connections.pop(name, None)
        self._tools.pop(name, None)
        if conn is not None:
            await conn.close()

    async def aclose(self) -> None:
        """Disconnect every server."""
        await asyncio.gather(*[self.detach(name) for name in self.servers])

    def tools(self, servers: Optional[Sequence[str]] = None) -> List[MCPClientTool]:
        """Tools of the given servers, or of every attached server."""
        names = self.servers if servers is None else servers
        return [t for name in names for t in self._tools[name]]

    async def call_tool(
        self, server: str, tool: str, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Call a tool by its remote name, reconnecting once if the session dropped.

        May be awaited from another event loop: the call runs on the loop the
        server was attached on.
        """
        coro = self._connections[server].call_tool(tool, arguments)
        if self._loop is None or self._loop is asyncio.get_running_loop():
            return await coro
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return await asyncio.wrap_future(future)

    @property
    def servers(self) -> List[str]:
        """Names of the attached servers."""
        return list(self._connections)
