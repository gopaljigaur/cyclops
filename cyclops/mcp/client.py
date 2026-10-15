"""Async MCP client that manages connections to several MCP servers."""

import asyncio
import functools
import weakref
from typing import Any, Dict, List, Optional, Sequence, Union

import httpx
from mcp.types import CallToolResult, Tool

from cyclops.mcp.connection import (
    DEFAULT_CALL_TIMEOUT,
    _Connection,
    ReconnectionOptions,
    ServerStatus,
    Transport,
)
from cyclops.mcp.tools import MCPClientTool


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
        # Every holder and waiter keeps a strong reference to its lock,
        # so a name's entry stays while any of them is in it.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

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
        auth: Optional[httpx.Auth] = None,
        prefix: Union[str, bool] = True,
        connect_timeout: float = 30.0,
        call_timeout: Optional[float] = DEFAULT_CALL_TIMEOUT,
        idle_timeout: Optional[float] = None,
        reconnection: ReconnectionOptions = ReconnectionOptions(),
    ) -> List[MCPClientTool]:
        """Connect to a server and return its tools.

        Pass ``command`` for a stdio server or ``url`` for a remote one.
        ``transport`` defaults to ``"stdio"`` for a command and ``"http"``
        (streamable HTTP) for a url; use ``"sse"`` for older servers.
        ``auth`` authenticates a remote server, for example the SDK's
        ``OAuthClientProvider``. ``prefix`` is ``True`` for the server name,
        a string, or ``False`` for none. ``call_timeout`` bounds each call;
        ``idle_timeout`` closes a session no call has used for that long.
        ``None`` disables either. Attaching an existing name replaces that
        server. The server's tools must be awaited on the loop that attached it.
        """
        if (command is None) == (url is None):
            raise ValueError("Pass exactly one of command or url")
        label = name if prefix is True else prefix
        async with self._locks.setdefault(name, asyncio.Lock()):
            await self._detach(name)
            conn = _Connection(
                name=name,
                transport=transport or ("stdio" if command else "http"),
                command=command,
                url=url,
                env=env,
                headers=headers,
                auth=auth,
                connect_timeout=connect_timeout,
                call_timeout=call_timeout,
                idle_timeout=idle_timeout,
                reconnection=reconnection,
                on_tools_changed=functools.partial(self._set_tools, name, label),
            )
            session = await conn.connect()
            try:
                remote_tools = await asyncio.wait_for(
                    conn.list_tools(session), connect_timeout
                )
                tools = self._set_tools(name, label, remote_tools)
            except BaseException:
                await conn.detach()
                raise
            self._connections[name] = conn
            return tools

    def _set_tools(
        self, name: str, label: Union[str, bool], remote_tools: List[Tool]
    ) -> List[MCPClientTool]:
        tools = [
            MCPClientTool(
                client=self,
                server=name,
                tool=t,
                name=f"{label}__{t.name}" if label else t.name,
            )
            for t in remote_tools
        ]
        taken = {t.name for t in self.tools() if t.server != name}
        clashes = sorted(taken & {t.name for t in tools})
        if clashes:
            raise ValueError(f"Tool names already attached: {clashes}")
        self._tools[name] = tools
        return tools

    async def detach(self, name: str) -> None:
        """Disconnect a server and drop its tools. Unknown names are ignored."""
        async with self._locks.setdefault(name, asyncio.Lock()):
            await self._detach(name)

    async def _detach(self, name: str) -> None:
        conn = self._connections.get(name)
        if conn is not None:
            # Unregister only once closed, so an interrupted detach can be retried.
            await conn.detach()
        self._connections.pop(name, None)
        self._tools.pop(name, None)

    async def aclose(self) -> None:
        """Disconnect every server, including ones still attaching."""
        names = dict.fromkeys([*self._connections, *self._locks])
        await asyncio.gather(*[self.detach(name) for name in names])

    def tools(self, servers: Optional[Sequence[str]] = None) -> List[MCPClientTool]:
        """Tools of the given servers, or of every attached server.

        A server's tools are listed again when it sends
        ``notifications/tools/list_changed``.
        """
        names = self.servers if servers is None else servers
        return [t for name in names for t in self._tools[name]]

    def status(self, server: str) -> ServerStatus:
        """The connection state of an attached server, and its last error."""
        return self._connections[server].status

    async def call_tool(
        self, server: str, tool: str, arguments: Dict[str, Any]
    ) -> CallToolResult:
        """Call a tool by its remote name, reconnecting if the session dropped."""
        conn = self._connections.get(server)
        if conn is None:
            raise RuntimeError(f"MCP server {server!r} is not attached")
        if conn.loop is not asyncio.get_running_loop():
            raise RuntimeError(
                f"MCP server {server!r} was attached on another event loop; "
                "await its tools on that loop, or use MCPBridge from sync code"
            )
        return await conn.call_tool(tool, arguments)

    @property
    def servers(self) -> List[str]:
        """Names of the attached servers."""
        return list(self._connections)
