"""Async MCP client that manages connections to several MCP servers."""

import asyncio
import contextlib
import functools
import logging
import random
from dataclasses import dataclass
from datetime import timedelta
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Union,
)

import anyio
import httpx
from mcp import McpError
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.types import (
    CONNECTION_CLOSED,
    CallToolResult,
    ErrorData,
    ServerNotification,
    Tool,
    ToolListChangedNotification,
)

from cyclops.mcp.tools import MCPClientTool

logger = logging.getLogger(__name__)

Transport = Literal["stdio", "http", "sse"]
State = Literal["connecting", "connected", "disconnected", "failed"]

# The TypeScript SDK's default request timeout.
DEFAULT_CALL_TIMEOUT = 60.0


@dataclass(frozen=True)
class ReconnectionOptions:
    """Exponential backoff with full jitter between reconnect attempts.

    Names follow the TypeScript SDK's ``StreamableHTTPReconnectionOptions``.
    """

    initial_reconnection_delay: float = 1.0
    max_reconnection_delay: float = 30.0
    reconnection_delay_grow_factor: float = 2.0
    max_retries: int = 5

    def delay(self, retry: int) -> float:
        """Seconds to wait before the given retry, counting from 0."""
        grown = self.initial_reconnection_delay * (
            self.reconnection_delay_grow_factor**retry
        )
        return random.uniform(0, min(self.max_reconnection_delay, grown))


@dataclass(frozen=True)
class ServerStatus:
    """A server's connection state, and the error behind it if any."""

    state: State
    error: Optional[str] = None


def _describe(error: BaseException) -> str:
    return str(error) or type(error).__name__


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
        auth: Optional[httpx.Auth],
        connect_timeout: float,
        call_timeout: Optional[float],
        idle_timeout: Optional[float],
        reconnection: ReconnectionOptions,
        on_tools_changed: Callable[[List[Tool]], Any],
    ) -> None:
        self.name = name
        self.transport = transport
        self.command = command
        self.url = url
        self.env = env
        self.headers = headers
        self.auth = auth
        self.connect_timeout = connect_timeout
        self.call_timeout = call_timeout
        self.idle_timeout = idle_timeout
        self.reconnection = reconnection
        self.on_tools_changed = on_tools_changed
        self.session: Optional[ClientSession] = None
        self.status = ServerStatus("connecting")
        self.loop = asyncio.get_running_loop()
        self._task: Optional[asyncio.Task] = None
        self._reconnecting: Optional[asyncio.Task] = None
        self._closing = asyncio.Event()
        self._detached = False
        self._busy = 0
        self._last_used = 0.0

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
            # A call holds its POST open until the result; call_timeout bounds it.
            timeout = httpx.Timeout(self.connect_timeout, read=None)
            http = create_mcp_http_client(
                headers=self.headers, timeout=timeout, auth=self.auth
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
                self.url,
                headers=self.headers,
                timeout=self.connect_timeout,
                auth=self.auth,
            ) as (read, write):
                yield read, write

    async def _run(
        self, ready: "asyncio.Future[ClientSession]", closing: asyncio.Event
    ) -> None:
        async def on_message(message: Any) -> None:
            if isinstance(message, ServerNotification) and isinstance(
                message.root, ToolListChangedNotification
            ):
                # In a task: the handler must return before the response is read.
                tg.start_soon(self._refresh_tools, session)

        try:
            async with self._streams() as (read, write):
                session = ClientSession(read, write, message_handler=on_message)
                async with session, anyio.create_task_group() as tg:
                    await session.initialize()
                    if not ready.done():
                        ready.set_result(session)
                    await self._serve(closing)
                    tg.cancel_scope.cancel()
        except Exception as e:
            if not ready.done():
                ready.set_exception(e)
            elif not closing.is_set() and not ready.cancelled():
                self._lost(ready.result(), e)

    async def _serve(self, closing: asyncio.Event) -> None:
        """Wait until closed, or close the session once idle for idle_timeout."""
        self._last_used = self.loop.time()
        if self.idle_timeout is None:
            await closing.wait()
            return
        while not closing.is_set():
            wait = self.idle_timeout
            if not self._busy:
                wait += self._last_used - self.loop.time()
                if wait <= 0:
                    logger.info("Closing idle MCP server %r", self.name)
                    self.session = None
                    self.status = ServerStatus("disconnected")
                    return
            with anyio.move_on_after(wait):
                await closing.wait()

    async def _refresh_tools(self, session: ClientSession) -> None:
        try:
            self.on_tools_changed(await self.list_tools(session))
        except Exception as e:
            logger.warning("Could not refresh tools of MCP server %r: %s", self.name, e)

    def _lost(self, session: ClientSession, error: BaseException) -> None:
        """Mark a dropped session; reconnect remote servers in the background."""
        if self.session is not session:
            return
        self.session = None
        self._closing.set()
        self.status = ServerStatus("disconnected", _describe(error))
        logger.warning("MCP server %r disconnected: %s", self.name, self.status.error)
        if self.transport != "stdio" and not self._detached:
            self._reconnect_task()

    async def connect(self) -> ClientSession:
        self._closing = asyncio.Event()
        ready: asyncio.Future[ClientSession] = self.loop.create_future()
        self._task = asyncio.create_task(self._run(ready, self._closing))
        try:
            self.session = await asyncio.wait_for(ready, self.connect_timeout)
        except BaseException:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
            raise
        self.status = ServerStatus("connected")
        return self.session

    async def close(self) -> None:
        self._closing.set()
        task, self._task, self.session = self._task, None, None
        if task is None:
            return
        try:
            await asyncio.wait({task}, timeout=5)
        finally:
            # Also runs if close is cancelled, so the owner task never outlives it.
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    async def detach(self) -> None:
        self._detached = True
        if self._reconnecting is not None:
            self._reconnecting.cancel()
            await asyncio.gather(self._reconnecting, return_exceptions=True)
        await self.close()

    @contextlib.contextmanager
    def _use(self) -> Iterator[None]:
        """Mark the session busy, so it is not closed as idle."""
        self._busy += 1
        try:
            yield
        finally:
            self._busy -= 1
            self._last_used = self.loop.time()

    async def list_tools(self, session: ClientSession) -> List[Tool]:
        tools: List[Tool] = []
        cursor: Optional[str] = None
        with self._use():
            while True:
                page = await session.list_tools(cursor=cursor)
                tools.extend(page.tools)
                cursor = page.nextCursor
                if not cursor:
                    return tools

    async def _reconnect(self) -> ClientSession:
        await self.close()
        retry = 0
        while True:
            logger.info("Reconnecting to MCP server %r", self.name)
            self.status = ServerStatus("connecting", self.status.error)
            try:
                return await self.connect()
            except Exception as e:
                if retry == self.reconnection.max_retries:
                    self.status = ServerStatus("failed", _describe(e))
                    raise
                self.status = ServerStatus("connecting", _describe(e))
            await asyncio.sleep(self.reconnection.delay(retry))
            retry += 1

    def _reconnect_task(self) -> "asyncio.Task[ClientSession]":
        """The running reconnect, started if there is none."""
        if self._reconnecting is None or self._reconnecting.done():
            self._reconnecting = asyncio.create_task(self._reconnect())
            # Read the outcome so a reconnect nobody awaits is not logged as unhandled.
            self._reconnecting.add_done_callback(
                lambda t: t.cancelled() or t.exception()
            )
        return self._reconnecting

    async def _session(self, stale: Optional[ClientSession]) -> ClientSession:
        """The live session, reconnecting if there is none or it is ``stale``.

        Concurrent callers share one reconnect, which goes on if they are
        cancelled.
        """
        if self.session is not None and self.session is not stale:
            return self.session
        if self._detached:
            raise RuntimeError(f"MCP server {self.name!r} is detached")
        task = self._reconnect_task()
        await asyncio.wait({task})
        if task.cancelled():
            raise RuntimeError(f"MCP server {self.name!r} is detached")
        return task.result()

    async def call_tool(self, tool: str, arguments: Dict[str, Any]) -> CallToolResult:
        with self._use():
            session = self.session or await self._session(None)
            try:
                return await self._call(session, tool, arguments)
            except McpError as e:
                # Dropped mid call: the tool may have run, so never retry it.
                if e.error.code == CONNECTION_CLOSED:
                    self._lost(session, e)
                raise
            except (anyio.ClosedResourceError, anyio.BrokenResourceError):
                # Dropped before sending: safe to retry once.
                session = await self._session(session)
                return await self._call(session, tool, arguments)

    async def _call(
        self, session: ClientSession, tool: str, arguments: Dict[str, Any]
    ) -> CallToolResult:
        timeout = (
            None if self.call_timeout is None else timedelta(seconds=self.call_timeout)
        )
        call = asyncio.ensure_future(
            session.call_tool(tool, arguments, read_timeout_seconds=timeout)
        )
        # A failed transport leaves its calls waiting, so also stop with the owner.
        watch = {call} if self._task is None else {call, self._task}
        try:
            await asyncio.wait(watch, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not call.done():
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
        if call.cancelled():
            raise McpError(
                ErrorData(code=CONNECTION_CLOSED, message="Connection closed")
            )
        return call.result()


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
        self._locks: Dict[str, asyncio.Lock] = {}

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
                await conn.close()
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
        await asyncio.gather(*[self.detach(name) for name in list(self._locks)])

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
