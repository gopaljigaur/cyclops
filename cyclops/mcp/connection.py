"""One MCP server connection: its session, reconnects and status."""

import asyncio
import contextlib
import logging
import random
import sys
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Set,
)

import anyio
import httpx2
from mcp import MCPError
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.shared._httpx_utils import create_mcp_http_client
from mcp.shared.jsonrpc_dispatcher import JSONRPCDispatcher
from mcp.types import (
    CONNECTION_CLOSED,
    CallToolResult,
    PaginatedRequestParams,
    Tool,
    ToolListChangedNotification,
)

from cyclops.utils.compat import asyncio_timeout

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

logger = logging.getLogger(__name__)

Transport = Literal["stdio", "http", "sse"]
State = Literal["connecting", "connected", "disconnected", "failed"]

# The TypeScript SDK's default request timeout.
DEFAULT_CALL_TIMEOUT = 60.0


@dataclass(frozen=True)
class ReconnectionOptions:
    """Exponential backoff with full jitter between reconnect attempts.

    Names follow the TypeScript SDK's ``StreamableHTTPReconnectionOptions``,
    in seconds. The defaults are cyclops' own: they reopen a whole session,
    so they allow for a server restart, where the SDK's resume one stream.
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


def _cause(error: BaseException) -> BaseException:
    """The single error inside nested task group errors, or ``error`` itself."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


@dataclass
class _Attempt:
    """One attempt at a call; ``refused`` once the server answers it 404."""

    refused: bool = False


# The call attempt of the running task; its requests carry this context.
_attempt: ContextVar[Optional[_Attempt]] = ContextVar("_attempt", default=None)


async def _flag_refused(response: httpx2.Response) -> None:
    # The spec's sign that the server no longer knows the session, as after a
    # restart: a 404 to a request that carried its id. That request was not run.
    attempt = _attempt.get()
    if (
        attempt is not None
        and response.status_code == 404
        and "mcp-session-id" in response.request.headers
    ):
        attempt.refused = True


class _Dispatcher(JSONRPCDispatcher):
    """Calls ``on_end`` once the transport's read stream ends."""

    def __init__(self, read: Any, write: Any, on_end: Callable[[], None]) -> None:
        super().__init__(read, write)
        self.on_end = on_end

    async def run(self, *args: Any, **kwargs: Any) -> None:
        try:
            await super().run(*args, **kwargs)
        finally:
            self.on_end()


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
        auth: Optional[httpx2.Auth],
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
        self._calls: Set[asyncio.Task] = set()
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
            timeout = httpx2.Timeout(self.connect_timeout, read=None)
            http = create_mcp_http_client(
                headers=self.headers, timeout=timeout, auth=self.auth
            )
            http.event_hooks["response"].append(_flag_refused)
            async with (
                http,
                streamable_http_client(self.url, http_client=http) as (read, write),
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
        def ended() -> None:
            # The server exited or dropped the connection.
            if not closing.is_set():
                self._lost(session, MCPError(CONNECTION_CLOSED, "Connection closed"))

        async def on_message(message: Any) -> None:
            # The session runs each notification in its own task, cancelled on close.
            if isinstance(message, ToolListChangedNotification):
                await self._refresh_tools(session)

        try:
            async with self._streams() as (read, write):
                session = ClientSession(
                    dispatcher=_Dispatcher(read, write, ended),
                    message_handler=on_message,
                )
                async with session:
                    await session.initialize()
                    if not ready.done():
                        ready.set_result(session)
                    await self._serve(closing)
        except Exception as group:
            e = _cause(group)
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
            async with asyncio_timeout(self.connect_timeout):
                self.session = await ready
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
                page = await session.list_tools(
                    params=PaginatedRequestParams(cursor=cursor)
                )
                tools.extend(page.tools)
                cursor = page.next_cursor
                if not cursor:
                    return tools

    async def _reconnect(self) -> ClientSession:
        if not self._closing.is_set():
            # Not a drop: calls already sent on the session get their own answer
            # first, so each one the server refused is retried, not cut off.
            self.session = None
            if self._calls:
                await asyncio.wait(self._calls)
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
            self._reconnecting.add_done_callback(self._reconnected)
        return self._reconnecting

    def _reconnected(self, task: "asyncio.Task[ClientSession]") -> None:
        if self._reconnecting is task:
            self._reconnecting = None
        # Read the outcome so a reconnect nobody awaits is not logged as unhandled.
        if not task.cancelled():
            task.exception()

    async def _session(self, stale: Optional[ClientSession]) -> ClientSession:
        """The live session, reconnecting if there is none or it is ``stale``.

        Waits up to call_timeout. Concurrent callers share one reconnect, which
        goes on if they stop waiting.
        """
        if self.session is not None and self.session is not stale:
            return self.session
        if self._detached:
            raise RuntimeError(f"MCP server {self.name!r} is detached")
        task = self._reconnect_task()
        await asyncio.wait({task}, timeout=self.call_timeout)
        if not task.done():
            raise TimeoutError(f"MCP server {self.name!r} is still reconnecting")
        if task.cancelled():
            raise RuntimeError(f"MCP server {self.name!r} is detached")
        return task.result()

    async def call_tool(self, tool: str, arguments: Dict[str, Any]) -> CallToolResult:
        with self._use():
            session = self.session or await self._session(None)
            attempt = _Attempt()
            token = _attempt.set(attempt)
            try:
                return await self._call(session, tool, arguments)
            except MCPError as e:
                # Dropped mid call: the tool may have run, so never retry it.
                if e.code == CONNECTION_CLOSED:
                    self._lost(session, e)
                if not attempt.refused:
                    raise
            finally:
                _attempt.reset(token)
            # Refused unrun: safe to retry once on a new session.
            session = await self._session(session)
            return await self._call(session, tool, arguments)

    async def _call(
        self, session: ClientSession, tool: str, arguments: Dict[str, Any]
    ) -> CallToolResult:
        call = asyncio.create_task(
            session.call_tool(tool, arguments, read_timeout_seconds=self.call_timeout)
        )
        self._calls.add(call)
        call.add_done_callback(self._calls.discard)
        # A failed transport leaves its calls waiting, so also stop with the owner.
        # Through a future of this call: asyncio.wait on the owner itself keeps
        # every caller alive on 3.14.0 to 3.14.6 (python/cpython#152569).
        stopped = self.loop.create_future()

        def stop(_: object) -> None:
            if not stopped.done():
                stopped.set_result(None)

        owner = self._task
        if owner is not None:
            owner.add_done_callback(stop)
        try:
            await asyncio.wait({call, stopped}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if owner is not None:
                owner.remove_done_callback(stop)
            if not call.done():
                call.cancel()
                await asyncio.gather(call, return_exceptions=True)
        if call.cancelled():
            raise MCPError(CONNECTION_CLOSED, "Connection closed")
        return call.result()
