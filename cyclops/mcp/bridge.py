"""Sync wrapper around MCPClient for code without an event loop."""

import asyncio
import threading
from typing import Any, Coroutine, List, Optional, Sequence, TypeVar

from cyclops.mcp.client import MCPClient
from cyclops.mcp.tools import MCPClientTool
from cyclops.toolkit.types import ToolResult

T = TypeVar("T")


class _BridgeTool(MCPClientTool):
    """Runs each call on the bridge's loop, so it can be awaited from any loop."""

    def __init__(self, loop: asyncio.AbstractEventLoop, tool: MCPClientTool) -> None:
        self._loop = loop
        super().__init__(tool.client, tool.server, tool.tool, tool.name)

    async def execute(self, **kwargs: Any) -> ToolResult:
        future = asyncio.run_coroutine_threadsafe(super().execute(**kwargs), self._loop)
        return await asyncio.wrap_future(future)


class MCPBridge:
    """Runs an MCPClient on an event loop in a daemon thread.

    Its tools run each call on that loop, so they work from any loop,
    including the one a sync ``agent.run()`` starts.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self.client = MCPClient()

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def attach(
        self, name: str, command: Optional[List[str]] = None, **kwargs: Any
    ) -> List[MCPClientTool]:
        """Sync ``MCPClient.attach``; takes the same arguments."""
        tools = self._run(self.client.attach(name, command, **kwargs))
        return [_BridgeTool(self._loop, t) for t in tools]

    def detach(self, name: str) -> None:
        self._run(self.client.detach(name))

    def tools(self, servers: Optional[Sequence[str]] = None) -> List[MCPClientTool]:
        return [_BridgeTool(self._loop, t) for t in self.client.tools(servers)]

    def stop(self) -> None:
        """Disconnect every server and stop the background loop."""
        self._run(self.client.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3)
        self._loop.close()

    @property
    def connected_names(self) -> List[str]:
        return self.client.servers
