"""Sync wrapper around MCPClient for code without an event loop."""

import asyncio
import threading
from typing import Any, Coroutine, List, Optional, Sequence, TypeVar

from cyclops.mcp.client import MCPClient
from cyclops.mcp.tools import MCPClientTool

T = TypeVar("T")


class MCPBridge:
    """Runs an MCPClient on an event loop in a daemon thread.

    Its tools work from any loop, including the one a sync ``agent.run()``
    starts, because MCPClient hands each call to the bridge's loop.
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
        return self._run(self.client.attach(name, command, **kwargs))

    def detach(self, name: str) -> None:
        self._run(self.client.detach(name))

    def tools(self, servers: Optional[Sequence[str]] = None) -> List[MCPClientTool]:
        return self.client.tools(servers)

    def stop(self) -> None:
        """Disconnect every server and stop the background loop."""
        self._run(self.client.aclose())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3)
        self._loop.close()

    @property
    def connected_names(self) -> List[str]:
        return self.client.servers
