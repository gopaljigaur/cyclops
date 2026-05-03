"""Sync bridge for async MCPClient — runs a persistent event loop in a background thread."""

import asyncio
import threading
from typing import Any, Dict, List, Optional

from cyclops.mcp.client import MCPClient


class MCPBridge:
    """Keeps one asyncio event loop alive in a daemon thread.

    All async MCP operations are submitted to that loop via
    run_coroutine_threadsafe so callers don't need to manage event loops.
    """

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._clients: Dict[str, MCPClient] = {}

    def _run(self, coro) -> Any:
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    # ── Connection management ─────────────────────────────────────────────────

    def connect(
        self,
        name: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
    ) -> MCPClient:
        """Spawn an MCP server process and connect to it."""
        if name in self._clients:
            self._run(self._clients[name].disconnect())
        client = MCPClient()
        self._run(client.connect_stdio(command, env=env))
        self._clients[name] = client
        return client

    def disconnect(self, name: str) -> None:
        if name in self._clients:
            self._run(self._clients[name].disconnect())
            del self._clients[name]

    def stop(self) -> None:
        """Disconnect all clients and stop the background loop."""
        for name in list(self._clients):
            try:
                self.disconnect(name)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=3)

    # ── Tool operations ───────────────────────────────────────────────────────

    def list_tools(self, client: MCPClient) -> List[Dict[str, Any]]:
        return self._run(client.list_tools())

    def call_tool(
        self, client: MCPClient, tool_name: str, arguments: Dict[str, Any]
    ) -> str:
        return self._run(client.call_tool(tool_name, arguments))

    @property
    def connected_names(self) -> List[str]:
        return list(self._clients.keys())
