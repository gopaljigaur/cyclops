"""MCP Client implementation"""

import contextlib
from typing import Any, Dict, List, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class MCPClient:
    """MCP Client wrapper for connecting to MCP servers"""

    def __init__(self):
        self.client: Optional[ClientSession] = None
        self._connected = False
        self._exit_stack: Optional[contextlib.AsyncExitStack] = None

    async def connect_stdio(
        self, command: List[str], env: Optional[Dict[str, str]] = None
    ):
        """Connect to MCP server via stdio"""
        server_params = StdioServerParameters(
            command=command[0], args=command[1:], env=env
        )
        self._exit_stack = contextlib.AsyncExitStack()
        read_stream, write_stream = await self._exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        client = await self._exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        self.client = client
        await client.initialize()
        self._connected = True

    async def disconnect(self):
        """Disconnect from MCP server"""
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None
        self.client = None
        self._connected = False

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server"""
        if not self._connected or not self.client:
            raise RuntimeError("Not connected to MCP server")

        response = await self.client.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Call a tool on the server"""
        if not self._connected or not self.client:
            raise RuntimeError("Not connected to MCP server")

        response = await self.client.call_tool(name=name, arguments=arguments)

        result_parts = []
        for content in response.content:
            if hasattr(content, "text"):
                result_parts.append(content.text)
            else:
                result_parts.append(str(content))

        return "\n".join(result_parts)

    @property
    def is_connected(self) -> bool:
        """Check if connected to server"""
        return self._connected
