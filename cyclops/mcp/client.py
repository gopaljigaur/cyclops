"""MCP Client implementation"""

from typing import Any, Dict, List, Optional
from mcp.client import Client
from mcp.client.stdio import stdio_client
from mcp.types import CallToolRequest, ListToolsRequest


class MCPClient:
    """MCP Client wrapper for connecting to MCP servers"""

    def __init__(self):
        self.client: Optional[Client] = None
        self._connected = False

    async def connect_stdio(
        self, command: List[str], env: Optional[Dict[str, str]] = None
    ):
        """Connect to MCP server via stdio"""
        read_stream, write_stream = await stdio_client(command, env or {})
        self.client = Client(read_stream, write_stream)
        await self.client.initialize()
        self._connected = True

    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.client:
            await self.client.close()
            self._connected = False

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server"""
        if not self._connected or not self.client:
            raise RuntimeError("Not connected to MCP server")

        response = await self.client.list_tools(ListToolsRequest())
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

        request = CallToolRequest(name=name, arguments=arguments)
        response = await self.client.call_tool(request)

        # Extract text content from response
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
