"""MCP Client implementation"""

from typing import Any, Dict, List, Optional
from mcp.client.session import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


class MCPClient:
    """MCP Client wrapper for connecting to MCP servers"""

    def __init__(self):
        self.client: Optional[ClientSession] = None
        self._connected = False

    async def connect_stdio(
        self, command: List[str], env: Optional[Dict[str, str]] = None
    ):
        """Connect to MCP server via stdio"""
        server_params = StdioServerParameters(
            command=command[0], args=command[1:], env=env
        )

        # Store the context manager for cleanup
        self._stdio_context = stdio_client(server_params)
        read_stream, write_stream = await self._stdio_context.__aenter__()

        self.client = ClientSession(read_stream, write_stream)
        await self.client.__aenter__()

        # Initialize the session
        await self.client.initialize()

        self._connected = True

    async def disconnect(self):
        """Disconnect from MCP server"""
        if self.client:
            await self.client.__aexit__(None, None, None)
        if hasattr(self, "_stdio_context"):
            await self._stdio_context.__aexit__(None, None, None)
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
