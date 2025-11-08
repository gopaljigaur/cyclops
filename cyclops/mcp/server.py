"""MCP Server implementation"""

from typing import Any, Dict, List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool as MCPTool, TextContent
from cyclops.toolkit.registry import ToolRegistry
from cyclops.toolkit.tool import BaseTool
from cyclops.toolkit.plugins import PluginManager


class MCPServer:
    """MCP Server wrapper for Cyclops tools"""

    def __init__(
        self,
        name: str = "cyclops-mcp-server",
        version: str = "0.1.0",
        load_plugins: bool = True,
    ):
        self.name = name
        self.version = version
        self.server = Server(name)
        self.tool_registry = ToolRegistry()

        # Initialize plugin system
        self.plugin_manager = PluginManager(self.tool_registry)

        if load_plugins:
            self.load_all_plugins()

        # Register MCP handlers
        self._setup_handlers()

    def load_all_plugins(self):
        """Load and register all toolkit plugins"""
        self.plugin_manager.load_plugins()
        self.plugin_manager.register_tools()

    def _setup_handlers(self):
        """Setup MCP protocol handlers"""

        @self.server.list_tools()
        async def list_tools() -> List[MCPTool]:
            """List available tools"""
            tools = []
            for tool_name, tool in self.tool_registry._tools.items():
                # Convert Cyclops tool to MCP tool format
                mcp_tool = MCPTool(
                    name=tool.name,
                    description=tool.description,
                    inputSchema={
                        "type": "object",
                        "properties": {
                            param.name: {
                                "type": param.type,
                                "description": param.description or "",
                            }
                            for param in tool.definition.parameters.values()
                        },
                        "required": [
                            param.name
                            for param in tool.definition.parameters.values()
                            if param.required
                        ],
                    },
                )
                tools.append(mcp_tool)
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Execute a tool"""
            try:
                result = await self.tool_registry.execute_tool(
                    tool_name=name, **arguments
                )
                return [TextContent(type="text", text=str(result))]
            except Exception as e:
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the server"""
        self.tool_registry.register(tool)

    def add_function_tool(self, name: str, description: str, func) -> None:
        """Add a function as a tool"""
        self.tool_registry.register_function(name, description, func)

    async def run_stdio(self):
        """Run the MCP server with stdio transport"""
        async with stdio_server() as streams:
            await self.server.run(
                streams[0],  # read stream
                streams[1],  # write stream
                self.server.create_initialization_options(),
            )
