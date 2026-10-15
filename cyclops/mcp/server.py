"""MCP Server implementation"""

from typing import List, Optional

from mcp import types as mcp_types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    Tool as MCPTool,
)

from cyclops.toolkit.registry import ToolRegistry
from cyclops.toolkit.tool import BaseTool
from cyclops.toolkit.plugins import PluginManager
from cyclops.toolkit.types import TextContent, ToolResult


def _to_call_result(result: ToolResult) -> CallToolResult:
    """Serve a ToolResult's content blocks as MCP text and image content."""
    if result.is_error:
        error = result.error or "tool execution failed"
        return CallToolResult(
            content=[mcp_types.TextContent(type="text", text=error)], is_error=True
        )
    content: List[mcp_types.ContentBlock] = [
        (
            mcp_types.TextContent(type="text", text=block.text)
            if isinstance(block, TextContent)
            else mcp_types.ImageContent(
                type="image", data=block.data, mime_type=block.mime_type
            )
        )
        for block in result.content
    ]
    if not content:
        # No blocks: the result itself, rendered as text or JSON.
        text = result.to_message_content()
        assert isinstance(text, str)
        content.append(mcp_types.TextContent(type="text", text=text))
    return CallToolResult(content=content)


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
        self.server = Server(
            name,
            version=version,
            on_list_tools=self._list_tools,
            on_call_tool=self._call_tool,
        )
        self.tool_registry = ToolRegistry()
        self.plugin_manager = PluginManager(self.tool_registry)

        if load_plugins:
            self.load_all_plugins()

    def load_all_plugins(self):
        """Load and register all toolkit plugins"""
        self.plugin_manager.load_plugins()
        self.plugin_manager.register_tools()

    async def _list_tools(
        self, ctx: ServerRequestContext, params: Optional[PaginatedRequestParams]
    ) -> ListToolsResult:
        tools = []
        for tool_name in self.tool_registry.list_tools():
            tool = self.tool_registry.get_tool(tool_name)
            if tool is None:
                continue
            tools.append(
                MCPTool(
                    name=tool.name,
                    description=tool.description,
                    input_schema=tool.definition.to_json_schema(),
                )
            )
        return ListToolsResult(tools=tools)

    async def _call_tool(
        self, ctx: ServerRequestContext, params: CallToolRequestParams
    ) -> CallToolResult:
        # The SDK answers a raised exception as a protocol error, which the
        # model never sees; an error result carries the text to it instead.
        try:
            result = await self.tool_registry.execute_tool(
                tool_name=params.name, **(params.arguments or {})
            )
        except Exception as e:
            return _to_call_result(ToolResult.error_result(str(e)))
        if not isinstance(result, ToolResult):
            result = ToolResult.success_result(result)
        return _to_call_result(result)

    def add_tool(self, tool: BaseTool) -> None:
        """Add a tool to the server"""
        self.tool_registry.register(tool)

    def add_function_tool(self, name: str, description: str, func) -> None:
        """Add a function as a tool"""
        self.tool_registry.register_function(name, description, func)

    async def run_stdio(self):
        """Run the MCP server with stdio transport"""
        async with stdio_server() as streams:
            read_stream, write_stream = streams
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )
