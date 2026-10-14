"""MCPClientTool: a remote MCP tool as a cyclops BaseTool."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mcp import McpError
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
    Tool,
)

from cyclops.toolkit.tool import BaseTool
from cyclops.toolkit.types import ToolDefinition, ToolParameter, ToolResult

if TYPE_CHECKING:
    from cyclops.mcp.bridge import MCPBridge
    from cyclops.mcp.client import MCPClient


def _mcp_schema_to_params(input_schema: Dict[str, Any]) -> Dict[str, ToolParameter]:
    props = input_schema.get("properties", {})
    required_set = set(input_schema.get("required", []))
    params: Dict[str, ToolParameter] = {}
    _valid = {"string", "integer", "number", "boolean", "array", "object"}
    for name, prop in props.items():
        json_type = prop.get("type", "string")
        if json_type not in _valid:
            json_type = "string"
        params[name] = ToolParameter(
            name=name,
            type=json_type,
            description=prop.get("description"),
            required=name in required_set,
        )
    return params


class MCPToolDefinition(ToolDefinition):
    """A tool definition whose JSON schema is the server's, unchanged."""

    input_schema: Dict[str, Any]

    def to_json_schema(self) -> Dict[str, Any]:
        return self.input_schema


def to_tool_result(result: CallToolResult) -> ToolResult:
    """Map an MCP tool result onto a ToolResult.

    Text only gives a string; with images the result is a list of
    OpenAI style content parts. ``isError`` gives an error result.
    """
    parts: List[Dict[str, Any]] = []
    for content in result.content:
        if isinstance(content, TextContent):
            parts.append({"type": "text", "text": content.text})
        elif isinstance(content, ImageContent):
            url = f"data:{content.mimeType};base64,{content.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
        elif isinstance(content, EmbeddedResource) and isinstance(
            content.resource, TextResourceContents
        ):
            parts.append({"type": "text", "text": content.resource.text})
        else:
            parts.append({"type": "text", "text": content.model_dump_json()})
    text = "\n".join(p["text"] for p in parts if p["type"] == "text")
    if result.isError:
        return ToolResult.error_result(text)
    if all(p["type"] == "text" for p in parts):
        return ToolResult.success_result(text)
    return ToolResult.success_result(parts)


class MCPClientTool(BaseTool):
    """A BaseTool backed by a tool on an attached MCP server."""

    def __init__(
        self,
        client: "MCPClient",
        server: str,
        tool: Tool,
        name: Optional[str] = None,
    ) -> None:
        self.client = client
        self.server = server
        self.tool = tool
        super().__init__(name or tool.name, tool.description or "")

    def _build_definition(self) -> ToolDefinition:
        return MCPToolDefinition(
            name=self.name,
            description=self.description,
            parameters=_mcp_schema_to_params(self.tool.inputSchema),
            input_schema=self.tool.inputSchema,
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self.client.call_tool(self.server, self.tool.name, kwargs)
        except McpError as e:
            return ToolResult.error_result(e.error.message)
        return to_tool_result(result)


def tools_from_server(
    bridge: "MCPBridge",
    name: str,
    command: List[str],
    env: Optional[Dict[str, str]] = None,
) -> tuple["MCPClient", List[MCPClientTool]]:
    """Attach a stdio server to the bridge and return (client, tools)."""
    tools = bridge.attach(name, command, env=env)
    return bridge.client, tools
