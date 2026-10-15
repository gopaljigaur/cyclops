"""MCPClientTool: a remote MCP tool as a cyclops BaseTool."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from mcp import types as mcp_types

from cyclops.toolkit.tool import BaseTool
from cyclops.toolkit.types import (
    ContentBlock,
    ImageContent,
    TextContent,
    ToolAnnotations,
    ToolDefinition,
    ToolParameter,
    ToolResult,
)

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
        if isinstance(json_type, list):
            # A union such as ["string", "null"]: describe the first non-null type.
            json_type = next((t for t in json_type if t != "null"), "string")
        if json_type not in _valid:
            json_type = "string"
        params[name] = ToolParameter(
            name=name,
            type=json_type,
            description=prop.get("description"),
            required=name in required_set,
        )
    return params


def to_tool_result(result: mcp_types.CallToolResult) -> ToolResult:
    """Map an MCP tool result onto a ToolResult.

    Text and images become content blocks and ``structuredContent`` becomes
    ``result``. ``isError`` gives an error result carrying the text.
    """
    blocks: List[ContentBlock] = []
    for content in result.content:
        if isinstance(content, mcp_types.TextContent):
            blocks.append(TextContent(text=content.text))
        elif isinstance(content, mcp_types.ImageContent):
            blocks.append(ImageContent(data=content.data, mime_type=content.mimeType))
        elif isinstance(content, mcp_types.EmbeddedResource) and isinstance(
            content.resource, mcp_types.TextResourceContents
        ):
            blocks.append(TextContent(text=content.resource.text))
        else:
            blocks.append(TextContent(text=content.model_dump_json()))
    if result.isError:
        text = "\n".join(b.text for b in blocks if isinstance(b, TextContent))
        return ToolResult.error_result(text)
    return ToolResult(result=result.structuredContent, content=blocks)


def _annotations(tool: mcp_types.Tool) -> ToolAnnotations:
    hints = tool.annotations
    if hints is None:
        return ToolAnnotations()
    return ToolAnnotations(
        read_only=hints.readOnlyHint,
        destructive=hints.destructiveHint,
        idempotent=hints.idempotentHint,
        open_world=hints.openWorldHint,
    )


class MCPClientTool(BaseTool):
    """A BaseTool backed by a tool on an attached MCP server."""

    def __init__(
        self,
        client: "MCPClient",
        server: str,
        tool: mcp_types.Tool,
        name: Optional[str] = None,
        defer_loading: bool = False,
    ) -> None:
        self.client = client
        self.server = server
        self.tool = tool
        super().__init__(
            name or tool.name,
            tool.description or "",
            _annotations(tool),
            defer_loading=defer_loading,
        )

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=_mcp_schema_to_params(self.tool.inputSchema),
            input_schema=self.tool.inputSchema,
            annotations=self.annotations,
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = await self.client.call_tool(self.server, self.tool.name, kwargs)
        except Exception as e:
            return ToolResult.error_result(str(e) or type(e).__name__)
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
