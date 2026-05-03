"""MCPClientTool — wraps a remote MCP tool as a cyclops BaseTool."""

from typing import TYPE_CHECKING, Any, Dict

from cyclops.toolkit.tool import BaseTool
from cyclops.toolkit.types import ToolDefinition, ToolParameter

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


class MCPClientTool(BaseTool):
    """A BaseTool backed by a remote MCP server tool."""

    def __init__(
        self,
        bridge: "MCPBridge",
        client: "MCPClient",
        name: str,
        description: str,
        input_schema: Dict[str, Any],
    ) -> None:
        self._bridge = bridge
        self._client = client
        self._input_schema = input_schema
        super().__init__(name, description)

    def _build_definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=_mcp_schema_to_params(self._input_schema),
        )

    async def execute(self, **kwargs: Any) -> str:
        return self._bridge.call_tool(self._client, self.name, kwargs)


def tools_from_server(
    bridge: "MCPBridge", name: str, command: list, env: dict | None = None
) -> tuple["MCPClient", list[MCPClientTool]]:
    """Connect to an MCP server and return (client, list_of_tools)."""
    client = bridge.connect(name, command, env=env)
    tool_infos = bridge.list_tools(client)
    tools = [
        MCPClientTool(
            bridge=bridge,
            client=client,
            name=t["name"],
            description=t.get("description", ""),
            input_schema=t.get("input_schema", {}),
        )
        for t in tool_infos
    ]
    return client, tools
