"""MCP-specific tool utilities"""

from typing import Any, Dict
from cyclops.toolkit.tool import BaseTool


class MCPTool(BaseTool):
    """Tool specifically designed for MCP integration"""

    def __init__(self, name: str, description: str, schema: Dict[str, Any]):
        super().__init__(name, description)
        self.schema = schema

    async def execute(self, **kwargs) -> Any:
        """Execute MCP tool - to be overridden by subclasses"""
        raise NotImplementedError("MCPTool subclasses must implement execute method")

    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert tool to MCP format"""
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.schema,
        }
