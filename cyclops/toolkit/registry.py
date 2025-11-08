"""Tool registry for managing available tools"""

from typing import Dict, List, Optional, Any
from cyclops.toolkit.tool import BaseTool, Tool, ToolDefinition


class ToolRegistry:
    """Registry for managing and organizing tools"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry"""
        self._tools[tool.name] = tool

    def register_function(self, name: str, description: str, func) -> None:
        """Register a function as a tool"""
        tool = Tool(name, description, func)
        self.register(tool)

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self._tools.keys())

    def get_definitions(self) -> Dict[str, ToolDefinition]:
        """Get all tool definitions"""
        return {name: tool.definition for name, tool in self._tools.items()}

    async def execute_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool by name"""
        tool = self.get_tool(tool_name)
        if not tool:
            raise ValueError(f"Tool '{tool_name}' not found")

        return await tool.execute(**kwargs)

    def remove_tool(self, name: str) -> bool:
        """Remove a tool from registry"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all tools from registry"""
        self._tools.clear()
