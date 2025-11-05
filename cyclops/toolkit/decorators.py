"""Decorators for creating tools"""

from typing import Callable, Optional
from cyclops.toolkit.tool import Tool
from cyclops.toolkit.registry import ToolRegistry


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
):
    """Decorator to convert a function into a tool"""

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Tool: {tool_name}"

        created_tool = Tool(tool_name, tool_description, func)

        if registry:
            registry.register(created_tool)

        return created_tool

    return decorator
