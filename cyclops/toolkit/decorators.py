"""Decorators for creating tools"""

from typing import Callable, Optional, Union
from cyclops.toolkit.tool import Tool
from cyclops.toolkit.registry import ToolRegistry


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
) -> Union[Tool, Callable[[Callable], Tool]]:
    """Decorator to convert a function into a tool

    Usage:
        # Use function name and docstring
        @tool
        def my_function():
            '''This is the description'''
            pass

        # Or with parentheses
        @tool()
        def my_function():
            '''This is the description'''
            pass

        # Override name and/or description
        @tool(name="custom_name", description="Custom description")
        def my_function():
            pass
    """

    def decorator(f: Callable) -> Tool:
        tool_name = name or f.__name__
        tool_description = description or (
            f.__doc__.strip() if f.__doc__ else f"Tool: {tool_name}"
        )

        created_tool = Tool(tool_name, tool_description, f)

        if registry:
            registry.register(created_tool)

        return created_tool

    # Handle both @tool and @tool()
    if func is not None:
        return decorator(func)
    return decorator
