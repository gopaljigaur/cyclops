"""Decorators for creating tools"""

import inspect
from typing import Callable, Optional, Union

from docstring_parser import parse as parse_docstring

from cyclops.toolkit.tool import Tool
from cyclops.toolkit.registry import ToolRegistry
from cyclops.toolkit.types import ToolAnnotations


def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
    annotations: Optional[ToolAnnotations] = None,
    timeout: Optional[float] = None,
    defer_loading: bool = False,
    max_output_chars: Optional[int] = None,
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

    Parameter descriptions are read from the docstring's Args/Parameters
    section; the rest of the docstring becomes the tool description.
    """

    def decorator(f: Callable) -> Tool:
        tool_name = name or f.__name__
        summary = parse_docstring(inspect.getdoc(f) or "").description
        tool_description = (
            description or (summary or "").strip() or f"Tool: {tool_name}"
        )

        created_tool = Tool(
            tool_name,
            tool_description,
            f,
            annotations,
            timeout,
            defer_loading,
            max_output_chars,
        )

        if registry:
            registry.register(created_tool)

        return created_tool

    # Handle both @tool and @tool()
    if func is not None:
        return decorator(func)
    return decorator
