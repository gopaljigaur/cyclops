"""Tool definitions and base classes"""

import inspect
import typing
from abc import ABC
from typing import Any, Callable

from cyclops.toolkit.types import ToolParameter, ToolDefinition


_PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _annotation_to_json_type(annotation) -> str:
    """Convert a Python type annotation to a JSON schema type string."""
    if annotation is inspect.Parameter.empty:
        return "string"
    origin = getattr(annotation, "__origin__", None)
    if origin is typing.Union:
        non_none = [a for a in annotation.__args__ if a is not type(None)]
        if non_none:
            return _PYTHON_TO_JSON_TYPE.get(non_none[0], "string")
    return _PYTHON_TO_JSON_TYPE.get(annotation, "string")


class BaseTool(ABC):
    """Abstract base tool class"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._definition = self._build_definition()

    async def execute(self, **kwargs):
        """Execute the tool. Subclasses should override with their own signature."""
        raise NotImplementedError(f"Tool '{self.name}' must implement execute()")

    def _build_definition(self) -> ToolDefinition:
        """Build tool definition from method signature"""
        sig = inspect.signature(self.execute)
        parameters = {}

        for param_name, param in sig.parameters.items():
            if param_name == "kwargs":
                continue

            param_type = _annotation_to_json_type(param.annotation)
            required = param.default == inspect.Parameter.empty
            default = param.default if not required else None

            parameters[param_name] = ToolParameter(
                name=param_name, type=param_type, required=required, default=default
            )

        return ToolDefinition(
            name=self.name, description=self.description, parameters=parameters
        )

    @property
    def definition(self) -> ToolDefinition:
        """Get tool definition"""
        return self._definition


class Tool(BaseTool):
    """Simple function-based tool"""

    def __init__(self, name: str, description: str, func: Callable):
        self.func = func
        super().__init__(name, description)

    def _build_definition(self) -> ToolDefinition:
        """Build tool definition from wrapped function signature"""
        sig = inspect.signature(self.func)
        parameters = {}

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "kwargs"):
                continue

            param_type = _annotation_to_json_type(param.annotation)
            required = param.default == inspect.Parameter.empty
            default = param.default if not required else None

            parameters[param_name] = ToolParameter(
                name=param_name, type=param_type, required=required, default=default
            )

        return ToolDefinition(
            name=self.name, description=self.description, parameters=parameters
        )

    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped function"""
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)
