"""Tool definitions and base classes"""

import inspect
import types
import typing
from abc import ABC
from typing import Any, Callable, Dict, get_args, get_origin

from cyclops.toolkit.types import ToolParameter, ToolDefinition


_PYTHON_TO_JSON_TYPE = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}

_ORIGIN_TO_JSON_TYPE = {
    list: "array",
    dict: "object",
}


def _annotation_to_json_type(annotation) -> str:
    """Convert a Python type annotation to a JSON schema type string."""
    if annotation is inspect.Parameter.empty:
        return "string"
    origin = get_origin(annotation)
    if origin is not None:
        is_union = origin is typing.Union
        if not is_union and hasattr(types, "UnionType"):
            is_union = isinstance(annotation, types.UnionType)
        if is_union:
            non_none = [a for a in get_args(annotation) if a is not type(None)]
            if non_none:
                return _annotation_to_json_type(non_none[0])
            return "string"
        return _ORIGIN_TO_JSON_TYPE.get(origin, "string")
    return _PYTHON_TO_JSON_TYPE.get(annotation, "string")


def _params_from_sig(sig) -> Dict[str, ToolParameter]:
    """Extract ToolParameter dict from an inspect.Signature."""
    parameters = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "kwargs"):
            continue
        param_type = _annotation_to_json_type(param.annotation)
        required = param.default == inspect.Parameter.empty
        parameters[param_name] = ToolParameter(
            name=param_name,
            type=param_type,
            required=required,
            default=param.default if not required else None,
        )
    return parameters


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
        parameters = _params_from_sig(inspect.signature(self.execute))
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
        self._is_async = inspect.iscoroutinefunction(func)
        super().__init__(name, description)

    def _build_definition(self) -> ToolDefinition:
        parameters = _params_from_sig(inspect.signature(self.func))
        return ToolDefinition(
            name=self.name, description=self.description, parameters=parameters
        )

    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped function"""
        if self._is_async:
            return await self.func(**kwargs)
        return self.func(**kwargs)
