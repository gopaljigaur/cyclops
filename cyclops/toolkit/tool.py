"""Tool definitions and base classes"""

import asyncio
import inspect
from abc import ABC
from typing import Any, Callable, Dict, Optional

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, Field, create_model

from cyclops.toolkit.types import ToolAnnotations, ToolDefinition, ToolParameter


def _args_model(name: str, func: Callable) -> type[BaseModel]:
    """Build a pydantic model of the function's parameters.

    Descriptions come from the Google or NumPy style docstring, or from
    ``Annotated[..., Field(description=...)]``.
    """
    docs = {
        p.arg_name: p.description
        for p in parse_docstring(inspect.getdoc(func) or "").params
    }
    fields: Dict[str, Any] = {}
    for param in inspect.signature(func, eval_str=True).parameters.values():
        if param.name == "self" or param.kind in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            continue
        annotation = Any if param.annotation is param.empty else param.annotation
        default = ... if param.default is param.empty else param.default
        # An explicit description=None would drop one set through Annotated
        if param.name in docs:
            default = Field(default, description=docs[param.name])
        fields[param.name] = (annotation, default)
    return create_model(name, **fields)


def _json_type(prop: Dict[str, Any], defs: Dict[str, Any]) -> str:
    """Top level JSON type of a property schema, for the parameter summary."""
    if "$ref" in prop:
        prop = defs[prop["$ref"].rsplit("/", 1)[-1]]
    for option in prop.get("anyOf", []):
        if option.get("type") != "null":
            return _json_type(option, defs)
    return prop.get("type", "string")


def _parameters(schema: Dict[str, Any]) -> Dict[str, ToolParameter]:
    defs = schema.get("$defs", {})
    required = set(schema.get("required", []))
    return {
        name: ToolParameter(
            name=name,
            type=_json_type(prop, defs),
            description=prop.get("description"),
            required=name in required,
            default=prop.get("default"),
        )
        for name, prop in schema["properties"].items()
    }


class BaseTool(ABC):
    """Abstract base tool class"""

    def __init__(
        self,
        name: str,
        description: str,
        annotations: Optional[ToolAnnotations] = None,
    ):
        self.name = name
        self.description = description
        self.annotations = annotations or ToolAnnotations()
        self._definition = self._build_definition()

    async def execute(self, **kwargs):
        """Execute the tool. Subclasses should override with their own signature."""
        raise NotImplementedError(f"Tool '{self.name}' must implement execute()")

    def _build_definition(self) -> ToolDefinition:
        return self._definition_for(self.execute)

    def _definition_for(self, func: Callable) -> ToolDefinition:
        self._args_model = _args_model(self.name, func)
        schema = self._args_model.model_json_schema()
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters=_parameters(schema),
            input_schema=schema,
            annotations=self.annotations,
        )

    @property
    def definition(self) -> ToolDefinition:
        """Get tool definition"""
        return self._definition


class Tool(BaseTool):
    """Simple function-based tool"""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        annotations: Optional[ToolAnnotations] = None,
    ):
        self.func = func
        self._is_async = inspect.iscoroutinefunction(func)
        super().__init__(name, description, annotations)

    def _build_definition(self) -> ToolDefinition:
        return self._definition_for(self.func)

    async def execute(self, **kwargs) -> Any:
        """Validate the arguments against the schema and call the function"""
        # Validation turns nested objects into the declared models
        kwargs.update(dict(self._args_model.model_validate(kwargs)))
        if self._is_async:
            return await self.func(**kwargs)
        # A blocking sync function must not stall the event loop
        return await asyncio.to_thread(self.func, **kwargs)
