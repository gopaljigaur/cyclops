"""Tool definitions and base classes"""

import asyncio
import inspect
from abc import ABC
from typing import Any, Callable, Dict, Literal, Optional, Tuple

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from cyclops.toolkit.types import ToolAnnotations, ToolDefinition, ToolParameter


def _args_model(name: str, func: Callable) -> Tuple[type[BaseModel], Dict[str, str]]:
    """Build a pydantic model of the function's parameters.

    Returns the model and a map from renamed fields back to parameter names.
    Descriptions come from the Google or NumPy style docstring, or from
    ``Annotated[..., Field(description=...)]``; the docstring wins.
    """
    docs = {
        p.arg_name: p.description
        for p in parse_docstring(inspect.getdoc(func) or "").params
    }
    fields: Dict[str, Any] = {}
    renames: Dict[str, str] = {}
    extra: Literal["allow", "forbid"] = "forbid"
    params = inspect.signature(func).parameters
    for i, param in enumerate(params.values()):
        if param.kind is param.VAR_KEYWORD:
            extra = "allow"
        if param.name == "self" or param.kind in (
            param.VAR_POSITIONAL,
            param.VAR_KEYWORD,
        ):
            continue
        annotation = Any if param.annotation is param.empty else param.annotation
        default = ... if param.default is param.empty else param.default
        field_name = param.name
        field_args: Dict[str, Any] = {}
        # Pydantic drops names with a leading underscore and warns on ones
        # that shadow BaseModel attributes, so those go through an alias
        if param.name.startswith("_") or hasattr(BaseModel, param.name):
            field_name = f"param_{i}"
            while field_name in params or field_name in fields:
                field_name += "_"
            renames[field_name] = param.name
            field_args["alias"] = param.name
        # An explicit description=None would drop one set through Annotated
        if param.name in docs:
            field_args["description"] = docs[param.name]
        if field_args:
            default = Field(default, **field_args)
        fields[field_name] = (annotation, default)
    # String annotations resolve in the function's module, like they would there
    model = create_model(
        name,
        __config__=ConfigDict(extra=extra),
        __module__=func.__module__,
        **fields,
    )
    return model, renames


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
        self._args_model, self._renames = _args_model(self.name, func)
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
        # Validation turns nested objects into the declared models and aliases
        # into parameter names; extra arguments pass only with **kwargs
        validated = self._args_model.model_validate(kwargs)
        kwargs = {
            self._renames.get(name, name): getattr(validated, name)
            for name in self._args_model.model_fields
        }
        extra = validated.model_extra or {}
        # An extra named like a declared parameter would replace its value
        clashes = [k for k in extra if k in kwargs]
        if clashes:
            raise ValidationError.from_exception_data(
                self.name,
                [
                    {"type": "extra_forbidden", "loc": (k,), "input": extra[k]}
                    for k in clashes
                ],
            )
        kwargs.update(extra)
        if self._is_async:
            return await self.func(**kwargs)
        # A blocking sync function must not stall the event loop
        return await asyncio.to_thread(self.func, **kwargs)
