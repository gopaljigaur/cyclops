"""Tests for cyclops toolkit: @tool decorator, BaseTool, and type annotation handling."""

import asyncio
import inspect
from typing import Optional

import pytest

from cyclops.toolkit.decorators import tool
from cyclops.toolkit.tool import BaseTool, Tool, _annotation_to_json_type, _PYTHON_TO_JSON_TYPE
from cyclops.toolkit.types import ToolDefinition, ToolParameter


# ---------------------------------------------------------------------------
# _annotation_to_json_type
# ---------------------------------------------------------------------------

class TestAnnotationToJsonType:
    """Unit tests for the _annotation_to_json_type helper."""

    def test_str_maps_to_string(self):
        assert _annotation_to_json_type(str) == "string"

    def test_int_maps_to_integer(self):
        assert _annotation_to_json_type(int) == "integer"

    def test_float_maps_to_number(self):
        assert _annotation_to_json_type(float) == "number"

    def test_bool_maps_to_boolean(self):
        assert _annotation_to_json_type(bool) == "boolean"

    def test_list_maps_to_array(self):
        assert _annotation_to_json_type(list) == "array"

    def test_dict_maps_to_object(self):
        assert _annotation_to_json_type(dict) == "object"

    def test_empty_annotation_defaults_to_string(self):
        assert _annotation_to_json_type(inspect.Parameter.empty) == "string"

    def test_unknown_type_defaults_to_string(self):
        class Foo:
            pass

        assert _annotation_to_json_type(Foo) == "string"

    def test_optional_str_maps_to_string(self):
        """Optional[str] is Union[str, None]; should resolve to 'string'."""
        assert _annotation_to_json_type(Optional[str]) == "string"

    def test_optional_int_maps_to_integer(self):
        assert _annotation_to_json_type(Optional[int]) == "integer"

    def test_optional_float_maps_to_number(self):
        assert _annotation_to_json_type(Optional[float]) == "number"

    def test_optional_bool_maps_to_boolean(self):
        assert _annotation_to_json_type(Optional[bool]) == "boolean"


# ---------------------------------------------------------------------------
# @tool decorator
# ---------------------------------------------------------------------------

class TestToolDecorator:
    """Tests for the @tool decorator behaviour."""

    def test_basic_decorator_uses_function_name(self):
        @tool
        def my_function(x: str) -> str:
            """Does something."""
            return x

        assert my_function.name == "my_function"

    def test_basic_decorator_uses_docstring(self):
        @tool
        def my_function(x: str) -> str:
            """Does something cool."""
            return x

        assert my_function.description == "Does something cool."

    def test_decorator_with_parentheses(self):
        @tool()
        def my_function(x: str) -> str:
            """Does something."""
            return x

        assert isinstance(my_function, Tool)

    def test_custom_name_override(self):
        @tool(name="custom_name")
        def my_function(x: str) -> str:
            """Does something."""
            return x

        assert my_function.name == "custom_name"

    def test_custom_description_override(self):
        @tool(description="Custom description")
        def my_function(x: str) -> str:
            """Ignored docstring."""
            return x

        assert my_function.description == "Custom description"

    def test_returns_tool_instance(self):
        @tool
        def my_function(x: str) -> str:
            """A tool."""
            return x

        assert isinstance(my_function, Tool)

    def test_no_docstring_fallback_description(self):
        @tool
        def mystery():
            pass

        assert mystery.description != ""

    def test_decorator_produces_correct_definition_type(self):
        @tool
        def greet(name: str) -> str:
            """Greet a person."""
            return f"Hello, {name}"

        assert isinstance(greet.definition, ToolDefinition)

    def test_parameter_types_extracted_correctly(self):
        @tool
        def calculator(a: int, b: float, label: str, active: bool) -> str:
            """Multi-type calculator."""
            return ""

        params = calculator.definition.parameters
        assert params["a"].type == "integer"
        assert params["b"].type == "number"
        assert params["label"].type == "string"
        assert params["active"].type == "boolean"

    def test_required_parameter_detection(self):
        @tool
        def func_with_default(x: str, y: str = "default") -> str:
            """Has a default."""
            return x

        params = func_with_default.definition.parameters
        assert params["x"].required is True
        assert params["y"].required is False

    def test_optional_annotation_is_not_required(self):
        @tool
        def func(name: str, nickname: Optional[str] = None) -> str:
            """Optional nickname."""
            return name

        params = func.definition.parameters
        assert params["name"].required is True
        assert params["nickname"].required is False
        assert params["nickname"].type == "string"

    def test_optional_str_annotation_type_is_string(self):
        @tool
        def func(value: Optional[str] = None) -> str:
            """Optional string param."""
            return value or ""

        params = func.definition.parameters
        assert params["value"].type == "string"


# ---------------------------------------------------------------------------
# BaseTool subclassing
# ---------------------------------------------------------------------------

class TestBaseTool:
    """Tests for direct BaseTool subclass usage."""

    def test_subclass_must_implement_execute(self):
        class MyTool(BaseTool):
            pass

        t = MyTool(name="my_tool", description="test")
        with pytest.raises(NotImplementedError):
            asyncio.run(t.execute())

    def test_subclass_execute_async(self):
        class EchoTool(BaseTool):
            async def execute(self, text: str) -> str:
                return text

        t = EchoTool(name="echo", description="Echoes text")
        result = asyncio.run(t.execute(text="hello"))
        assert result == "hello"

    def test_build_definition_from_subclass_signature(self):
        class MultiplyTool(BaseTool):
            async def execute(self, a: int, b: int) -> int:
                return a * b

        t = MultiplyTool(name="multiply", description="Multiply two numbers")
        defn = t.definition
        assert defn.name == "multiply"
        assert defn.description == "Multiply two numbers"
        assert "a" in defn.parameters
        assert "b" in defn.parameters
        assert defn.parameters["a"].type == "integer"
        assert defn.parameters["b"].type == "integer"

    def test_definition_property_returns_correct_object(self):
        class SimpleTool(BaseTool):
            async def execute(self, x: str) -> str:
                return x

        t = SimpleTool(name="simple", description="Simple tool")
        assert t.definition is t._definition


# ---------------------------------------------------------------------------
# Tool (function-based)
# ---------------------------------------------------------------------------

class TestFunctionTool:
    """Tests for the function-based Tool class."""

    def test_sync_function_executes_correctly(self):
        def add(a: int, b: int) -> int:
            return a + b

        t = Tool(name="add", description="Add numbers", func=add)
        result = asyncio.run(t.execute(a=2, b=3))
        assert result == 5

    def test_async_function_executes_correctly(self):
        async def multiply(a: int, b: int) -> int:
            return a * b

        t = Tool(name="multiply", description="Multiply", func=multiply)
        result = asyncio.run(t.execute(a=4, b=5))
        assert result == 20

    def test_definition_excludes_self_parameter(self):
        def func(x: str) -> str:
            return x

        t = Tool(name="t", description="d", func=func)
        assert "self" not in t.definition.parameters

    def test_definition_name_matches(self):
        def my_func(x: str) -> str:
            return x

        t = Tool(name="my_tool", description="desc", func=my_func)
        assert t.definition.name == "my_tool"

    def test_definition_description_matches(self):
        def my_func(x: str) -> str:
            return x

        t = Tool(name="my_tool", description="A description", func=my_func)
        assert t.definition.description == "A description"

    def test_no_parameters_tool(self):
        def ping() -> str:
            return "pong"

        t = Tool(name="ping", description="Ping the server", func=ping)
        assert t.definition.parameters == {}

    def test_all_json_types_in_single_tool(self):
        def complex_func(
            s: str,
            i: int,
            f: float,
            b: bool,
            lst: list,
            dct: dict,
        ) -> str:
            return ""

        t = Tool(name="complex", description="Complex func", func=complex_func)
        params = t.definition.parameters
        assert params["s"].type == "string"
        assert params["i"].type == "integer"
        assert params["f"].type == "number"
        assert params["b"].type == "boolean"
        assert params["lst"].type == "array"
        assert params["dct"].type == "object"
