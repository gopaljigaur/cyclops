"""Tests for cyclops toolkit: @tool decorator, BaseTool, schemas, results and registry."""

import asyncio
import time
from enum import Enum
from typing import Annotated, List, Literal, Optional

import pytest
from pydantic import BaseModel, Field, ValidationError

from cyclops.toolkit.decorators import tool
from cyclops.toolkit.plugins import PluginManager, Toolkit
from cyclops.toolkit.registry import ToolRegistry
from cyclops.toolkit.tool import BaseTool, Tool
from cyclops.toolkit.types import (
    ImageContent,
    TextContent,
    ToolAnnotations,
    ToolDefinition,
    ToolResult,
)
from tests import forward_ref_tools

# ---------------------------------------------------------------------------
# Parameter schemas
# ---------------------------------------------------------------------------


class Color(str, Enum):
    RED = "red"
    BLUE = "blue"


class Point(BaseModel):
    x: int
    y: int


class TestParameterSchema:
    """The JSON schema built from a function signature."""

    def test_list_has_items(self):
        @tool
        def f(tags: List[str]) -> str:
            """F."""
            return ""

        prop = f.definition.to_json_schema()["properties"]["tags"]
        assert prop["type"] == "array"
        assert prop["items"] == {"type": "string"}

    def test_literal_becomes_enum(self):
        @tool
        def f(mode: Literal["fast", "slow"]) -> str:
            """F."""
            return ""

        prop = f.definition.to_json_schema()["properties"]["mode"]
        assert prop["enum"] == ["fast", "slow"]

    def test_enum_is_referenced(self):
        @tool
        def f(color: Color) -> str:
            """F."""
            return ""

        schema = f.definition.to_json_schema()
        assert schema["$defs"]["Color"]["enum"] == ["red", "blue"]
        assert f.definition.parameters["color"].type == "string"

    def test_optional_allows_null(self):
        @tool
        def f(note: Optional[str] = None) -> str:
            """F."""
            return ""

        prop = f.definition.to_json_schema()["properties"]["note"]
        assert {"type": "null"} in prop["anyOf"]
        assert f.definition.parameters["note"].type == "string"

    def test_nested_model(self):
        @tool
        def f(point: Point) -> str:
            """F."""
            return ""

        schema = f.definition.to_json_schema()
        assert schema["$defs"]["Point"]["required"] == ["x", "y"]
        assert f.definition.parameters["point"].type == "object"

    def test_defaults_are_kept(self):
        @tool
        def f(limit: int = 5) -> str:
            """F."""
            return ""

        schema = f.definition.to_json_schema()
        assert schema["properties"]["limit"]["default"] == 5
        assert "limit" not in schema.get("required", [])

    def test_google_docstring_descriptions(self):
        @tool
        def f(query: str, limit: int = 5) -> str:
            """Search the index.

            Args:
                query: Text to search for.
                limit: Maximum number of hits.
            """
            return ""

        props = f.definition.to_json_schema()["properties"]
        assert props["query"]["description"] == "Text to search for."
        assert props["limit"]["description"] == "Maximum number of hits."
        assert f.description == "Search the index."

    def test_numpy_docstring_descriptions(self):
        @tool
        def f(query: str) -> str:
            """Search the index.

            Parameters
            ----------
            query : str
                Text to search for.
            """
            return ""

        props = f.definition.to_json_schema()["properties"]
        assert props["query"]["description"] == "Text to search for."
        assert f.description == "Search the index."

    def test_annotated_field_description(self):
        @tool
        def f(query: Annotated[str, Field(description="Text to search for.")]) -> str:
            """Search."""
            return ""

        props = f.definition.to_json_schema()["properties"]
        assert props["query"]["description"] == "Text to search for."
        assert f.definition.parameters["query"].description == "Text to search for."

    def test_unannotated_parameter_accepts_anything(self):
        def f(x):
            return x

        t = Tool(name="f", description="d", func=f)
        assert "type" not in t.definition.to_json_schema()["properties"]["x"]

    def test_execute_validates_nested_model(self):
        @tool
        def f(point: Point) -> int:
            """F."""
            return point.x + point.y

        assert asyncio.run(f.execute(point={"x": 1, "y": 2})) == 3

    def test_postponed_annotations_from_another_module(self):
        # list["Point"] resolves in the defining module; the return type,
        # imported only for type checking, is never evaluated
        t = Tool("total", "Sum", forward_ref_tools.total)
        schema = t.definition.to_json_schema()
        assert schema["properties"]["points"]["items"] == {"$ref": "#/$defs/Point"}
        assert asyncio.run(t.execute(points=[{"x": 1, "y": 2}])) == 3

    def test_execute_accepts_alias(self):
        @tool
        def f(query: Annotated[str, Field(alias="q")]) -> str:
            """F."""
            return query

        assert "q" in f.definition.to_json_schema()["properties"]
        assert asyncio.run(f.execute(q="hi")) == "hi"

    def test_execute_rejects_unknown_arguments(self):
        @tool
        def f(n: int) -> int:
            """F."""
            return n

        with pytest.raises(ValidationError):
            asyncio.run(f.execute(n=1, extra=2))

    def test_execute_passes_extra_arguments_to_var_keyword(self):
        @tool
        def f(n: int, **rest) -> dict:
            """F."""
            return {"n": n, **rest}

        assert asyncio.run(f.execute(n=1, extra=2)) == {"n": 1, "extra": 2}

    def test_docstring_description_wins_over_annotated(self):
        @tool
        def f(query: Annotated[str, Field(description="From Annotated.")]) -> str:
            """F.

            Args:
                query: From the docstring.
            """
            return query

        props = f.definition.to_json_schema()["properties"]
        assert props["query"]["description"] == "From the docstring."

    def test_list_of_models(self):
        @tool
        def f(points: List[Point]) -> int:
            """F."""
            return sum(p.x for p in points)

        prop = f.definition.to_json_schema()["properties"]["points"]
        assert prop["items"] == {"$ref": "#/$defs/Point"}
        assert asyncio.run(f.execute(points=[{"x": 1, "y": 0}, {"x": 2, "y": 0}])) == 3

    def test_any_parameter_name_works(self):
        @tool
        def f(_private: int, schema: str, json: str = "j") -> str:
            """F.

            Args:
                _private: A number.
            """
            return f"{_private}{schema}{json}"

        props = f.definition.to_json_schema()["properties"]
        assert set(props) == {"_private", "schema", "json"}
        assert props["_private"]["description"] == "A number."
        assert set(f.definition.parameters) == {"_private", "schema", "json"}
        assert asyncio.run(f.execute(_private=1, schema="s")) == "1sj"

    def test_execute_rejects_invalid_arguments(self):
        @tool
        def f(n: int) -> int:
            """F."""
            return n

        with pytest.raises(ValidationError):
            asyncio.run(f.execute(n="not a number"))


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
            async def execute(self, text: str) -> str:  # type: ignore[override]
                return text

        t = EchoTool(name="echo", description="Echoes text")
        result = asyncio.run(t.execute(text="hello"))
        assert result == "hello"

    def test_build_definition_from_subclass_signature(self):
        class MultiplyTool(BaseTool):
            async def execute(self, a: int, b: int) -> int:  # type: ignore[override]
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
            async def execute(self, x: str) -> str:  # type: ignore[override]
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

    def test_sync_function_does_not_block_event_loop(self):
        def slow() -> str:
            time.sleep(0.2)
            return "done"

        t = Tool(name="slow", description="Slow", func=slow)

        async def main():
            ticks = 0

            async def tick():
                nonlocal ticks
                while True:
                    await asyncio.sleep(0.01)
                    ticks += 1

            ticker = asyncio.create_task(tick())
            result = await t.execute()
            ticker.cancel()
            return result, ticks

        result, ticks = asyncio.run(main())
        assert result == "done"
        assert ticks > 5

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


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


class TestAnnotations:
    def test_default_annotations_are_unset(self):
        @tool
        def f() -> str:
            """F."""
            return ""

        assert f.annotations == ToolAnnotations()
        assert f.annotations.read_only is None

    def test_decorator_annotations(self):
        @tool(annotations=ToolAnnotations(read_only=True, open_world=False))
        def f() -> str:
            """F."""
            return ""

        assert f.annotations.read_only is True
        assert f.definition.annotations.open_world is False

    def test_base_tool_annotations(self):
        class Delete(BaseTool):
            async def execute(self, path: str) -> str:  # type: ignore[override]
                return path

        t = Delete("delete", "Delete a file", ToolAnnotations(destructive=True))
        assert t.definition.annotations.destructive is True


# ---------------------------------------------------------------------------
# ToolResult
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success_result(self):
        r = ToolResult.success_result({"a": 1})
        assert r.success is True
        assert r.is_error is False
        assert r.to_message_content() == '{"a": 1}'

    def test_string_result_is_not_quoted(self):
        assert ToolResult.success_result("done").to_message_content() == "done"

    def test_error_result(self):
        r = ToolResult.error_result("boom")
        assert r.is_error is True
        assert r.to_message_content() == "Error: boom"

    def test_failure_without_error_text(self):
        r = ToolResult(success=False, content=[TextContent(text="denied")])
        assert r.to_message_content() == "Error: tool execution failed\ndenied"
        assert ToolResult(success=False).to_message_content() == (
            "Error: tool execution failed"
        )

    def test_error_with_content(self):
        r = ToolResult(
            success=False,
            error="partial failure",
            content=[TextContent(text="1 of 2 done")],
        )
        assert r.to_message_content() == "Error: partial failure\n1 of 2 done"

    def test_mixed_content_keeps_order(self):
        image = ImageContent(data="aGk=", mime_type="image/png")
        r = ToolResult(
            content=[TextContent(text="before"), image, TextContent(text="after")]
        )
        assert [p["type"] for p in r.to_message_content()] == [
            "text",
            "image_url",
            "text",
        ]
        assert r.to_message_content()[2] == {"type": "text", "text": "after"}

    def test_text_content(self):
        r = ToolResult(content=[TextContent(text="a"), TextContent(text="b")])
        assert r.to_message_content() == "a\nb"

    def test_content_takes_precedence_over_structured_data(self):
        r = ToolResult(content=[TextContent(text="2 rows")], result=[1, 2])
        assert r.to_message_content() == "2 rows"

    def test_image_content_renders_parts(self):
        r = ToolResult(
            content=[
                TextContent(text="chart"),
                ImageContent(data="aGk=", mime_type="image/png"),
            ]
        )
        assert r.to_message_content() == [
            {"type": "text", "text": "chart"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}},
        ]

    def test_content_blocks_from_dicts(self):
        r = ToolResult.model_validate(
            {"content": [{"type": "image", "data": "x", "mime_type": "image/jpeg"}]}
        )
        assert isinstance(r.content[0], ImageContent)

    def test_empty_result(self):
        assert ToolResult().to_message_content() == ""


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class TestToolRegistry:
    def test_duplicate_name_is_rejected(self):
        registry = ToolRegistry()
        registry.register(Tool("ping", "Ping", lambda: "pong"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(Tool("ping", "Ping again", lambda: "pong"))

    def test_name_is_free_after_removal(self):
        registry = ToolRegistry()
        registry.register(Tool("ping", "Ping", lambda: "pong"))
        registry.remove_tool("ping")
        registry.register(Tool("ping", "Ping again", lambda: "pong"))
        assert registry.get_tool("ping").description == "Ping again"

    def test_plugin_duplicate_skips_only_that_tool(self):
        class Tools(Toolkit):
            a = Tool("a", "A", lambda: "a")
            b = Tool("b", "B", lambda: "b")

        manager = PluginManager()
        manager.registry.register(Tool("a", "Existing", lambda: "x"))
        manager.register(Tools())
        manager.register_tools()
        assert manager.registry.get_tool("a").description == "Existing"
        assert manager.registry.get_tool("b") is not None
