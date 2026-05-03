"""Tests for MCPBridge and MCPClientTool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


from cyclops.mcp.bridge import MCPBridge
from cyclops.mcp.tools import MCPClientTool, _mcp_schema_to_params
from cyclops.toolkit.types import ToolDefinition


# ---------------------------------------------------------------------------
# _mcp_schema_to_params
# ---------------------------------------------------------------------------


class TestMCPSchemaToParams:
    def test_empty_schema_returns_empty_params(self):
        assert _mcp_schema_to_params({}) == {}

    def test_basic_string_property(self):
        schema = {"properties": {"name": {"type": "string", "description": "A name"}}}
        params = _mcp_schema_to_params(schema)
        assert "name" in params
        assert params["name"].type == "string"
        assert params["name"].description == "A name"

    def test_required_fields_marked_correctly(self):
        schema = {
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a"],
        }
        params = _mcp_schema_to_params(schema)
        assert params["a"].required is True
        assert params["b"].required is False

    def test_all_json_types_preserved(self):
        schema = {
            "properties": {
                "s": {"type": "string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "arr": {"type": "array"},
                "obj": {"type": "object"},
            }
        }
        params = _mcp_schema_to_params(schema)
        assert params["s"].type == "string"
        assert params["i"].type == "integer"
        assert params["n"].type == "number"
        assert params["b"].type == "boolean"
        assert params["arr"].type == "array"
        assert params["obj"].type == "object"

    def test_unknown_type_falls_back_to_string(self):
        schema = {"properties": {"x": {"type": "exotic_type"}}}
        params = _mcp_schema_to_params(schema)
        assert params["x"].type == "string"

    def test_missing_type_falls_back_to_string(self):
        schema = {"properties": {"x": {"description": "no type field"}}}
        params = _mcp_schema_to_params(schema)
        assert params["x"].type == "string"


# ---------------------------------------------------------------------------
# MCPClientTool
# ---------------------------------------------------------------------------


def _make_tool(
    name="echo",
    description="Echo a string",
    input_schema=None,
):
    bridge = MagicMock(spec=MCPBridge)
    client = MagicMock()
    schema = input_schema or {
        "properties": {"text": {"type": "string", "description": "Input text"}},
        "required": ["text"],
    }
    return MCPClientTool(
        bridge=bridge,
        client=client,
        name=name,
        description=description,
        input_schema=schema,
    )


class TestMCPClientTool:
    def test_name_and_description_set(self):
        t = _make_tool(name="my_tool", description="Does stuff")
        assert t.name == "my_tool"
        assert t.description == "Does stuff"

    def test_definition_is_tool_definition(self):
        t = _make_tool()
        assert isinstance(t.definition, ToolDefinition)

    def test_definition_params_from_schema(self):
        t = _make_tool(
            input_schema={
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "string"},
                },
                "required": ["a"],
            }
        )
        params = t.definition.parameters
        assert params["a"].type == "integer"
        assert params["a"].required is True
        assert params["b"].type == "string"
        assert params["b"].required is False

    def test_definition_json_schema_shape(self):
        t = _make_tool()
        schema = t.definition.to_json_schema()
        assert schema["type"] == "object"
        assert "text" in schema["properties"]
        assert "text" in schema["required"]

    def test_execute_delegates_to_bridge(self):
        bridge = MagicMock(spec=MCPBridge)
        bridge.call_tool.return_value = "hello"
        client = MagicMock()
        t = MCPClientTool(
            bridge=bridge,
            client=client,
            name="echo",
            description="echo",
            input_schema={
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        )
        result = asyncio.run(t.execute(text="hi"))
        bridge.call_tool.assert_called_once_with(client, "echo", {"text": "hi"})
        assert result == "hello"

    def test_execute_passes_all_kwargs(self):
        bridge = MagicMock(spec=MCPBridge)
        bridge.call_tool.return_value = "ok"
        client = MagicMock()
        t = MCPClientTool(
            bridge=bridge,
            client=client,
            name="calc",
            description="calc",
            input_schema={
                "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}
            },
        )
        asyncio.run(t.execute(a=1, b=2))
        bridge.call_tool.assert_called_once_with(client, "calc", {"a": 1, "b": 2})


# ---------------------------------------------------------------------------
# MCPBridge (unit — no real subprocess)
# ---------------------------------------------------------------------------


class TestMCPBridgeUnit:
    """Test MCPBridge sync/async bridging logic without a real MCP server."""

    def test_bridge_starts_background_thread(self):
        bridge = MCPBridge()
        assert bridge._thread.is_alive()
        bridge.stop()

    def test_bridge_stop_terminates_thread(self):
        bridge = MCPBridge()
        bridge.stop()
        bridge._thread.join(timeout=2)
        assert not bridge._thread.is_alive()

    def test_run_coroutine_returns_value(self):
        bridge = MCPBridge()
        try:

            async def _coro():
                return 42

            assert bridge._run(_coro()) == 42
        finally:
            bridge.stop()

    def test_connected_names_empty_on_init(self):
        bridge = MCPBridge()
        try:
            assert bridge.connected_names == []
        finally:
            bridge.stop()

    def test_disconnect_unknown_name_is_noop(self):
        bridge = MCPBridge()
        try:
            bridge.disconnect("nonexistent")  # should not raise
        finally:
            bridge.stop()

    def test_connect_stores_client(self):
        bridge = MCPBridge()
        try:
            mock_client = MagicMock()
            mock_client.is_connected = True

            with patch("cyclops.mcp.bridge.MCPClient") as MockClass:
                instance = MockClass.return_value
                instance.connect_stdio = AsyncMock()
                instance.disconnect = AsyncMock()
                bridge.connect("myserver", ["echo", "hello"])
                assert "myserver" in bridge._clients
        finally:
            bridge.stop()
