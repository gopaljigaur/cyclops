---
title: Tools
description: Build tools for your agent.
---

## @tool decorator

The easiest way to create a tool is the `@tool` decorator. It wraps any Python function (sync or async) and derives the tool name from the function name and the description from the docstring.

```python
from cyclops.toolkit import tool


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Sunny, 72F in {location}"
```

Both forms work — with and without parentheses:

```python
@tool()
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Sunny, 72F in {location}"
```

To override the name or description, pass keyword arguments:

```python
@tool(name="weather_lookup", description="Fetch live weather data for any city.")
def _internal_get_weather(location: str) -> str:
    return f"Sunny, 72F in {location}"
```

## BaseTool subclass

For tools that need to hold state or require complex initialisation, subclass `BaseTool` and override `execute()`.

```python
from cyclops.toolkit.tool import BaseTool


class DatabaseTool(BaseTool):
    def __init__(self):
        super().__init__(name="query_db", description="Query an in-memory database.")
        self.data = {"users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]}

    async def execute(self, table: str, user_id: int = None) -> str:
        rows = self.data.get(table, [])
        if user_id is not None:
            rows = [r for r in rows if r.get("id") == user_id]
        return str(rows)


db_tool = DatabaseTool()
```

Define `execute()` with the exact parameter signature you want exposed to the LLM. Cyclops inspects that signature to build the JSON schema automatically.

## Sync vs async

Both sync functions and async coroutines work. The agent runs sync tools inside `asyncio.run()` when needed, and awaits async tools directly in async contexts. Prefer async for I/O-bound work.

```python
import httpx
from cyclops.toolkit import tool


@tool
async def fetch_url(url: str) -> str:
    """Fetch the text content of a URL."""
    async with httpx.AsyncClient() as client:
        r = await client.get(url, timeout=10)
        return r.text[:500]
```

## Type annotations and JSON schema

Cyclops converts Python type annotations to JSON schema types automatically.

| Python type | JSON schema type |
|---|---|
| `str` | `"string"` |
| `int` | `"integer"` |
| `float` | `"number"` |
| `bool` | `"boolean"` |
| `list` | `"array"` |
| `dict` | `"object"` |
| unannotated | `"string"` (default) |

`Optional[T]` is treated the same as `T`. The parameter becomes optional in the schema if it has a default value.

```python
@tool
def calculate(operation: str, a: float, b: float) -> str:
    """Perform a basic math operation: add, subtract, multiply, divide."""
    ops = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}
    return str(ops.get(operation, "unknown operation"))
```

This produces a JSON schema with `operation` as `"string"` and `a`, `b` as `"number"`, all required.

Parameters with default values become optional in the schema:

```python
@tool
def search(query: str, max_results: int = 5) -> str:
    """Search for information."""
    return f"Results for '{query}' (max {max_results})"
```

Here `query` is required and `max_results` is optional with a default of `5`.

## ToolRegistry

`ToolRegistry` is a named collection of tools. It is useful when managing tools separately from any single agent, such as in a plugin or a shared tool library.

```python
from cyclops.toolkit.registry import ToolRegistry
from cyclops.toolkit import tool


registry = ToolRegistry()


@tool(registry=registry)
def ping(host: str) -> str:
    """Ping a host and return the round-trip time."""
    return f"pong from {host} in 12ms"


# Or register an existing tool manually:
registry.register(ping)

# Inspect:
print(registry.list_tools())       # ['ping']
print(registry.get_tool("ping"))   # <Tool ping>

# Execute a tool by name:
import asyncio
result = asyncio.run(registry.execute_tool("ping", host="example.com"))
```

## Passing tools to Agent

Pass any list of tools (decorated functions or `BaseTool` instances) to the `Agent` constructor.

```python
from cyclops import Agent, AgentConfig

config = AgentConfig(model="groq/llama-3.1-8b-instant")
agent = Agent(config, tools=[get_weather, db_tool, calculate])

print(agent.run("What is the weather in Tokyo?"))
```

Pull tools from a registry:

```python
tools = [registry.get_tool(name) for name in registry.list_tools()]
agent = Agent(config, tools=tools)
```

:::note
If the model does not support native function calling (some Ollama models, for example), set `tool_mode="naive"` in `AgentConfig`. Cyclops also falls back to prompt-based tool invocation automatically when `tool_mode="auto"` and the API returns a function-calling error.
:::
