# MCP

The Model Context Protocol (MCP) is an open standard for exposing tools to LLMs. Cyclops ships a lightweight client and server so you can either consume tools from any MCP server or publish your Cyclops tools as an MCP server.

## MCPClient

`MCPClient` connects to an external MCP server over stdio and lets you list and call its tools programmatically.

### connect_stdio

Pass a command list (`[executable, *args]`) and optionally a dict of environment variables:

```python
import asyncio
from cyclops import MCPClient


async def main():
    client = MCPClient()
    await client.connect_stdio(["python", "my_mcp_server.py"])
    # ...
    await client.disconnect()


asyncio.run(main())
```

### list_tools

Returns a list of dicts, each with `"name"`, `"description"`, and `"input_schema"` keys:

```python
tools = await client.list_tools()
for t in tools:
    print(t["name"], "—", t["description"])
```

### call_tool

Calls a named tool with a dict of arguments and returns the result as a string:

```python
result = await client.call_tool("add", {"a": 5, "b": 3})
print(result)  # "8.0"
```

### disconnect

Always disconnect when done to release the subprocess:

```python
await client.disconnect()
```

### Full example

```python
import asyncio
from cyclops import MCPClient


async def main():
    client = MCPClient()
    try:
        await client.connect_stdio(["python", "mcp_server.py"])

        tools = await client.list_tools()
        print("Available tools:", [t["name"] for t in tools])

        result = await client.call_tool("greet", {"name": "World"})
        print(result)  # "Hello, World!"
    finally:
        await client.disconnect()


asyncio.run(main())
```

## MCPServer

`MCPServer` wraps a `ToolRegistry` and exposes its tools over the stdio MCP protocol. Any MCP-compatible client (Claude Desktop, Cursor, tiny-agents) can connect to it.

### add_tool

Register a Cyclops tool with the server:

```python
import asyncio
from cyclops import MCPServer
from cyclops.toolkit import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


async def main():
    server = MCPServer(name="math-server", load_plugins=False)
    server.add_tool(add)
    server.add_tool(greet)
    await server.run_stdio()


asyncio.run(main())
```

### run_stdio

Starts the MCP server and blocks until the client disconnects. Run this script as a subprocess from your MCP host configuration.

### Plugin auto-discovery

Set `load_plugins=True` (the default) to have `MCPServer` automatically load all installed Cyclops toolkit plugins via entry points. This is the zero-config way to expose a whole ecosystem of tools.

```python
server = MCPServer(name="my-agent-tools")  # load_plugins=True by default
await server.run_stdio()
```

## Using MCP tools with Agent

To bridge an MCP server into a Cyclops agent, connect the client, convert its tools into `Tool` objects, and pass them to the agent.

```python
import asyncio
from cyclops import Agent, AgentConfig, MCPClient
from cyclops.toolkit.tool import Tool, BaseTool
from typing import Any


class MCPProxyTool(BaseTool):
    """Wraps a single MCP tool so the agent can call it."""

    def __init__(self, client: MCPClient, name: str, description: str):
        super().__init__(name=name, description=description)
        self._client = client

    async def execute(self, **kwargs: Any) -> str:
        return await self._client.call_tool(self.name, kwargs)


async def main():
    client = MCPClient()
    await client.connect_stdio(["python", "mcp_server.py"])

    # Build proxy tools from the server's tool list.
    mcp_tools = await client.list_tools()
    proxy_tools = [
        MCPProxyTool(client, t["name"], t["description"] or "")
        for t in mcp_tools
    ]

    config = AgentConfig(model="groq/llama-3.1-8b-instant")
    agent = Agent(config, tools=proxy_tools)

    response = await agent.arun("Add 17 and 25, then greet Alice.")
    print(response)

    await client.disconnect()


asyncio.run(main())
```

## Connecting to the filesystem MCP server

The [MCP filesystem server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) is a popular MCP server that exposes file-system operations. Connect to it with:

```python
import asyncio
from cyclops import MCPClient


async def main():
    client = MCPClient()
    # The npx invocation downloads and runs the official filesystem server.
    await client.connect_stdio(
        ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )

    tools = await client.list_tools()
    print("Filesystem tools:", [t["name"] for t in tools])

    # List files in /tmp.
    result = await client.call_tool("list_directory", {"path": "/tmp"})
    print(result)

    await client.disconnect()


asyncio.run(main())
```

/// warning
Only point the filesystem MCP server at directories you intend to expose. It has full read/write access within the paths it is given.
///
