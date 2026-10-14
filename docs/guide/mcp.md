# MCP

The Model Context Protocol (MCP) is an open standard for exposing tools to LLMs. Cyclops ships a client and a server so you can either consume tools from any MCP server or publish your Cyclops tools as an MCP server.

## MCPClient

`MCPClient` is an async client that runs on your event loop. Attach servers by name, pass their tools to an agent, and detach them when you are done. Leaving the `async with` block disconnects every server.

```python
import asyncio
from cyclops import Agent, AgentConfig, MCPClient


async def main():
    async with MCPClient() as mcp:
        await mcp.attach("math", ["python", "mcp_server.py"])

        agent = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"), tools=mcp.tools())
        response = await agent.arun("Add 17 and 25, then greet Alice.")
        print(response)


asyncio.run(main())
```

### Transports

Pass `command` for a stdio server, or `url` for a remote one. A url uses streamable HTTP; set `transport="sse"` for servers that only speak the older SSE transport. Headers are set per server:

```python
await mcp.attach("local", ["python", "mcp_server.py"], env={"LOG_LEVEL": "debug"})
await mcp.attach("remote", url="https://example.com/mcp", headers={"Authorization": "Bearer ..."})
await mcp.attach("legacy", url="https://example.com/sse", transport="sse")
```

### Timeouts and reconnects

`connect_timeout` (default 30 seconds) bounds connecting and the HTTP requests; `call_timeout` (default none) bounds each tool call. A call that times out returns an error result.

When a session drops, the next call reconnects. A call that was sent before the drop returns an error result instead of being retried, since the tool may already have run.

### Tool names

Tools are named `<server>__<tool>`, so tools from different servers do not collide. Set `prefix` to use another prefix, or `prefix=False` to keep the server's names:

```python
await mcp.attach("math", ["python", "mcp_server.py"])                 # math__add
await mcp.attach("calc", ["python", "mcp_server.py"], prefix="m")     # m__add
await mcp.attach("plain", ["python", "mcp_server.py"], prefix=False)  # add
```

The server's input schema is sent to the model as is, including nested objects and `$defs`.

### Results

`execute()` returns a `ToolResult`. A result with `isError` set becomes an error result. Text content becomes a string; when the result has images, it becomes a list of content parts (`{"type": "text", ...}` and `{"type": "image_url", ...}` with a data URL).

### Server sets per agent or per run

`attach` returns the server's tools and `tools()` takes a list of server names, so one client can serve agents with different server sets. Servers can be attached and detached at any point, from any task:

```python
await mcp.attach("files", ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
await mcp.attach("search", url="https://example.com/mcp")

reader = Agent(config, tools=mcp.tools(["files"]))
researcher = Agent(config, tools=mcp.tools(["files", "search"]))

await mcp.detach("search")
```

### Calling tools directly

`call_tool` returns the raw MCP `CallToolResult`:

```python
result = await mcp.call_tool("math", "add", {"a": 5, "b": 3})
print(result.content[0].text)  # "8.0"
```

## MCPBridge

For scripts without an event loop, `MCPBridge` runs an `MCPClient` on a background thread. Its tools work with the sync `agent.run()`:

```python
from cyclops import Agent, AgentConfig, MCPBridge

bridge = MCPBridge()
try:
    tools = bridge.attach("math", ["python", "mcp_server.py"])
    agent = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"), tools=tools)
    print(agent.run("Add 17 and 25."))
finally:
    bridge.stop()
```

`bridge.attach` takes the same arguments as `MCPClient.attach`.

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

## Connecting to the filesystem MCP server

The [MCP filesystem server](https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem) exposes file system operations:

```python
import asyncio
from cyclops import MCPClient


async def main():
    async with MCPClient() as mcp:
        # The npx invocation downloads and runs the filesystem server.
        tools = await mcp.attach(
            "fs", ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        )
        print("Filesystem tools:", [t.name for t in tools])

        result = await mcp.call_tool("fs", "list_directory", {"path": "/tmp"})
        print(result.content[0].text)


asyncio.run(main())
```

/// warning
Only point the filesystem MCP server at directories you intend to expose. It has full read/write access within the paths it is given.
///
