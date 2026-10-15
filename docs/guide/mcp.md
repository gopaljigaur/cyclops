# MCP

The [Model Context Protocol](https://modelcontextprotocol.io) (MCP) is an open standard for exposing tools to LLMs. Cyclops ships a client and a server so you can either consume tools from any MCP server or publish your Cyclops tools as an MCP server.

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

### Timeouts

`connect_timeout` (default 30 seconds) bounds connecting, listing the server's tools, and opening HTTP connections. `call_timeout` (default 60 seconds, the TypeScript SDK's default request timeout) bounds each tool call, so a hung server cannot stall a run; it is the only limit on how long a streamable HTTP call may take. Raise it for long running tools, or pass `None` for no limit. A call that times out returns an error result. A timed out or cancelled call also sends the server `notifications/cancelled`, so the server can stop the tool.

`idle_timeout` (default none) closes a session that no call has used for that many seconds, and the next call opens a new one. It is off by default because a server may keep state per session, which a new session loses; turn it on to stop idle stdio processes or free remote connections.

### Reconnects

A stdio or SSE server that exits is seen at once. A dropped streamable HTTP session is seen by the call it breaks; the client does not watch idle streamable HTTP sessions. The next call then reconnects, and for a remote server the reconnect starts in the background right away. Calls share one reconnect and wait for it up to `call_timeout`; if it is still running then, the call returns an error result and the reconnect goes on. Attempts back off exponentially with full jitter and stop after `max_retries`. The names follow the TypeScript SDK's reconnection options, in seconds; the defaults are cyclops' own, long enough to ride out a server restart:

```python
from cyclops.mcp import ReconnectionOptions

await mcp.attach(
    "remote",
    url="https://example.com/mcp",
    reconnection=ReconnectionOptions(
        initial_reconnection_delay=1.0,     # first retry waits up to 1 second
        max_reconnection_delay=30.0,        # no retry waits longer than this
        reconnection_delay_grow_factor=2.0,
        max_retries=5,
    ),
)
```

A call that meets a dropped connection returns an error result instead of being retried, since the tool may already have run. A call that a restarted streamable HTTP server answered with a 404, because it no longer knows the session, is retried once on a new session. Calls already sent on that session first get their own answer, so each of them that was refused is retried too.

A call can wait for a reconnect twice, each time up to `call_timeout`: once when it starts without a session, and once more when its first attempt hits a terminated session. So the worst case is three times `call_timeout`: two reconnect waits plus the call itself. A server that stays down fails the first wait, so with the defaults it costs each call 60 seconds at most, while the reconnect behind it sleeps up to 31 seconds in all, plus up to `max_retries + 1` times `connect_timeout` against a host that does not answer.

### Status

`status(name)` gives a server's `ServerStatus`: its `state` (`"connecting"`, `"connected"`, `"disconnected"` or `"failed"`) and the `error` behind it. It reflects the last call or reconnect, and the exit of a stdio or SSE server: a streamable HTTP server that went away while idle reads `"connected"` until a call finds out. A server is `"disconnected"` after a drop or an idle close, `"connecting"` while reconnecting, and `"failed"` once every retry failed; the next call starts reconnecting again.

```python
for name in mcp.servers:
    status = mcp.status(name)
    print(name, status.state, status.error or "")
```

### Tool list changes

When a server sends `notifications/tools/list_changed`, the client lists its tools again, and `tools()` returns the new list. The client connects with the `initialize` handshake, so it gets the notification a server sends on the session; it does not open the `subscriptions/listen` stream of the 2026-07-28 protocol revision. Agents built before the change keep the tools they were given, so pass `mcp.tools()` per run to pick up changes.

### Authentication

`auth` takes any `httpx2.Auth` for a remote server, since the MCP SDK sends its requests with `httpx2`. For OAuth, pass the MCP SDK's `OAuthClientProvider`: it refreshes an expired access token, and on a 401 response it authorizes again and retries the request once.

```python
from mcp.client.auth import OAuthClientProvider

oauth = OAuthClientProvider(server_url, client_metadata, storage, redirect_handler, callback_handler)
await mcp.attach("remote", url=server_url, auth=oauth)
```

### Tool names

Tools are named `<server>__<tool>`, so tools from different servers do not collide. Set `prefix` to use another prefix, or `prefix=False` to keep the server's names. `attach` raises `ValueError` if a name is already taken by another attached server:

```python
await mcp.attach("math", ["python", "mcp_server.py"])                 # math__add
await mcp.attach("calc", ["python", "mcp_server.py"], prefix="m")     # m__add
await mcp.attach("plain", ["python", "mcp_server.py"], prefix=False)  # add
```

The server's input schema is sent to the model as is, including nested objects and `$defs`.

### Deferred tools

A server with many tools can keep them out of every completion until the model searches for them. `defer_loading=True` defers each tool of the server, including ones it lists later; see [Deferred tools](tools.md#deferred-tools):

```python
await mcp.attach("crm", url="https://example.com/mcp", defer_loading=True)
```

### Results

`execute()` returns a `ToolResult`. Text and image content become its `content` blocks (`TextContent`, `ImageContent`) and `structuredContent` becomes its `result`; the agent sends it to the model like any other tool result. A result with `isError` set becomes an error result carrying the server's text. The server's tool annotations (`readOnlyHint` and the others) become the tool's `ToolAnnotations`.

### Server sets per agent or per run

`attach` returns the server's tools and `tools()` takes a list of server names, so one client can serve agents with different server sets. Servers can be attached and detached at any point, from any task on the client's loop. A server's tools must be awaited on the loop that attached it; from sync code, use `MCPBridge`:

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

A tool that returns a `ToolResult` has its text and image content blocks served as MCP content. A tool that raises, or returns an error result, is served with `isError` set.

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
