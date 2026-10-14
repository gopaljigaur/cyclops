# API Reference

This page provides a concise reference for every public class and function in Cyclops. Each section links to the source module and shows the class signature.

---

## cyclops.core

### Agent

The main agent class. Wraps an LLM and orchestrates tool calls, conversation history, streaming, and cost tracking.

```python
class Agent:
    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List] = None,
        memory: Optional[Memory] = None,
    ): ...
```

**Methods:** `run`, `arun`, `stream`, `astream`, `run_with_response`, `arun_with_response`, `reset`

**Properties:** `messages`

See [Agents guide](guide/agents.md) for full documentation.

---

### AgentConfig

Pydantic model that holds all LLM and agent runtime settings.

```python
class AgentConfig(BaseModel):
    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tool_mode: Literal["auto", "native", "naive"] = "auto"
    router: Optional[Any] = None   # LiteLLM Router
    max_iterations: int = 10
    hooks: Optional[AgentHooks] = None
```

---

### AgentResponse

Returned by `run_with_response()` and `arun_with_response()`.

```python
class AgentResponse(BaseModel):
    content: str
    tool_calls: List[ToolCall]
    model: str
    stop_reason: Literal["completed", "max_iterations"]
    tokens_used: Optional[int]
    cost: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    cached_tokens: Optional[int]
```

---

### ToolCall

Records a single tool invocation within an `AgentResponse`.

```python
class ToolCall(BaseModel):
    id: str
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any]
```

---

### AgentHooks

Lifecycle callback base class. Subclass and override any methods you need.

```python
class AgentHooks:
    def on_run_start(self, input_message: str) -> None: ...
    def on_run_end(self, content: str) -> None: ...
    def on_llm_start(self, messages: List[Dict[str, Any]]) -> None: ...
    def on_llm_end(self, response: Any) -> None: ...
    def on_llm_error(self, error: Exception) -> None: ...
    def on_tool_start(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]: ...
    def on_tool_end(self, tool_name: str, args: Dict[str, Any], result: str) -> None: ...
    def on_tool_error(self, tool_name: str, args: Dict[str, Any], error: Exception) -> None: ...
```

See [Hooks guide](guide/hooks.md) for full documentation.

---

### Message

Represents a single conversation turn. Covers all LiteLLM message types.

```python
class Message(BaseModel):
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
```

---

## cyclops.core.memory

### Memory

Abstract base class for all memory backends.

```python
class Memory(ABC):
    async def store(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None: ...
    async def retrieve(self, key: str) -> Optional[Any]: ...
    async def list_keys(self) -> List[str]: ...
    async def clear(self) -> None: ...
```

---

### InMemoryStorage

In-process dict-backed storage. Data is lost on process exit.

```python
class InMemoryStorage(Memory):
    def __init__(self): ...
```

---

### FileStorage

JSON-file-backed persistent storage.

```python
class FileStorage(Memory):
    def __init__(self, path: str): ...
```

See [Memory guide](guide/memory.md) for examples.

---

## cyclops.toolkit

### tool decorator

```python
def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
    annotations: Optional[ToolAnnotations] = None,
) -> Union[Tool, Callable[[Callable], Tool]]: ...
```

---

### BaseTool

Abstract base class for all tools. Subclass this to create stateful tools.

```python
class BaseTool(ABC):
    def __init__(
        self,
        name: str,
        description: str,
        annotations: Optional[ToolAnnotations] = None,
    ): ...
    async def execute(self, **kwargs) -> Any: ...

    @property
    def definition(self) -> ToolDefinition: ...
```

---

### Tool

Function-wrapping tool created by the `@tool` decorator.

```python
class Tool(BaseTool):
    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        annotations: Optional[ToolAnnotations] = None,
    ): ...
    async def execute(self, **kwargs) -> Any: ...
```

---

### ToolResult

What a tool may return instead of a plain value. See [Tool results](guide/tools.md#tool-results).

```python
class ToolResult(BaseModel):
    success: bool = True
    result: Any = None                   # structured data
    error: Optional[str] = None
    content: List[TextContent | ImageContent] = []

    @property
    def is_error(self) -> bool: ...
    @classmethod
    def success_result(cls, result: Any) -> "ToolResult": ...
    @classmethod
    def error_result(cls, error: str, result: Any = None) -> "ToolResult": ...
    def to_message_content(self) -> Union[str, List[Dict[str, Any]]]: ...
```

---

### ToolAnnotations

Behaviour hints mirroring MCP tool annotations. Each field defaults to `None`.

```python
class ToolAnnotations(BaseModel):
    read_only: Optional[bool] = None
    destructive: Optional[bool] = None
    idempotent: Optional[bool] = None
    open_world: Optional[bool] = None
```

---

### ToolRegistry

Named collection of tools with lookup and execution helpers. `register()` raises `ValueError` for a name that is already registered.

```python
class ToolRegistry:
    def register(self, tool: BaseTool) -> None: ...
    def register_function(self, name: str, description: str, func) -> None: ...
    def get_tool(self, name: str) -> Optional[BaseTool]: ...
    def list_tools(self) -> List[str]: ...
    def get_definitions(self) -> Dict[str, ToolDefinition]: ...
    async def execute_tool(self, tool_name: str, **kwargs) -> Any: ...
    def remove_tool(self, name: str) -> bool: ...
    def clear(self) -> None: ...
```

---

### Toolkit

Base class for plugin packages. Define `BaseTool` instances as class attributes.

```python
class Toolkit:
    def get_tools(self) -> List[BaseTool]: ...
```

---

### PluginManager

Loads and manages `Toolkit` plugins; populates a `ToolRegistry`.

```python
class PluginManager:
    def __init__(self, registry: Optional[ToolRegistry] = None): ...
    def register(self, plugin: Toolkit, name: Optional[str] = None) -> None: ...
    def load_plugins(self) -> None: ...       # discovers entry points
    def register_tools(self) -> None: ...     # pushes tools into registry
    def get_plugin_names(self) -> List[str]: ...
```

See [Plugins guide](guide/plugins.md) for full documentation.

---

## cyclops.mcp

### MCPClient

Async client for MCP servers over stdio, streamable HTTP or SSE, running on the caller's event loop.

```python
class MCPClient:
    async def attach(
        self,
        name: str,
        command: Optional[List[str]] = None,
        *,
        url: Optional[str] = None,
        transport: Optional[Literal["stdio", "http", "sse"]] = None,
        env: Optional[Dict[str, str]] = None,
        headers: Optional[Dict[str, str]] = None,
        prefix: Union[str, bool] = True,
        connect_timeout: float = 30.0,
        call_timeout: Optional[float] = None,
    ) -> List[MCPClientTool]: ...
    async def detach(self, name: str) -> None: ...
    async def aclose(self) -> None: ...
    def tools(self, servers: Optional[Sequence[str]] = None) -> List[MCPClientTool]: ...
    async def call_tool(
        self, server: str, tool: str, arguments: Dict[str, Any]
    ) -> CallToolResult: ...

    @property
    def servers(self) -> List[str]: ...
```

`MCPClient` is an async context manager; leaving it calls `aclose()`.

---

### MCPClientTool

A `BaseTool` for one tool on an attached server. Await `execute()` on the loop that attached the server. It returns a `ToolResult`: an error result when the server sets `isError` or the call fails, otherwise text and image `content` blocks with `structuredContent` as `result`. `definition.input_schema` is the server's input schema, unchanged.

---

### MCPBridge

Sync wrapper that runs an `MCPClient` on a background event loop. Its tools run each call on that loop, so they can be awaited from any loop.

```python
class MCPBridge:
    client: MCPClient
    def attach(self, name: str, command: Optional[List[str]] = None, **kwargs) -> List[MCPClientTool]: ...
    def detach(self, name: str) -> None: ...
    def tools(self, servers: Optional[Sequence[str]] = None) -> List[MCPClientTool]: ...
    def stop(self) -> None: ...

    @property
    def connected_names(self) -> List[str]: ...
```

---

### MCPServer

Exposes Cyclops tools as an MCP server over stdio. A tool's `ToolResult` content blocks are served as MCP text and image content; a tool that raises or returns an error result is served with `isError` set.

```python
class MCPServer:
    def __init__(
        self,
        name: str = "cyclops-mcp-server",
        version: str = "0.1.0",
        load_plugins: bool = True,
    ): ...
    def add_tool(self, tool: BaseTool) -> None: ...
    def add_function_tool(self, name: str, description: str, func) -> None: ...
    async def run_stdio(self) -> None: ...
```

See [MCP guide](guide/mcp.md) for full documentation.
