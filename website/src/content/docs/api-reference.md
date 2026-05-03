---
title: API Reference
description: Complete reference for all public classes and functions in Cyclops.
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

**Methods**

| Method | Signature | Description |
|---|---|---|
| `run` | `run(input_message: str, response_model: Optional[Type] = None) -> Any` | Run synchronously. Returns `str` or a Pydantic model instance. |
| `arun` | `arun(input_message: str, response_model: Optional[Type] = None) -> Any` | Run asynchronously. |
| `run_with_response` | `run_with_response(input_message: str) -> AgentResponse` | Run synchronously and return full metadata. |
| `arun_with_response` | `arun_with_response(input_message: str) -> AgentResponse` | Run asynchronously and return full metadata. |
| `stream` | `stream(input_message: str) -> Iterator[str]` | Sync token stream. True streaming for no-tools agents; tool loop then stream for tool agents. |
| `astream` | `astream(input_message: str) -> AsyncIterator[str]` | Async token stream. |
| `reset` | `reset() -> None` | Clear conversation history. |

**Properties**

| Property | Type | Description |
|---|---|---|
| `messages` | `List[Dict[str, Any]]` | Shallow copy of the current conversation history. |

---

### AgentConfig

Pydantic model holding all LLM and agent runtime settings.

```python
class AgentConfig(BaseModel):
    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tool_mode: Literal["auto", "native", "naive"] = "auto"
    router: Optional[Any] = None  # LiteLLM Router
    max_iterations: int = 10
    hooks: Optional[AgentHooks] = None
```

| Field | Default | Description |
|---|---|---|
| `model` | required | LiteLLM model string. |
| `temperature` | `0.1` | Sampling temperature. |
| `max_tokens` | `None` | Max response tokens. `None` uses the model default. |
| `system_prompt` | `None` | Prepended to every conversation. |
| `tool_mode` | `"auto"` | `"auto"` auto-detects native function-calling support; `"native"` forces it; `"naive"` uses prompt-based fallback. |
| `router` | `None` | Optional LiteLLM Router for fallback and load balancing. |
| `max_iterations` | `10` | Maximum tool-call rounds per run. |
| `hooks` | `None` | `AgentHooks` instance for lifecycle callbacks and tool approval. |

---

### AgentResponse

Returned by `run_with_response()` and `arun_with_response()`.

```python
class AgentResponse(BaseModel):
    content: str
    tool_calls: List[ToolCall]
    model: str
    tokens_used: Optional[int]
    cost: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
```

| Field | Type | Description |
|---|---|---|
| `content` | `str` | The final text response. |
| `tool_calls` | `List[ToolCall]` | Every tool call made during the run. |
| `model` | `str` | Model identifier used. |
| `tokens_used` | `int` or `None` | Total tokens consumed. |
| `prompt_tokens` | `int` or `None` | Tokens in the prompt/context. |
| `completion_tokens` | `int` or `None` | Tokens in the generated response. |
| `cost` | `float` or `None` | Estimated USD cost. `None` for models not in LiteLLM's pricing table. |

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

| Field | Description |
|---|---|
| `id` | Unique call identifier assigned by the LLM. |
| `name` | Name of the tool that was called. |
| `arguments` | Arguments the model passed to the tool. |
| `result` | The value returned by the tool. |

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

| Method | When it fires | Return value |
|---|---|---|
| `on_run_start` | Once at the start of `run()`, `arun()`, `stream()`, or `astream()` | `None` |
| `on_run_end` | After `run()` or `arun()` returns | `None` |
| `on_llm_start` | Before each LiteLLM completion call | `None` |
| `on_llm_end` | After each non-streaming completion call | `None` |
| `on_llm_error` | When a LiteLLM call raises an exception | `None` |
| `on_tool_start` | Before each tool execution | `"deny"` to block, anything else to allow |
| `on_tool_end` | After a tool executes successfully | `None` |
| `on_tool_error` | When a tool raises an exception | `None` |

See [Hooks guide](/guides/hooks/) for full documentation.

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

Abstract base class for all memory backends. Implement all four async methods to create a custom backend.

```python
class Memory(ABC):
    async def store(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None: ...
    async def retrieve(self, key: str) -> Optional[Any]: ...
    async def list_keys(self) -> List[str]: ...
    async def clear(self) -> None: ...
```

| Method | Description |
|---|---|
| `store(key, value, metadata=None)` | Write a value. Overwrites any existing entry for that key. |
| `retrieve(key)` | Read a value. Returns `None` if the key does not exist. |
| `list_keys()` | Return a list of all stored keys. |
| `clear()` | Delete all entries. |

---

### InMemoryStorage

In-process dict-backed storage. Data is lost on process exit.

```python
class InMemoryStorage(Memory):
    def __init__(self): ...
```

---

### FileStorage

JSON-file-backed persistent storage. Loads existing data at construction; writes to disk after every `store()`.

```python
class FileStorage(Memory):
    def __init__(self, path: str): ...
```

---

## cyclops.toolkit

### tool

Decorator that wraps any Python function (sync or async) into a `Tool`.

```python
def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
) -> Union[Tool, Callable[[Callable], Tool]]: ...
```

| Parameter | Description |
|---|---|
| `func` | The function to wrap (set automatically when using `@tool` without parentheses). |
| `name` | Override the tool name. Defaults to `func.__name__`. |
| `description` | Override the tool description. Defaults to the docstring. |
| `registry` | If provided, the tool is automatically registered in this registry. |

---

### BaseTool

Abstract base class for all tools. Subclass this to create stateful tools.

```python
class BaseTool(ABC):
    def __init__(self, name: str, description: str): ...
    async def execute(self, **kwargs) -> Any: ...

    @property
    def definition(self) -> ToolDefinition: ...
```

| Member | Description |
|---|---|
| `__init__(name, description)` | Set the tool name and description. Builds the `ToolDefinition` from the `execute()` signature. |
| `execute(**kwargs)` | Override with the exact parameter signature you want exposed to the LLM. |
| `definition` | `ToolDefinition` derived from the `execute()` signature. |

---

### Tool

Function-wrapping tool created by the `@tool` decorator.

```python
class Tool(BaseTool):
    def __init__(self, name: str, description: str, func: Callable): ...
    async def execute(self, **kwargs) -> Any: ...
```

---

### ToolRegistry

Named collection of tools with lookup and execution helpers.

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

| Method | Description |
|---|---|
| `register(tool)` | Add a `BaseTool` instance to the registry. |
| `register_function(name, description, func)` | Wrap a function in a `Tool` and register it. |
| `get_tool(name)` | Return the named tool, or `None` if not found. |
| `list_tools()` | Return a list of all registered tool names. |
| `get_definitions()` | Return a dict mapping tool names to `ToolDefinition` objects. |
| `execute_tool(tool_name, **kwargs)` | Execute the named tool with the given arguments. |
| `remove_tool(name)` | Remove a tool by name. Returns `True` if it existed. |
| `clear()` | Remove all registered tools. |

---

### Toolkit

Base class for plugin packages. Define `BaseTool` instances as class attributes.

```python
class Toolkit:
    def get_tools(self) -> List[BaseTool]: ...
```

`get_tools()` scans the class for any attribute that is a `BaseTool` instance and returns them as a list.

---

### PluginManager

Loads and manages `Toolkit` plugins; populates a `ToolRegistry`.

```python
class PluginManager:
    def __init__(self, registry: Optional[ToolRegistry] = None): ...
    def register(self, plugin: Toolkit, name: Optional[str] = None) -> None: ...
    def load_plugins(self) -> None: ...
    def register_tools(self) -> None: ...
    def get_plugin_names(self) -> List[str]: ...
```

| Method | Description |
|---|---|
| `__init__(registry)` | Create a manager. Uses a fresh `ToolRegistry` if none is provided. |
| `register(plugin, name)` | Manually register a `Toolkit` instance. `name` defaults to the class name. |
| `load_plugins()` | Discover and load all toolkits registered under the `"cyclops.toolkits"` entry-point group. |
| `register_tools()` | Push all tools from loaded plugins into the registry. |
| `get_plugin_names()` | Return a list of registered plugin names. |

---

## cyclops.mcp

### MCPClient

Connects to an external MCP server over stdio.

```python
class MCPClient:
    async def connect_stdio(
        self, command: List[str], env: Optional[Dict[str, str]] = None
    ) -> None: ...
    async def disconnect(self) -> None: ...
    async def list_tools(self) -> List[Dict[str, Any]]: ...
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> str: ...

    @property
    def is_connected(self) -> bool: ...
```

| Method / Property | Description |
|---|---|
| `connect_stdio(command, env)` | Connect to an MCP server via a subprocess command. `env` overrides environment variables. |
| `disconnect()` | Cleanly shut down the connection and release the subprocess. |
| `list_tools()` | Return available tools as a list of dicts with `"name"`, `"description"`, and `"input_schema"`. |
| `call_tool(name, arguments)` | Call a tool and return its result as a string. |
| `is_connected` | `True` if currently connected to a server. |

---

### MCPServer

Exposes Cyclops tools as an MCP server over stdio.

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

| Method | Description |
|---|---|
| `__init__(name, version, load_plugins)` | Create the server. Set `load_plugins=False` to skip entry-point discovery. |
| `add_tool(tool)` | Register a `BaseTool` instance with the server. |
| `add_function_tool(name, description, func)` | Wrap a function in a tool and register it. |
| `run_stdio()` | Start the MCP server over stdio. Blocks until the client disconnects. |
