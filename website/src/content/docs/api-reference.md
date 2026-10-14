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
    ): ...
```

**Methods** (each run method also takes a keyword-only `session: Optional[Session] = None`)

| Method | Signature | Description |
|---|---|---|
| `start` | `start(input_message: str, *, stream: bool = True, response_model: Optional[Type] = None, session: Optional[Session] = None) -> RunHandle` | Start a run on the running event loop; await, iterate, steer or cancel it. |
| `run` | `run(input_message: str, response_model: Optional[Type] = None) -> Any` | Run synchronously. Returns `str` or a Pydantic model instance. |
| `arun` | `arun(input_message: str, response_model: Optional[Type] = None) -> Any` | Run asynchronously. |
| `run_with_response` | `run_with_response(input_message: str) -> RunResult` | Run synchronously; return the result, stop reason, usage and cost. |
| `arun_with_response` | `arun_with_response(input_message: str) -> RunResult` | Run asynchronously; return the result, stop reason, usage and cost. |
| `stream` | `stream(input_message: str) -> Iterator[str]` | Sync token stream. True streaming for no-tools agents; tool loop then stream for tool agents. |
| `astream` | `astream(input_message: str) -> AsyncIterator[str]` | Async token stream. |
| `stream_events` | `stream_events(input_message: str) -> Iterator[Event]` | Sync stream of typed run events, ending with `RunFinished`. |
| `astream_events` | `astream_events(input_message: str) -> AsyncIterator[Event]` | Async stream of typed run events. |
| `reset` | `reset() -> None` | Clear the history of runs without a session. |

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
    max_steps: int = 10
    max_total_tokens: Optional[int] = None
    max_cost: Optional[float] = None   # USD
    hooks: Optional[AgentHooks] = None
    middleware: List[ToolMiddleware] = []
    guards: List[Guard] = []
    max_guard_rounds: int = 3
```

| Field | Default | Description |
|---|---|---|
| `model` | required | LiteLLM model string. |
| `temperature` | `0.1` | Sampling temperature. |
| `max_tokens` | `None` | Max response tokens. `None` uses the model default. |
| `system_prompt` | `None` | Prepended to every conversation. |
| `tool_mode` | `"auto"` | `"auto"` auto-detects native function-calling support; `"native"` forces it; `"naive"` uses prompt-based fallback. |
| `router` | `None` | Optional LiteLLM Router for fallback and load balancing. |
| `max_steps` | `10` | Most LLM completions per run. |
| `max_total_tokens` | `None` | Most tokens per run, checked between steps. |
| `max_cost` | `None` | Most USD per run, checked between steps. |
| `hooks` | `None` | `AgentHooks` instance for lifecycle callbacks and tool approval. |
| `middleware` | `[]` | `ToolMiddleware` run around every tool call, in order. |
| `guards` | `[]` | Async functions that can send the model back when it ends its turn. |
| `max_guard_rounds` | `3` | How many times per run guards may send the model back. |

---

### Session

`cyclops.core.session`. One conversation, passed to any run with `session=`. See [Sessions](/guides/sessions/).

```python
class Session(BaseModel):
    version: Literal[1] = 1
    id: str                        # a random hex id by default
    messages: List[Dict[str, Any]]
    usage: Optional[Usage]


class SessionStore(Protocol):
    async def load(self, session_id: str) -> Optional[Session]: ...
    async def save(self, session: Session) -> None: ...
    async def delete(self, session_id: str) -> None: ...


class InMemorySessionStore: ...                  # SessionStore in this process
class FileSessionStore:                          # <directory>/<id>.json
    def __init__(self, directory: str | Path): ...
```

---

### RunHandle

Returned by `Agent.start(input_message, *, stream=True, response_model=None, session=None)`. See [Agents](/guides/agents/#start-steer-and-cancel-a-run).

```python
class RunHandle:
    def __await__(self): ...            # await handle -> RunResult
    def __aiter__(self): ...            # async for event in handle (once)
    def steer(self, message: str) -> None: ...
    def cancel(self) -> None: ...
```

---

### RunResult

Returned by `run_with_response()` and `arun_with_response()`. See [Cost Tracking](/guides/cost-tracking/) for the fields.

```python
class RunResult(BaseModel):
    content: str
    tool_calls: List[ToolCall]
    model: str
    run_id: str
    stop_reason: Literal["completed", "max_steps", "budget", "guard", "cancelled", "error"]
    steps: int
    usage: Optional[Usage]
    error: Optional[Exception]
    # Read-only properties over usage:
    tokens_used, cost, prompt_tokens, completion_tokens, cached_tokens


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    total_tokens: int
    cost: Optional[float]  # None when LiteLLM can't price a completion
```

---

### Events

`cyclops.core.events`: what `stream_events()` and `astream_events()` yield. Every event has `agent_path: Tuple[str, ...]`, empty for the agent that was run. See [Streaming](/guides/streaming/#run-events).

```python
class TextDelta(Event):        text: str
class ReasoningDelta(Event):   text: str
class ToolCallStarted(Event):  call_id: Optional[str]; name: str; arguments: Dict[str, Any]
class ToolCallFinished(Event): call_id: Optional[str]; name: str; result: str; is_error: bool
class GuardRejected(Event):    feedback: str
class StepFinished(Event):     step: int; usage: Optional[Usage]
class RunFinished(Event):      result: RunResult
```

---

### ToolCall

Records a single tool invocation within a `RunResult`.

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
    def on_run_error(self, error: BaseException) -> None: ...
    def on_llm_start(self, messages: List[Dict[str, Any]]) -> None: ...
    def on_llm_end(self, response: Any) -> None: ...
    def on_llm_error(self, error: Exception) -> None: ...
    def on_tool_start(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]: ...
    def on_tool_end(self, tool_name: str, args: Dict[str, Any], result: str) -> None: ...
    def on_tool_error(self, tool_name: str, args: Dict[str, Any], error: BaseException) -> None: ...
```

| Method | When it fires | Return value |
|---|---|---|
| `on_run_start` | Once at the start of `run()`, `arun()`, `stream()`, or `astream()` | `None` |
| `on_run_end` | When a run finishes, streamed or not | `None` |
| `on_run_error` | Instead of `on_run_end` when a run raises, is cancelled, or its stream is closed early. A run its caller stopped (a stream closed early) gets a `CancelledError` whose `args` are `(RUN_STOPPED,)`, from `cyclops.core.hooks`. | `None` |
| `on_llm_start` | Before each LiteLLM completion call | `None` |
| `on_llm_end` | After each completion; for a stream, with the assembled response | `None` |
| `on_llm_error` | When a LiteLLM call raises an exception | `None` |
| `on_tool_start` | Before each tool execution | `"deny"` to block, anything else to allow |
| `on_tool_end` | After a tool executes successfully | `None` |
| `on_tool_error` | When a tool raises an exception or is cancelled | `None` |

See [Hooks guide](/guides/hooks/) for full documentation.

---

### ToolMiddleware

Async middleware around every tool call, set with `AgentConfig.middleware`.

```python
class ToolMiddleware:
    async def before_tool(self, call: ToolCallContext) -> Union[None, Deny, Dict[str, Any]]: ...
    async def after_tool(self, call: ToolCallContext, result: ToolResult, error: Optional[BaseException]) -> Optional[ToolResult]: ...

@dataclass
class ToolCallContext:
    tool_name: str
    arguments: Dict[str, Any]
    call_id: Optional[str]
    run_id: str
    annotations: ToolAnnotations
    agent_path: Tuple[str, ...]

@dataclass
class Deny:
    reason: str
```

A guard is `Callable[[GuardContext], Awaitable[Optional[str]]]`; `GuardContext` has `content`, `messages`, `run_id` and `agent_path`. See [Hooks guide](/guides/hooks/#tool-middleware).

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
