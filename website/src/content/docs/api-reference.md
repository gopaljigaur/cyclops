---
title: API Reference
description: Complete reference for all public classes and functions in Cyclops.
---

## cyclops.core

### Agent

The main agent class. Wraps an LLM and orchestrates tool calls, sessions, streaming, and cost tracking.

```python
class Agent:
    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List] = None,
        skills: Optional[Skills] = None,
        *,
        session_store: Optional[SessionStore] = None,
        durability: Literal["sync", "async", "exit"] = "async",
    ): ...
```

With a `session_store`, every run with a session saves it there as `durability` says. See [Sessions](/guides/sessions/#saving-automatically).

**Methods** (each run method also takes a keyword-only `session: Optional[Session] = None`)

| Method | Signature | Description |
|---|---|---|
| `start` | `start(input_message: str, *, stream: bool = True, response_model: Optional[Type] = None, session: Optional[Session] = None) -> RunHandle` | Start a run on the running event loop; await, iterate, steer or cancel it. |
| `run` | `run(input_message: str, response_model: Optional[Type] = None) -> Any` | Run synchronously. Returns `str` or a Pydantic model instance. |
| `arun` | `arun(input_message: str, response_model: Optional[Type] = None) -> Any` | Run asynchronously. |
| `run_with_response` | `run_with_response(input_message: str) -> RunResult` | Run synchronously; return the result, stop reason, usage and cost. |
| `arun_with_response` | `arun_with_response(input_message: str) -> RunResult` | Run asynchronously; return the result, stop reason, usage and cost. |
| `stream` | `stream(input_message: str) -> Iterator[str]` | Sync token stream of every completion in the run, including steps that call tools. |
| `astream` | `astream(input_message: str) -> AsyncIterator[str]` | Async token stream. |
| `stream_events` | `stream_events(input_message: str) -> Iterator[Event]` | Sync stream of typed run events, ending with `RunFinished`. |
| `astream_events` | `astream_events(input_message: str) -> AsyncIterator[Event]` | Async stream of typed run events. |
| `compact` | `compact(session: Session) -> bool` | Summarize the session's older messages now; `acompact` is the async form. See [Context window](/guides/agents/#context-window). |
| `as_tool` | `as_tool(name: str, description: str, *, annotations: Optional[ToolAnnotations] = None, timeout: Optional[float] = None) -> AgentTool` | This agent as a tool that runs it on its `input` argument, nested in the calling agent's run. See [Subagents](/guides/subagents/). |

---

### AgentTool

`cyclops.core.subagents`: the tool `Agent.as_tool()` returns. See [Subagents](/guides/subagents/).

```python
class AgentTool(BaseTool):
    agent: Agent
    async def execute(self, input: str) -> Any: ...
```

---

### AgentConfig

Pydantic model holding all LLM and agent runtime settings.

```python
class AgentConfig(BaseModel):
    model: str
    fallback_models: List[str] = []
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tool_mode: Literal["auto", "native", "naive"] = "auto"
    router: Optional[Any] = None  # LiteLLM Router
    max_steps: int = 10
    max_total_tokens: Optional[int] = None
    max_cost: Optional[float] = None   # USD
    max_subagent_depth: int = 5
    max_concurrent_subagents: int = 10
    max_parallel_tool_calls: int = 10
    tool_timeout: Optional[float] = None   # seconds
    hooks: Optional[AgentHooks] = None
    num_retries: int = 2
    timeout: float = 600   # seconds
    stream_idle_timeout: float = 120   # seconds
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]] = None
    thinking_budget: Optional[int] = None   # tokens
    completion_kwargs: Dict[str, Any] = {}   # extra LiteLLM completion arguments
    prompt_caching: bool | Literal["5m", "1h"] = True
    compaction: Optional[Compaction] = None
    max_tool_output_chars: Optional[int] = None
    middleware: List[ToolMiddleware] = []
    guards: List[Guard] = []
    max_guard_rounds: int = 3
```

| Field | Default | Description |
|---|---|---|
| `model` | required | LiteLLM model string. |
| `temperature` | `None` | Sampling temperature. `None` uses the model default. |
| `max_tokens` | `None` | Max response tokens. `None` uses the model default. |
| `system_prompt` | `None` | Prepended to every conversation. |
| `tool_mode` | `"auto"` | `"auto"` auto-detects native function-calling support; `"native"` forces it; `"naive"` uses prompt-based fallback. |
| `router` | `None` | Optional LiteLLM Router for fallback and load balancing. |
| `max_steps` | `10` | Most LLM completions per run. |
| `max_total_tokens` | `None` | Most tokens per run, checked between steps. |
| `max_cost` | `None` | Most USD per run, checked between steps. |
| `max_subagent_depth` | `5` | Most levels of subagent runs below one run. |
| `max_concurrent_subagents` | `10` | Most subagent runs one run has going at once. |
| `max_parallel_tool_calls` | `10` | Most tool calls of one step that run at the same time. |
| `tool_timeout` | `None` | Seconds a tool call may run before it is cancelled and fails. A tool's own `timeout` wins. |
| `hooks` | `None` | `AgentHooks` instance for lifecycle callbacks and tool approval. |
| `num_retries` | `2` | Retries of a completion after a rate limit, overload, 5xx, timeout or connection error. |
| `timeout` | `600` | Seconds LiteLLM's HTTP client waits to connect and for each read or write. |
| `stream_idle_timeout` | `120` | Seconds a stream may go without a chunk after its first. |
| `reasoning_effort` | `None` | Reasoning effort for models that take it; LiteLLM maps it per provider. See [Reasoning](/guides/agents/#reasoning). |
| `thinking_budget` | `None` | Thinking tokens for models that take a budget; not with `reasoning_effort`. See [Reasoning](/guides/agents/#reasoning). |
| `completion_kwargs` | `{}` | Extra LiteLLM completion arguments sent with every call. |
| `compaction` | `None` | Summarize older messages once a request nears the model's input limit. See [Context window](/guides/agents/#context-window). |
| `max_tool_output_chars` | `None` | Longest tool result text sent to the model; longer text is cut in the middle. |
| `prompt_caching` | `True` | Cache breakpoints for providers that cache at markers; `"5m"` or `"1h"` sets the time to live, `False` sends none. See [Prompt caching](/guides/agents/#prompt-caching). |
| `middleware` | `[]` | `ToolMiddleware` run around every tool call, in order. |
| `guards` | `[]` | Async functions that can send the model back when it ends its turn. |
| `max_guard_rounds` | `3` | How many times per run guards may send the model back. |

---

### Compaction

When a run summarizes the older part of its transcript, and how. See [Context window](/guides/agents/#context-window).

```python
class Compaction(BaseModel):
    threshold: float = 0.8   # of the model's max input tokens
    keep: float = 0.2        # of the model's max input tokens, below threshold
    model: Optional[str] = None   # the agent's model when None
    prompt: str = ...        # the summarizing instruction
```

---

### Session

`cyclops.core.session`. One conversation, passed to any run with `session=`. See [Sessions](/guides/sessions/).

```python
class Session(BaseModel):
    version: Literal[1] = 1
    id: str                        # a random hex id by default
    messages: List[Dict[str, Any]]
    usage: Optional[Usage]
    def fork(self) -> Session: ...                # a copy with a new id
    def truncate(self, index: int) -> None: ...   # keep messages[:index]


class SessionStore(Protocol):
    async def load(self, session_id: str) -> Optional[Session]: ...
    async def save(self, session: Session) -> None: ...
    async def add_messages(self, session: Session, messages: List[Dict[str, Any]]) -> None: ...
    async def delete(self, session_id: str) -> None: ...


class InMemorySessionStore: ...                  # SessionStore in this process
class FileSessionStore:                          # <directory>/<id>.jsonl
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
    denied_tool_calls: List[DeniedToolCall]
    model: str
    run_id: str
    stop_reason: Literal[
        "completed", "max_tokens", "content_filter",
        "max_steps", "budget", "guard", "cancelled", "error",
    ]
    finish_reason: Optional[str]
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

### StreamIdleTimeout

`cyclops.core.completion`, exported from `cyclops`. A subclass of `litellm.Timeout`, raised when a stream sends no chunk for `AgentConfig.stream_idle_timeout` seconds after its first. A stalled stream is retried once if none of it was emitted, so a run ends with this error when the retry stalls too or content had already been emitted. See [Retries and timeouts](/guides/agents/#retries-and-timeouts).

---

### Events

`cyclops.core.events`: what `stream_events()` and `astream_events()` yield. Every event has `agent_path: Tuple[str, ...]`, empty for the agent that was run. See [Streaming](/guides/streaming/#run-events).

```python
class TextDelta(Event):        text: str
class ReasoningDelta(Event):   text: str
class ToolCallStarted(Event):  call_id: Optional[str]; name: str; arguments: Dict[str, Any]
class ToolCallFinished(Event): call_id: Optional[str]; name: str; result: str; is_error: bool
class GuardRejected(Event):    feedback: str
class ModelFallback(Event):    model: str; fallback_model: str; error: Optional[Exception]; usage: Optional[Usage]
class StepFinished(Event):     step: int; model: Optional[str]; usage: Optional[Usage]; finish_reason: Optional[str]
class Compacted(Event):        summary: str; replaced_messages: int; tokens_before: int; tokens_after: int
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

`RunResult.denied_tool_calls` holds a `DeniedToolCall` for each call that middleware denied:

```python
class DeniedToolCall(BaseModel):
    id: Optional[str]  # None in naive tool mode
    name: str
    arguments: Dict[str, Any]
    reason: str
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
    def on_compaction(self, summary: str, tokens_before: int, tokens_after: int) -> None: ...
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
| `on_compaction` | After a summary replaced older messages | `None` |
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

`ToolPermissions` is a `ToolMiddleware` that allows, asks about or denies each call by tool name. See [Tool permissions](/guides/hooks/#tool-permissions).

```python
class ToolPermissions(ToolMiddleware):
    def __init__(
        self,
        *,
        allow: Iterable[str] = (),
        ask: Iterable[str] = (),
        deny: Iterable[str] = (),
        approve: Optional[Callable[[ToolCallContext], Awaitable[Union[bool, Deny]]]] = None,
        default: Literal["allow", "ask", "deny"] = "allow",
    ): ...
    def permission(self, tool_name: str) -> Literal["allow", "ask", "deny"]: ...
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
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[List[Dict[str, Any]]] = None
```

---

## cyclops.skills

### Skills

Skills found in the subfolders of one or more directories. Each skill is a folder with a `SKILL.md`. Raises `FileNotFoundError` for a missing directory and `ValueError` for a duplicate name.

```python
class Skills:
    def __init__(self, *dirs: Union[str, Path]): ...
    skills: Dict[str, Skill]
    tool: Tool  # load_skill
    def prompt(self) -> str: ...
    def load(self, name: str, file: Optional[str] = None) -> str: ...
```

See [Skills guide](/guides/skills/) for full documentation.

---

### Skill

One parsed `SKILL.md`. Extra frontmatter fields are ignored.

```python
class Skill(BaseModel):
    name: str          # lowercase letters, digits and hyphens; matches the folder
    description: str   # 1 to 1024 characters
    path: Path         # the skill's folder
    body: str          # the markdown after the frontmatter

    @classmethod
    def from_file(cls, file: Path) -> "Skill": ...
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
    annotations: Optional[ToolAnnotations] = None,
    timeout: Optional[float] = None,
    defer_loading: bool = False,
) -> Union[Tool, Callable[[Callable], Tool]]: ...
```

| Parameter | Description |
|---|---|
| `func` | The function to wrap (set automatically when using `@tool` without parentheses). |
| `name` | Override the tool name. Defaults to `func.__name__`. |
| `description` | Override the tool description. Defaults to the docstring. |
| `registry` | If provided, the tool is automatically registered in this registry. |
| `annotations` | `ToolAnnotations` behaviour hints; `read_only` lets calls run in parallel. |
| `timeout` | Seconds a call may run; overrides `AgentConfig.tool_timeout`. |
| `defer_loading` | Send the tool only once the model finds it with `search_tools`. See [Deferred tools](/guides/tools/#deferred-tools). |

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
        timeout: Optional[float] = None,
        defer_loading: bool = False,
    ): ...
    async def execute(self, **kwargs) -> Any: ...

    @property
    def definition(self) -> ToolDefinition: ...
```

| Member | Description |
|---|---|
| `__init__(name, description, annotations, timeout, defer_loading)` | Set the tool name, description, behaviour hints, call timeout in seconds and whether it is deferred. Builds the `ToolDefinition` from the `execute()` signature. |
| `execute(**kwargs)` | Override with the exact parameter signature you want exposed to the LLM. |
| `definition` | `ToolDefinition` derived from the `execute()` signature. |

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
        timeout: Optional[float] = None,
        defer_loading: bool = False,
    ): ...
    async def execute(self, **kwargs) -> Any: ...
```

---

### ToolResult

What a tool may return instead of a plain value. See [Tool results](/guides/tools/#tool-results).

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
    @staticmethod
    def message_text(content: Union[str, List[Dict[str, Any]]]) -> str: ...   # without images
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

| Method | Description |
|---|---|
| `register(tool)` | Add a `BaseTool` instance to the registry. Raises `ValueError` if the name is taken. |
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
| `register(plugin, name)` | Manually register a `Toolkit` instance. `name` defaults to the class name; a name already in use raises `ValueError`. |
| `load_plugins()` | Instantiate and register every toolkit under the `"cyclops.toolkits"` entry-point group. Call once per manager. |
| `register_tools()` | Push all tools from loaded plugins into the registry. |
| `get_plugin_names()` | Return a list of registered plugin names. |

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
        auth: Optional[httpx.Auth] = None,
        prefix: Union[str, bool] = True,
        connect_timeout: float = 30.0,
        call_timeout: Optional[float] = 60.0,
        idle_timeout: Optional[float] = None,
        reconnection: ReconnectionOptions = ReconnectionOptions(),
        defer_loading: bool = False,
    ) -> List[MCPClientTool]: ...
    async def detach(self, name: str) -> None: ...
    async def aclose(self) -> None: ...
    def tools(self, servers: Optional[Sequence[str]] = None) -> List[MCPClientTool]: ...
    def status(self, server: str) -> ServerStatus: ...
    async def call_tool(
        self, server: str, tool: str, arguments: Dict[str, Any]
    ) -> CallToolResult: ...

    @property
    def servers(self) -> List[str]: ...
```

`MCPClient` is an async context manager; leaving it calls `aclose()`.

| Method / Property | Description |
|---|---|
| `attach(name, command, *, url, ...)` | Connect to a server and return its tools. Pass `command` for stdio or `url` for streamable HTTP (`transport="sse"` for SSE). Attaching an existing name replaces that server. `defer_loading=True` defers every tool of the server; see [Deferred tools](/guides/tools/#deferred-tools). |
| `detach(name)` | Disconnect a server and drop its tools. Unknown names are ignored. |
| `aclose()` | Disconnect every server. |
| `tools(servers)` | Tools of the given servers, or of every attached server. |
| `status(server)` | The server's `ServerStatus`: its connection state and last error. |
| `call_tool(server, tool, arguments)` | Call a tool by its remote name and return the raw `CallToolResult`. |
| `servers` | Names of the attached servers. |

```python
@dataclass(frozen=True)
class ReconnectionOptions:
    initial_reconnection_delay: float = 1.0
    max_reconnection_delay: float = 30.0
    reconnection_delay_grow_factor: float = 2.0
    max_retries: int = 5


@dataclass(frozen=True)
class ServerStatus:
    state: Literal["connecting", "connected", "disconnected", "failed"]
    error: Optional[str] = None
```

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
    def status(self, name: str) -> ServerStatus: ...
    def stop(self) -> None: ...

    @property
    def connected_names(self) -> List[str]: ...
```

| Method / Property | Description |
|---|---|
| `attach(name, command, **kwargs)` | Sync `MCPClient.attach`; takes the same arguments. |
| `detach(name)` | Disconnect a server and drop its tools. |
| `tools(servers)` | Tools of the given servers, or of every attached server. |
| `status(name)` | The server's `ServerStatus`. |
| `stop()` | Disconnect every server and stop the background loop. |
| `connected_names` | Names of the attached servers. |

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

| Method | Description |
|---|---|
| `__init__(name, version, load_plugins)` | Create the server. Set `load_plugins=False` to skip entry-point discovery. |
| `add_tool(tool)` | Register a `BaseTool` instance with the server. |
| `add_function_tool(name, description, func)` | Wrap a function in a tool and register it. |
| `run_stdio()` | Start the MCP server over stdio. Blocks until the client disconnects. |
