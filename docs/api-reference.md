# API Reference

This page provides a concise reference for every public class and function in Cyclops. Each section links to the source module and shows the class signature.

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
    ): ...
```

Every run method takes a keyword-only `session: Optional[Session] = None`.

**Methods:** `start`, `run`, `arun`, `stream`, `astream`, `stream_events`, `astream_events`, `run_with_response`, `arun_with_response`, `as_tool`

`as_tool(name, description, *, annotations=None, timeout=None) -> AgentTool` returns a tool that runs the agent on its `input` argument, nested in the calling agent's run. See [Subagents](guide/subagents.md).

See [Agents guide](guide/agents.md) for full documentation.

---

### AgentTool

`cyclops.core.subagents`: the tool `Agent.as_tool()` returns. See [Subagents](guide/subagents.md).

```python
class AgentTool(BaseTool):
    agent: Agent
    async def execute(self, input: str) -> Any: ...
```

---

### AgentConfig

Pydantic model that holds all LLM and agent runtime settings.

```python
class AgentConfig(BaseModel):
    model: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tool_mode: Literal["auto", "native", "naive"] = "auto"
    router: Optional[Any] = None   # LiteLLM Router
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
    middleware: List[ToolMiddleware] = []
    guards: List[Guard] = []
    max_guard_rounds: int = 3
```

---

### Session

`cyclops.core.session`. One conversation, passed to any run with `session=`. See [Sessions](guide/sessions.md).

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

Returned by `Agent.start(input_message, *, stream=True, response_model=None, session=None)`. See [Agents](guide/agents.md#start-steer-and-cancel-a-run).

```python
class RunHandle:
    def __await__(self): ...            # await handle -> RunResult
    def __aiter__(self): ...            # async for event in handle (once)
    def steer(self, message: str) -> None: ...
    def cancel(self) -> None: ...
```

---

### RunResult

Returned by `run_with_response()` and `arun_with_response()`. See [Cost Tracking](guide/cost-tracking.md) for the fields.

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

`cyclops.core.completion`, exported from `cyclops`. A subclass of `litellm.Timeout`, raised when a stream sends no chunk for `AgentConfig.stream_idle_timeout` seconds after its first. A stalled stream is retried once if none of it was emitted, so a run ends with this error when the retry stalls too or content had already been emitted. See [Retries and timeouts](guide/agents.md#retries-and-timeouts).

---

### Events

`cyclops.core.events`: what `stream_events()` and `astream_events()` yield. Every event has `agent_path: Tuple[str, ...]`, empty for the agent that was run. See [Streaming](guide/streaming.md#run-events).

```python
class TextDelta(Event):        text: str
class ReasoningDelta(Event):   text: str
class ToolCallStarted(Event):  call_id: Optional[str]; name: str; arguments: Dict[str, Any]
class ToolCallFinished(Event): call_id: Optional[str]; name: str; result: str; is_error: bool
class GuardRejected(Event):    feedback: str
class StepFinished(Event):     step: int; usage: Optional[Usage]; finish_reason: Optional[str]
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

See [Hooks guide](guide/hooks.md) for full documentation.

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

`ToolPermissions` is a `ToolMiddleware` that allows, asks about or denies each call by tool name. See [Tool permissions](guide/hooks.md#tool-permissions).

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

A guard is `Callable[[GuardContext], Awaitable[Optional[str]]]`; `GuardContext` has `content`, `messages`, `run_id` and `agent_path`. See [Hooks guide](guide/hooks.md#tool-middleware).

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

See [Skills guide](guide/skills.md) for full documentation.

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

### tool decorator

```python
def tool(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    registry: Optional[ToolRegistry] = None,
    annotations: Optional[ToolAnnotations] = None,
    timeout: Optional[float] = None,
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
        timeout: Optional[float] = None,
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
        timeout: Optional[float] = None,
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
        auth: Optional[httpx.Auth] = None,
        prefix: Union[str, bool] = True,
        connect_timeout: float = 30.0,
        call_timeout: Optional[float] = 60.0,
        idle_timeout: Optional[float] = None,
        reconnection: ReconnectionOptions = ReconnectionOptions(),
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
