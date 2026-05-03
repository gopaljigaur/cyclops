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
    tokens_used: Optional[int]
    cost: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
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
) -> Union[Tool, Callable[[Callable], Tool]]: ...
```

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

See [MCP guide](guide/mcp.md) for full documentation.
