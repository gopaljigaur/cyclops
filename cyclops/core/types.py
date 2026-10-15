"""Core type definitions"""

import uuid
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from cyclops.core.caching import PromptCaching
from cyclops.core.hooks import AgentHooks
from cyclops.core.middleware import Guard, ToolMiddleware


class AgentConfig(BaseModel):
    """Configuration for an agent"""

    # forbid: a misspelled or removed setting (max_iterations) fails loudly.
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tool_mode: Literal["auto", "native", "naive"] = "auto"
    router: Optional[Any] = None
    max_steps: int = Field(
        default=10,
        ge=1,
        description="Most LLM completions in one run, guard rounds included",
    )
    max_total_tokens: Optional[int] = Field(
        default=None, description="Most tokens one run may use, checked between steps"
    )
    max_cost: Optional[float] = Field(
        default=None, description="Most USD one run may cost, checked between steps"
    )
    max_subagent_depth: int = Field(
        default=5, ge=1, description="Most levels of subagent runs below one run"
    )
    max_concurrent_subagents: int = Field(
        default=10, ge=1, description="Most subagent runs one run has going at once"
    )
    max_parallel_tool_calls: int = Field(
        default=10,
        ge=1,
        description="Most tool calls of one step that run at the same time",
    )
    tool_timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Seconds a tool call may run; a tool's own timeout wins",
    )
    hooks: Optional[AgentHooks] = None
    middleware: List[ToolMiddleware] = Field(default_factory=list)
    guards: List[Guard] = Field(default_factory=list)
    max_guard_rounds: int = Field(
        default=3, description="Most times per run guards may send the model back"
    )
    num_retries: int = Field(
        default=2,
        ge=0,
        description="Retries of a completion after a rate limit, overload, 5xx, "
        "timeout or connection error, before any of it has streamed",
    )
    timeout: float = Field(
        default=600,
        gt=0,
        description="Seconds LiteLLM's HTTP client waits to connect and for "
        "each read or write; a timeout is retried like other transient errors",
    )
    stream_idle_timeout: float = Field(
        default=120,
        gt=0,
        description="Seconds a stream may go without a chunk after its first; "
        "a stall is retried once while nothing of it has been emitted",
    )
    # Extra LiteLLM completion arguments sent with every call, e.g. api_key,
    # api_base, extra_headers, reasoning_effort, thinking.
    completion_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # Mark cache breakpoints for providers that cache at cache_control markers:
    # True, or a time to live of "5m" or "1h"; False sends none.
    prompt_caching: PromptCaching = True


class Message(BaseModel):
    """A single conversation message. Covers all LiteLLM message types."""

    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        known = set(cls.model_fields)
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ToolCall(BaseModel):
    """Tool call tracking"""

    id: str = Field(description="Unique identifier for the tool call")
    name: str = Field(description="Name of the tool being called")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments passed to the tool"
    )
    result: Optional[Any] = Field(
        default=None, description="Result from tool execution"
    )


class DeniedToolCall(BaseModel):
    """A tool call that middleware denied"""

    id: Optional[str] = Field(description="The call's id; None in naive tool mode")
    name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(description="What the model was told")


class Usage(BaseModel):
    """Tokens and cost of one or more LLM completions."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    cost: Optional[float] = Field(
        default=0.0, description="USD; None when LiteLLM can't price a completion"
    )

    def __add__(self, other: "Usage") -> "Usage":
        cost = None
        if self.cost is not None and other.cost is not None:
            cost = self.cost + other.cost
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=cost,
        )


StopReason = Literal[
    "completed",
    "max_tokens",
    "content_filter",
    "max_steps",
    "budget",
    "guard",
    "cancelled",
    "error",
]


class RunResult(BaseModel):
    """What one agent run produced and why it stopped."""

    # forbid: the former token and cost fields are now properties over usage.
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    content: str = Field(
        default="", description="The final answer; empty unless the run completed"
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tools called during the run"
    )
    denied_tool_calls: List[DeniedToolCall] = Field(
        default_factory=list, description="Tool calls that middleware denied"
    )
    model: str = Field(description="Model used for generation")
    run_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex, description="Unique id of the run"
    )
    stop_reason: StopReason = Field(
        default="completed", description="Why the run stopped"
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="The last completion's finish reason as LiteLLM normalizes "
        "it (stop, length, tool_calls, content_filter); None before any",
    )
    steps: int = Field(default=0, description="LLM completions made in the run")
    usage: Optional[Usage] = Field(
        default_factory=Usage,
        description="Summed over every completion, nested runs included; None if "
        "one reported no usage",
    )
    error: Optional[Exception] = Field(
        default=None, exclude=True, description="What ended the run with 'error'"
    )

    @property
    def tokens_used(self) -> Optional[int]:
        return self.usage.total_tokens if self.usage else None

    @property
    def cost(self) -> Optional[float]:
        return self.usage.cost if self.usage else None

    @property
    def prompt_tokens(self) -> Optional[int]:
        return self.usage.prompt_tokens if self.usage else None

    @property
    def completion_tokens(self) -> Optional[int]:
        return self.usage.completion_tokens if self.usage else None

    @property
    def cached_tokens(self) -> Optional[int]:
        return self.usage.cached_tokens if self.usage else None
