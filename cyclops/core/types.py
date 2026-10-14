"""Core type definitions"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import AliasChoices, BaseModel, Field

from cyclops.core.hooks import AgentHooks


class AgentConfig(BaseModel):
    """Configuration for an agent"""

    model_config = {"arbitrary_types_allowed": True}

    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tool_mode: Literal["auto", "native", "naive"] = "auto"
    router: Optional[Any] = None
    max_steps: int = Field(
        default=10,
        # The field's former name is still accepted.
        validation_alias=AliasChoices("max_steps", "max_iterations"),
        description="Most LLM completions in one run; a run without tools makes one",
    )
    max_total_tokens: Optional[int] = Field(
        default=None, description="Most tokens one run may use, checked between steps"
    )
    max_cost: Optional[float] = Field(
        default=None, description="Most USD one run may cost, checked between steps"
    )
    hooks: Optional[AgentHooks] = None
    # Extra LiteLLM completion arguments sent with every call, e.g. api_key,
    # api_base, extra_headers, num_retries, timeout, reasoning_effort, thinking.
    completion_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # Mark the system prompt and the latest message as cache breakpoints, for
    # providers that cache at cache_control markers.
    prompt_caching: bool = False


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


StopReason = Literal["completed", "max_steps", "budget", "cancelled", "error"]


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
    model: str = Field(description="Model used for generation")
    stop_reason: StopReason = Field(
        default="completed", description="Why the run stopped"
    )
    steps: int = Field(default=0, description="LLM completions made in the run")
    usage: Optional[Usage] = Field(
        default_factory=Usage,
        description="Summed over every completion; None if one reported no usage",
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


# The former name of RunResult.
AgentResponse = RunResult
