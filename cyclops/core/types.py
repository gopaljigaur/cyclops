"""Core type definitions"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Tuple, Union
from litellm.types.llms.openai import REASONING_EFFORT
from pydantic import BaseModel, Field, model_validator

from cyclops.core.caching import PromptCaching
from cyclops.core.hooks import AgentHooks
from cyclops.core.middleware import Guard, ToolMiddleware

_SUMMARY_PROMPT = (
    "Summarize the conversation below so that it can go on from your summary "
    "alone. Keep the user's goals and instructions, the decisions made, the "
    "facts and results the tools returned, names, identifiers and numbers that "
    "matter, and what is left to do. Write only the summary."
)


class Compaction(BaseModel):
    """When a run summarizes the older part of its transcript, and how."""

    model_config = {"extra": "forbid"}

    threshold: float = Field(
        default=0.8,
        gt=0,
        le=1,
        description="Fraction of the model's max input tokens a request may "
        "reach before its older messages are summarized",
    )
    keep: float = Field(
        default=0.2,
        gt=0,
        lt=1,
        description="Fraction of the model's max input tokens kept as the most "
        "recent messages, unsummarized",
    )
    model: Optional[str] = Field(
        default=None, description="Model that writes the summary; None uses the agent's"
    )
    prompt: str = Field(
        default=_SUMMARY_PROMPT, description="Instruction the summary is written to"
    )

    @model_validator(mode="after")
    def _keep_below_threshold(self) -> "Compaction":
        if self.keep >= self.threshold:
            raise ValueError("keep must be below threshold")
        return self


class AgentConfig(BaseModel):
    """Configuration for an agent"""

    # forbid: a misspelled or removed setting (max_iterations) fails loudly.
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    model: str
    fallback_models: List[str] = Field(
        default_factory=list,
        description="Models a step's completion goes to in turn when the model "
        "is still overloaded or down after its retries, or refuses; every step "
        "starts with model",
    )
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tool_mode: Literal["auto", "native", "naive"] = "auto"
    router: Optional[Any] = None
    max_steps: int = Field(
        default=10,
        ge=1,
        description="Most LLM completions in one run, guard rounds and output "
        "retries included",
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
    hook_timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Seconds each async hook call may run before it raises "
        "TimeoutError",
    )
    middleware: List[ToolMiddleware] = Field(default_factory=list)
    guards: List[Guard] = Field(default_factory=list)
    max_guard_rounds: int = Field(
        default=3, description="Most times per run guards may send the model back"
    )
    output_retries: int = Field(
        default=1,
        ge=0,
        description="Most times per run an answer that fails response_model "
        "validation is sent back with the errors",
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
    reasoning_effort: Optional[REASONING_EFFORT] = Field(
        default=None,
        description="How much the model reasons, sent to models LiteLLM lists "
        "with reasoning_effort; None sends nothing",
    )
    thinking_budget: Optional[int] = Field(
        default=None,
        gt=0,
        description="Tokens the model may think, sent as LiteLLM's thinking to "
        "models that take it; not with reasoning_effort",
    )
    # Extra LiteLLM completion arguments sent with every call, e.g. api_key,
    # api_base, extra_headers.
    completion_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # Mark cache breakpoints for providers that cache at cache_control markers:
    # True, or a time to live of "5m" or "1h"; False sends none.
    prompt_caching: PromptCaching = True
    compaction: Optional[Compaction] = Field(
        default=None,
        description="Summarize older messages once a request nears the model's "
        "input limit; None never does",
    )
    max_tool_output_chars: Optional[int] = Field(
        default=None,
        gt=0,
        description="Longest tool result text sent to the model; longer text "
        "keeps its start and end around a note of what was cut",
    )

    @model_validator(mode="after")
    def _one_reasoning_option(self) -> "AgentConfig":
        # LiteLLM maps each onto the same provider setting.
        if self.reasoning_effort is not None and self.thinking_budget is not None:
            raise ValueError("set reasoning_effort or thinking_budget, not both")
        return self

    @model_validator(mode="after")
    def _one_fallback_option(self) -> "AgentConfig":
        if self.router is not None and self.fallback_models:
            raise ValueError("set fallbacks on the router, not fallback_models")
        return self


class Message(BaseModel):
    """A single conversation message. Covers all LiteLLM message types."""

    role: str
    # Text, or a list of content parts such as text and image_url.
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
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
    """Tokens, cost and API time of one or more LLM completions."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = Field(
        default=0, description="Prompt tokens read from the provider's cache"
    )
    cache_write_tokens: int = Field(
        default=0, description="Prompt tokens written to the provider's cache"
    )
    reasoning_tokens: int = Field(
        default=0, description="Completion tokens the model spent reasoning"
    )
    total_tokens: int = 0
    cost: Optional[float] = Field(
        default=0.0, description="USD; None when LiteLLM can't price a completion"
    )
    api_duration: float = Field(
        default=0.0,
        description="Seconds from each request to its whole response, retries "
        "included; on a paced stream, also the caller's time reading it",
    )

    def __add__(self, other: "Usage") -> "Usage":
        cost = None
        if self.cost is not None and other.cost is not None:
            cost = self.cost + other.cost
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=cost,
            api_duration=self.api_duration + other.api_duration,
        )


def _added(total: Optional[Usage], usage: Optional[Usage]) -> Optional[Usage]:
    """The sum of two usages; None once either is unknown."""
    if total is None or usage is None:
        return None
    return total + usage


class ContextUsage(BaseModel):
    """How much of the model's context window a session's next request takes."""

    tokens: int = Field(
        description="Tokens of the system prompt, tools and transcript, counted "
        "as compaction counts them"
    )
    max_input_tokens: int = Field(
        description="The model's input limit; through a router, the smallest "
        "of its deployments'"
    )
    compaction_threshold: Optional[int] = Field(
        description="Tokens at which a run compacts; None without compaction"
    )


# A Go duration, as OpenAI's reset headers send it: 1s, 6m0s, 120ms.
_DURATION = re.compile(r"(?:\d+(?:\.\d+)?(?:ms|h|m|s))+")
_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_UNIT_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def _count(value: Optional[str]) -> Optional[int]:
    return int(value) if value is not None and value.isdigit() else None


def _duration(value: Optional[str]) -> Optional[float]:
    if value is None or not _DURATION.fullmatch(value):
        return None
    return sum(float(n) * _UNIT_SECONDS[u] for n, u in _DURATION_PART.findall(value))


def _until(value: Optional[str]) -> Optional[float]:
    """Seconds until an RFC 3339 time. Before 3.11 fromisoformat reads no Z."""
    if value is None:
        return None
    try:
        reset = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return max((reset - datetime.now(timezone.utc)).total_seconds(), 0.0)


class RateLimit(BaseModel):
    """The rate limits a provider reported with a response or an error."""

    remaining_requests: Optional[int] = None
    reset_requests: Optional[float] = Field(
        default=None, description="Seconds until the request limit resets"
    )
    remaining_tokens: Optional[int] = None
    reset_tokens: Optional[float] = Field(
        default=None, description="Seconds until the token limit resets"
    )

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> Optional["RateLimit"]:
        """The limits in a response's headers; None if they report none.

        Reads OpenAI's x-ratelimit-* headers, which OpenAI-compatible providers
        send too, and Anthropic's anthropic-ratelimit-* headers.
        """
        fields = {}
        for field, readers in _RATE_LIMIT_HEADERS.items():
            for header, read in readers:
                value = read(headers.get(header))
                if value is not None:
                    fields[field] = value
                    break
        return cls(**fields) if fields else None


# Each RateLimit field's headers, and how to read them.
_RATE_LIMIT_HEADERS: Dict[str, List[Tuple[str, Callable[[Optional[str]], Any]]]] = {
    "remaining_requests": [
        ("x-ratelimit-remaining-requests", _count),
        ("anthropic-ratelimit-requests-remaining", _count),
    ],
    "reset_requests": [
        ("x-ratelimit-reset-requests", _duration),
        ("anthropic-ratelimit-requests-reset", _until),
    ],
    "remaining_tokens": [
        ("x-ratelimit-remaining-tokens", _count),
        ("anthropic-ratelimit-tokens-remaining", _count),
    ],
    "reset_tokens": [
        ("x-ratelimit-reset-tokens", _duration),
        ("anthropic-ratelimit-tokens-reset", _until),
    ],
}


StopReason = Literal[
    "completed",
    "max_tokens",
    "content_filter",
    "max_steps",
    "budget",
    "guard",
    "output_validation",
    "cancelled",
    "error",
]


class RunResult(BaseModel):
    """What one agent run produced and why it stopped."""

    # forbid: the former token and cost fields are now properties over usage.
    model_config = {"arbitrary_types_allowed": True, "extra": "forbid"}

    content: str = Field(
        default="",
        description="The final answer; empty unless the run completed, or the "
        "last answer when it failed response_model validation",
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
    usage_by_model: Dict[str, Optional[Usage]] = Field(
        default_factory=dict,
        description="Usage split by the model, as configured, that made each "
        "completion; a model's is None if one of its completions reported none",
    )
    duration: float = Field(default=0.0, description="Wall-clock seconds of the run")
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
