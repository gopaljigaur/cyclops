"""Core type definitions"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

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
    max_iterations: int = 10
    hooks: Optional[AgentHooks] = None
    # Extra LiteLLM completion arguments sent with every call, e.g. api_key,
    # api_base, extra_headers, num_retries, timeout, reasoning_effort, thinking.
    completion_kwargs: Dict[str, Any] = Field(default_factory=dict)
    # Mark the system prompt and the latest message as cache breakpoints.
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


class AgentResponse(BaseModel):
    """Response from agent execution"""

    content: str = Field(description="The text response from the agent")
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="Tools called during execution"
    )
    model: str = Field(description="Model used for generation")
    stop_reason: Literal["completed", "max_iterations"] = Field(
        default="completed",
        description="Why the run stopped: a final answer, or max_iterations reached",
    )
    tokens_used: Optional[int] = Field(
        default=None, description="Number of tokens used across the run"
    )
    cost: Optional[float] = Field(
        default=None, description="Estimated cost of every completion in the run"
    )
    prompt_tokens: Optional[int] = Field(
        default=None, description="Number of prompt tokens used"
    )
    completion_tokens: Optional[int] = Field(
        default=None, description="Number of completion tokens used"
    )
    cached_tokens: Optional[int] = Field(
        default=None, description="Number of prompt tokens served from cache"
    )
