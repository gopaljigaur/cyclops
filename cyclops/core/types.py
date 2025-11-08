"""Core type definitions"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Configuration for an agent"""

    model: str
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    tool_mode: str = "auto"
    router: Optional[Any] = None


class Message(BaseModel):
    """Message representation"""

    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
    tokens_used: Optional[int] = Field(
        default=None, description="Number of tokens used"
    )
