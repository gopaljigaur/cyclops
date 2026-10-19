"""Typed events a run emits while it progresses."""

from typing import Annotated, Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from cyclops.core.types import RateLimit, RunResult, Usage


class Event(BaseModel):
    """Base of every run event."""

    agent_path: Tuple[str, ...] = Field(
        default=(),
        description="Names of the nested agents the event came from, outermost "
        "first; empty for the agent that was run",
    )


class TextDelta(Event):
    """A piece of the model's answer text."""

    text: str


class ReasoningDelta(Event):
    """A piece of the model's reasoning, for models that return it."""

    text: str


class ToolCallStarted(Event):
    """A tool call is about to run."""

    call_id: Optional[str] = Field(description="None in naive tool mode")
    name: str
    arguments: Dict[str, Any]


class ToolCallFinished(Event):
    """A tool call finished, failed or was cancelled."""

    call_id: Optional[str] = Field(description="None in naive tool mode")
    name: str
    result: str = Field(description="The text of the result sent to the model")
    is_error: bool


class GuardRejected(Event):
    """A guard sent the model back after its answer; that answer is not the result."""

    feedback: str


class OutputRejected(Event):
    """The answer failed response_model validation, so the model was sent back
    with the errors; that answer is not the result."""

    feedback: str


_Headers = Annotated[
    Dict[str, str],
    Field(
        default_factory=dict,
        description="The provider's HTTP response headers, as LiteLLM passes them on",
    ),
]
_RateLimit = Annotated[
    Optional[RateLimit],
    Field(
        default=None,
        description="The rate limits in headers; None if they report none",
    ),
]


class CompletionRetry(Event):
    """A completion failed with a transient error and is sent again after delay."""

    model_config = {"arbitrary_types_allowed": True}

    model: str
    delay: float = Field(description="Seconds before the next attempt")
    status_code: Optional[int] = Field(description="The error's HTTP status")
    headers: _Headers
    rate_limit: _RateLimit
    error: Exception = Field(exclude=True)


class ModelFallback(Event):
    """A model failed or refused, so the step's completion goes to the next model."""

    model_config = {"arbitrary_types_allowed": True}

    model: str = Field(description="The model that failed or refused")
    fallback_model: str = Field(description="The model tried next")
    error: Optional[Exception] = Field(
        default=None, exclude=True, description="What it raised; None if it refused"
    )
    usage: Optional[Usage] = Field(
        default=None, description="The refused completion's usage, counted in the run"
    )
    status_code: Optional[int] = Field(
        default=None, description="The error's HTTP status; None if it refused"
    )
    headers: _Headers
    rate_limit: _RateLimit


class StepFinished(Event):
    """One LLM completion and the tool calls it asked for are done."""

    step: int = Field(description="1 for the first step of the run")
    model: Optional[str] = Field(
        default=None, description="The model that answered, as configured"
    )
    usage: Optional[Usage] = Field(description="None if the completion had none")
    finish_reason: Optional[str] = Field(
        default=None,
        description="The completion's finish reason as LiteLLM normalizes it",
    )
    headers: _Headers
    rate_limit: _RateLimit


class Compacted(Event):
    """The older part of the transcript was replaced by a summary of it."""

    summary: str
    replaced_messages: int = Field(description="Messages the summary replaced")
    tokens_before: int = Field(description="Request tokens before, as counted")
    tokens_after: int = Field(description="Request tokens after, as counted")


class RunFinished(Event):
    """The run ended; always the last event."""

    result: RunResult


__all__ = [
    "Event",
    "TextDelta",
    "ReasoningDelta",
    "ToolCallStarted",
    "ToolCallFinished",
    "GuardRejected",
    "OutputRejected",
    "CompletionRetry",
    "ModelFallback",
    "StepFinished",
    "Compacted",
    "RunFinished",
]
