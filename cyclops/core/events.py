"""Typed events a run emits while it progresses."""

from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from cyclops.core.types import RunResult, Usage


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


class StepFinished(Event):
    """One LLM completion and the tool calls it asked for are done."""

    step: int = Field(description="1 for the first step of the run")
    usage: Optional[Usage] = Field(description="None if the completion had none")


class RunFinished(Event):
    """The run ended; always the last event."""

    result: RunResult


__all__ = [
    "Event",
    "TextDelta",
    "ReasoningDelta",
    "ToolCallStarted",
    "ToolCallFinished",
    "StepFinished",
    "RunFinished",
]
