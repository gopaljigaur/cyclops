"""Agent lifecycle hooks — standard event callbacks for the agent run loop."""

from contextvars import ContextVar
from typing import Any, Dict, List, Optional

_tool_call_id: ContextVar[Optional[str]] = ContextVar(
    "cyclops_tool_call_id", default=None
)

# The message of the CancelledError a run raises when its caller stops it, e.g.
# by closing its event stream early; on_run_error receives that error.
RUN_STOPPED = "run stopped by its caller"


def current_tool_call_id() -> Optional[str]:
    """The id of the native tool call the tool hooks are firing for, else None.

    Each native tool call runs in its own task, so this is exact in parallel calls.
    Naive mode calls have no id.
    """
    return _tool_call_id.get()


class AgentHooks:
    """Base class for agent lifecycle hooks.

    Override any method to intercept events. All methods are no-ops by default.

    on_tool_start is the exception: it controls execution. Return "deny" to
    block the tool call; any other return value (including None) allows it.
    """

    def on_run_start(self, input_message: str) -> None:
        """Fired once when agent.run / agent.stream is called."""

    def on_run_end(self, content: str) -> None:
        """Fired when a run completes, streamed or not."""

    def on_run_error(self, error: BaseException) -> None:
        """Fired instead of on_run_end when a run raises or is cancelled.

        A run its caller stopped, such as a stream closed early, gets a
        CancelledError whose args are (RUN_STOPPED,).
        """

    def on_llm_start(self, messages: List[Dict[str, Any]]) -> None:
        """Fired before each LiteLLM completion call."""

    def on_llm_end(self, response: Any) -> None:
        """Fired after each LiteLLM completion; for a stream, with the assembled response."""

    def on_llm_error(self, error: Exception) -> None:
        """Fired when a LiteLLM call raises."""

    def on_compaction(
        self, summary: str, tokens_before: int, tokens_after: int
    ) -> None:
        """Fired when a summary replaced the older part of the transcript."""

    def on_tool_start(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Fired before each tool execution.

        Return "deny" to block the tool call. Any other value (or None) allows it.
        """
        return "allow"

    def on_tool_end(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        """Fired after successful tool execution."""

    def on_tool_error(
        self, tool_name: str, args: Dict[str, Any], error: BaseException
    ) -> None:
        """Fired when a tool raises an exception or is cancelled."""


__all__ = ["AgentHooks", "RUN_STOPPED", "current_tool_call_id"]
