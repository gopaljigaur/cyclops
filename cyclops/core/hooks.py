"""Agent lifecycle hooks — standard event callbacks for the agent run loop."""

from typing import Any, Dict, List, Optional


class AgentHooks:
    """Base class for agent lifecycle hooks.

    Override any method to intercept events. All methods are no-ops by default.

    on_tool_start is the exception: it controls execution. Return "deny" to
    block the tool call; any other return value (including None) allows it.
    """

    def on_run_start(self, input_message: str) -> None:
        """Fired once when agent.run / agent.stream is called."""

    def on_run_end(self, content: str) -> None:
        """Fired when a non-streaming run completes (not called for stream/astream)."""

    def on_llm_start(self, messages: List[Dict[str, Any]]) -> None:
        """Fired before each LiteLLM completion call."""

    def on_llm_end(self, response: Any) -> None:
        """Fired after each non-streaming LiteLLM completion call."""

    def on_llm_error(self, error: Exception) -> None:
        """Fired when a LiteLLM call raises."""

    def on_tool_start(self, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """Fired before each tool execution.

        Return "deny" to block the tool call. Any other value (or None) allows it.
        """
        return "allow"

    def on_tool_end(self, tool_name: str, args: Dict[str, Any], result: str) -> None:
        """Fired after successful tool execution."""

    def on_tool_error(
        self, tool_name: str, args: Dict[str, Any], error: Exception
    ) -> None:
        """Fired when a tool raises an exception."""


__all__ = ["AgentHooks"]
