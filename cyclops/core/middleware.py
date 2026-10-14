"""Tool-call middleware and end-of-run guards."""

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from cyclops.toolkit.types import ToolAnnotations, ToolResult


@dataclass
class ToolCallContext:
    """One tool call, as middleware sees it."""

    tool_name: str
    arguments: Dict[str, Any]
    # None for naive mode calls, which carry no id.
    call_id: Optional[str]
    run_id: str
    annotations: ToolAnnotations = field(default_factory=ToolAnnotations)
    # Names of the agents that made the call, outermost first; empty until
    # agents can be nested.
    agent_path: Tuple[str, ...] = ()


@dataclass
class Deny:
    """Returned by before_tool to block a call; the model sees `reason` as an error."""

    reason: str


class ToolMiddleware:
    """Wraps every tool call. Override either method; both are no-ops by default.

    Middleware in `AgentConfig.middleware` run in list order before the call
    and in reverse order after it.
    """

    async def before_tool(
        self, call: ToolCallContext
    ) -> Union[None, Deny, Dict[str, Any]]:
        """Return None to allow, Deny to block, or a dict to replace the arguments."""
        return None

    async def after_tool(
        self, call: ToolCallContext, result: ToolResult, error: Optional[BaseException]
    ) -> Optional[ToolResult]:
        """See the result, and the exception if the tool raised or was cancelled.

        Return a ToolResult to replace the result; ignored for a cancelled call.

        Runs, in reverse order, for every middleware whose before_tool allowed
        the call, so the outer ones also see a call denied further in.
        """
        return None


@dataclass
class GuardContext:
    """The end of a model turn, as a guard sees it."""

    content: str
    messages: List[Dict[str, Any]]
    run_id: str
    # Empty until agents can be nested.
    agent_path: Tuple[str, ...] = ()


# Returns feedback that sends the model back into the loop, or None to let the run end.
Guard = Callable[[GuardContext], Awaitable[Optional[str]]]


__all__ = ["Deny", "Guard", "GuardContext", "ToolCallContext", "ToolMiddleware"]
