"""Agent lifecycle hooks: standard event callbacks for the agent run loop."""

import inspect
from contextvars import ContextVar
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from cyclops.utils.compat import asyncio_timeout

if TYPE_CHECKING:
    from cyclops.core.content import UserContent
    from cyclops.core.types import AgentConfig, RunResult

_T = TypeVar("_T")
# What a hook returns, sync or async.
MaybeAwaitable = Union[_T, Awaitable[_T]]

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

    Override any method to intercept events, as a plain or an async method.
    All methods are no-ops by default. on_run_start may replace the run's
    input and on_tool_start may block a tool call; the others only observe.
    """

    def on_run_start(
        self, input_message: Union[str, List[Dict[str, Any]]]
    ) -> MaybeAwaitable[Optional["UserContent"]]:
        """Fired once when agent.run / agent.stream is called.

        input_message is the run's text, or its list of content parts. Return
        text or content parts to replace it, or None to keep it.
        """
        return None

    def on_run_end(self, content: str) -> MaybeAwaitable[None]:
        """Fired when a run completes, streamed or not."""

    def on_run_error(self, error: BaseException) -> MaybeAwaitable[None]:
        """Fired instead of on_run_end when a run raises or is cancelled.

        A run its caller stopped, such as a stream closed early, gets a
        CancelledError whose args are (RUN_STOPPED,).
        """

    def on_subagent_start(
        self, agent_path: Tuple[str, ...], input_message: str
    ) -> MaybeAwaitable[None]:
        """Fired before a subagent run nested in this agent's run starts.

        agent_path names the nested agents down to it, outermost first.
        """

    def on_subagent_end(
        self, agent_path: Tuple[str, ...], result: "RunResult"
    ) -> MaybeAwaitable[None]:
        """Fired when that subagent run ends, with its result, however it ended."""

    def on_llm_start(self, messages: List[Dict[str, Any]]) -> MaybeAwaitable[None]:
        """Fired before each LiteLLM completion call."""

    def on_llm_end(self, response: Any) -> MaybeAwaitable[None]:
        """Fired after each LiteLLM completion; for a stream, with the assembled response."""

    def on_llm_error(self, error: Exception) -> MaybeAwaitable[None]:
        """Fired when a LiteLLM call raises."""

    def on_compaction(
        self, summary: str, tokens_before: int, tokens_after: int
    ) -> MaybeAwaitable[None]:
        """Fired when a summary replaced the older part of the transcript."""

    def on_tool_start(
        self, tool_name: str, args: Dict[str, Any]
    ) -> MaybeAwaitable[Optional[str]]:
        """Fired before each tool execution.

        Return "deny" to block the tool call. Any other value (or None) allows it.
        """
        return "allow"

    def on_tool_end(
        self, tool_name: str, args: Dict[str, Any], result: str
    ) -> MaybeAwaitable[None]:
        """Fired after successful tool execution."""

    def on_tool_error(
        self, tool_name: str, args: Dict[str, Any], error: BaseException
    ) -> MaybeAwaitable[None]:
        """Fired when a tool raises an exception or is cancelled."""


async def _call_hook(config: "AgentConfig", name: str, *args: Any) -> Any:
    """Call config.hooks' method name, if any hooks are set.

    An async hook gets config.hook_timeout and raises TimeoutError past it, as
    a hook that raises does. A plain hook runs inline, since nothing can stop it.
    """
    if config.hooks is None:
        return None
    hook = getattr(config.hooks, name)
    if not inspect.iscoroutinefunction(hook):
        return hook(*args)
    async with asyncio_timeout(config.hook_timeout):
        return await hook(*args)


__all__ = ["AgentHooks", "RUN_STOPPED", "current_tool_call_id"]
