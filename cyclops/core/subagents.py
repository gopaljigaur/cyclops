"""Agents as tools: a tool that runs an agent, nested in the calling agent's run."""

import asyncio
from typing import TYPE_CHECKING, Any, Optional
from weakref import WeakKeyDictionary

from cyclops.core.hooks import RUN_STOPPED
from cyclops.core.run import _current_loop, _RunLoop
from cyclops.core.run_state import _Run
from cyclops.core.types import RunResult
from cyclops.toolkit.tool import BaseTool
from cyclops.toolkit.types import ToolAnnotations, ToolResult

if TYPE_CHECKING:
    from cyclops.core.agent import Agent

# Each run's slots for the subagent runs it has going at once.
_slots: "WeakKeyDictionary[_Run, asyncio.Semaphore]" = WeakKeyDictionary()


def _too_deep(run: _Run, depth: int = 1) -> bool:
    """True if a run depth levels below run would pass its or an outer run's limit."""
    if depth > run.config.max_subagent_depth:
        return True
    return run.parent is not None and _too_deep(run.parent, depth + 1)


class AgentTool(BaseTool):
    """Runs an agent on the input it is called with and returns its answer.

    Made by Agent.as_tool(). Each call is a run of its own with a fresh
    transcript. Called by another agent, the run is nested in that agent's
    run: it is cancelled with it, streams when it streams, its events go to
    that run's events with agent_path extended by the tool name, and its
    usage counts in that run's usage and budgets. A run that fails or stops
    without an answer gives an error result.
    """

    def __init__(
        self,
        agent: "Agent",
        name: str,
        description: str,
        annotations: Optional[ToolAnnotations] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.agent = agent
        super().__init__(name, description, annotations, timeout)

    async def execute(self, input: str) -> Any:  # type: ignore[override]
        """Run the agent.

        Args:
            input: The task, with everything needed to do it.
        """
        result = await self._run(input)
        if result.stop_reason == "cancelled":
            # Only the calling run can cancel a nested run.
            raise asyncio.CancelledError(RUN_STOPPED)
        if result.error is not None:
            raise result.error
        if result.stop_reason != "completed":
            return ToolResult.error_result(
                f"{self.name} stopped without an answer: {result.stop_reason}"
            )
        return result.content

    async def _run(self, input_message: str) -> RunResult:
        """Run the agent with a fresh transcript, nested in the calling run if any."""
        ref = _current_loop.get()
        caller = None if ref is None else ref()
        config = self.agent.config
        if caller is None:
            run = _Run(config, [], None)
            return await _RunLoop(self.agent, run, False).execute(input_message)
        parent = caller.run
        if _too_deep(parent):
            raise RuntimeError(
                "subagents are nested as deep as max_subagent_depth allows"
            )
        if parent not in _slots:
            _slots[parent] = asyncio.Semaphore(parent.config.max_concurrent_subagents)
        async with _slots[parent]:
            run = _Run(config, [], None, parent=parent, name=self.name)
            return await _RunLoop(self.agent, run, caller.stream).execute(input_message)


__all__ = ["AgentTool"]
