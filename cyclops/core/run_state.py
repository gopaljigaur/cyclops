"""A run in progress: the state its step loop shares, and the RunHandle its caller holds."""

import asyncio
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple

from cyclops.core.events import Event, RunFinished
from cyclops.core.hooks import RUN_STOPPED
from cyclops.core.session import Session
from cyclops.core.types import AgentConfig, RunResult, Usage


class _Run:
    """One run in progress: its transcript, result so far, events and steering.

    A run nested in another (a subagent's) shares the outermost run's events
    and pacing, and counts its usage in every run above it.
    """

    def __init__(
        self,
        config: AgentConfig,
        history: List[Dict[str, Any]],
        session: Optional[Session],
        paced: bool = False,
        parent: Optional["_Run"] = None,
        name: str = "",
    ) -> None:
        self.config = config
        self.result = RunResult(model=config.model)
        self.history = history
        self.session = session
        self.parent = parent
        self.agent_path: Tuple[str, ...] = ()
        self.events: asyncio.Queue
        if parent is None:
            self.events = asyncio.Queue()
            # True for astream_events, whose caller takes the events as they come.
            self.paced = paced
        else:
            self.agent_path = parent.agent_path + (name,)
            self.events = parent.events
            self.paced = parent.paced
        self.steering: List[str] = []
        self.started = False
        self.cancelled = False
        # Budgets count what is known, even once the reported totals are not.
        self.spent_tokens = 0
        self.spent_cost = 0.0

    def emit(self, event: Event) -> None:
        if self.agent_path:
            event = event.model_copy(update={"agent_path": self.agent_path})
        self.events.put_nowait(event)

    async def taken(self) -> None:
        """When paced, wait until the caller has taken every event emitted so far."""
        if self.paced:
            await self.events.join()

    def spend(self, usage: Optional[Usage]) -> None:
        """Count one completion in this run and every run it is nested in."""
        if usage is not None:
            self.spent_tokens += usage.total_tokens
            self.spent_cost += usage.cost or 0.0
        if self.result.usage is None or usage is None:
            self.result.usage = None
        else:
            self.result.usage += usage
        if self.parent is not None:
            self.parent.spend(usage)

    def over_budget(self) -> bool:
        """True once this run or one it is nested in reached its token or cost limit."""
        max_tokens = self.config.max_total_tokens
        if max_tokens is not None and self.spent_tokens >= max_tokens:
            return True
        max_cost = self.config.max_cost
        if max_cost is not None and self.spent_cost >= max_cost:
            return True
        return self.parent is not None and self.parent.over_budget()


class RunHandle:
    """A run started with Agent.start().

    Await it for the RunResult, and iterate it (once) for the run's events.
    """

    def __init__(self, run: _Run, task: "asyncio.Task[RunResult]") -> None:
        self._run = run
        self._task = task
        self._iterated = False
        # Marks the end of a run that stopped without RunFinished.
        task.add_done_callback(run.events.put_nowait)

    def __await__(self) -> Generator[Any, None, RunResult]:
        return self._task.__await__()

    def __aiter__(self) -> AsyncGenerator[Event, None]:
        if self._iterated:
            raise RuntimeError("A run's events can be iterated only once")
        self._iterated = True
        return self._events()

    async def _events(self) -> AsyncGenerator[Event, None]:
        events = self._run.events
        while True:
            event = await events.get()
            if event is self._task:
                self._task.result()
            yield event
            events.task_done()
            if isinstance(event, RunFinished) and not event.agent_path:
                return

    def steer(self, message: str) -> None:
        """Add a user message that the model sees in its next completion.

        A message that arrives during a completion goes into the one after,
        once the step's tool calls finish; if that completion was the final
        answer, the run takes another step to answer it when its limits allow,
        and otherwise leaves it in the history for the next run. Steering a
        finished run raises RuntimeError.
        """
        if self._task.done():
            raise RuntimeError("The run has finished")
        self._run.steering.append(message)

    def cancel(self) -> None:
        """Stop the run; awaiting it then gives stop_reason "cancelled".

        Tool calls in progress are cancelled and get a cancelled result. A
        second call does nothing.
        """
        if self._run.cancelled:
            return
        self._run.cancelled = True
        if self._run.started:
            self._task.cancel(RUN_STOPPED)
