"""A run in progress: the state its step loop shares, and the RunHandle its caller holds."""

import asyncio
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional

from cyclops.core.events import Event, RunFinished
from cyclops.core.hooks import RUN_STOPPED
from cyclops.core.session import Session
from cyclops.core.types import RunResult


class _Run:
    """One run in progress: its transcript, result so far, events and steering."""

    def __init__(
        self, model: str, history: List[Dict[str, Any]], session: Optional[Session]
    ) -> None:
        self.result = RunResult(model=model)
        self.history = history
        self.session = session
        self.events: asyncio.Queue = asyncio.Queue()
        # Set by astream_events, whose caller takes the events as they come.
        self.paced = False
        self.steering: List[str] = []
        self.started = False
        self.cancelled = False

    def emit(self, event: Event) -> None:
        self.events.put_nowait(event)

    async def taken(self) -> None:
        """When paced, wait until the caller has taken every event emitted so far."""
        if self.paced:
            await self.events.join()


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
            if isinstance(event, RunFinished):
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
