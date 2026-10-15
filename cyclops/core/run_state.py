"""A run in progress: the state its step loop shares, and the RunHandle its caller holds."""

import asyncio
import io
from collections import deque
from typing import Any, AsyncGenerator, Deque, Dict, Generator, List, Optional, Tuple

from cyclops.core.content import UserContent, _Content, _user_content
from cyclops.core.events import Event, ReasoningDelta, RunFinished, TextDelta
from cyclops.core.hooks import RUN_STOPPED
from cyclops.core.session import Session
from cyclops.core.types import AgentConfig, RunResult, Usage, _added


class _Events(asyncio.Queue):
    """A run's events.

    Until the handle is iterated, consecutive deltas are kept as one and
    nested runs' events are dropped, so a run that is only awaited holds a few
    events per step instead of one per token. put_nowait counts a merged
    delta as unfinished, which join() would wait on; only paced runs join, and
    their handle is iterated before the run starts.
    """

    def _init(self, maxsize: int) -> None:
        # Each item with the text of the deltas merged into it, if any.
        self._queue: Deque[Tuple[Any, Optional[io.StringIO]]] = deque()
        self.listening = False

    def _put(self, item: Any) -> None:
        if not self.listening and self._queue:
            last, text = self._queue[-1]
            if isinstance(item, (TextDelta, ReasoningDelta)) and type(item) is type(
                last
            ):
                if text is None:
                    text = io.StringIO()
                    self._queue[-1] = (last, text)
                text.write(item.text)
                return
        self._queue.append((item, None))

    def _get(self) -> Any:
        item, text = self._queue.popleft()
        if text is not None:
            item = item.model_copy(update={"text": item.text + text.getvalue()})
        return item


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
        self.events: _Events
        if parent is None:
            self.events = _Events()
            # True for astream_events, whose caller takes the events as they come.
            self.paced = paced
        else:
            self.agent_path = parent.agent_path + (name,)
            self.events = parent.events
            self.paced = parent.paced
        self.steering: List[_Content] = []
        self.started = False
        self.cancelled = False
        # Budgets count what is known, even once the reported totals are not.
        self.spent_tokens = 0
        self.spent_cost = 0.0

    def emit(self, event: Event) -> None:
        if self.agent_path:
            if not self.events.listening:
                return
            event = event.model_copy(update={"agent_path": self.agent_path})
        self.events.put_nowait(event)

    async def taken(self, ahead: int = 0) -> None:
        """When paced, wait until the caller has taken every event emitted so far.

        With ahead, wait only once more than that many are left to take.
        """
        if self.paced and self.events.qsize() > ahead:
            await self.events.join()

    def spend(self, model: str, usage: Optional[Usage]) -> None:
        """Count one completion of model in this run and every run it is nested in."""
        if usage is not None:
            self.spent_tokens += usage.total_tokens
            self.spent_cost += usage.cost or 0.0
        result = self.result
        result.usage = _added(result.usage, usage)
        by_model = result.usage_by_model
        by_model[model] = _added(by_model.get(model, Usage()), usage)
        if self.parent is not None:
            self.parent.spend(model, usage)

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
        self._run.events.listening = True
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

    def steer(self, message: UserContent) -> None:
        """Add a user message that the model sees in its next completion.

        A message that arrives during a completion goes into the one after,
        once the step's tool calls finish; if that completion was the final
        answer, the run takes another step to answer it when its limits allow,
        and otherwise leaves it in the session for its next run; without a
        session it is dropped. Steering a finished run raises RuntimeError, and
        content the model can't take raises ValueError, as run input does.
        """
        if self._task.done():
            raise RuntimeError("The run has finished")
        self._run.steering.append(_user_content(self._run.config, message))

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
