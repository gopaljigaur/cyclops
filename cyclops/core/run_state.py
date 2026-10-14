"""The state of one run in progress, shared by the step loop and the code it calls."""

import asyncio
from typing import Any, Dict, List, Optional

from cyclops.core.events import Event
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
