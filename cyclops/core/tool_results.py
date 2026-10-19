"""Oversized tool results: stored whole, sent as a preview, read in pages by handle."""

import uuid
from typing import Annotated, Any, Dict, List, Optional, Protocol

from pydantic import Field

from cyclops.core.types import AgentConfig
from cyclops.toolkit.tool import Tool
from cyclops.toolkit.types import ToolAnnotations

READ_TOOL_NAME = "read_tool_result"

_DESCRIPTION = (
    "Read part of a tool result that was too long to send whole, by the "
    "handle in its note."
)


class ToolResultStore(Protocol):
    """Where a run keeps the tool results too long to send whole.

    A run deletes what it stored when it ends, so a handle lasts one run.
    """

    async def put(self, handle: str, text: str) -> None:
        """Store text under handle."""

    async def get(self, handle: str) -> Optional[str]:
        """The text stored under handle, or None if there is none."""

    async def delete(self, handle: str) -> None:
        """Remove the text stored under handle, if there is any."""


class InMemoryToolResultStore:
    """Keeps tool results in this process."""

    def __init__(self) -> None:
        self._results: Dict[str, str] = {}

    async def put(self, handle: str, text: str) -> None:
        self._results[handle] = text

    async def get(self, handle: str) -> Optional[str]:
        return self._results.get(handle)

    async def delete(self, handle: str) -> None:
        self._results.pop(handle, None)


def _read_on(handle: str, offset: int) -> str:
    return f'{READ_TOOL_NAME}(handle="{handle}", offset={offset})'


def _tail(text: str, limit: int) -> int:
    """Where the preview's end part of text starts."""
    return len(text) - (limit - limit // 2)


def _preview(text: str, handle: str, limit: int) -> str:
    """The start and end of text, limit characters in all, around a note of the rest."""
    head, tail = limit // 2, _tail(text, limit)
    note = (
        f"[Characters {head} to {tail} of {len(text)} omitted. "
        f"Read them with {_read_on(handle, head)}.]"
    )
    return f"{text[:head]}\n{note}\n{text[tail:]}"


def _page(text: str, handle: str, offset: int, limit: int, stop: int) -> str:
    """The characters of text from offset, at most limit and up to stop, and where they are."""
    end = min(offset + limit, stop)
    where = f"Characters {offset} to {end} of {len(text)}"
    if end == len(text):
        note = f"[{where}, the end of the result.]"
    elif end == stop:
        note = f"[{where}. The rest, from {end}, was in the preview.]"
    else:
        note = f"[{where}. Read on with {_read_on(handle, end)}.]"
    return f"{text[offset:end]}\n{note}"


def _limits_output(config: AgentConfig, tools: List[Any]) -> bool:
    """True if the agent or any of its tools has an output limit."""
    return config.max_tool_output_chars is not None or any(
        t.max_output_chars is not None for t in tools
    )


class _ToolResults:
    """The results one run stored, and the tool that reads them."""

    def __init__(self, store: Optional[ToolResultStore]) -> None:
        self.store = store or InMemoryToolResultStore()
        # Each stored result's page size: the limit of the tool that returned it.
        self._pages: Dict[str, int] = {}
        self.tool = Tool(
            READ_TOOL_NAME, _DESCRIPTION, self.read, ToolAnnotations(read_only=True)
        )

    async def preview(self, text: str, limit: int) -> str:
        """Store text and return its preview, with its handle."""
        handle = uuid.uuid4().hex
        await self.store.put(handle, text)
        self._pages[handle] = limit
        return _preview(text, handle, limit)

    async def read(
        self,
        handle: str,
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Optional[Annotated[int, Field(gt=0)]] = None,
    ) -> str:
        """Read a stored tool result from a character offset.

        Args:
            handle: The handle in the result's note.
            offset: The first character to read.
            limit: Most characters to read; at most, and by default, a page.
        """
        page = self._pages.get(handle)
        # Only this run's results, though the store may be shared.
        text = None if page is None else await self.store.get(handle)
        if page is None or text is None:
            raise ValueError(f"No stored tool result has the handle '{handle}'")
        if offset >= len(text):
            raise ValueError(f"offset {offset} is past the end, at {len(text)}")
        # Pages from inside the omitted range stop where the preview's end part starts.
        tail = _tail(text, page)
        stop = tail if offset < tail else len(text)
        return _page(text, handle, offset, min(limit or page, page), stop)

    async def free(self) -> None:
        """Delete every result this run stored."""
        for handle in self._pages:
            await self.store.delete(handle)
        self._pages.clear()


__all__ = ["InMemoryToolResultStore", "ToolResultStore"]
