"""Oversized tool results: stored whole, sent as a first page, read on by handle."""

import uuid
from typing import Annotated, Dict, Optional, Protocol

from pydantic import Field

from cyclops.core.tool_search import _ToolSearch
from cyclops.toolkit.tool import Tool
from cyclops.toolkit.types import ToolAnnotations

READ_TOOL_NAME = "read_tool_result"

_DESCRIPTION = (
    "Read more of a tool result that was too long to send whole, by the handle "
    "its first page ends with."
)


class ToolResultStore(Protocol):
    """Where a run keeps the tool results too long to send whole.

    A run deletes what it stored when it ends.
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


def _page(text: str, handle: str, offset: int, limit: int) -> str:
    """The characters of text from offset, at most limit, and where they are."""
    end = min(offset + limit, len(text))
    where = f"Characters {offset} to {end} of {len(text)}"
    if end == len(text):
        note = f"[{where}, the end of the result.]"
    else:
        note = (
            f"[{where}. Read on with {READ_TOOL_NAME}"
            f'(handle="{handle}", offset={end}).]'
        )
    return f"{text[offset:end]}\n{note}"


class _ToolResults:
    """The results one run stored, and the tool that reads them.

    The tool joins the run's tools once a result is stored.
    """

    def __init__(self, store: Optional[ToolResultStore], tools: _ToolSearch) -> None:
        self.store = store or InMemoryToolResultStore()
        self.tools = tools
        # Each stored result's page size: the limit of the tool that returned it.
        self._pages: Dict[str, int] = {}
        self.tool = Tool(
            READ_TOOL_NAME, _DESCRIPTION, self.read, ToolAnnotations(read_only=True)
        )

    async def first_page(self, text: str, limit: int) -> str:
        """Store text and return its first limit characters with its handle."""
        handle = uuid.uuid4().hex
        await self.store.put(handle, text)
        self._pages[handle] = limit
        if self.tool not in self.tools.loaded:
            self.tools.add(self.tool)
        return _page(text, handle, 0, limit)

    async def read(
        self,
        handle: str,
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Optional[Annotated[int, Field(gt=0)]] = None,
    ) -> str:
        """Read a stored tool result from a character offset.

        Args:
            handle: The handle the result's first page ends with.
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
        return _page(text, handle, offset, min(limit or page, page))

    async def free(self) -> None:
        """Delete every result this run stored."""
        for handle in self._pages:
            await self.store.delete(handle)
        self._pages.clear()


__all__ = ["InMemoryToolResultStore", "ToolResultStore"]
