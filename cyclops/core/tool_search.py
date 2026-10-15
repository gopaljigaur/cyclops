"""Deferred tools: a search tool that loads them into a run as the model finds them."""

import re
from typing import Any, Dict, List, Set

from cyclops.toolkit.tool import Tool
from cyclops.toolkit.types import ToolAnnotations

SEARCH_TOOL_NAME = "search_tools"

# Most tools one search loads.
_MAX_RESULTS = 5

_DESCRIPTION = (
    "Search for more tools by keywords that describe what you need. The "
    "tools found are loaded and can be called from the next step on."
)


def _words(text: str) -> Set[str]:
    """Lowercased words, splitting names like read_file at underscores."""
    return set(re.findall(r"[^\W_]+", text.lower()))


class _ToolSearch:
    """The tools of one run: those always sent, plus the deferred ones found so far.

    With any tool deferred, the tools sent include a search tool over them.
    """

    def __init__(self, tools: List[Any], tools_by_name: Dict[str, Any]) -> None:
        self.loaded = [t for t in tools if not t.defer_loading]
        self.tools_by_name = tools_by_name
        # Each deferred tool with the words of its name and description.
        self._index = [
            (t, _words(f"{t.name} {t.description}")) for t in tools if t.defer_loading
        ]
        # True until the step loop has read loaded, and again after a search adds to it.
        self.changed = True
        if self._index:
            search = Tool(
                SEARCH_TOOL_NAME,
                _DESCRIPTION,
                self.search,
                ToolAnnotations(read_only=True),
            )
            self.loaded.append(search)
            self.tools_by_name = {**tools_by_name, SEARCH_TOOL_NAME: search}

    async def search(self, query: str) -> str:
        """Find deferred tools by the words their name and description share with query.

        Args:
            query: Keywords such as "read file" or "calendar event".
        """
        words = _words(query)
        scores = [(len(words & tool_words), t) for t, tool_words in self._index]
        found = [
            t
            for score, t in sorted(scores, key=lambda s: -s[0])[:_MAX_RESULTS]
            if score
        ]
        if not found:
            return f"No tools match '{query}'."
        new = [t for t in found if t not in self.loaded]
        if new:
            self.loaded += new
            self.changed = True
        listing = "".join(f"\n- {t.name}: {t.description}" for t in found)
        return f"These tools can now be called:{listing}"
