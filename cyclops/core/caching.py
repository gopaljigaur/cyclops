"""Prompt caching breakpoints for providers that cache at cache_control markers."""

from typing import Any, Dict, List, Literal, Union

# True caches for the provider's default time (5 minutes on Anthropic models),
# "5m" or "1h" sets it, False sends no markers.
PromptCaching = Union[bool, Literal["5m", "1h"]]


def _cache_points(caching: PromptCaching) -> List[Dict[str, Any]]:
    """LiteLLM injection points for the system prompt and the latest message.

    Providers cache tools, system prompt and messages in that order, so the
    system marker covers the tools. LiteLLM marks a copy of the messages and
    skips any marker past the provider's limit.
    """
    control = {"type": "ephemeral"}
    if isinstance(caching, str):
        control["ttl"] = caching
    return [
        {"location": "message", "role": "system", "control": control},
        {"location": "message", "index": -1, "control": control},
    ]
