"""Keep a run's requests within the model's context window by summarizing older messages."""

from typing import Any, Dict, List, Optional, Tuple

import litellm

from cyclops.core.completion import _acompletion, _completion_kwargs, _usage
from cyclops.core.events import Compacted
from cyclops.core.types import AgentConfig, Compaction, Usage
from cyclops.toolkit.types import ToolResult

# The summary's message starts with this, as LangChain's summaries do.
_SUMMARY_PREFIX = "Here is a summary of the conversation to date:\n\n"


def _window(config: AgentConfig) -> Optional[Tuple[str, int]]:
    """The model to count tokens for and its max input tokens.

    Through a router, the deployment with the smallest window. None if LiteLLM
    knows no window for some deployment.
    """
    windows = []
    if config.router is None:
        try:
            info: Any = litellm.get_model_info(config.model)
        except Exception:  # not mapped
            return None
        windows.append((config.model, info.get("max_input_tokens")))
    else:
        for deployment in config.router.get_model_list(config.model) or []:
            try:
                info = config.router.get_router_model_info(deployment, config.model)
            except Exception:  # not mapped
                return None
            model = deployment["litellm_params"]["model"]
            windows.append((model, info.get("max_input_tokens")))
    if not windows or any(tokens is None for _, tokens in windows):
        return None
    return min(windows, key=lambda w: w[1])


def _text(message: Dict[str, Any]) -> str:
    """A message as a line of the transcript the summary is written from."""
    role = message["role"]
    content = message.get("content")
    text = ToolResult.message_text(content) if content else ""
    if role == "tool":
        return f"tool result ({message.get('name')}): {text}"
    calls = [
        f"called {c['function']['name']}({c['function']['arguments']})"
        for c in message.get("tool_calls") or ()
    ]
    return f"{role}: " + "\n".join(filter(None, [text, *calls]))


async def _summarize(
    config: AgentConfig, compaction: Compaction, messages: List[Dict[str, Any]]
) -> Tuple[str, Optional[Usage]]:
    """A summary of messages, written by the compaction model, and its usage."""
    if compaction.model is not None:
        config = config.model_copy(update={"model": compaction.model})
    kwargs = _completion_kwargs(config, [], "none", None)
    # A one-off request gains nothing from a cache write.
    kwargs.pop("cache_control_injection_points", None)
    transcript = "\n\n".join(_text(m) for m in messages)
    prompt = f"{compaction.prompt}\n\n<conversation>\n{transcript}\n</conversation>"
    response = await _acompletion(
        config, messages=[{"role": "user", "content": prompt}], **kwargs
    )
    if config.hooks:
        config.hooks.on_llm_end(response)
    return response.choices[0].message.content or "", _usage(response)


class _ContextWindow:
    """Counts the tokens of a run's requests and compacts its transcript.

    The system prompt and tools are counted once, and each transcript message
    once, when it is first seen.
    """

    def __init__(
        self,
        config: AgentConfig,
        compaction: Compaction,
        model: str,
        max_input_tokens: int,
        system_prompt: Optional[str],
        tools: Optional[List[Any]],
    ) -> None:
        self.config = config
        self.compaction = compaction
        self.model = model
        self.max_input_tokens = max_input_tokens
        system = [{"role": "system", "content": system_prompt}] if system_prompt else []
        # Also the tokens that prime the reply.
        self._fixed = litellm.token_counter(model=model, messages=system, tools=tools)
        # Tokens of each counted message, in transcript order.
        self._counts: List[int] = []
        self._total = 0

    def _count(self, message: Dict[str, Any]) -> int:
        return litellm.token_counter(
            model=self.model,
            messages=[message],
            count_response_tokens=True,
            use_default_image_token_count=True,
        )

    def tokens(self, history: List[Dict[str, Any]]) -> int:
        """Tokens of a request with this transcript."""
        for message in history[len(self._counts) :]:
            count = self._count(message)
            self._counts.append(count)
            self._total += count
        return self._fixed + self._total

    async def compact_if_needed(
        self, history: List[Dict[str, Any]]
    ) -> Optional[Tuple[Compacted, Optional[Usage]]]:
        """Compact once the request reaches the compaction threshold."""
        limit = self.compaction.threshold * self.max_input_tokens
        if self.tokens(history) < limit:
            return None
        return await self.compact(history)

    async def compact(
        self, history: List[Dict[str, Any]]
    ) -> Optional[Tuple[Compacted, Optional[Usage]]]:
        """Replace the messages older than the kept ones with a summary of them.

        Returns the Compacted event and the summary's usage, or None if every
        message is kept.
        """
        before = self.tokens(history)
        cut = self._cut(history)
        if cut == 0:
            return None
        summary, usage = await _summarize(self.config, self.compaction, history[:cut])
        message = {"role": "user", "content": _SUMMARY_PREFIX + summary}
        history[:cut] = [message]
        self._counts[:cut] = [self._count(message)]
        self._total = sum(self._counts)
        after = self.tokens(history)
        if self.config.hooks:
            self.config.hooks.on_compaction(summary, before, after)
        event = Compacted(
            summary=summary,
            replaced_messages=cut,
            tokens_before=before,
            tokens_after=after,
        )
        return event, usage

    def _cut(self, history: List[Dict[str, Any]]) -> int:
        """Index of the first kept message: the most recent ones within keep.

        A tool result is never the first kept message, so a tool call and its
        results are summarized or kept together.
        """
        budget = self.compaction.keep * self.max_input_tokens
        cut, kept = len(history), 0
        while cut > 0 and kept + self._counts[cut - 1] <= budget:
            cut -= 1
            kept += self._counts[cut]
        while cut < len(history) and history[cut]["role"] == "tool":
            cut += 1
        return cut


def _context_window(
    config: AgentConfig,
    compaction: Compaction,
    system_prompt: Optional[str],
    tools: Optional[List[Any]],
) -> Optional[_ContextWindow]:
    """A context window for the config's model, or None if LiteLLM knows none."""
    window = _window(config)
    if window is None:
        return None
    return _ContextWindow(config, compaction, *window, system_prompt, tools)
