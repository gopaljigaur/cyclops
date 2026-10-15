"""Keep a run's requests within the model's context window by summarizing older messages."""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import litellm
from litellm.constants import (
    DEFAULT_IMAGE_TOKEN_COUNT,
    DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET,
)

from cyclops.core.completion import _acompletion, _completion_kwargs, _usage
from cyclops.core.events import Compacted
from cyclops.core.types import AgentConfig, Compaction, Usage
from cyclops.toolkit.types import ToolResult

logger = logging.getLogger(__name__)

# The summary's message starts with this, as LangChain's summaries do.
_SUMMARY_PREFIX = "Here is a summary of the conversation to date:\n\n"

# A model to count tokens for, its max input tokens and max output tokens.
_Limits = Tuple[str, int, Optional[int]]


def _window(config: AgentConfig) -> Optional[_Limits]:
    """The limits of the config's model; None if LiteLLM knows no input limit.

    Through a router, those of the deployment with the smallest input limit.
    """
    windows = []
    if config.router is None:
        try:
            info: Any = litellm.get_model_info(config.model)
        except Exception:  # not mapped
            return None
        windows.append((config.model, info))
    else:
        for deployment in config.router.get_model_list(config.model) or []:
            try:
                info = config.router.get_router_model_info(deployment, config.model)
            except Exception:  # not mapped
                return None
            windows.append((deployment["litellm_params"]["model"], info))
    limits = [
        (model, info.get("max_input_tokens"), info.get("max_output_tokens"))
        for model, info in windows
    ]
    if not limits or any(max_input is None for _, max_input, _ in limits):
        return None
    return min(limits, key=lambda limit: limit[1])


def _known_window(config: AgentConfig) -> _Limits:
    """The limits of the config's model, which compaction needs."""
    limits = _window(config)
    if limits is None:
        raise ValueError(
            f"Compaction needs the input limit of {config.model!r}, which LiteLLM "
            "does not know; add it with litellm.register_model"
        )
    return limits


def _is_summary(message: Dict[str, Any]) -> bool:
    content = message.get("content")
    return isinstance(content, str) and content.startswith(_SUMMARY_PREFIX)


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


def _trimmed(lines: List[str], counts: List[int], budget: int) -> List[str]:
    """The newest lines within budget tokens, with a note of the oldest left out.

    A newest line longer than the budget keeps its end. Empty if nothing fits.
    """
    kept: List[str] = []
    total = 0
    for line, count in zip(reversed(lines), reversed(counts)):
        if total + count > budget:
            end = line[len(line) - len(line) * max(budget, 0) // count :]
            if not kept and end:
                kept.append(end)
            break
        kept.append(line)
        total += count
    if not kept:
        return []
    omitted = len(lines) - len(kept)
    if omitted:
        kept.append(f"[{omitted} earlier messages omitted]")
    return kept[::-1]


class _ContextWindow:
    """Counts the tokens of a run's requests and compacts its transcript.

    The system prompt and tools are counted when they change, and each
    transcript message once, when it is first seen.
    """

    def __init__(
        self,
        config: AgentConfig,
        limits: _Limits,
        spend: Callable[[Optional[Usage]], None],
    ) -> None:
        self.config = config
        self.model, self.max_input_tokens, self.max_output_tokens = limits
        # Where the summaries' usage goes.
        self.spend = spend
        # The system prompt and tools, and their tokens once counted.
        self._request: Tuple[Optional[str], Optional[List[Any]]] = (None, None)
        self._fixed: Optional[int] = None
        # Tokens of each counted message, in transcript order.
        self._counts: List[int] = []
        self._total = 0
        # Set once tokens() counted the transcript.
        self.counting = False
        # Set once a compaction left the request past the threshold.
        self._stuck = False

    def request(self, system_prompt: Optional[str], tools: Optional[List[Any]]) -> None:
        """Note the system prompt and tools the requests now carry."""
        self._request = (system_prompt, tools)
        self._fixed = None

    def _count(self, message: Dict[str, Any]) -> int:
        return litellm.token_counter(
            model=self.model,
            messages=[message],
            count_response_tokens=True,
            use_default_image_token_count=True,
            # Content LiteLLM can't count, such as input_audio parts, counts
            # as one image does.
            default_token_count=DEFAULT_IMAGE_TOKEN_COUNT,
        )

    def tokens(self, history: List[Dict[str, Any]]) -> int:
        """Tokens of a request with this transcript."""
        self.counting = True
        if self._fixed is None:
            system_prompt, tools = self._request
            system = [{"role": "system", "content": system_prompt}]
            # Also the tokens that prime the reply.
            self._fixed = litellm.token_counter(
                model=self.model, messages=system if system_prompt else [], tools=tools
            )
        for message in history[len(self._counts) :]:
            count = self._count(message)
            self._counts.append(count)
            self._total += count
        return self._fixed + self._total

    def fitting_max_tokens(
        self, history: List[Dict[str, Any]], kwargs: Dict[str, Any]
    ) -> Optional[int]:
        """The room the request leaves for output, if a lower max_tokens fits it.

        That is when the room is below max_tokens (or else the model's output
        limit) and above a thinking budget, which max_tokens has to exceed.
        """
        room = self.max_input_tokens - self.tokens(history)
        limit = kwargs.get("max_tokens") or self.max_output_tokens
        budget = (kwargs.get("thinking") or {}).get("budget_tokens") or 0
        if kwargs.get("reasoning_effort") not in (None, "none"):
            # LiteLLM turns an effort into a thinking budget for some
            # providers; this is the largest it uses.
            budget = max(budget, DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET)
        if budget < room and (limit is None or room < limit):
            return room
        return None

    async def compact_if_needed(
        self, history: List[Dict[str, Any]], compaction: Compaction
    ) -> Optional[Compacted]:
        """Compact once the request reaches the compaction threshold."""
        threshold = compaction.threshold * self.max_input_tokens
        if self._stuck or self.tokens(history) < threshold:
            return None
        return await self.compact(history, compaction)

    async def compact(
        self, history: List[Dict[str, Any]], compaction: Compaction
    ) -> Optional[Compacted]:
        """Replace the messages older than the kept ones with a summary of them.

        Returns the Compacted event, or None if nothing but the latest summary
        is older than the kept messages. Raises, leaving the transcript as it
        was, when the summary is empty or cut.
        """
        before = self.tokens(history)
        cut = self._cut(history, compaction.keep)
        if cut == 0 or (cut == 1 and _is_summary(history[0])):
            return None
        summary = await self._summarize(compaction, history[:cut], self._counts[:cut])
        message = {"role": "user", "content": _SUMMARY_PREFIX + summary}
        history[:cut] = [message]
        self._counts[:cut] = [self._count(message)]
        self._total = sum(self._counts)
        after = self.tokens(history)
        threshold = compaction.threshold * self.max_input_tokens
        if after >= threshold:
            self._stuck = True
            logger.warning(
                "Compaction left the request at %d tokens, past the threshold of "
                "%d; it is not compacted again in this run",
                after,
                threshold,
            )
        if self.config.hooks:
            self.config.hooks.on_compaction(summary, before, after)
        return Compacted(
            summary=summary,
            replaced_messages=cut,
            tokens_before=before,
            tokens_after=after,
        )

    def _cut(self, history: List[Dict[str, Any]], keep: float) -> int:
        """Index of the first kept message: the most recent ones within keep.

        The last message is always kept. A tool result is never the first kept
        message: the cut moves back to the call, so a call and its results are
        summarized or kept together.
        """
        if not history:
            return 0
        budget = keep * self.max_input_tokens
        cut = len(history) - 1
        kept = self._counts[cut]
        while cut > 0 and kept + self._counts[cut - 1] <= budget:
            cut -= 1
            kept += self._counts[cut]
        while cut > 0 and history[cut]["role"] == "tool":
            cut -= 1
        return cut

    async def _summarize(
        self,
        compaction: Compaction,
        messages: List[Dict[str, Any]],
        counts: List[int],
    ) -> str:
        """A summary of messages, written by the compaction model.

        The newest messages that fit the summarizing model's window are sent.
        """
        config = _summary_config(self.config, compaction)
        kwargs = _completion_kwargs(config, [], "none", None)
        # A one-off request gains nothing from a cache write, and its
        # max_tokens below could fall under a thinking budget.
        for key in ("cache_control_injection_points", "thinking", "reasoning_effort"):
            kwargs.pop(key, None)
        model, max_input, max_output = _known_window(config)
        # As long as the kept messages at most, so the summary leaves room.
        kwargs["max_tokens"] = int(compaction.keep * self.max_input_tokens)
        if max_output is not None:
            kwargs["max_tokens"] = min(kwargs["max_tokens"], max_output)
        head = f"{compaction.prompt}\n\n<conversation>\n"
        tail = "\n</conversation>"
        budget = (
            max_input
            - kwargs["max_tokens"]
            - litellm.token_counter(model=model, text=head + tail)
        )
        lines = _trimmed([_text(m) for m in messages], counts, budget)
        if not lines:
            raise RuntimeError(
                "Compaction found no room for the conversation in the "
                f"summarizing model's window of {max_input} tokens"
            )
        prompt = head + "\n\n".join(lines) + tail
        response = await _acompletion(
            config, messages=[{"role": "user", "content": prompt}], **kwargs
        )
        if config.hooks:
            config.hooks.on_llm_end(response)
        self.spend(_usage(response))
        choice = response.choices[0]
        summary = choice.message.content
        if not summary or choice.finish_reason in ("length", "content_filter"):
            raise RuntimeError(
                "Compaction got no usable summary (finish_reason "
                f"{choice.finish_reason!r}); the transcript is unchanged"
            )
        return summary


def _summary_config(config: AgentConfig, compaction: Compaction) -> AgentConfig:
    if compaction.model is None:
        return config
    return config.model_copy(update={"model": compaction.model})


def _compaction_window(
    config: AgentConfig,
    compaction: Compaction,
    spend: Callable[[Optional[Usage]], None],
) -> _ContextWindow:
    """A context window that compacts; raises ValueError when LiteLLM knows no
    input limit for the model or the summarizing model."""
    _known_window(_summary_config(config, compaction))
    return _ContextWindow(config, _known_window(config), spend)


def _context_window(
    config: AgentConfig,
    compaction: Optional[Compaction],
    spend: Callable[[Optional[Usage]], None],
) -> Optional[_ContextWindow]:
    """A context window for the config's model; without compaction, None when
    LiteLLM knows no input limit for it."""
    if compaction is not None:
        return _compaction_window(config, compaction, spend)
    limits = _window(config)
    return None if limits is None else _ContextWindow(config, limits, spend)
