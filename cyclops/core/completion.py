"""Build a completion's arguments, call LiteLLM, and assemble streamed responses."""

import asyncio
from contextlib import aclosing
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

import litellm

from cyclops.core.events import ReasoningDelta, TextDelta
from cyclops.core.providers import (
    _every_deployment,
    _marker_caching_supported,
    _native_schema,
)
from cyclops.core.run_state import _Run
from cyclops.core.types import AgentConfig, Usage

# Cache the system prompt (with the tools before it) and the conversation so far.
_CACHE_POINTS = [
    {"location": "message", "role": "system"},
    {"location": "message", "index": -1},
]


def _completion_kwargs(
    config: AgentConfig,
    tools: List[Any],
    mode: str,
    response_model: Optional[Type],
) -> Dict[str, Any]:
    """The arguments of every completion in a run, besides its messages."""
    kwargs: Dict[str, Any] = {
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
    # A router has its own timeouts and retries. Elsewhere _acompletion's
    # retries replace those of the OpenAI client LiteLLM uses for some providers.
    if config.router is None:
        kwargs.update(timeout=config.timeout, max_retries=0)
    kwargs.update(config.completion_kwargs)
    if config.prompt_caching and _marker_caching_supported(config):
        kwargs["cache_control_injection_points"] = _CACHE_POINTS
    # With tools only where LiteLLM sends the schema natively: elsewhere it
    # may emulate it with a forced tool call, which blocks the agent's tools.
    if response_model is not None and (
        (mode == "none" and _every_deployment(config, litellm.supports_response_schema))
        or (mode == "native" and _every_deployment(config, _native_schema))
    ):
        kwargs["response_format"] = response_model
    if mode == "native":
        kwargs["tools"] = [_tool_to_openai_format(t) for t in tools]
    return kwargs


def _tool_to_openai_format(tool) -> Dict[str, Any]:
    """Convert a Tool to OpenAI function-calling format."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.definition.to_json_schema(),
        },
    }


def _build_messages(
    history: List[Dict[str, Any]], system_prompt: Optional[str]
) -> List[Dict[str, Any]]:
    """Build message list for LiteLLM, prepending the system prompt if any."""
    messages = list(history)
    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages
    return messages


class _StreamStalled(litellm.Timeout):
    """A stream sent no chunk for config.stream_idle_timeout seconds."""


def _retry_delay(
    config: AgentConfig, error: Exception, retries: int
) -> Optional[float]:
    """Seconds to wait before retrying a failed completion, or None to raise."""
    status = getattr(error, "status_code", None)
    if (
        config.router is not None
        or retries == config.num_retries
        or status is None
        # 408, 409, 429 and 5xx, as the OpenAI client retries.
        or not litellm._should_retry(status)
        # OpenAI reports a spent quota as a rate limit.
        or "insufficient_quota" in str(error)
    ):
        return None
    # Exponential backoff with jitter, or the retry-after header up to 60s.
    return litellm._calculate_retry_after(
        remaining_retries=config.num_retries - retries,
        max_retries=config.num_retries,
        response_headers=getattr(error, "litellm_response_headers", None),
    )


async def _attempt(config: AgentConfig, /, **kwargs):
    if config.hooks:
        config.hooks.on_llm_start(kwargs.get("messages", []))
    try:
        if config.router:
            return await config.router.acompletion(model=config.model, **kwargs)
        return await litellm.acompletion(model=config.model, **kwargs)
    except Exception as e:
        if config.hooks:
            config.hooks.on_llm_error(e)
        raise


async def _acompletion(config: AgentConfig, /, **kwargs):
    """Make a completion, retrying transient provider errors."""
    retries = 0
    while True:
        try:
            return await _attempt(config, **kwargs)
        except Exception as e:
            delay = _retry_delay(config, e, retries)
            if delay is None:
                raise
            retries += 1
            await asyncio.sleep(delay)


async def _astream_completion(
    config: AgentConfig, messages, kwargs
) -> AsyncGenerator[Any, None]:
    response = await _attempt(
        config,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **kwargs,
    )
    try:
        # The request timeout bounds the wait for the first chunk.
        idle_timeout = None
        while True:
            try:
                chunk = await asyncio.wait_for(response.__anext__(), idle_timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise _StreamStalled(
                    message=f"no stream chunk for {idle_timeout}s",
                    model=config.model,
                    llm_provider=None,
                ) from None
            yield chunk
            idle_timeout = config.stream_idle_timeout
    except Exception as e:
        if config.hooks:
            config.hooks.on_llm_error(e)
        raise
    finally:
        await response.aclose()


async def _streamed_completion(config: AgentConfig, messages, kwargs, run: _Run):
    """Stream a completion, emitting its deltas, and build the whole response.

    Until a delta is emitted, a transient error is retried as in _acompletion
    and a stalled stream once.
    """
    retries, stalled = 0, False
    while True:
        chunks = []
        emitted = False
        try:
            async with aclosing(
                _astream_completion(config, messages, kwargs)
            ) as completion:
                async for chunk in completion:
                    chunks.append(chunk)
                    delta = chunk.choices[0].delta if chunk.choices else None
                    # LiteLLM sets reasoning_content on a delta only when present.
                    reasoning = getattr(delta, "reasoning_content", None)
                    if reasoning:
                        emitted = True
                        run.emit(ReasoningDelta(text=reasoning))
                    if delta and delta.content:
                        emitted = True
                        run.emit(TextDelta(text=delta.content))
                    await run.taken()
        except _StreamStalled:
            if emitted or stalled:
                raise
            stalled = True
        except Exception as e:
            delay = None if emitted else _retry_delay(config, e, retries)
            if delay is None:
                raise
            retries += 1
            await asyncio.sleep(delay)
        else:
            return litellm.stream_chunk_builder(chunks, messages=messages)


def _usage(response) -> Optional[Usage]:
    """Tokens and cost of one completion; None if it reports no usage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    try:
        cost: Optional[float] = litellm.completion_cost(completion_response=response)
    except Exception:
        # LiteLLM has no pricing for this model.
        cost = None
    return Usage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=getattr(details, "cached_tokens", None) or 0,
        total_tokens=usage.total_tokens or 0,
        cost=cost,
    )
