"""Build a completion's arguments, call LiteLLM, and assemble streamed responses."""

import asyncio
from contextlib import aclosing
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Protocol,
    Type,
)

import litellm

# The Router's retry policy and backoff. Imported here so a LiteLLM that
# renames them fails at import, not inside the retry's except handler.
from litellm import _calculate_retry_after, _should_retry

from cyclops.core.caching import _cache_points
from cyclops.core.events import CompletionRetry, Event, ReasoningDelta, TextDelta
from cyclops.core.providers import (
    _caches_at_markers,
    _every_deployment,
    _native_schema,
    _response_schema,
    _takes_param,
)
from cyclops.core.types import AgentConfig, RateLimit, Usage
from cyclops.utils.compat import asyncio_timeout


def _completion_kwargs(
    config: AgentConfig,
    tools: List[Any],
    mode: str,
    response_model: Optional[Type],
) -> Dict[str, Any]:
    """The arguments of every completion in a run, besides its messages."""
    # Unset values stay out: through a Router, None would replace a deployment's own.
    sampling = {"temperature": config.temperature, "max_tokens": config.max_tokens}
    kwargs: Dict[str, Any] = {k: v for k, v in sampling.items() if v is not None}
    # A router has its own timeouts and retries. Elsewhere _acompletion's
    # retries replace those of the OpenAI client LiteLLM uses for some providers.
    if config.router is None:
        kwargs.update(timeout=config.timeout, max_retries=0)
    kwargs.update(config.completion_kwargs)
    # Injection points from completion_kwargs win, as over LiteLLM's own defaults.
    if config.prompt_caching and _every_deployment(config, _caches_at_markers):
        kwargs.setdefault(
            "cache_control_injection_points", _cache_points(config.prompt_caching)
        )
    # Reasoning arguments in completion_kwargs replace the typed options.
    if not {"reasoning_effort", "thinking"} & kwargs.keys():
        kwargs.update(_reasoning_kwargs(config))
    # With tools only where LiteLLM sends the schema natively: elsewhere it
    # may emulate it with a forced tool call, which blocks the agent's tools.
    if response_model is not None and (
        (mode == "none" and _every_deployment(config, _response_schema))
        or (mode == "native" and _every_deployment(config, _native_schema))
    ):
        kwargs["response_format"] = response_model
    if mode == "native":
        kwargs["tools"] = [_tool_to_openai_format(t) for t in tools]
    return kwargs


def _reasoning_kwargs(config: AgentConfig) -> Dict[str, Any]:
    """LiteLLM's reasoning arguments for the typed options the model takes."""
    if config.reasoning_effort and _every_deployment(
        config, _takes_param, "reasoning_effort"
    ):
        return {"reasoning_effort": config.reasoning_effort}
    if config.thinking_budget and _every_deployment(config, _takes_param, "thinking"):
        return {
            "thinking": {"type": "enabled", "budget_tokens": config.thinking_budget}
        }
    return {}


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


class StreamIdleTimeout(litellm.Timeout):
    """A stream sent no chunk for AgentConfig.stream_idle_timeout seconds."""


def _quota_spent(error: Exception) -> bool:
    """OpenAI's 429 for a spent quota, wherever LiteLLM's mapping kept it."""
    body = getattr(error, "body", None)
    if isinstance(body, dict) and body.get("code") == "insufficient_quota":
        return True
    # Providers LiteLLM calls over plain HTTP keep the error JSON in the message.
    return "insufficient_quota" in str(error)


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
        or not _should_retry(status)
        or _quota_spent(error)
        # A spent budget, whatever status LiteLLM gives it (429 before 1.104).
        or isinstance(error, litellm.BudgetExceededError)
    ):
        return None
    # Exponential backoff with jitter, or the retry-after header up to 60s.
    return _calculate_retry_after(
        remaining_retries=config.num_retries - retries,
        max_retries=config.num_retries,
        response_headers=getattr(error, "litellm_response_headers", None),
    )


# LiteLLM's prefix on the provider's own headers in a response's hidden params.
_PROVIDER_HEADER = "llm_provider-"


def _response_headers(response) -> Dict[str, str]:
    """The provider's HTTP headers of a completion."""
    headers = response._hidden_params.get("additional_headers") or {}
    return {
        k[len(_PROVIDER_HEADER) :]: v
        for k, v in headers.items()
        if k.startswith(_PROVIDER_HEADER)
    }


def _error_headers(error: Exception) -> Dict[str, str]:
    """The provider's HTTP headers of a failed completion, if it got a response."""
    return dict(getattr(error, "litellm_response_headers", None) or {})


def _retry_event(
    config: AgentConfig, error: Exception, delay: float
) -> CompletionRetry:
    headers = _error_headers(error)
    return CompletionRetry(
        model=config.model,
        delay=delay,
        status_code=getattr(error, "status_code", None),
        headers=headers,
        rate_limit=RateLimit.from_headers(headers),
        error=error,
    )


class _Output(Protocol):
    """Where a completion's events go: a run, as _Run implements it."""

    def emit(self, event: Event) -> None: ...

    async def taken(self, ahead: int = 0) -> None: ...


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


async def _acompletion(config: AgentConfig, run: Optional[_Output], /, **kwargs):
    """Make a completion, retrying transient provider errors.

    Each retry emits CompletionRetry to run, if given.
    """
    retries = 0
    while True:
        try:
            return await _attempt(config, **kwargs)
        except Exception as e:
            delay = _retry_delay(config, e, retries)
            if delay is None:
                raise
            retries += 1
            if run is not None:
                run.emit(_retry_event(config, e, delay))
            await asyncio.sleep(delay)


async def _next(chunks: AsyncIterator[Any], timeout: Optional[float]) -> Any:
    async with asyncio_timeout(timeout):
        return await chunks.__anext__()


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
        # The first chunk may follow minutes of thinking, so the idle timeout
        # starts after it.
        idle_timeout = None
        while True:
            try:
                chunk = await _next(response, idle_timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise StreamIdleTimeout(
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


# Deltas a paced run emits before it waits for the caller to take them.
_READ_AHEAD = 16


async def _streamed_completion(config: AgentConfig, messages, kwargs, run: _Output):
    """Stream a completion, emitting its deltas, and build the whole response.

    Until a delta is emitted, a transient error is retried as in _acompletion
    and a stalled stream once, each retry emitting CompletionRetry.
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
                    # LiteLLM ends the stream in an AnyIO worker thread, and AnyIO
                    # keeps a worker whose job is cancelled before it starts, so
                    # a run cancelled there leaks one (#103). Wait here instead.
                    finished = chunk.choices and chunk.choices[0].finish_reason
                    await run.taken(ahead=0 if finished else _READ_AHEAD)
        except StreamIdleTimeout as e:
            if emitted or stalled:
                raise
            stalled = True
            run.emit(_retry_event(config, e, 0))
        except Exception as e:
            delay = None if emitted else _retry_delay(config, e, retries)
            if delay is None:
                raise
            retries += 1
            run.emit(_retry_event(config, e, delay))
            await asyncio.sleep(delay)
        else:
            # The step's tool calls and the next completion wait for the caller.
            await run.taken()
            return litellm.stream_chunk_builder(chunks, messages=messages)


def _usage(response, api_duration: float) -> Optional[Usage]:
    """Tokens, cost and API time of one completion; None if it reports no usage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    prompt = getattr(usage, "prompt_tokens_details", None)
    completion = getattr(usage, "completion_tokens_details", None)
    # LiteLLM prices the response as it builds it, None if it has no pricing.
    hidden = response._hidden_params
    cost: Optional[float]
    if "response_cost" in hidden:
        cost = hidden["response_cost"]
    else:
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            cost = None
    return Usage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=getattr(prompt, "cached_tokens", None) or 0,
        cache_write_tokens=getattr(prompt, "cache_write_tokens", None) or 0,
        reasoning_tokens=getattr(completion, "reasoning_tokens", None) or 0,
        total_tokens=usage.total_tokens or 0,
        cost=cost,
        api_duration=api_duration,
    )
