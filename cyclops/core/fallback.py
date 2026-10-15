"""Send a step's completion to the fallback models when a model is down or refuses."""

import time
from typing import Any, Dict, List, Optional, Tuple, Type

import litellm
from litellm import _should_retry

from cyclops.core.completion import (
    _acompletion,
    _completion_kwargs,
    _error_headers,
    _Output,
    _response_headers,
    _streamed_completion,
    _usage,
)
from cyclops.core.events import Event, ModelFallback, ReasoningDelta, TextDelta
from cyclops.core.run_state import _Run
from cyclops.core.types import AgentConfig, RateLimit, Usage

# A model's config and the arguments of its completions.
_Request = Tuple[AgentConfig, Dict[str, Any]]


def _requests(
    config: AgentConfig, tools: List[Any], mode: str, response_model: Optional[Type]
) -> List[_Request]:
    """The model's request, then each fallback model's, built for what it supports."""
    configs = [config] + [
        config.model_copy(update={"model": model}) for model in config.fallback_models
    ]
    return [(c, _completion_kwargs(c, tools, mode, response_model)) for c in configs]


def _falls_back(error: Exception) -> bool:
    """True for a content policy block, or an overload or outage that outlasted
    the retries: the errors they retry, and a spent quota."""
    if isinstance(error, litellm.ContentPolicyViolationError):
        return True
    status = getattr(error, "status_code", None)
    return status is not None and _should_retry(status)


class _Watched:
    """A run, noting whether a streamed completion emitted a delta to it."""

    def __init__(self, run: _Run) -> None:
        self.run = run
        self.emitted = False

    def emit(self, event: Event) -> None:
        if isinstance(event, (TextDelta, ReasoningDelta)):
            self.emitted = True
        self.run.emit(event)

    async def taken(self, ahead: int = 0) -> None:
        await self.run.taken(ahead)


async def _send(
    request: _Request, messages: List[Dict[str, Any]], run: _Output, streamed: bool
) -> Tuple[Any, Optional[Usage]]:
    """A completion and its usage."""
    config, kwargs = request
    start = time.perf_counter()
    if streamed:
        response = await _streamed_completion(config, messages, kwargs, run)
    else:
        response = await _acompletion(config, run, messages=messages, **kwargs)
    return response, _usage(response, time.perf_counter() - start)


async def _complete_with_fallbacks(
    requests: List[_Request],
    messages: List[Dict[str, Any]],
    run: _Run,
    streamed: bool,
) -> Optional[Tuple[str, Any, Optional[Usage]]]:
    """The model that answered a step, its response and usage; None if a
    refusal spent the run's budget before the next model.

    The models are tried in order, each after its own retries. Once a model
    emitted part of a stream, its error or refusal ends the step as without
    fallbacks.
    """
    for request, (fallback, _) in zip(requests, requests[1:]):
        config = request[0]
        watched = _Watched(run)
        try:
            response, usage = await _send(request, messages, watched, streamed)
        except Exception as e:
            if watched.emitted or not _falls_back(e):
                raise
            headers = _error_headers(e)
            run.emit(
                ModelFallback(
                    model=config.model,
                    fallback_model=fallback.model,
                    error=e,
                    status_code=getattr(e, "status_code", None),
                    headers=headers,
                    rate_limit=RateLimit.from_headers(headers),
                )
            )
            continue
        if watched.emitted or response.choices[0].finish_reason != "content_filter":
            return config.model, response, usage
        if config.hooks:
            config.hooks.on_llm_end(response)
        # A refused completion is billed too.
        run.spend(config.model, usage)
        if run.over_budget():
            return None
        headers = _response_headers(response)
        run.emit(
            ModelFallback(
                model=config.model,
                fallback_model=fallback.model,
                usage=usage,
                headers=headers,
                rate_limit=RateLimit.from_headers(headers),
            )
        )
    response, usage = await _send(requests[-1], messages, run, streamed)
    return requests[-1][0].model, response, usage
