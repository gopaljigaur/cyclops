"""Provider rate limit headers on step, retry and fallback events."""

import asyncio
from datetime import datetime, timedelta, timezone
from json import dumps, loads
from unittest.mock import patch

import httpx
import pytest

from cyclops import Agent, AgentConfig, RateLimit
from cyclops.core import completion
from cyclops.core.events import CompletionRetry, ModelFallback, StepFinished

OPENAI = "gpt-4o-mini"
ANTHROPIC = "anthropic/claude-sonnet-4-5"

OPENAI_HEADERS = {
    "x-ratelimit-limit-requests": "500",
    "x-ratelimit-remaining-requests": "499",
    "x-ratelimit-reset-requests": "120ms",
    "x-ratelimit-limit-tokens": "30000",
    "x-ratelimit-remaining-tokens": "29000",
    "x-ratelimit-reset-tokens": "6m0s",
    "x-request-id": "req_1",
}


def _anthropic_headers():
    soon = datetime.now(timezone.utc) + timedelta(seconds=30)
    return {
        "anthropic-ratelimit-requests-limit": "50",
        "anthropic-ratelimit-requests-remaining": "49",
        "anthropic-ratelimit-requests-reset": soon.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "anthropic-ratelimit-tokens-limit": "8000",
        "anthropic-ratelimit-tokens-remaining": "7000",
        "anthropic-ratelimit-tokens-reset": soon.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "request-id": "req_a",
    }


def _sse(events):
    return "".join(f"data: {dumps(e)}\n\n" for e in events) + "data: [DONE]\n\n"


def _openai_chunk(delta, finish_reason=None):
    choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
    return {
        "id": "c",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": OPENAI,
        "choices": [choice],
    }


def _anthropic_event(event):
    return f"event: {event['type']}\ndata: {dumps(event)}\n\n"


# Each provider's reply as JSON and as a stream.
REPLIES = {
    OPENAI: (
        {
            "id": "c",
            "object": "chat.completion",
            "created": 1,
            "model": OPENAI,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        _sse(
            [
                _openai_chunk({"role": "assistant", "content": "hi"}),
                _openai_chunk({}, "stop"),
            ]
        ),
    ),
    ANTHROPIC: (
        {
            "id": "m",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        "".join(
            _anthropic_event(e)
            for e in [
                {
                    "type": "message_start",
                    "message": {
                        "id": "m",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5",
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "hi"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 1},
                },
                {"type": "message_stop"},
            ]
        ),
    ),
}

RATE_LIMITED = {"error": {"type": "rate_limit_error", "message": "slow down"}}


def _headers(model):
    return OPENAI_HEADERS if model == OPENAI else _anthropic_headers()


@pytest.fixture
def provider():
    """Answer each HTTP request with the next (status, headers, stream) reply.

    The provider's own parser in LiteLLM reads each reply, as from the network.
    """
    replies = []

    async def send(self, request, **kwargs):
        status, headers, model = replies.pop(0)
        reply, stream = REPLIES[model]
        if status != 200:
            return httpx.Response(
                status, json=RATE_LIMITED, headers=headers, request=request
            )
        if loads(request.content).get("stream"):
            headers = {**headers, "content-type": "text/event-stream"}
            return httpx.Response(
                200, content=stream.encode(), headers=headers, request=request
            )
        return httpx.Response(200, json=reply, headers=headers, request=request)

    with patch.object(httpx.AsyncClient, "send", send):
        yield replies


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    """The retries don't wait."""
    monkeypatch.setattr(completion, "_calculate_retry_after", lambda **kwargs: 0)


def _events(config, stream):
    async def collect():
        handle = Agent(config).start("hi", stream=stream)
        return [event async for event in handle]

    return asyncio.run(collect())


def _config(model, **kwargs):
    return AgentConfig(model=model, completion_kwargs={"api_key": "test"}, **kwargs)


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("model", [OPENAI, ANTHROPIC])
def test_step_finished_has_the_response_headers(provider, model, stream):
    headers = _headers(model)
    provider.append((200, headers, model))

    (step,) = [
        e for e in _events(_config(model), stream) if isinstance(e, StepFinished)
    ]

    assert step.headers.items() >= headers.items()
    if model == OPENAI:
        assert step.rate_limit == RateLimit(
            remaining_requests=499,
            reset_requests=0.12,
            remaining_tokens=29000,
            reset_tokens=360,
        )
    else:
        limit = step.rate_limit
        assert (limit.remaining_requests, limit.remaining_tokens) == (49, 7000)
        assert 25 < limit.reset_requests <= 30
        assert 25 < limit.reset_tokens <= 30


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
@pytest.mark.parametrize("model", [OPENAI, ANTHROPIC])
def test_a_retry_carries_the_429_headers(provider, model, stream):
    headers = {**_headers(model), "retry-after": "7"}
    provider += [(429, headers, model), (200, {}, model)]

    events = _events(_config(model), stream)

    (retry,) = [e for e in events if isinstance(e, CompletionRetry)]
    assert retry.model == model
    assert retry.status_code == 429
    assert retry.headers.items() >= headers.items()
    assert retry.rate_limit.remaining_tokens == (29000 if model == OPENAI else 7000)
    assert retry.delay == 0
    assert events[-1].result.content == "hi"


def test_a_fallback_carries_the_error_status_and_headers(provider):
    provider += [(429, OPENAI_HEADERS, OPENAI), (200, {}, OPENAI)]
    config = _config(OPENAI, num_retries=0, fallback_models=["gpt-4o"])

    events = _events(config, stream=False)

    (fallback,) = [e for e in events if isinstance(e, ModelFallback)]
    assert fallback.status_code == 429
    assert fallback.headers["x-request-id"] == "req_1"
    assert fallback.rate_limit.remaining_requests == 499


def test_a_streamed_retry_still_falls_back(provider):
    # A retry event is no delta, so the stream may still fall back after it.
    provider += [
        (429, OPENAI_HEADERS, OPENAI),
        (429, OPENAI_HEADERS, OPENAI),
        (200, {}, OPENAI),
    ]
    config = _config(OPENAI, num_retries=1, fallback_models=["gpt-4o"])

    events = _events(config, stream=True)

    assert [type(e) for e in events if not isinstance(e, StepFinished)][:2] == [
        CompletionRetry,
        ModelFallback,
    ]
    assert events[-1].result.content == "hi"


def test_a_response_without_rate_limit_headers_has_none(provider):
    provider.append((200, {"x-request-id": "req_2"}, OPENAI))

    (step,) = [
        e for e in _events(_config(OPENAI), False) if isinstance(e, StepFinished)
    ]

    assert step.headers["x-request-id"] == "req_2"
    assert step.rate_limit is None


@pytest.mark.parametrize(
    "value, seconds",
    [("1s", 1), ("6m0s", 360), ("120ms", 0.12), ("1h2m3.5s", 3723.5), ("0s", 0)],
)
def test_openai_reset_durations_are_read(value, seconds):
    limit = RateLimit.from_headers({"x-ratelimit-reset-tokens": value})

    assert limit.reset_tokens == pytest.approx(seconds)


def test_a_past_anthropic_reset_is_zero_seconds_away():
    limit = RateLimit.from_headers(
        {"anthropic-ratelimit-tokens-reset": "2020-01-01T00:00:00Z"}
    )

    assert limit.reset_tokens == 0


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"retry-after": "7"},
        {"x-ratelimit-remaining-requests": "many"},
        {"x-ratelimit-reset-requests": "soon"},
        {"anthropic-ratelimit-requests-reset": "soon"},
    ],
)
def test_headers_without_readable_limits_give_none(headers):
    assert RateLimit.from_headers(headers) is None
