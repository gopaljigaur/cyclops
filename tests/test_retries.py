"""Retries of transient provider errors, request timeouts and stalled streams."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import litellm
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

from cyclops.core.agent import Agent
from cyclops.core.hooks import AgentHooks
from cyclops.core.types import AgentConfig

MODEL = "gpt-4o-mini"


def _response(text: str = "ok"):
    return litellm.ModelResponse(
        model=MODEL, choices=[{"message": {"role": "assistant", "content": text}}]
    )


def _chunk(text: str = ""):
    return ModelResponseStream(
        model=MODEL, choices=[StreamingChoices(delta=Delta(content=text))]
    )


async def _stream(*items, stall_before=None):
    """Yield chunks, raising an exception item; never returns at stall_before."""
    for i, item in enumerate(items):
        if i == stall_before:
            await asyncio.Event().wait()
        if isinstance(item, BaseException):
            raise item
        yield item


def _rate_limit(message: str = "slow down"):
    return litellm.RateLimitError(message=message, llm_provider="openai", model=MODEL)


def _overloaded():
    return litellm.InternalServerError(
        message="Overloaded", llm_provider="anthropic", model=MODEL
    )


@pytest.fixture
def delays(monkeypatch):
    """The backoff LiteLLM computed for each retry; the retries don't wait."""
    computed = []
    calculate = litellm._calculate_retry_after

    def record(**kwargs):
        computed.append(calculate(**kwargs))
        return 0

    monkeypatch.setattr(litellm, "_calculate_retry_after", record)
    return computed


async def _streamed_text(agent: Agent) -> str:
    return "".join([text async for text in agent.astream("hi")])


def test_request_sets_timeout_and_disables_client_retries():
    mock = AsyncMock(return_value=_response())
    with patch("litellm.acompletion", new=mock):
        Agent(AgentConfig(model=MODEL)).run("hi")

    assert mock.call_args.kwargs["timeout"] == 600
    assert mock.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize(
    "error",
    [
        _rate_limit(),
        _overloaded(),
        litellm.ServiceUnavailableError(
            message="unavailable", llm_provider="openai", model=MODEL
        ),
        litellm.Timeout(message="timed out", llm_provider="openai", model=MODEL),
        litellm.APIConnectionError(message="reset", llm_provider="openai", model=MODEL),
    ],
)
def test_transient_error_is_retried(error, delays):
    mock = AsyncMock(side_effect=[error, _response("done")])
    with patch("litellm.acompletion", new=mock):
        assert Agent(AgentConfig(model=MODEL)).run("hi") == "done"

    assert mock.call_count == 2
    assert len(delays) == 1


def test_backoff_grows_and_the_last_error_is_raised(delays):
    errors = [_overloaded() for _ in range(3)]
    mock = AsyncMock(side_effect=errors)
    with patch("litellm.acompletion", new=mock):
        result = Agent(AgentConfig(model=MODEL)).run_with_response("hi")

    assert mock.call_count == 3
    assert result.error is errors[-1]
    # 0.5s then 1s, each with up to 0.75s of jitter.
    assert 0.5 <= delays[0] < 1.25
    assert 1 <= delays[1] < 1.75


def test_retry_after_header_sets_the_delay(delays):
    error = _rate_limit()
    # As LiteLLM's exception mapping attaches the provider's headers.
    error.litellm_response_headers = httpx.Headers({"retry-after": "7"})
    mock = AsyncMock(side_effect=[error, _response()])
    with patch("litellm.acompletion", new=mock):
        Agent(AgentConfig(model=MODEL)).run("hi")

    assert 7 <= delays[0] < 7.75


@pytest.mark.parametrize(
    "error",
    [
        litellm.AuthenticationError(
            message="bad key", llm_provider="openai", model=MODEL
        ),
        litellm.PermissionDeniedError(
            message="forbidden",
            llm_provider="openai",
            model=MODEL,
            response=httpx.Response(403, request=httpx.Request("POST", "http://x")),
        ),
        litellm.BadRequestError(message="bad", llm_provider="openai", model=MODEL),
        _rate_limit("You exceeded your current quota. 'code': 'insufficient_quota'"),
        litellm.BudgetExceededError(current_cost=2.0, max_budget=1.0),
    ],
)
def test_auth_spend_and_request_errors_are_not_retried(error, delays):
    mock = AsyncMock(side_effect=[error, _response()])
    with patch("litellm.acompletion", new=mock):
        result = Agent(AgentConfig(model=MODEL)).run_with_response("hi")

    assert result.error is error
    assert mock.call_count == 1
    assert delays == []


def test_num_retries_zero_disables_retries():
    error = _overloaded()
    mock = AsyncMock(side_effect=[error, _response()])
    with patch("litellm.acompletion", new=mock):
        result = Agent(AgentConfig(model=MODEL, num_retries=0)).run_with_response("hi")

    assert result.error is error
    assert mock.call_count == 1


def test_router_retries_by_its_own_policy():
    router = litellm.Router(
        model_list=[{"model_name": "smart", "litellm_params": {"model": MODEL}}]
    )
    error = _rate_limit()
    router.acompletion = AsyncMock(side_effect=[error, _response()])
    result = Agent(AgentConfig(model="smart", router=router)).run_with_response("hi")

    assert result.error is error
    assert router.acompletion.call_count == 1
    assert "timeout" not in router.acompletion.call_args.kwargs
    assert "max_retries" not in router.acompletion.call_args.kwargs


def test_each_attempt_fires_the_llm_hooks(delays):
    hooks = MagicMock(spec=AgentHooks)
    error = _overloaded()
    with patch("litellm.acompletion", new=AsyncMock(side_effect=[error, _response()])):
        Agent(AgentConfig(model=MODEL, hooks=hooks)).run("hi")

    assert hooks.on_llm_start.call_count == 2
    hooks.on_llm_error.assert_called_once_with(error)
    hooks.on_llm_end.assert_called_once()


async def test_stream_that_fails_to_open_is_retried(delays):
    mock = AsyncMock(side_effect=[_rate_limit(), _stream(_chunk("hello"))])
    with patch("litellm.acompletion", new=mock):
        assert await _streamed_text(Agent(AgentConfig(model=MODEL))) == "hello"

    assert mock.call_count == 2


async def test_stream_error_before_content_is_retried(delays):
    first = _stream(_chunk(), _overloaded())
    mock = AsyncMock(side_effect=[first, _stream(_chunk("hello"))])
    with patch("litellm.acompletion", new=mock):
        assert await _streamed_text(Agent(AgentConfig(model=MODEL))) == "hello"

    assert mock.call_count == 2


async def test_stream_error_after_content_is_raised(delays):
    error = _overloaded()
    mock = AsyncMock(side_effect=[_stream(_chunk("hel"), error), _stream(_chunk())])
    received = []
    with patch("litellm.acompletion", new=mock):
        with pytest.raises(litellm.InternalServerError):
            async for text in Agent(AgentConfig(model=MODEL)).astream("hi"):
                received.append(text)

    assert received == ["hel"]
    assert mock.call_count == 1
    assert delays == []


async def test_stalled_stream_is_retried_once():
    config = AgentConfig(model=MODEL, stream_idle_timeout=0.01)
    stalled = _stream(_chunk(), _chunk("never"), stall_before=1)
    mock = AsyncMock(side_effect=[stalled, _stream(_chunk("hello"))])
    with patch("litellm.acompletion", new=mock):
        assert await _streamed_text(Agent(config)) == "hello"

    assert mock.call_count == 2


async def test_stream_stalled_twice_raises_a_timeout():
    config = AgentConfig(model=MODEL, stream_idle_timeout=0.01)
    streams = [_stream(_chunk(), _chunk(), stall_before=1) for _ in range(3)]
    mock = AsyncMock(side_effect=streams)
    with patch("litellm.acompletion", new=mock):
        with pytest.raises(litellm.Timeout):
            await _streamed_text(Agent(config))

    assert mock.call_count == 2


async def test_stream_stalled_after_content_raises_a_timeout():
    config = AgentConfig(model=MODEL, stream_idle_timeout=0.01)
    stalled = _stream(_chunk("hel"), _chunk("lo"), stall_before=1)
    mock = AsyncMock(side_effect=[stalled, _stream(_chunk("hello"))])
    with patch("litellm.acompletion", new=mock):
        with pytest.raises(litellm.Timeout):
            await _streamed_text(Agent(config))

    assert mock.call_count == 1


async def test_idle_timeout_starts_after_the_first_chunk():
    async def slow_start():
        await asyncio.sleep(0.05)
        yield _chunk("hello")

    config = AgentConfig(model=MODEL, stream_idle_timeout=0.01)
    with patch("litellm.acompletion", new=AsyncMock(return_value=slow_start())):
        assert await _streamed_text(Agent(config)) == "hello"
