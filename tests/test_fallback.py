"""Fallback models: when a step's completion goes to the next model."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode
from pydantic import BaseModel

from cyclops.core import completion
from cyclops.core.agent import Agent
from cyclops.core.events import ModelFallback, StepFinished
from cyclops.core.hooks import AgentHooks
from cyclops.core.types import AgentConfig
from cyclops.observability import TelemetryHooks
from cyclops.toolkit.tool import Tool
from cyclops.toolkit.types import ImageContent, TextContent, ToolResult

PRIMARY = "gpt-4o-mini"
FALLBACK = "gpt-4o"
USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}


def _response(model: str, text: str = "ok", finish_reason: str = "stop"):
    return litellm.ModelResponse(
        model=model,
        choices=[
            {
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        usage=USAGE,
    )


def _chunk(text: str = "", finish_reason=None):
    choice = StreamingChoices(delta=Delta(content=text), finish_reason=finish_reason)
    return ModelResponseStream(model=FALLBACK, choices=[choice])


async def _stream(*items):
    for item in items:
        if isinstance(item, BaseException):
            raise item
        yield item


def _tool_call(name: str):
    return litellm.ModelResponse(
        model=FALLBACK,
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": name, "arguments": "{}"},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    )


def _overloaded():
    return litellm.InternalServerError(
        message="Overloaded", llm_provider="anthropic", model=PRIMARY
    )


def _quota_spent():
    error = litellm.RateLimitError(
        message="insufficient_quota", llm_provider="openai", model=PRIMARY
    )
    error.body = {"code": "insufficient_quota"}
    return error


def _blocked():
    return litellm.ContentPolicyViolationError(
        message="blocked", llm_provider="openai", model=PRIMARY
    )


def _config(**kwargs) -> AgentConfig:
    kwargs.setdefault("num_retries", 1)
    return AgentConfig(model=PRIMARY, fallback_models=[FALLBACK], **kwargs)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(completion, "_calculate_retry_after", lambda **_: 0)


def _models(mock) -> list:
    return [call.kwargs["model"] for call in mock.call_args_list]


async def _events(agent: Agent) -> list:
    return [e async for e in agent.start("hi", stream=False)]


async def test_persistent_overload_switches_after_the_retries():
    error = _overloaded()
    mock = AsyncMock(side_effect=[error, error, _response(FALLBACK)])
    with patch("litellm.acompletion", new=mock):
        result = await Agent(_config()).arun_with_response("hi")

    assert result.content == "ok"
    assert _models(mock) == [PRIMARY, PRIMARY, FALLBACK]


@pytest.mark.parametrize("error", [_quota_spent(), _blocked()])
async def test_spent_quota_and_content_policy_switch_at_once(error):
    mock = AsyncMock(side_effect=[error, _response(FALLBACK)])
    with patch("litellm.acompletion", new=mock):
        result = await Agent(_config()).arun_with_response("hi")

    assert result.content == "ok"
    assert _models(mock) == [PRIMARY, FALLBACK]


async def test_refusal_switches_and_its_usage_is_counted():
    refused = _response(PRIMARY, "", "content_filter")
    mock = AsyncMock(side_effect=[refused, _response(FALLBACK)])
    hooks = MagicMock(spec=AgentHooks)
    with patch("litellm.acompletion", new=mock):
        result = await Agent(_config(hooks=hooks)).arun_with_response("hi")

    assert result.content == "ok"
    assert result.stop_reason == "completed"
    assert result.usage.total_tokens == 2 * USAGE["total_tokens"]
    assert [c.args[0].model for c in hooks.on_llm_end.call_args_list] == [
        PRIMARY,
        FALLBACK,
    ]


async def test_switch_emits_model_fallback():
    refused = _response(PRIMARY, "", "content_filter")
    error = _blocked()
    answer = _response(FALLBACK)
    mock = AsyncMock(side_effect=[error, answer, refused, answer])
    with patch("litellm.acompletion", new=mock):
        first = await _events(Agent(_config()))
        second = await _events(Agent(_config()))

    (failed,) = [e for e in first if isinstance(e, ModelFallback)]
    assert (failed.model, failed.fallback_model) == (PRIMARY, FALLBACK)
    assert failed.error is error and failed.usage is None
    (refusal,) = [e for e in second if isinstance(e, ModelFallback)]
    assert refusal.error is None
    assert refusal.usage.total_tokens == USAGE["total_tokens"]


async def test_cost_and_step_are_the_answering_models():
    mock = AsyncMock(side_effect=[_quota_spent(), _response(FALLBACK)])
    with patch("litellm.acompletion", new=mock):
        events = await _events(Agent(_config()))

    (step,) = [e for e in events if isinstance(e, StepFinished)]
    assert step.model == FALLBACK
    expected = litellm.completion_cost(completion_response=_response(FALLBACK))
    assert step.usage.cost == pytest.approx(expected)
    assert events[-1].result.cost == pytest.approx(expected)


async def test_every_step_starts_with_the_model():
    tool = Tool("work", "Does work", lambda: "done")
    call = _tool_call("work")
    mock = AsyncMock(side_effect=[_quota_spent(), call, _response(PRIMARY)])
    with patch("litellm.acompletion", new=mock):
        events = await _events(Agent(_config(), tools=[tool]))

    assert _models(mock) == [PRIMARY, FALLBACK, PRIMARY]
    steps = [e.model for e in events if isinstance(e, StepFinished)]
    assert steps == [FALLBACK, PRIMARY]


async def test_each_model_gets_the_options_it_supports():
    config = AgentConfig(
        model="anthropic/claude-sonnet-4-5",
        fallback_models=[PRIMARY],
        reasoning_effort="low",
        num_retries=0,
    )
    mock = AsyncMock(side_effect=[_overloaded(), _response(PRIMARY)])
    with patch("litellm.acompletion", new=mock):
        await Agent(config).arun("hi")

    first, second = (call.kwargs for call in mock.call_args_list)
    assert first["reasoning_effort"] == "low"
    assert "cache_control_injection_points" in first
    assert "reasoning_effort" not in second
    assert "cache_control_injection_points" not in second


@pytest.mark.parametrize(
    "error",
    [
        litellm.AuthenticationError(message="no", llm_provider="openai", model=PRIMARY),
        litellm.BadRequestError(message="bad", llm_provider="openai", model=PRIMARY),
    ],
)
async def test_other_errors_do_not_switch(error):
    mock = AsyncMock(side_effect=[error, _response(FALLBACK)])
    with patch("litellm.acompletion", new=mock):
        with pytest.raises(type(error)):
            await Agent(_config()).arun("hi")

    assert _models(mock) == [PRIMARY]


async def test_last_models_error_and_refusal_stand():
    error = _quota_spent()
    config = AgentConfig(model=PRIMARY, fallback_models=[FALLBACK, "gpt-4.1"])
    mock = AsyncMock(side_effect=[error, error, error])
    with patch("litellm.acompletion", new=mock):
        with pytest.raises(litellm.RateLimitError):
            await Agent(config).arun("hi")
    assert _models(mock) == [PRIMARY, FALLBACK, "gpt-4.1"]

    refused = _response(FALLBACK, "", "content_filter")
    with patch("litellm.acompletion", new=AsyncMock(side_effect=[error, refused])):
        result = await Agent(_config()).arun_with_response("hi")
    assert result.stop_reason == "content_filter"


async def test_without_fallbacks_the_error_is_raised():
    error = _quota_spent()
    mock = AsyncMock(side_effect=[error, _response(FALLBACK)])
    with patch("litellm.acompletion", new=mock):
        events = await _events(Agent(AgentConfig(model=PRIMARY)))

    assert events[-1].result.error is error
    assert not any(isinstance(e, ModelFallback) for e in events)
    assert _models(mock) == [PRIMARY]


async def test_stream_that_fails_before_output_switches():
    first = _stream(_chunk(), _overloaded())
    mock = AsyncMock(side_effect=[first, _stream(_chunk("hello"))])
    with patch("litellm.acompletion", new=mock):
        agent = Agent(_config(num_retries=0))
        text = "".join([t async for t in agent.astream("hi")])

    assert text == "hello"
    assert _models(mock) == [PRIMARY, FALLBACK]


async def test_stream_refused_before_output_switches():
    refused = _stream(_chunk(finish_reason="content_filter"))
    mock = AsyncMock(side_effect=[refused, _stream(_chunk("hello"))])
    with patch("litellm.acompletion", new=mock):
        text = "".join([t async for t in Agent(_config()).astream("hi")])

    assert text == "hello"
    assert _models(mock) == [PRIMARY, FALLBACK]


async def test_stream_error_or_refusal_after_output_stands():
    cut = _stream(_chunk("hel"), _overloaded())
    mock = AsyncMock(side_effect=[cut, _stream(_chunk("hello"))])
    with patch("litellm.acompletion", new=mock):
        with pytest.raises(litellm.InternalServerError):
            async for _ in Agent(_config()).astream("hi"):
                pass
    assert _models(mock) == [PRIMARY]

    refused = _stream(_chunk("I can"), _chunk(finish_reason="content_filter"))
    mock = AsyncMock(side_effect=[refused, _stream(_chunk("hello"))])
    with patch("litellm.acompletion", new=mock):
        events = [e async for e in Agent(_config()).astream_events("hi")]
    assert events[-1].result.stop_reason == "content_filter"
    assert _models(mock) == [PRIMARY]


async def test_telemetry_spans_name_the_answering_model():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    hooks = TelemetryHooks(tracer_provider=provider)
    mock = AsyncMock(side_effect=[_quota_spent(), _response(FALLBACK)])
    with patch("litellm.acompletion", new=mock):
        await Agent(_config(hooks=hooks)).arun("hi")

    failed, answered = [s for s in exporter.get_finished_spans() if s.name == "chat"]
    assert failed.status.status_code == StatusCode.ERROR
    assert answered.attributes["gen_ai.response.model"] == FALLBACK


def test_fallback_models_and_a_router_are_not_both_set():
    with pytest.raises(ValueError, match="router"):
        AgentConfig(model=PRIMARY, fallback_models=[FALLBACK], router=MagicMock())


async def test_tool_images_stay_in_the_tool_message_only_if_every_model_takes_them():
    photo = Tool(
        "photo",
        "An image",
        lambda: ToolResult(
            content=[
                TextContent(text="a pixel"),
                ImageContent(data="AAAA", mime_type="image/png"),
            ]
        ),
    )
    sent = []
    for fallbacks in ([], [FALLBACK]):
        config = AgentConfig(
            model="anthropic/claude-sonnet-4-5", fallback_models=fallbacks
        )
        mock = AsyncMock(side_effect=[_tool_call("photo"), _response(PRIMARY)])
        with patch("litellm.acompletion", new=mock):
            await Agent(config, tools=[photo]).arun("hi")
        messages = mock.call_args_list[1].kwargs["messages"]
        (tool_message,) = [m for m in messages if m["role"] == "tool"]
        sent.append(tool_message["content"])

    alone, chained = sent
    assert isinstance(alone, list)
    assert chained == "a pixel"


async def test_refusal_past_the_budget_stops_the_run():
    refused = _response(PRIMARY, "", "content_filter")
    mock = AsyncMock(side_effect=[refused, _response(FALLBACK)])
    with patch("litellm.acompletion", new=mock):
        events = await _events(Agent(_config(max_total_tokens=1000)))

    result = events[-1].result
    assert result.stop_reason == "budget"
    assert result.usage.total_tokens == USAGE["total_tokens"]
    assert _models(mock) == [PRIMARY]
    assert not any(isinstance(e, ModelFallback) for e in events)


async def test_cancel_during_a_fallback_attempt():
    started = asyncio.Event()

    async def complete(**kwargs):
        if kwargs["model"] == PRIMARY:
            raise _quota_spent()
        started.set()
        await asyncio.Event().wait()

    with patch("litellm.acompletion", new=AsyncMock(side_effect=complete)):
        handle = Agent(_config()).start("hi", stream=False)
        await started.wait()
        handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"


async def test_each_model_gets_the_response_format_it_supports():
    class Answer(BaseModel):
        text: str

    config = AgentConfig(model=PRIMARY, fallback_models=["gpt-4"], num_retries=0)
    mock = AsyncMock(side_effect=[_quota_spent(), _response("gpt-4", '{"text": "ok"}')])
    with patch("litellm.acompletion", new=mock):
        await Agent(config).arun("hi", response_model=Answer)

    first, second = (call.kwargs for call in mock.call_args_list)
    assert first["response_format"] is Answer
    assert "response_format" not in second
