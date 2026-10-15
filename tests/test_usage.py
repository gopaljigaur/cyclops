"""Tests for usage accounting: cache and reasoning tokens, per-model usage, durations"""

import asyncio
from json import dumps
from unittest.mock import AsyncMock, patch

import httpx
import litellm
import pytest
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

from cyclops import Agent, AgentConfig, Session, tool
from cyclops.core.events import StepFinished

ANTHROPIC_USAGE = {
    "input_tokens": 100,
    "cache_creation_input_tokens": 1000,
    "cache_read_input_tokens": 500,
    "output_tokens": 20,
}


def _sse(*events):
    return "".join(f"event: {e['type']}\ndata: {dumps(e)}\n\n" for e in events).encode()


def _anthropic_reply(stream: bool) -> bytes:
    """An Anthropic answer of "hi", as a JSON body or an SSE stream."""
    message = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": ANTHROPIC_USAGE,
    }
    if not stream:
        return dumps(message).encode()
    usage = {**ANTHROPIC_USAGE, "output_tokens": 1}
    start = {**message, "content": [], "stop_reason": None, "usage": usage}
    return _sse(
        {"type": "message_start", "message": start},
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
            "usage": {"output_tokens": 20},
        },
        {"type": "message_stop"},
    )


@pytest.mark.parametrize("stream", [False, True], ids=["complete", "stream"])
async def test_anthropic_cache_writes_and_reads_are_counted(stream):
    """End to end through LiteLLM's Anthropic response and SSE parsers."""

    async def post(self, url, data=None, json=None, **kwargs):
        content = _anthropic_reply(stream)
        return httpx.Response(200, content=content, request=httpx.Request("POST", url))

    config = AgentConfig(
        model="anthropic/claude-sonnet-4-5", completion_kwargs={"api_key": "sk-test"}
    )
    with patch.object(AsyncHTTPHandler, "post", post):
        handle = Agent(config).start("hello", stream=stream)
        events = [e async for e in handle]

    (step,) = [e for e in events if isinstance(e, StepFinished)]
    usage = events[-1].result.usage
    assert usage == step.usage
    # LiteLLM counts cache reads and writes in the prompt tokens.
    assert usage.prompt_tokens == 1600
    assert usage.cache_write_tokens == 1000
    assert usage.cached_tokens == 500
    assert usage.completion_tokens == 20
    assert usage.reasoning_tokens == 0


async def test_openai_cached_and_reasoning_tokens_are_counted():
    reply = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1,
        "model": "o3-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 2000,
            "completion_tokens": 300,
            "total_tokens": 2300,
            "prompt_tokens_details": {"cached_tokens": 1500},
            "completion_tokens_details": {"reasoning_tokens": 200},
        },
    }

    async def send(self, request, **kwargs):
        return httpx.Response(200, json=reply, request=request)

    config = AgentConfig(model="o3-mini", completion_kwargs={"api_key": "sk-test"})
    with patch.object(httpx.AsyncClient, "send", send):
        result = await Agent(config).arun_with_response("hello")

    usage = result.usage
    assert (usage.prompt_tokens, usage.cached_tokens) == (2000, 1500)
    assert (usage.completion_tokens, usage.reasoning_tokens) == (300, 200)
    assert usage.cache_write_tokens == 0
    assert result.usage_by_model == {"o3-mini": usage}


def _response(model: str, finish_reason: str = "stop", **usage):
    return litellm.ModelResponse(
        model=model,
        choices=[
            {
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": finish_reason,
            }
        ],
        usage=litellm.Usage(**usage),
    )


async def test_a_fallback_run_splits_its_usage_by_model():
    refused = _response(
        "gpt-4o-mini",
        "content_filter",
        prompt_tokens=1000,
        completion_tokens=10,
        total_tokens=1010,
    )
    answer = _response(
        "gpt-4o", prompt_tokens=1000, completion_tokens=100, total_tokens=1100
    )
    config = AgentConfig(model="gpt-4o-mini", fallback_models=["gpt-4o"])
    with patch("litellm.acompletion", new=AsyncMock(side_effect=[refused, answer])):
        result = await Agent(config).arun_with_response("hi")

    by_model = result.usage_by_model
    assert list(by_model) == ["gpt-4o-mini", "gpt-4o"]
    assert by_model["gpt-4o-mini"].completion_tokens == 10
    assert by_model["gpt-4o"].completion_tokens == 100
    assert by_model["gpt-4o-mini"] + by_model["gpt-4o"] == result.usage


def _replies(*replies):
    """Patch litellm.acompletion to answer each call with the next mock reply."""
    real = litellm.acompletion
    queue = list(replies)

    async def complete(**kwargs):
        return await real(**kwargs, **queue.pop(0))

    return patch("litellm.acompletion", new=AsyncMock(side_effect=complete))


def _call(name: str, arguments: str):
    return {
        "mock_response": "",
        "mock_tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


async def test_a_subagent_on_another_model_counts_under_that_model():
    child = Agent(AgentConfig(model="gpt-4o"))
    lead = Agent(
        AgentConfig(model="gpt-4o-mini"),
        tools=[child.as_tool("researcher", "Researches a question")],
    )
    replies = [
        _call("researcher", '{"input": "find x"}'),
        {"mock_response": "5"},
        {"mock_response": "x is 5"},
    ]
    with _replies(*replies):
        result = await lead.arun_with_response("what is x?")

    assert result.content == "x is 5"
    by_model = result.usage_by_model
    # mock_response reports 30 tokens a completion: two of the lead's, one of the child's.
    assert by_model["gpt-4o-mini"].total_tokens == 60
    assert by_model["gpt-4o"].total_tokens == 30
    assert result.usage.total_tokens == 90


async def test_a_session_accumulates_every_runs_usage():
    session = Session()
    answer = _response(
        "claude-sonnet-4-5",
        prompt_tokens=1600,
        completion_tokens=20,
        total_tokens=1620,
        cache_creation_input_tokens=1000,
        cache_read_input_tokens=500,
    )
    agent = Agent(AgentConfig(model="anthropic/claude-sonnet-4-5"))
    durations = []
    with patch("litellm.acompletion", new=AsyncMock(return_value=answer)):
        for _ in range(2):
            await agent.arun("hi", session=session)
            durations.append(session.usage.api_duration)

    assert session.usage.cache_write_tokens == 2000
    assert session.usage.cached_tokens == 1000
    assert session.usage.prompt_tokens == 3200
    assert 0 < durations[0] < durations[1]


@tool
async def wait() -> str:
    """Wait a little."""
    await asyncio.sleep(0.05)
    return "done"


async def test_durations_are_measured():
    agent = Agent(AgentConfig(model="gpt-4o-mini"), tools=[wait])
    with _replies(_call("wait", "{}"), {"mock_response": "done"}):
        events = [e async for e in agent.start("wait", stream=False)]

    steps = [e.usage.api_duration for e in events if isinstance(e, StepFinished)]
    result = events[-1].result
    assert len(steps) == 2 and all(d > 0 for d in steps)
    assert result.usage.api_duration == pytest.approx(sum(steps))
    # The wall-clock time also covers the tool's wait, which no completion does.
    assert result.duration - result.usage.api_duration >= 0.04


async def test_a_streamed_completion_is_timed():
    agent = Agent(AgentConfig(model="gpt-4o-mini"))
    with _replies({"mock_response": "done"}):
        events = [e async for e in agent.start("hi", stream=True)]

    (step,) = [e for e in events if isinstance(e, StepFinished)]
    result = events[-1].result
    assert 0 < step.usage.api_duration == result.usage.api_duration <= result.duration
