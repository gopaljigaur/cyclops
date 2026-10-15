"""Tests for completion kwargs, reasoning history, prompt caching and response schemas"""

import sys
from base64 import b64decode
from json import dumps, loads
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Choices,
    Delta,
    Function,
    Message,
    ModelResponse,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)
import pytest
from pydantic import BaseModel

from cyclops.core.agent import Agent
from cyclops.core.providers import (
    _caches_at_markers,
    _every_deployment,
    _holds,
    _lacks_tools,
    _native_schema,
    _response_schema,
    _takes_param,
    _takes_tool_message_images,
)
from cyclops.core.session import Session
from cyclops.core.types import AgentConfig, Message as CyclopsMessage
from cyclops.toolkit import tool


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class Answer(BaseModel):
    value: int


THINKING = [{"type": "thinking", "thinking": "Add them.", "signature": "sig-1"}]


def _response(content=None, tool_calls=None, **message_fields):
    return ModelResponse(
        choices=[
            Choices(
                message=Message(
                    content=content, tool_calls=tool_calls, **message_fields
                )
            )
        ],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _add_call():
    return ChatCompletionMessageToolCall(
        id="call_1", function=Function(name="add", arguments='{"a": 1, "b": 2}')
    )


def _tool_round(**message_fields):
    return [
        _response(tool_calls=[_add_call()], **message_fields),
        _response("3"),
    ]


# ---------------------------------------------------------------------------
# completion_kwargs
# ---------------------------------------------------------------------------


def test_completion_kwargs_go_to_every_call():
    config = AgentConfig(
        model="gpt-4o-mini",
        completion_kwargs={
            "api_key": "sk-test",
            "api_base": "https://llm.example.com/v1",
            "extra_headers": {"X-Trace": "1"},
            "timeout": 30,
            "max_tokens": 512,
        },
    )
    agent = Agent(config, tools=[add])
    mock = AsyncMock(side_effect=_tool_round())
    with patch("litellm.acompletion", new=mock):
        agent.run("1 + 2?")

    assert mock.call_count == 2
    for call in mock.call_args_list:
        kwargs = dict(call.kwargs)
        del kwargs["messages"], kwargs["tools"]
        assert kwargs == {
            "model": "gpt-4o-mini",
            "max_tokens": 512,
            "api_key": "sk-test",
            "api_base": "https://llm.example.com/v1",
            "extra_headers": {"X-Trace": "1"},
            "timeout": 30,
            "max_retries": 0,
        }


def test_completion_kwargs_go_through_a_router():
    router = litellm.Router(
        model_list=[{"model_name": "smart", "litellm_params": {"model": "gpt-4o"}}]
    )
    router.acompletion = AsyncMock(return_value=_response("hi"))
    config = AgentConfig(
        model="smart",
        router=router,
        completion_kwargs={"reasoning_effort": "low", "timeout": 10},
    )
    Agent(config).run("hello")

    router.acompletion.assert_called_once_with(
        model="smart",
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort="low",
        timeout=10,
    )


# ---------------------------------------------------------------------------
# Reasoning and thinking blocks
# ---------------------------------------------------------------------------


def test_thinking_blocks_are_sent_back_with_the_tool_result():
    config = AgentConfig(
        model="anthropic/claude-sonnet-4-5",
        completion_kwargs={"thinking": {"type": "enabled", "budget_tokens": 1024}},
    )
    agent = Agent(config, tools=[add])
    mock = AsyncMock(
        side_effect=_tool_round(reasoning_content="Add them.", thinking_blocks=THINKING)
    )
    with patch("litellm.acompletion", new=mock):
        agent.run("1 + 2?")

    assert mock.call_args_list[1].kwargs["messages"] == [
        {"role": "user", "content": "1 + 2?"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "Add them.",
            "thinking_blocks": THINKING,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "3", "name": "add"},
    ]


def test_final_answer_keeps_its_reasoning():
    agent = Agent(AgentConfig(model="deepseek/deepseek-reasoner"))
    session = Session()
    response = _response("Paris", reasoning_content="The capital is Paris.")
    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        agent.run("Capital of France?", session=session)

    assert session.messages[-1] == {
        "role": "assistant",
        "content": "Paris",
        "reasoning_content": "The capital is Paris.",
    }


def test_streamed_thinking_blocks_keep_their_signature():
    chunks = [
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(
                        reasoning_content="Add them.",
                        thinking_blocks=[{"type": "thinking", "thinking": "Add them."}],
                    )
                )
            ]
        ),
        ModelResponseStream(
            choices=[
                StreamingChoices(
                    delta=Delta(
                        thinking_blocks=[
                            {"type": "thinking", "thinking": "", "signature": "sig-1"}
                        ]
                    )
                )
            ]
        ),
        ModelResponseStream(choices=[StreamingChoices(delta=Delta(content="3"))]),
    ]

    async def stream():
        for chunk in chunks:
            yield chunk

    agent = Agent(AgentConfig(model="anthropic/claude-sonnet-4-5"))
    session = Session()
    with patch("litellm.acompletion", new=AsyncMock(return_value=stream())):
        assert "".join(agent.stream("1 + 2?", session=session)) == "3"

    assert session.messages[-1]["thinking_blocks"] == THINKING


def _sse(*events):
    return "".join(f"event: {e['type']}\ndata: {dumps(e)}\n\n" for e in events).encode()


def _anthropic_stream(*blocks, stop_reason):
    """An Anthropic SSE stream of the given content blocks, each a list of deltas."""
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-5",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        }
    ]
    for index, (start, deltas) in enumerate(blocks):
        events.append(
            {"type": "content_block_start", "index": index, "content_block": start}
        )
        events += [
            {"type": "content_block_delta", "index": index, "delta": d} for d in deltas
        ]
        events.append({"type": "content_block_stop", "index": index})
    events += [
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]
    return _sse(*events)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="LiteLLM repeats streamed thinking at its signature (#108)",
)
def test_streamed_anthropic_tool_loop_sends_back_the_signed_thinking():
    """End to end through LiteLLM's Anthropic SSE parser and request builder."""
    streams = [
        _anthropic_stream(
            (
                {"type": "thinking", "thinking": ""},
                [
                    {"type": "thinking_delta", "thinking": "Add "},
                    {"type": "thinking_delta", "thinking": "them."},
                    {"type": "signature_delta", "signature": "sig-1"},
                ],
            ),
            (
                {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {}},
                [{"type": "input_json_delta", "partial_json": '{"a": 1, "b": 2}'}],
            ),
            stop_reason="tool_use",
        ),
        _anthropic_stream(
            (
                {"type": "text", "text": ""},
                [{"type": "text_delta", "text": "3"}],
            ),
            stop_reason="end_turn",
        ),
    ]
    bodies = []

    async def post(self, url, data=None, json=None, **kwargs):
        bodies.append(loads(data) if data else json)
        return httpx.Response(
            200,
            content=streams[len(bodies) - 1],
            request=httpx.Request("POST", url),
        )

    config = AgentConfig(
        model="anthropic/claude-sonnet-4-5",
        thinking_budget=1024,
        completion_kwargs={"api_key": "sk-test"},
    )
    with patch.object(AsyncHTTPHandler, "post", post):
        assert "".join(Agent(config, tools=[add]).stream("1 + 2?")) == "3"

    assert bodies[1]["messages"][1]["content"][0] == THINKING[0]


def test_message_round_trips_reasoning():
    d = {
        "role": "assistant",
        "content": "3",
        "reasoning_content": "Add them.",
        "thinking_blocks": THINKING,
    }
    assert CyclopsMessage.from_dict(d).to_dict() == d


def test_anthropic_request_puts_signed_thinking_before_the_tool_use():
    """End to end through LiteLLM's Anthropic request builder, with thinking_budget."""
    replies = [
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "stop_reason": "tool_use",
            "content": [
                THINKING[0],
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "add",
                    "input": {"a": 1, "b": 2},
                },
            ],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
        {
            "id": "msg_2",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": "3"}],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    ]
    bodies = []

    async def post(self, url, data=None, json=None, **kwargs):
        bodies.append(loads(data) if data else json)
        return httpx.Response(
            200, json=replies[len(bodies) - 1], request=httpx.Request("POST", url)
        )

    config = AgentConfig(
        model="anthropic/claude-sonnet-4-5",
        system_prompt="Be brief.",
        prompt_caching=True,
        thinking_budget=1024,
        completion_kwargs={"api_key": "sk-test", "max_tokens": 4096},
    )
    with patch.object(AsyncHTTPHandler, "post", post):
        assert Agent(config, tools=[add]).run("1 + 2?") == "3"

    second = bodies[1]
    assert second["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    assert second["system"] == [
        {"type": "text", "text": "Be brief.", "cache_control": {"type": "ephemeral"}}
    ]
    assert second["messages"][1] == {
        "role": "assistant",
        "content": [
            THINKING[0],
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "add",
                "input": {"a": 1, "b": 2},
            },
        ],
    }
    assert second["messages"][2]["content"][0]["cache_control"] == {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Reasoning options
# ---------------------------------------------------------------------------

# A one pixel PNG.
PNG = b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

REPLIES = {
    "api.openai.com": {
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
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "api.anthropic.com": {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": "hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
    "generativelanguage.googleapis.com": {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "hi"}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2,
        },
    },
    "bedrock-runtime.us-east-1.amazonaws.com": {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
    },
}


def _request_body(monkeypatch, model, message="hello", **config):
    """The body LiteLLM builds for one completion, answered without a network.

    An image URL in the message is served a one pixel PNG.
    """
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    bodies = []

    async def send(self, request, **kwargs):
        if request.url.host == "example.com":
            headers = {"content-type": "image/png"}
            return httpx.Response(200, content=PNG, headers=headers, request=request)
        bodies.append(loads(request.content))
        return httpx.Response(200, json=REPLIES[request.url.host], request=request)

    config = AgentConfig(model=model, completion_kwargs={"api_key": "test"}, **config)
    with patch.object(httpx.AsyncClient, "send", send):
        assert Agent(config).run(message) == "hi"
    (body,) = bodies
    return body


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", ["gpt-4o-mini", "o3-mini", "gpt-5"])
def test_request_has_no_temperature_by_default(monkeypatch, model):
    body = _request_body(monkeypatch, model)

    assert "temperature" not in body


def test_request_takes_a_set_temperature(monkeypatch):
    body = _request_body(monkeypatch, "gpt-4o-mini", temperature=0.3)

    assert body["temperature"] == 0.3


def test_router_request_keeps_the_deployment_sampling_params(monkeypatch):
    bodies = []

    async def send(self, request, **kwargs):
        bodies.append(loads(request.content))
        return httpx.Response(200, json=REPLIES[request.url.host], request=request)

    router = litellm.Router(
        model_list=[
            {
                "model_name": "smart",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                    "api_key": "test",
                    "temperature": 0.7,
                    "max_tokens": 77,
                },
            }
        ]
    )
    with patch.object(httpx.AsyncClient, "send", send):
        assert Agent(AgentConfig(model="smart", router=router)).run("hello") == "hi"

    (body,) = bodies
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 77


def test_anthropic_thinking_request_has_no_temperature_by_default(monkeypatch):
    body = _request_body(
        monkeypatch, "anthropic/claude-sonnet-4-5", thinking_budget=2048
    )

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert "temperature" not in body


def test_openai_request_takes_the_reasoning_effort(monkeypatch):
    body = _request_body(monkeypatch, "o3-mini", reasoning_effort="high")

    assert body["reasoning_effort"] == "high"


def test_gpt_5_request_takes_the_reasoning_effort(monkeypatch):
    body = _request_body(monkeypatch, "gpt-5", reasoning_effort="high")

    assert body["reasoning_effort"] == "high"


def test_anthropic_request_turns_the_reasoning_effort_into_a_budget(monkeypatch):
    body = _request_body(
        monkeypatch, "anthropic/claude-sonnet-4-5", reasoning_effort="low"
    )

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}


def test_anthropic_request_takes_the_thinking_budget(monkeypatch):
    body = _request_body(
        monkeypatch, "anthropic/claude-sonnet-4-5", thinking_budget=2048
    )

    assert body["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    # LiteLLM leaves room for the answer after the budget.
    assert body["max_tokens"] > 2048


def test_anthropic_request_sends_effort_to_adaptive_thinking_models(monkeypatch):
    body = _request_body(
        monkeypatch, "anthropic/claude-opus-4-6", reasoning_effort="high"
    )

    assert body["thinking"]["type"] == "adaptive"
    assert body["output_config"] == {"effort": "high"}


def test_anthropic_request_sends_adaptive_thinking_on_opus_4_7(monkeypatch):
    body = _request_body(
        monkeypatch, "anthropic/claude-opus-4-7", reasoning_effort="high"
    )

    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}


def test_anthropic_request_turns_a_budget_into_an_effort_on_opus_4_7(monkeypatch):
    body = _request_body(monkeypatch, "anthropic/claude-opus-4-7", thinking_budget=2048)

    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "medium"}


def test_gemini_request_turns_the_reasoning_effort_into_a_budget(monkeypatch):
    body = _request_body(monkeypatch, "gemini/gemini-2.5-flash", reasoning_effort="low")

    assert body["generationConfig"]["thinkingConfig"] == {
        "thinkingBudget": 1024,
        "includeThoughts": True,
    }


def test_gemini_request_takes_the_thinking_budget(monkeypatch):
    body = _request_body(monkeypatch, "gemini/gemini-2.5-flash", thinking_budget=2048)

    assert body["generationConfig"]["thinkingConfig"] == {
        "thinkingBudget": 2048,
        "includeThoughts": True,
    }


def test_gemini_3_request_takes_the_reasoning_effort_as_a_level(monkeypatch):
    body = _request_body(
        monkeypatch, "gemini/gemini-3-pro-preview", reasoning_effort="high"
    )

    assert body["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "high",
        "includeThoughts": True,
    }


def test_gemini_3_request_drops_the_thinking_budget(monkeypatch):
    body = _request_body(
        monkeypatch, "gemini/gemini-3-pro-preview", thinking_budget=2048
    )

    assert body["generationConfig"]["thinkingConfig"] == {"includeThoughts": True}


@pytest.mark.parametrize(
    "option, budget",
    [({"reasoning_effort": "medium"}, 2048), ({"thinking_budget": 4096}, 4096)],
)
def test_bedrock_request_takes_reasoning_options(monkeypatch, option, budget):
    body = _request_body(
        monkeypatch, "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0", **option
    )

    assert body["additionalModelRequestFields"]["thinking"] == {
        "type": "enabled",
        "budget_tokens": budget,
    }


def _reasoning_kwargs(model, **config):
    mock = AsyncMock(return_value=_response("hi"))
    with patch("litellm.acompletion", new=mock):
        Agent(AgentConfig(model=model, **config)).run("hello")
    kwargs = mock.call_args.kwargs
    return {k: kwargs[k] for k in ("reasoning_effort", "thinking") if k in kwargs}


def test_reasoning_options_send_nothing_by_default():
    assert _reasoning_kwargs("anthropic/claude-sonnet-4-5") == {}


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4o-mini",
        "gemini/gemini-2.0-flash",
        "anthropic/claude-3-5-haiku-20241022",
        "bedrock/meta.llama3-1-70b-instruct-v1:0",
    ],
)
@pytest.mark.parametrize(
    "option", [{"reasoning_effort": "high"}, {"thinking_budget": 2048}]
)
def test_reasoning_options_are_not_sent_to_models_without_reasoning(model, option):
    kwargs = _reasoning_kwargs(model, **option)

    assert kwargs == {}


def test_reasoning_options_need_every_router_deployment():
    router = litellm.Router(
        model_list=[
            {"model_name": "smart", "litellm_params": {"model": "gpt-5"}},
            {"model_name": "smart", "litellm_params": {"model": "gpt-4o"}},
        ]
    )
    router.acompletion = AsyncMock(return_value=_response("hi"))
    Agent(AgentConfig(model="smart", router=router, reasoning_effort="low")).run("hi")

    assert "reasoning_effort" not in router.acompletion.call_args.kwargs


@pytest.mark.parametrize(
    "completion_kwargs",
    [
        {"reasoning_effort": "high"},
        {"thinking": {"type": "enabled", "budget_tokens": 4096}},
    ],
)
def test_reasoning_options_give_way_to_completion_kwargs(completion_kwargs):
    kwargs = _reasoning_kwargs(
        "anthropic/claude-sonnet-4-5",
        reasoning_effort="low",
        completion_kwargs=completion_kwargs,
    )

    assert kwargs == completion_kwargs


def test_reasoning_options_are_validated():
    AgentConfig(model="gpt-5", reasoning_effort="high")
    AgentConfig(model="gpt-5", thinking_budget=1)
    with pytest.raises(ValueError):
        AgentConfig(model="gpt-5", reasoning_effort="extreme")
    with pytest.raises(ValueError):
        AgentConfig(model="gpt-5", thinking_budget=0)


def test_reasoning_options_are_one_or_the_other():
    with pytest.raises(ValueError, match="not both"):
        AgentConfig(model="gpt-5", reasoning_effort="high", thinking_budget=2048)


# ---------------------------------------------------------------------------
# Prompt caching
# ---------------------------------------------------------------------------

MARKER_MODELS = [
    "anthropic/claude-sonnet-4-5",
    "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "vertex_ai/claude-sonnet-4-5",
]


def _cached_kwargs(model, **config):
    mock = AsyncMock(return_value=_response("hi"))
    with patch("litellm.acompletion", new=mock):
        Agent(AgentConfig(model=model, **config), tools=[add]).run("hello")
    return mock.call_args.kwargs


@pytest.mark.parametrize("model", MARKER_MODELS)
def test_prompt_caching_is_on_by_default(model):
    kwargs = _cached_kwargs(model)

    control = {"type": "ephemeral"}
    # The system marker covers the tools before it.
    assert "cache_control" not in kwargs["tools"][-1]
    assert kwargs["cache_control_injection_points"] == [
        {"location": "message", "role": "system", "control": control},
        {"location": "message", "index": -1, "control": control},
    ]


def test_prompt_caching_sets_the_time_to_live():
    kwargs = _cached_kwargs("anthropic/claude-sonnet-4-5", prompt_caching="1h")

    control = {"type": "ephemeral", "ttl": "1h"}
    assert all(
        p["control"] == control for p in kwargs["cache_control_injection_points"]
    )


def test_prompt_caching_can_be_turned_off():
    kwargs = _cached_kwargs("anthropic/claude-sonnet-4-5", prompt_caching=False)

    assert "cache_control_injection_points" not in kwargs
    assert "cache_control" not in kwargs["tools"][-1]


def test_prompt_caching_rejects_an_unknown_time_to_live():
    with pytest.raises(ValueError):
        AgentConfig(model="anthropic/claude-sonnet-4-5", prompt_caching="10m")


@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-2.5-flash",
        "vertex_ai/gemini-2.5-flash",
        "gpt-4o-mini",
        "bedrock/meta.llama3-1-70b-instruct-v1:0",
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    ],
)
def test_prompt_caching_sends_no_markers_where_they_do_not_cache(model):
    """Gemini would create a billed cache resource, OpenAI caches without markers
    and LiteLLM lists these Bedrock models without prompt caching."""
    kwargs = _cached_kwargs(model)

    assert "cache_control_injection_points" not in kwargs
    assert "cache_control" not in kwargs["tools"][-1]


def test_prompt_caching_needs_every_router_deployment():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "smart",
                "litellm_params": {"model": "anthropic/claude-sonnet-4-5"},
            },
            {
                "model_name": "smart",
                "litellm_params": {"model": "gemini/gemini-2.5-flash"},
            },
        ]
    )
    router.acompletion = AsyncMock(return_value=_response("hi"))
    Agent(AgentConfig(model="smart", router=router)).run("hello")

    assert "cache_control_injection_points" not in router.acompletion.call_args.kwargs


def _cache_markers(body):
    return dumps(body).count('"cache_control"') + dumps(body).count('"cachePoint"')


def test_anthropic_request_marks_system_prompt_and_latest_message():
    """End to end through LiteLLM's Anthropic request builder, over two steps."""
    replies = [
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "add",
                    "input": {"a": 1, "b": 2},
                }
            ],
            "stop_reason": "tool_use",
        },
        {"content": [{"type": "text", "text": "3"}], "stop_reason": "end_turn"},
    ]
    bodies = []

    async def post(self, url, data=None, json=None, **kwargs):
        bodies.append(loads(data) if data else json)
        reply = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            **replies[len(bodies) - 1],
        }
        return httpx.Response(200, json=reply, request=httpx.Request("POST", url))

    config = AgentConfig(
        model="anthropic/claude-sonnet-4-5",
        system_prompt="Be brief.",
        prompt_caching="1h",
        completion_kwargs={"api_key": "sk-test"},
    )
    agent = Agent(config, tools=[add])
    session = Session()
    with patch.object(AsyncHTTPHandler, "post", post):
        assert agent.run("1 + 2?", session=session) == "3"

    control = {"type": "ephemeral", "ttl": "1h"}
    for body in bodies:
        assert "cache_control" not in body["tools"][-1]
        assert body["system"][-1]["cache_control"] == control
        assert body["messages"][-1]["content"][-1]["cache_control"] == control
        # The tail marker moves on; earlier ones are not kept in the history.
        assert _cache_markers(body) == 2
    assert bodies[1]["messages"][-1]["content"][-1]["type"] == "tool_result"
    assert "cache_control" not in dumps(session.messages)


def test_bedrock_request_marks_system_prompt_and_latest_message(monkeypatch):
    """End to end through LiteLLM's Bedrock Converse request builder."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    bodies = []

    async def post(self, url, data=None, json=None, **kwargs):
        bodies.append(loads(data) if data else json)
        reply = {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }
        return httpx.Response(200, json=reply, request=httpx.Request("POST", url))

    config = AgentConfig(
        model="bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        system_prompt="Be brief.",
        prompt_caching="1h",
    )
    with patch.object(AsyncHTTPHandler, "post", post):
        assert Agent(config, tools=[add]).run("hello") == "hi"

    (body,) = bodies
    point = {"cachePoint": {"type": "default", "ttl": "1h"}}
    assert "cachePoint" not in body["toolConfig"]["tools"][-1]
    assert body["system"][-1] == point
    assert body["messages"][-1]["content"][-1] == point
    assert _cache_markers(body) == 2


def test_prompt_caching_keeps_the_callers_injection_points():
    points = [{"location": "message", "index": 0}]
    kwargs = _cached_kwargs(
        "anthropic/claude-sonnet-4-5",
        completion_kwargs={"cache_control_injection_points": points},
    )

    assert kwargs["cache_control_injection_points"] == points


def test_anthropic_request_keeps_four_caller_markers_at_four():
    """Anthropic rejects a fifth breakpoint; LiteLLM skips ours past the limit."""
    bodies = []

    async def post(self, url, data=None, json=None, **kwargs):
        bodies.append(loads(data) if data else json)
        reply = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
        }
        return httpx.Response(200, json=reply, request=httpx.Request("POST", url))

    marked = [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]
    session = Session(
        messages=[
            {"role": role, "content": marked}
            for role in ("user", "assistant", "user", "assistant")
        ]
    )
    config = AgentConfig(
        model="anthropic/claude-sonnet-4-5",
        system_prompt="Be brief.",
        completion_kwargs={"api_key": "sk-test"},
    )
    with patch.object(AsyncHTTPHandler, "post", post):
        assert Agent(config, tools=[add]).run("hello", session=session) == "hi"

    (body,) = bodies
    assert _cache_markers(body) == 4


# ---------------------------------------------------------------------------
# response_model schemas
# ---------------------------------------------------------------------------


def test_response_model_sends_a_schema_when_supported():
    mock = AsyncMock(return_value=_response('{"value": 3}'))
    with patch("litellm.acompletion", new=mock):
        result = Agent(AgentConfig(model="gpt-4o-mini")).run(
            "1 + 2?", response_model=Answer
        )

    assert result == Answer(value=3)
    assert mock.call_args.kwargs["response_format"] is Answer


def test_response_model_only_validates_when_unsupported():
    mock = AsyncMock(return_value=_response('{"value": 3}'))
    with patch("litellm.acompletion", new=mock):
        result = Agent(AgentConfig(model="gpt-3.5-turbo")).run(
            "1 + 2?", response_model=Answer
        )

    assert result == Answer(value=3)
    assert "response_format" not in mock.call_args.kwargs


def _anthropic_bodies(model, **completion_kwargs):
    """Run an Anthropic agent with a tool and return the request bodies LiteLLM sends."""
    replies = [
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "add",
                    "input": {"a": 1, "b": 2},
                }
            ],
            "stop_reason": "tool_use",
        },
        {
            "content": [{"type": "text", "text": '{"value": 3}'}],
            "stop_reason": "end_turn",
        },
    ]
    bodies = []

    async def post(self, url, data=None, json=None, **kwargs):
        bodies.append(loads(data) if data else json)
        reply = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "usage": {"input_tokens": 1, "output_tokens": 1},
            **replies[len(bodies) - 1],
        }
        return httpx.Response(200, json=reply, request=httpx.Request("POST", url))

    config = AgentConfig(model=model, completion_kwargs=completion_kwargs)
    with patch.object(AsyncHTTPHandler, "post", post):
        result = Agent(config, tools=[add]).run("1 + 2?", response_model=Answer)

    assert result == Answer(value=3)
    return bodies


def test_response_model_sends_a_native_schema_alongside_tools():
    bodies = _anthropic_bodies("anthropic/claude-sonnet-4-5", api_key="sk-test")

    for body in bodies:
        assert [t["name"] for t in body["tools"]] == ["add"]
        assert "tool_choice" not in body
        assert body["output_format"] == {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"value": {"type": "integer", "title": "Value"}},
                "required": ["value"],
                "title": "Answer",
                "additionalProperties": False,
            },
        }


def test_response_model_sends_no_emulated_schema_alongside_tools():
    # LiteLLM would turn the schema into a forced tool call for this model.
    bodies = _anthropic_bodies("anthropic/claude-sonnet-4-20250514", api_key="sk-test")

    for body in bodies:
        assert [t["name"] for t in body["tools"]] == ["add"]
        assert "tool_choice" not in body
        assert "output_format" not in body


def test_response_model_sends_no_schema_through_vertex_ai_where_it_is_emulated():
    # The Anthropic entry lists this model as native, the Vertex AI one does not.
    token = ("token", "project")
    # LiteLLM before 1.105 imports the optional Vertex AI SDK here, unused.
    sdk = {"vertexai": MagicMock()}
    with (
        patch.object(VertexBase, "_ensure_access_token", return_value=token),
        patch.dict(sys.modules, sdk),
    ):
        bodies = _anthropic_bodies(
            "vertex_ai/claude-sonnet-4-5",
            vertex_project="project",
            vertex_location="us-east5",
        )

    for body in bodies:
        assert [t["name"] for t in body["tools"]] == ["add"]
        assert "tool_choice" not in body
        assert "output_format" not in body


def test_response_model_sends_no_schema_alongside_tools_unless_listed_native():
    mock = AsyncMock(
        side_effect=[_response(tool_calls=[_add_call()]), _response('{"value": 3}')]
    )
    with patch("litellm.acompletion", new=mock):
        result = Agent(AgentConfig(model="gpt-4o-mini"), tools=[add]).run(
            "1 + 2?", response_model=Answer
        )

    assert result == Answer(value=3)
    for call in mock.call_args_list:
        assert "response_format" not in call.kwargs


def test_response_model_schema_needs_every_router_deployment():
    router = litellm.Router(
        model_list=[
            {"model_name": "smart", "litellm_params": {"model": "gpt-4o-mini"}},
            {"model_name": "smart", "litellm_params": {"model": "gpt-3.5-turbo"}},
        ]
    )
    router.acompletion = AsyncMock(return_value=_response('{"value": 3}'))
    Agent(AgentConfig(model="smart", router=router)).run(
        "1 + 2?", response_model=Answer
    )

    assert "response_format" not in router.acompletion.call_args.kwargs


def test_response_model_native_schema_needs_every_router_deployment():
    router = litellm.Router(
        model_list=[
            {
                "model_name": "smart",
                "litellm_params": {"model": "anthropic/claude-sonnet-4-5"},
            },
            {
                "model_name": "smart",
                "litellm_params": {"model": "anthropic/claude-sonnet-4-20250514"},
            },
        ]
    )
    router.acompletion = AsyncMock(
        side_effect=[_response(tool_calls=[_add_call()]), _response('{"value": 3}')]
    )
    Agent(AgentConfig(model="smart", router=router), tools=[add]).run(
        "1 + 2?", response_model=Answer
    )

    for call in router.acompletion.call_args_list:
        assert "response_format" not in call.kwargs


_CHECKS = [
    (_response_schema,),
    (_native_schema,),
    (_takes_param, "reasoning_effort"),
    (_takes_param, "thinking"),
    (_lacks_tools,),
    (_takes_tool_message_images,),
    (_caches_at_markers,),
]
_MODELS = [
    ("gpt-4o-mini", None),
    ("anthropic/claude-sonnet-4-5", None),
    ("claude-sonnet-4-5", "anthropic"),
    ("bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0", None),
    ("vertex_ai/gemini-2.5-pro", None),
    ("vertex_ai/claude-sonnet-4-5", None),
    ("gemini/gemini-2.5-flash", None),
    ("ollama/llama3", None),
]


def _router(*deployments):
    return litellm.Router(
        model_list=[
            {
                "model_name": "smart",
                "litellm_params": {"model": m}
                | ({"custom_llm_provider": p} if p else {}),
            }
            for m, p in deployments
        ]
    )


@pytest.mark.parametrize(
    "check", _CHECKS, ids=lambda c: "-".join([c[0].__name__, *c[1:]])
)
def test_every_deployment_decides_as_each_check(check):
    """The memoized helper decides as the plain check, alone or through a router."""
    _holds.cache_clear()
    for model, provider in _MODELS:
        expected = check[0](model, provider, *check[1:])
        if provider is None:
            assert _every_deployment(AgentConfig(model=model), *check) is expected
        config = AgentConfig(model="smart", router=_router((model, provider)))
        assert _every_deployment(config, *check) is expected
    for pair in zip(_MODELS, _MODELS[1:]):
        expected = all(check[0](m, p, *check[1:]) for m, p in pair)
        config = AgentConfig(model="smart", router=_router(*pair))
        assert _every_deployment(config, *check) is expected


@pytest.mark.parametrize("model, provider", _MODELS)
def test_the_response_schema_check_decides_as_litellm(model, provider):
    expected = litellm.supports_response_schema(model, provider)
    assert _response_schema(model, provider) is expected


def test_every_deployment_is_false_without_deployments():
    config = AgentConfig(model="other", router=_router(("gpt-4o-mini", None)))
    assert _every_deployment(config, _takes_param, "reasoning_effort") is False


def test_every_deployment_asks_litellm_once_per_model_and_provider():
    _holds.cache_clear()
    config = AgentConfig(
        model="smart",
        router=_router(("gpt-4o-mini", None), ("claude-sonnet-4-5", "anthropic")),
    )
    with patch(
        "litellm.get_supported_openai_params",
        wraps=litellm.get_supported_openai_params,
    ) as lookup:
        for _ in range(3):
            _every_deployment(config, _takes_param, "tools")
            _every_deployment(AgentConfig(model="gpt-4o-mini"), _lacks_tools)

    assert lookup.call_count == 3


@pytest.mark.parametrize(
    "model, fallback",
    [
        ("anthropic/claude-sonnet-4-5", "gpt-4o-mini"),
        ("gpt-4o-mini", "anthropic/claude-sonnet-4-5"),
        ("anthropic/claude-sonnet-4-5", "gemini/gemini-2.5-flash"),
    ],
)
def test_every_deployment_counts_fallback_models_when_asked(model, fallback):
    config = AgentConfig(model=model, fallback_models=[fallback])
    check = _takes_tool_message_images

    assert _every_deployment(config, check, fallbacks=True) is (
        check(model, None) and check(fallback, None)
    )
    assert _every_deployment(config, check) is check(model, None)
