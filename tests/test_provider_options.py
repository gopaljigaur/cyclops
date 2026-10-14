"""Tests for completion kwargs, reasoning history, prompt caching and response schemas"""

from json import loads
from unittest.mock import AsyncMock, patch

import httpx
import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
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
            "num_retries": 2,
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
            "temperature": 0.1,
            "max_tokens": 512,
            "api_key": "sk-test",
            "api_base": "https://llm.example.com/v1",
            "extra_headers": {"X-Trace": "1"},
            "num_retries": 2,
            "timeout": 30,
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
        temperature=0.1,
        max_tokens=None,
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
    response = _response("Paris", reasoning_content="The capital is Paris.")
    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        agent.run("Capital of France?")

    assert agent.messages[-1] == {
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
    with patch("litellm.acompletion", new=AsyncMock(return_value=stream())):
        assert "".join(agent.stream("1 + 2?")) == "3"

    assert agent.messages[-1]["thinking_blocks"] == THINKING


def test_message_round_trips_reasoning():
    d = {
        "role": "assistant",
        "content": "3",
        "reasoning_content": "Add them.",
        "thinking_blocks": THINKING,
    }
    assert CyclopsMessage.from_dict(d).to_dict() == d


def test_anthropic_request_puts_signed_thinking_before_the_tool_use():
    """End to end through LiteLLM's Anthropic request builder."""
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
        temperature=1,
        system_prompt="Be brief.",
        prompt_caching=True,
        completion_kwargs={
            "api_key": "sk-test",
            "max_tokens": 4096,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
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
# Prompt caching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "anthropic/claude-sonnet-4-5",
        "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "vertex_ai/claude-sonnet-4-5",
    ],
)
def test_prompt_caching_marks_system_prompt_and_latest_message(model):
    agent = Agent(AgentConfig(model=model, prompt_caching=True))
    mock = AsyncMock(return_value=_response("hi"))
    with patch("litellm.acompletion", new=mock):
        agent.run("hello")

    assert mock.call_args.kwargs["cache_control_injection_points"] == [
        {"location": "message", "role": "system"},
        {"location": "message", "index": -1},
    ]


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
    and these Bedrock models reject cache points."""
    mock = AsyncMock(return_value=_response("hi"))
    with patch("litellm.acompletion", new=mock):
        Agent(AgentConfig(model=model, prompt_caching=True)).run("hello")

    assert "cache_control_injection_points" not in mock.call_args.kwargs


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
    Agent(AgentConfig(model="smart", router=router, prompt_caching=True)).run("hello")

    assert "cache_control_injection_points" not in router.acompletion.call_args.kwargs


def test_prompt_caching_is_off_by_default():
    mock = AsyncMock(return_value=_response("hi"))
    with patch("litellm.acompletion", new=mock):
        Agent(AgentConfig(model="anthropic/claude-sonnet-4-5")).run("hello")

    assert "cache_control_injection_points" not in mock.call_args.kwargs


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


def test_response_model_sends_no_schema_alongside_tools():
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
