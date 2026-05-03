"""Tests for the Agent class"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from cyclops.core.agent import Agent
from cyclops.core.types import AgentConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(model: str = "gpt-3.5-turbo", **kwargs) -> AgentConfig:
    return AgentConfig(model=model, **kwargs)


def _make_completion_response(content: str, tool_calls=None):
    """Build a minimal mock litellm completion response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = msg

    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    usage.total_tokens = 30

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_tool_call(id: str, name: str, arguments: str):
    """Build a minimal mock tool_call object."""
    fn = MagicMock()
    fn.name = name
    fn.arguments = arguments

    tc = MagicMock()
    tc.id = id
    tc.function = fn
    return tc


def _make_simple_tool(name: str = "add", description: str = "Add two numbers"):
    """Return a simple mock tool."""
    from cyclops.toolkit.tool import Tool

    def add(a: int, b: int) -> int:
        return a + b

    return Tool(name=name, description=description, func=add)


# ---------------------------------------------------------------------------
# test_run_no_tools
# ---------------------------------------------------------------------------


def test_run_no_tools():
    agent = Agent(config=_make_config())
    response = _make_completion_response("Hello, world!")

    with patch("litellm.completion", return_value=response) as mock_comp:
        result = agent.run("Say hi")

    assert result == "Hello, world!"
    mock_comp.assert_called_once()
    # History should have user + assistant
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[1]["role"] == "assistant"
    assert agent.messages[1]["content"] == "Hello, world!"


# ---------------------------------------------------------------------------
# test_run_with_tools_loop
# ---------------------------------------------------------------------------


def test_run_with_tools_loop():
    """Agent loops through tool calls on first response, then returns final answer."""
    tool = _make_simple_tool()
    agent = Agent(config=_make_config(), tools=[tool])

    tc = _make_tool_call("tc_1", "add", '{"a": 2, "b": 3}')
    first_response = _make_completion_response(content=None, tool_calls=[tc])
    second_response = _make_completion_response(content="The answer is 5.")

    with patch("litellm.completion", side_effect=[first_response, second_response]):
        result = agent.run("What is 2 + 3?")

    assert result == "The answer is 5."

    # Verify history: user → assistant (with tool_calls) → tool → assistant
    roles = [m["role"] for m in agent.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]

    tool_msg = agent.messages[2]
    assert tool_msg["tool_call_id"] == "tc_1"
    assert tool_msg["name"] == "add"
    assert tool_msg["content"] == "5"


# ---------------------------------------------------------------------------
# test_run_with_tools_multi_iteration
# ---------------------------------------------------------------------------


def test_run_with_tools_multi_iteration():
    """Agent correctly handles two sequential tool call rounds before final answer."""
    tool = _make_simple_tool()
    agent = Agent(config=_make_config(), tools=[tool])

    tc1 = _make_tool_call("tc_1", "add", '{"a": 1, "b": 2}')
    tc2 = _make_tool_call("tc_2", "add", '{"a": 3, "b": 4}')
    first_response = _make_completion_response(content=None, tool_calls=[tc1])
    second_response = _make_completion_response(content=None, tool_calls=[tc2])
    third_response = _make_completion_response(content="Done")

    with patch(
        "litellm.completion",
        side_effect=[first_response, second_response, third_response],
    ):
        result = agent.run("Do two additions")

    assert result == "Done"
    roles = [m["role"] for m in agent.messages]
    # user, asst(tc1), tool, asst(tc2), tool, asst(final)
    assert roles.count("tool") == 2
    assert roles[-1] == "assistant"
    assert agent.messages[-1]["content"] == "Done"


# ---------------------------------------------------------------------------
# test_stream_no_tools
# ---------------------------------------------------------------------------


def test_stream_no_tools():
    """stream() yields individual chunks when there are no tools."""
    agent = Agent(config=_make_config())

    chunks = ["Hello", ", ", "world", "!"]

    def _make_stream_chunk(text: str):
        delta = MagicMock()
        delta.content = text
        choice = MagicMock()
        choice.delta = delta
        chunk = MagicMock()
        chunk.choices = [choice]
        return chunk

    stream_response = iter(_make_stream_chunk(c) for c in chunks)

    with patch("litellm.completion", return_value=stream_response):
        result = list(agent.stream("Say hi"))

    assert result == chunks
    # History should record the joined content
    assert agent.messages[-1]["content"] == "Hello, world!"


# ---------------------------------------------------------------------------
# test_run_with_response_model
# ---------------------------------------------------------------------------


class AnswerModel(BaseModel):
    answer: str
    confidence: float


def test_run_with_response_model():
    """run() parses JSON response into a Pydantic model when response_model is given."""
    agent = Agent(config=_make_config())
    json_content = '{"answer": "Paris", "confidence": 0.99}'
    response = _make_completion_response(json_content)

    with patch("litellm.completion", return_value=response):
        result = agent.run("What is the capital of France?", response_model=AnswerModel)

    assert isinstance(result, AnswerModel)
    assert result.answer == "Paris"
    assert result.confidence == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# test_reset
# ---------------------------------------------------------------------------


def test_reset():
    """reset() clears conversation history."""
    agent = Agent(config=_make_config())
    response = _make_completion_response("Hi")

    with patch("litellm.completion", return_value=response):
        agent.run("Hello")

    assert len(agent.messages) == 2
    agent.reset()
    assert agent.messages == []


# ---------------------------------------------------------------------------
# test_conversation_history
# ---------------------------------------------------------------------------


def test_conversation_history():
    """Multi-turn conversations accumulate history correctly."""
    agent = Agent(config=_make_config())

    resp1 = _make_completion_response("I'm Claude.")
    resp2 = _make_completion_response("The sky is blue.")

    with patch("litellm.completion", side_effect=[resp1, resp2]):
        result1 = agent.run("Who are you?")
        result2 = agent.run("What color is the sky?")

    assert result1 == "I'm Claude."
    assert result2 == "The sky is blue."

    roles = [m["role"] for m in agent.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert agent.messages[0]["content"] == "Who are you?"
    assert agent.messages[2]["content"] == "What color is the sky?"


# ---------------------------------------------------------------------------
# test_arun_no_tools (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_no_tools():
    agent = Agent(config=_make_config())
    response = _make_completion_response("Async hello!")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = await agent.arun("Hi async")

    assert result == "Async hello!"
    assert agent.messages[-1]["content"] == "Async hello!"


# ---------------------------------------------------------------------------
# test_arun_with_tools_loop (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_with_tools_loop():
    tool = _make_simple_tool()
    agent = Agent(config=_make_config(), tools=[tool])

    tc = _make_tool_call("tc_async_1", "add", '{"a": 10, "b": 5}')
    first_response = _make_completion_response(content=None, tool_calls=[tc])
    second_response = _make_completion_response(content="Result is 15.")

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[first_response, second_response]),
    ):
        result = await agent.arun("What is 10 + 5?")

    assert result == "Result is 15."
    roles = [m["role"] for m in agent.messages]
    assert "tool" in roles


# ---------------------------------------------------------------------------
# test_run_with_response_method
# ---------------------------------------------------------------------------


def test_run_with_response_method():
    """run_with_response() returns an AgentResponse with metadata."""
    agent = Agent(config=_make_config())
    response = _make_completion_response("Hello!")

    with patch("litellm.completion", return_value=response):
        with patch("litellm.completion_cost", return_value=0.0001):
            agent_response = agent.run_with_response("Hi")

    assert agent_response.content == "Hello!"
    assert agent_response.model == "gpt-3.5-turbo"
    assert agent_response.prompt_tokens == 10
    assert agent_response.completion_tokens == 20
    assert agent_response.tokens_used == 30
    assert agent_response.cost == pytest.approx(0.0001)
