"""Tests for the Agent class"""

from unittest.mock import AsyncMock, MagicMock, patch

import litellm
import pytest
from pydantic import BaseModel

from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponseStream,
    StreamingChoices,
)

from cyclops.core.agent import Agent
from cyclops.core.hooks import AgentHooks
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
    usage.prompt_tokens_details = None

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


def _text_chunk(text: str):
    return ModelResponseStream(
        model="gpt-3.5-turbo", choices=[StreamingChoices(delta=Delta(content=text))]
    )


def _tool_call_chunk(index: int, arguments: str, id=None, name=None):
    delta = ChatCompletionDeltaToolCall(
        index=index,
        id=id,
        type="function" if id else None,
        function=Function(name=name, arguments=arguments),
    )
    return ModelResponseStream(
        model="gpt-3.5-turbo",
        choices=[StreamingChoices(delta=Delta(tool_calls=[delta]))],
    )


async def _stream(chunks):
    for chunk in chunks:
        yield chunk


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

    with patch(
        "litellm.acompletion", new=AsyncMock(return_value=response)
    ) as mock_comp:
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

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[first_response, second_response]),
    ):
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
        "litellm.acompletion",
        new=AsyncMock(side_effect=[first_response, second_response, third_response]),
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

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_stream([_text_chunk(c) for c in chunks])),
    ):
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

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
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

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
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

    with patch("litellm.acompletion", new=AsyncMock(side_effect=[resp1, resp2])):
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

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        with patch("litellm.completion_cost", return_value=0.0001):
            agent_response = agent.run_with_response("Hi")

    assert agent_response.content == "Hello!"
    assert agent_response.model == "gpt-3.5-turbo"
    assert agent_response.prompt_tokens == 10
    assert agent_response.completion_tokens == 20
    assert agent_response.tokens_used == 30
    assert agent_response.cost == pytest.approx(0.0001)


# ---------------------------------------------------------------------------
# Loop behaviour
# ---------------------------------------------------------------------------


class _RecordingHooks(AgentHooks):
    def __init__(self):
        self.events = []

    def on_run_end(self, content):
        self.events.append(("run_end", content))

    def on_llm_end(self, response):
        self.events.append(("llm_end", None))


def test_sync_tool_runs_in_worker_thread():
    """A tool with a sync execute runs off the event loop thread."""
    import threading

    from cyclops.toolkit.tool import BaseTool

    seen = []

    class SyncTool(BaseTool):
        def execute(self, x: int) -> int:  # type: ignore[override]
            seen.append(threading.get_ident())
            return x * 2

    agent = Agent(config=_make_config(), tools=[SyncTool("double", "Double x")])
    tc = _make_tool_call("tc_1", "double", '{"x": 4}')
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="8"),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        assert agent.run("double 4") == "8"

    assert agent.messages[2]["content"] == "8"
    assert seen and seen[0] != threading.get_ident()


def test_run_inside_running_loop():
    """run() works when called from code that already runs an event loop."""
    import asyncio

    agent = Agent(config=_make_config())
    response = _make_completion_response("ok")

    async def caller():
        return agent.run("hi")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        assert asyncio.run(caller()) == "ok"


def test_run_inside_running_loop_keeps_runtime_error():
    """A RuntimeError from the run is raised as is when a loop is already running."""
    import asyncio

    agent = Agent(config=_make_config())

    async def caller():
        return agent.run("hi")

    with patch(
        "litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("provider down"))
    ):
        with pytest.raises(RuntimeError, match="provider down"):
            asyncio.run(caller())


def _unsupported_tools_error():
    return litellm.UnsupportedParamsError(
        message="does not support parameters: ['tools']",
        llm_provider="custom",
        model="local-model",
    )


def test_fallback_to_naive_on_unsupported_tools():
    """The unsupported-params error switches only this agent to naive mode."""
    tool = _make_simple_tool()
    agent = Agent(config=_make_config("local-model"), tools=[tool])
    other = Agent(config=_make_config("local-model"), tools=[tool])
    naive_call = _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 2}}')
    final = _make_completion_response("It is 3.")
    mock = AsyncMock(side_effect=[_unsupported_tools_error(), naive_call, final])

    with (
        patch("litellm.acompletion", new=mock),
        patch("litellm.supports_function_calling", return_value=False),
    ):
        assert agent.run("1 + 2?") == "It is 3."

    assert "tools" not in mock.call_args.kwargs
    user_messages = [m for m in agent.messages if m["content"] == "1 + 2?"]
    assert len(user_messages) == 1
    assert agent.messages[-2]["content"] == "Tool result: 3"
    assert agent._get_tool_mode() == "naive"
    assert other._get_tool_mode() == "native"


def test_unrelated_error_keeps_native_mode():
    """An error that mentions tools but is not the unsupported-params error is raised."""
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    error = litellm.BadRequestError(
        message="invalid tool_use block", model="gpt-3.5-turbo", llm_provider="openai"
    )

    with patch("litellm.acompletion", new=AsyncMock(side_effect=error)):
        with pytest.raises(litellm.BadRequestError):
            agent.run("hi")

    assert agent._get_tool_mode() == "native"


def test_unsupported_params_on_tool_capable_model_is_raised():
    """A model that supports tools rejected some other parameter: no fallback."""
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])

    with (
        patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=_unsupported_tools_error()),
        ),
        patch("litellm.supports_function_calling", return_value=True),
    ):
        with pytest.raises(litellm.UnsupportedParamsError):
            agent.run("hi")

    assert agent._get_tool_mode() == "native"


def test_invalid_tool_arguments_return_parse_error():
    """Unparseable arguments do not run the tool; the model sees the parse error."""
    tool = MagicMock(wraps=_make_simple_tool())
    tool.name = "add"
    agent = Agent(config=_make_config(), tools=[tool])
    tc = _make_tool_call("tc_1", "add", '{"a": 1,')
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="Sorry."),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        agent.run("add")

    tool.execute.assert_not_called()
    assert "add are not valid JSON" in agent.messages[2]["content"]


def test_non_object_tool_arguments_return_error():
    """JSON arguments that are not an object get an error result, not a crash."""
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    tc = _make_tool_call("tc_1", "add", "[1, 2]")
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="Sorry."),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        result = agent.run_with_response("add")

    assert (
        agent.messages[2]["content"] == "Error: arguments for add must be a JSON object"
    )
    assert result.tool_calls[0].arguments == {}


def test_failing_end_hook_becomes_tool_error():
    """An on_tool_end hook that raises yields a tool error and a valid transcript."""

    class FailingHooks(AgentHooks):
        def on_tool_end(self, tool_name, args, result):
            raise ValueError("hook broke")

    agent = Agent(
        config=_make_config(hooks=FailingHooks()), tools=[_make_simple_tool()]
    )
    tc = _make_tool_call("tc_1", "add", '{"a": 1, "b": 2}')
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="ok"),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        assert agent.run("add") == "ok"

    assert agent.messages[2]["content"] == "Error executing add: hook broke"


def test_max_iterations_sets_stop_reason():
    """Hitting max_iterations is a stop reason, not answer text."""
    agent = Agent(config=_make_config(max_iterations=2), tools=[_make_simple_tool()])
    responses = [
        _make_completion_response(
            content=None,
            tool_calls=[_make_tool_call(f"tc_{i}", "add", '{"a": 1, "b": 1}')],
        )
        for i in range(2)
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        result = agent.run_with_response("loop")

    assert result.stop_reason == "max_iterations"
    assert result.content == ""
    assert len(result.tool_calls) == 2
    assert agent.messages[-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_cancelled_run_closes_pending_tool_calls():
    """Cancelling during tool execution leaves a result for every tool call."""
    import asyncio

    from cyclops.toolkit.tool import Tool

    started = asyncio.Event()

    async def fast() -> str:
        return "fast done"

    async def slow() -> str:
        started.set()
        await asyncio.sleep(10)
        return "done"

    agent = Agent(
        config=_make_config(),
        tools=[Tool("fast", "Fast tool", fast), Tool("slow", "Slow tool", slow)],
    )
    calls = [
        _make_tool_call("tc_1", "fast", "{}"),
        _make_tool_call("tc_2", "slow", "{}"),
    ]
    response = _make_completion_response(content=None, tool_calls=calls)

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        task = asyncio.create_task(agent.arun("go"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_messages] == [
        ("tc_1", "fast done"),
        ("tc_2", "Tool call cancelled."),
    ]


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


def test_usage_sums_every_completion():
    """Tokens and cost cover every completion of a tool-using run."""
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    tc = _make_tool_call("tc_1", "add", '{"a": 2, "b": 3}')
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="5"),
    ]

    with (
        patch("litellm.acompletion", new=AsyncMock(side_effect=responses)),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        result = agent.run_with_response("2 + 3?")

    assert result.prompt_tokens == 20
    assert result.completion_tokens == 40
    assert result.tokens_used == 60
    assert result.cost == pytest.approx(0.002)
    assert result.stop_reason == "completed"


def test_usage_in_naive_mode():
    """Naive tool mode reports usage too."""
    agent = Agent(config=_make_config(tool_mode="naive"), tools=[_make_simple_tool()])
    responses = [
        _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 2}}'),
        _make_completion_response("3"),
    ]

    with (
        patch("litellm.acompletion", new=AsyncMock(side_effect=responses)),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        result = agent.run_with_response("1 + 2?")

    assert result.content == "3"
    assert result.tokens_used == 60
    assert result.cost == pytest.approx(0.002)


# ---------------------------------------------------------------------------
# Streaming with tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_astream_with_tools_streams_the_real_loop():
    """Each step is streamed with tools; tool calls are assembled from deltas."""
    hooks = _RecordingHooks()
    agent = Agent(config=_make_config(hooks=hooks), tools=[_make_simple_tool()])
    first = [
        _text_chunk("Adding. "),
        _tool_call_chunk(0, '{"a": 2,', id="tc_1", name="add"),
        _tool_call_chunk(0, ' "b": 3}'),
    ]
    second = [_text_chunk("The answer "), _text_chunk("is 5.")]
    mock = AsyncMock(side_effect=[_stream(first), _stream(second)])

    with patch("litellm.acompletion", new=mock):
        chunks = [c async for c in agent.astream("2 + 3?")]

    assert chunks == ["Adding. ", "The answer ", "is 5."]
    assert mock.call_count == 2
    assert all("tools" in call.kwargs for call in mock.call_args_list)
    roles = [m["role"] for m in agent.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert agent.messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"a": 2, "b": 3}'
    )
    assert agent.messages[2]["content"] == "5"
    assert agent.messages[3]["content"] == "The answer is 5."
    assert hooks.events == [
        ("llm_end", None),
        ("llm_end", None),
        ("run_end", "The answer is 5."),
    ]


def test_stream_with_tools_sync():
    """The sync stream() wrapper drives the same loop."""
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    first = [_tool_call_chunk(0, '{"a": 1, "b": 1}', id="tc_1", name="add")]
    second = [_text_chunk("2")]

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[_stream(first), _stream(second)]),
    ):
        assert list(agent.stream("1 + 1?")) == ["2"]

    assert agent.messages[2]["content"] == "2"
