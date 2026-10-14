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


def _unsupported_params_error(param: str = "tools"):
    return litellm.UnsupportedParamsError(
        message=f"does not support parameters: ['{param}']",
        llm_provider="ollama",
        model="llama2",
    )


def test_fallback_to_naive_on_unsupported_tools():
    """The unsupported-params error switches only this agent to naive mode."""
    tool = _make_simple_tool()
    # LiteLLM lists no `tools` support for the ollama provider.
    agent = Agent(config=_make_config("ollama/llama2"), tools=[tool])
    other = Agent(config=_make_config("ollama/llama2"), tools=[tool])
    naive_call = _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 2}}')
    final = _make_completion_response("It is 3.")
    mock = AsyncMock(side_effect=[_unsupported_params_error(), naive_call, final])

    with patch("litellm.acompletion", new=mock):
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


@pytest.mark.parametrize("model", ["gpt-3.5-turbo", "openai/my-fine-tune"])
def test_unsupported_params_on_tool_capable_model_is_raised(model):
    """A model whose provider supports tools rejected another parameter: no fallback."""
    agent = Agent(config=_make_config(model), tools=[_make_simple_tool()])
    error = _unsupported_params_error("temperature")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=error)):
        with pytest.raises(litellm.UnsupportedParamsError):
            agent.run("hi")

    assert agent._get_tool_mode() == "native"


@pytest.mark.parametrize(
    "deployment, expected_mode",
    [("openai/gpt-5", "native"), ("ollama/llama2", "naive")],
)
def test_fallback_through_router_checks_the_deployment(deployment, expected_mode):
    """Through a Router, the alias is resolved to its deployment before deciding."""
    router = litellm.Router(
        model_list=[{"model_name": "smart", "litellm_params": {"model": deployment}}]
    )
    final = _make_completion_response("done")
    router.acompletion = AsyncMock(side_effect=[_unsupported_params_error(), final])
    agent = Agent(
        config=_make_config("smart", router=router), tools=[_make_simple_tool()]
    )

    if expected_mode == "native":
        with pytest.raises(litellm.UnsupportedParamsError):
            agent.run("hi")
    else:
        assert agent.run("hi") == "done"

    assert agent._get_tool_mode() == expected_mode


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

    assert agent.messages[2]["content"] == "Error: add failed: hook broke"


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


def test_cached_tokens_are_summed():
    """Cached prompt tokens add up across the completions of a run."""
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    tc = _make_tool_call("tc_1", "add", '{"a": 2, "b": 3}')
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="5"),
    ]
    for response in responses:
        response.usage.prompt_tokens_details = MagicMock(cached_tokens=4)

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        result = agent.run_with_response("2 + 3?")

    assert result.cached_tokens == 8


def test_litellm_callbacks_fire_for_sync_runs():
    """Sync calls share one loop, so LiteLLM's async success logging completes."""
    import functools
    import time

    from litellm.integrations.custom_logger import CustomLogger

    events = []

    class Recorder(CustomLogger):
        async def async_log_success_event(self, *args, **kwargs):
            events.append("success")

    agent = Agent(config=_make_config())
    acompletion = functools.partial(litellm.acompletion, mock_response="hi")
    with (
        patch.object(litellm, "callbacks", [Recorder()]),
        patch("litellm.acompletion", new=acompletion),
    ):
        for _ in range(3):
            agent.run("hi")
        deadline = time.monotonic() + 5
        while len(events) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)

    assert events == ["success"] * 3


def test_stream_inside_running_loop():
    """stream() works when called from code that already runs an event loop."""
    import asyncio

    agent = Agent(config=_make_config())

    async def caller():
        return list(agent.stream("hi"))

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_stream([_text_chunk("a"), _text_chunk("b")])),
    ):
        assert asyncio.run(caller()) == ["a", "b"]


def test_closing_sync_stream_early_closes_the_completion_stream():
    """Leaving a sync stream early closes the underlying LiteLLM stream."""
    closed = []

    async def completion_stream():
        try:
            for text in ["a", "b", "c"]:
                yield _text_chunk(text)
        finally:
            closed.append(True)

    agent = Agent(config=_make_config())
    with patch("litellm.acompletion", new=AsyncMock(return_value=completion_stream())):
        stream = agent.stream("hi")
        assert next(stream) == "a"
        stream.close()

    assert closed == [True]


def test_sync_run_from_an_async_tool_raises():
    """A sync run started on the shared loop raises instead of deadlocking."""
    from cyclops.toolkit.tool import Tool

    inner = Agent(config=_make_config())

    async def ask() -> str:
        return inner.run("nested")

    agent = Agent(config=_make_config(), tools=[Tool("ask", "Ask", ask)])
    tc = _make_tool_call("tc_1", "ask", "{}")
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="ok"),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        assert agent.run("go") == "ok"

    assert "Use the async Agent methods" in agent.messages[2]["content"]


@pytest.mark.asyncio
async def test_tool_raising_cancelled_stops_sibling_calls():
    """A tool that raises CancelledError also stops the calls still running."""
    import asyncio

    from cyclops.toolkit.tool import Tool

    sibling_cancelled = asyncio.Event()

    async def cancels() -> str:
        await asyncio.sleep(0)
        raise asyncio.CancelledError

    async def slow() -> str:
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
        return "done"

    agent = Agent(
        config=_make_config(),
        tools=[Tool("cancels", "Cancels", cancels), Tool("slow", "Slow", slow)],
    )
    calls = [
        _make_tool_call("tc_1", "cancels", "{}"),
        _make_tool_call("tc_2", "slow", "{}"),
    ]
    response = _make_completion_response(content=None, tool_calls=calls)

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        with pytest.raises(asyncio.CancelledError):
            await agent.arun("go")

    assert sibling_cancelled.is_set()
    tool_messages = [m for m in agent.messages if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["Tool call cancelled."] * 2


def test_interrupted_sync_run_cancels_the_run():
    """An interrupt while run() waits cancels the run and waits for it to unwind."""
    import asyncio
    import threading

    from cyclops.toolkit.tool import Tool

    started = threading.Event()
    cleaned = threading.Event()
    original_wait = threading.Event.wait
    interrupts = []

    async def slow() -> str:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            cleaned.set()
            raise
        return "done"

    def interrupted(self, timeout=None):
        if self is started or interrupts:
            return original_wait(self, timeout)
        interrupts.append(True)
        original_wait(started, 5)
        raise KeyboardInterrupt

    agent = Agent(config=_make_config(), tools=[Tool("slow", "Slow", slow)])
    tc = _make_tool_call("tc_1", "slow", "{}")
    response = _make_completion_response(content=None, tool_calls=[tc])

    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=response)),
        patch.object(threading.Event, "wait", interrupted),
    ):
        with pytest.raises(KeyboardInterrupt):
            agent.run("go")

    assert cleaned.is_set()
    assert agent.messages[-1]["content"] == "Tool call cancelled."


def test_run_without_tools_ignores_max_iterations():
    """max_iterations limits tool rounds; a run without tools still answers."""
    agent = Agent(config=_make_config(max_iterations=0))

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("hi")),
    ):
        result = agent.run_with_response("hello")

    assert result.content == "hi"
    assert result.stop_reason == "completed"


def test_sync_stream_keeps_hook_context():
    """Hooks of a sync stream run in one context, so a ContextVar token resets."""
    import contextvars

    var: contextvars.ContextVar = contextvars.ContextVar("run")

    class ContextHooks(AgentHooks):
        def on_run_start(self, input_message):
            self.token = var.set(input_message)

        def on_run_end(self, content):
            var.reset(self.token)

    agent = Agent(config=_make_config(hooks=ContextHooks()))
    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_stream([_text_chunk("a"), _text_chunk("b")])),
    ):
        assert list(agent.stream("hi")) == ["a", "b"]


def test_interrupted_sync_stream_waits_for_cleanup():
    """An interrupt during a sync stream unwinds the run before returning."""
    import asyncio
    import queue
    import threading

    from cyclops.toolkit.tool import Tool

    started = threading.Event()
    cleaned = threading.Event()

    async def slow() -> str:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)
            cleaned.set()
            raise
        return "done"

    def interrupted(self, *args, **kwargs):
        started.wait(5)
        raise KeyboardInterrupt

    agent = Agent(config=_make_config(), tools=[Tool("slow", "Slow", slow)])
    first = [_tool_call_chunk(0, "{}", id="tc_1", name="slow")]

    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=_stream(first))),
        patch.object(queue.Queue, "get", interrupted),
    ):
        with pytest.raises(KeyboardInterrupt):
            next(agent.stream("go"))

    assert cleaned.is_set()
    assert agent.messages[-1]["content"] == "Tool call cancelled."


@pytest.mark.asyncio
async def test_cancelled_tools_finish_their_cleanup():
    """Cancelling a run cancels each tool once, so slower cleanup still completes."""
    import asyncio

    from cyclops.toolkit.tool import Tool

    started = []
    cleaned = []

    def make(name: str, cleanup: float):
        async def tool() -> str:
            started.append(name)
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await asyncio.sleep(cleanup)
                cleaned.append(name)
                raise
            return "done"

        return Tool(name, name, tool)

    agent = Agent(config=_make_config(), tools=[make("quick", 0), make("slow", 0.05)])
    calls = [
        _make_tool_call("tc_1", "quick", "{}"),
        _make_tool_call("tc_2", "slow", "{}"),
    ]
    response = _make_completion_response(content=None, tool_calls=calls)

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        task = asyncio.create_task(agent.arun("go"))
        while len(started) < 2:
            await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert sorted(cleaned) == ["quick", "slow"]


def test_usage_is_unknown_when_a_step_has_none():
    """Totals are None when any completion of the run reports no usage."""
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    tc = _make_tool_call("tc_1", "add", '{"a": 2, "b": 3}')
    first = _make_completion_response(content=None, tool_calls=[tc])
    first.usage = None
    responses = [first, _make_completion_response(content="5")]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        result = agent.run_with_response("2 + 3?")

    assert result.tokens_used is None
    assert result.prompt_tokens is None


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded")
def test_forked_child_gets_its_own_loop():
    """After fork, the child starts a new shared loop instead of reusing a dead one."""
    import os

    if not hasattr(os, "fork"):
        pytest.skip("needs os.fork")

    agent = Agent(config=_make_config())
    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("hi")),
    ):
        agent.run("start the loop")
        pid = os.fork()
        if pid == 0:
            try:
                os._exit(0 if agent.run("in the child") == "hi" else 1)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(pid, 0)

    assert os.WEXITSTATUS(status) == 0


def test_sync_tool_can_run_another_agent():
    """A sync @tool runs in a worker thread, so it may call another agent's run()."""
    from cyclops.toolkit import tool

    inner = Agent(config=_make_config())

    @tool
    def ask(question: str) -> str:
        """Ask the inner agent."""
        return inner.run(question)

    agent = Agent(config=_make_config(), tools=[ask])
    tc = _make_tool_call("tc_1", "ask", '{"question": "capital of France?"}')
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="Paris"),
        _make_completion_response(content="It is Paris."),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        assert agent.run("Ask about France") == "It is Paris."

    assert agent.messages[2]["content"] == "Paris"


# ---------------------------------------------------------------------------
# Tool results sent to the model
# ---------------------------------------------------------------------------

_IMAGE_PART = {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGk="}}


def _chart_tool():
    from cyclops.toolkit import ImageContent, TextContent, ToolResult, tool

    @tool
    def chart() -> ToolResult:
        """Draw a chart."""
        return ToolResult(
            content=[
                TextContent(text="a chart"),
                ImageContent(data="aGk=", mime_type="image/png"),
            ]
        )

    return chart


def _run_tool_calls(agent, *calls):
    """Run one round of native tool calls followed by a final answer."""
    tool_calls = [
        _make_tool_call(f"tc_{i}", name, "{}") for i, name in enumerate(calls)
    ]
    responses = [
        _make_completion_response(content=None, tool_calls=tool_calls),
        _make_completion_response(content="done"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)) as mock:
        assert agent.run("go") == "done"
    return mock


def test_duplicate_tool_names_are_rejected():
    with pytest.raises(ValueError, match="Duplicate tool name: 'add'"):
        Agent(config=_make_config(), tools=[_make_simple_tool(), _make_simple_tool()])


def test_tool_results_are_rendered_for_the_model():
    from cyclops.toolkit import TextContent, ToolResult, tool

    @tool
    def data() -> dict:
        """Return data."""
        return {"rows": 2}

    @tool
    def text() -> ToolResult:
        """Return text."""
        return ToolResult(content=[TextContent(text="two rows")])

    @tool
    def fail() -> ToolResult:
        """Fail."""
        return ToolResult.error_result("no access")

    hooks = MagicMock(spec=AgentHooks)
    hooks.on_tool_start.return_value = "allow"
    agent = Agent(config=_make_config(hooks=hooks), tools=[data, text, fail])
    _run_tool_calls(agent, "data", "text", "fail")

    tool_messages = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert tool_messages == ['{"rows": 2}', "two rows", "Error: no access"]
    assert [c.args[2] for c in hooks.on_tool_end.call_args_list] == tool_messages


def test_raising_tool_is_marked_as_error():
    from cyclops.toolkit import tool

    @tool
    def boom() -> str:
        """Raise."""
        raise RuntimeError("disk full")

    agent = Agent(config=_make_config(), tools=[boom])
    _run_tool_calls(agent, "boom")
    assert agent.messages[2]["content"] == "Error: boom failed: disk full"


def test_tool_images_stay_in_tool_message_when_provider_accepts_them():
    agent = Agent(
        config=_make_config(model="anthropic/claude-3-5-sonnet-latest"),
        tools=[_chart_tool()],
    )
    _run_tool_calls(agent, "chart")

    assert [m["role"] for m in agent.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert agent.messages[2]["content"] == [
        {"type": "text", "text": "a chart"},
        _IMAGE_PART,
    ]


def test_tool_images_move_to_user_message_for_text_only_tool_messages():
    agent = Agent(
        config=_make_config(model="gpt-4o"),
        tools=[_chart_tool(), _make_simple_tool()],
    )
    tool_calls = [
        _make_tool_call("tc_1", "chart", "{}"),
        _make_tool_call("tc_2", "add", '{"a": 1, "b": 2}'),
    ]
    responses = [
        _make_completion_response(content=None, tool_calls=tool_calls),
        _make_completion_response(content="done"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)) as mock:
        agent.run("go")

    # The images follow every tool message of the round
    assert [m["role"] for m in agent.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
    ]
    assert agent.messages[2]["content"] == "a chart"
    assert agent.messages[3]["content"] == "3"
    assert agent.messages[4]["content"] == [
        {"type": "text", "text": "Images returned by chart:"},
        _IMAGE_PART,
    ]
    sent = mock.call_args_list[1].kwargs["messages"]
    assert sent[-1] == agent.messages[4]


def test_tool_images_in_naive_mode():
    agent = Agent(config=_make_config(tool_mode="naive"), tools=[_chart_tool()])
    responses = [
        _make_completion_response('{"tool": "chart", "args": {}}'),
        _make_completion_response("done"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        agent.run("go")

    assert agent.messages[2] == {
        "role": "user",
        "content": [{"type": "text", "text": "Tool result: a chart"}, _IMAGE_PART],
    }


def test_streamed_run_renders_tool_results():
    from cyclops.toolkit import ToolResult, tool

    @tool
    def fail() -> ToolResult:
        """Fail."""
        return ToolResult.error_result("no access")

    agent = Agent(config=_make_config(), tools=[fail])
    first = [_tool_call_chunk(0, "{}", id="tc_1", name="fail")]
    second = [_text_chunk("sorry")]
    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[_stream(first), _stream(second)]),
    ):
        assert list(agent.stream("go")) == ["sorry"]

    assert agent.messages[2]["content"] == "Error: no access"


def test_denied_and_unknown_tools_are_reported_as_errors():
    class Deny(AgentHooks):
        def on_tool_start(self, tool_name, args):
            return "deny"

    agent = Agent(config=_make_config(hooks=Deny()), tools=[_make_simple_tool()])
    _run_tool_calls(agent, "add", "missing")

    tool_messages = [m["content"] for m in agent.messages if m["role"] == "tool"]
    assert tool_messages == [
        "Error: Tool 'add' was not approved",
        "Error: Tool 'missing' not found",
    ]


@pytest.mark.parametrize(
    "deployments, kept",
    [
        (["anthropic/claude-3-5-sonnet-latest", "gemini/gemini-1.5-pro"], True),
        (["anthropic/claude-3-5-sonnet-latest", "openai/gpt-4o"], False),
        (["vertex_ai/gemini-1.5-pro"], True),
        (["vertex_ai/meta/llama3-405b-instruct-maas"], False),
    ],
)
def test_tool_image_route_checks_every_router_deployment(deployments, kept):
    router = litellm.Router(
        model_list=[
            {"model_name": "smart", "litellm_params": {"model": d}} for d in deployments
        ]
    )
    tool_calls = [_make_tool_call("tc_1", "chart", "{}")]
    router.acompletion = AsyncMock(
        side_effect=[
            _make_completion_response(content=None, tool_calls=tool_calls),
            _make_completion_response(content="done"),
        ]
    )
    agent = Agent(config=_make_config("smart", router=router), tools=[_chart_tool()])
    agent.run("go")

    roles = [m["role"] for m in agent.messages]
    if kept:
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert agent.messages[2]["content"][1] == _IMAGE_PART
    else:
        assert roles == ["user", "assistant", "tool", "user", "assistant"]
        assert agent.messages[2]["content"] == "a chart"
