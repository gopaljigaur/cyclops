"""Tests for the Agent class"""

import contextvars
import gc
import traceback
import weakref
from typing import Optional
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
from cyclops.core.session import Session
from cyclops.core.sync import _run_coroutine_sync
from cyclops.core.types import AgentConfig
from cyclops.toolkit.types import ToolAnnotations

# Read-only tools run concurrently, so tests of parallel calls use them.
READ_ONLY = ToolAnnotations(read_only=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(model: str = "gpt-3.5-turbo", **kwargs) -> AgentConfig:
    return AgentConfig(model=model, **kwargs)


def _make_completion_response(
    content: Optional[str], tool_calls=None, finish_reason: Optional[str] = None
):
    """Build a minimal mock litellm completion response."""
    msg = MagicMock(reasoning_content=None, thinking_blocks=None)
    msg.content = content
    msg.tool_calls = tool_calls or []

    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason or ("tool_calls" if tool_calls else "stop")

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
    session = Session()
    agent = Agent(config=_make_config())
    response = _make_completion_response("Hello, world!")

    with patch(
        "litellm.acompletion", new=AsyncMock(return_value=response)
    ) as mock_comp:
        result = agent.run("Say hi", session=session)

    assert result == "Hello, world!"
    mock_comp.assert_called_once()
    # History should have user + assistant
    assert session.messages[0]["role"] == "user"
    assert session.messages[1]["role"] == "assistant"
    assert session.messages[1]["content"] == "Hello, world!"


# ---------------------------------------------------------------------------
# test_run_with_tools_loop
# ---------------------------------------------------------------------------


def test_run_with_tools_loop():
    """Agent loops through tool calls on first response, then returns final answer."""
    session = Session()
    tool = _make_simple_tool()
    agent = Agent(config=_make_config(), tools=[tool])

    tc = _make_tool_call("tc_1", "add", '{"a": 2, "b": 3}')
    first_response = _make_completion_response(content=None, tool_calls=[tc])
    second_response = _make_completion_response(content="The answer is 5.")

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[first_response, second_response]),
    ):
        result = agent.run("What is 2 + 3?", session=session)

    assert result == "The answer is 5."

    # Verify history: user → assistant (with tool_calls) → tool → assistant
    roles = [m["role"] for m in session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]

    tool_msg = session.messages[2]
    assert tool_msg["tool_call_id"] == "tc_1"
    assert tool_msg["name"] == "add"
    assert tool_msg["content"] == "5"


# ---------------------------------------------------------------------------
# test_run_with_tools_multi_iteration
# ---------------------------------------------------------------------------


def test_run_with_tools_multi_iteration():
    """Agent correctly handles two sequential tool call rounds before final answer."""
    session = Session()
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
        result = agent.run("Do two additions", session=session)

    assert result == "Done"
    roles = [m["role"] for m in session.messages]
    # user, asst(tc1), tool, asst(tc2), tool, asst(final)
    assert roles.count("tool") == 2
    assert roles[-1] == "assistant"
    assert session.messages[-1]["content"] == "Done"


# ---------------------------------------------------------------------------
# test_stream_no_tools
# ---------------------------------------------------------------------------


def test_stream_no_tools():
    """stream() yields individual chunks when there are no tools."""
    session = Session()
    agent = Agent(config=_make_config())
    chunks = ["Hello", ", ", "world", "!"]

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_stream([_text_chunk(c) for c in chunks])),
    ):
        result = list(agent.stream("Say hi", session=session))

    assert result == chunks
    # History should record the joined content
    assert session.messages[-1]["content"] == "Hello, world!"


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
# test_arun_no_tools (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_no_tools():
    session = Session()
    agent = Agent(config=_make_config())
    response = _make_completion_response("Async hello!")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = await agent.arun("Hi async", session=session)

    assert result == "Async hello!"
    assert session.messages[-1]["content"] == "Async hello!"


# ---------------------------------------------------------------------------
# test_arun_with_tools_loop (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arun_with_tools_loop():
    session = Session()
    tool = _make_simple_tool()
    agent = Agent(config=_make_config(), tools=[tool])

    tc = _make_tool_call("tc_async_1", "add", '{"a": 10, "b": 5}')
    first_response = _make_completion_response(content=None, tool_calls=[tc])
    second_response = _make_completion_response(content="Result is 15.")

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[first_response, second_response]),
    ):
        result = await agent.arun("What is 10 + 5?", session=session)

    assert result == "Result is 15."
    roles = [m["role"] for m in session.messages]
    assert "tool" in roles


# ---------------------------------------------------------------------------
# test_run_with_response_method
# ---------------------------------------------------------------------------


def test_run_with_response_method():
    """run_with_response() returns a RunResult with metadata."""
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
    session = Session()
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
        assert agent.run("double 4", session=session) == "8"

    assert session.messages[2]["content"] == "8"
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
    session = Session()
    tool = _make_simple_tool()
    # LiteLLM lists no `tools` support for the ollama provider.
    agent = Agent(config=_make_config("ollama/llama2"), tools=[tool])
    other = Agent(config=_make_config("ollama/llama2"), tools=[tool])
    naive_call = _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 2}}')
    final = _make_completion_response("It is 3.")
    mock = AsyncMock(side_effect=[_unsupported_params_error(), naive_call, final])

    with patch("litellm.acompletion", new=mock):
        assert agent.run("1 + 2?", session=session) == "It is 3."

    assert "tools" not in mock.call_args.kwargs
    user_messages = [m for m in session.messages if m["content"] == "1 + 2?"]
    assert len(user_messages) == 1
    assert session.messages[-2]["content"] == "Tool result: 3"
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
    session = Session()
    tool = MagicMock(wraps=_make_simple_tool())
    tool.name = "add"
    agent = Agent(config=_make_config(), tools=[tool])
    tc = _make_tool_call("tc_1", "add", '{"a": 1,')
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="Sorry."),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        agent.run("add", session=session)

    tool.execute.assert_not_called()
    assert "add are not valid JSON" in session.messages[2]["content"]


def test_non_object_tool_arguments_return_error():
    """JSON arguments that are not an object get an error result, not a crash."""
    session = Session()
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    tc = _make_tool_call("tc_1", "add", "[1, 2]")
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="Sorry."),
    ]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        result = agent.run_with_response("add", session=session)

    assert (
        session.messages[2]["content"]
        == "Error: arguments for add must be a JSON object"
    )
    assert result.tool_calls[0].arguments == {}


def test_failing_end_hook_becomes_tool_error():
    """An on_tool_end hook that raises yields a tool error and a valid transcript."""
    session = Session()

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
        assert agent.run("add", session=session) == "ok"

    assert session.messages[2]["content"] == "Error: add failed: hook broke"


def _tool_call_responses(n):
    return [
        _make_completion_response(
            content=None,
            tool_calls=[_make_tool_call(f"tc_{i}", "add", '{"a": 1, "b": 1}')],
        )
        for i in range(n)
    ]


def test_max_steps_sets_stop_reason():
    """Hitting max_steps is a stop reason, not answer text."""
    session = Session()
    agent = Agent(config=_make_config(max_steps=2), tools=[_make_simple_tool()])

    with patch(
        "litellm.acompletion", new=AsyncMock(side_effect=_tool_call_responses(2))
    ):
        result = agent.run_with_response("loop", session=session)

    assert result.stop_reason == "max_steps"
    assert result.steps == 2
    assert result.content == ""
    assert len(result.tool_calls) == 2
    assert session.messages[-1]["role"] == "tool"


def test_unknown_config_fields_are_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _make_config(max_iterations=3)


def test_token_budget_ends_the_run_between_steps():
    """Reaching max_total_tokens stops before the next completion."""
    session = Session()
    agent = Agent(config=_make_config(max_total_tokens=50), tools=[_make_simple_tool()])
    mock = AsyncMock(side_effect=_tool_call_responses(3))

    with patch("litellm.acompletion", new=mock):
        result = agent.run_with_response("loop", session=session)

    assert result.stop_reason == "budget"
    assert result.steps == 2
    assert result.tokens_used == 60
    assert mock.call_count == 2
    # Every tool call of the last step still has its result.
    assert session.messages[-1]["role"] == "tool"


def test_cost_budget_ends_the_run_between_steps():
    agent = Agent(config=_make_config(max_cost=0.001), tools=[_make_simple_tool()])

    with (
        patch(
            "litellm.acompletion", new=AsyncMock(side_effect=_tool_call_responses(3))
        ),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        result = agent.run_with_response("loop")

    assert result.stop_reason == "budget"
    assert result.steps == 1
    assert result.cost == pytest.approx(0.001)


def test_unknown_cost_does_not_end_the_run():
    agent = Agent(config=_make_config(max_cost=0.001), tools=[_make_simple_tool()])
    responses = [*_tool_call_responses(1), _make_completion_response("2")]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        result = agent.run_with_response("1 + 1?")

    assert result.cost is None
    assert result.stop_reason == "completed"
    assert result.content == "2"


def test_final_answer_completes_even_over_budget():
    agent = Agent(config=_make_config(max_total_tokens=1))

    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("hi")),
    ):
        result = agent.run_with_response("hello")

    assert result.stop_reason == "completed"
    assert result.content == "hi"
    assert result.steps == 1


def test_token_budget_counts_steps_after_one_without_usage():
    """A completion without usage doesn't switch the token limit off."""
    agent = Agent(config=_make_config(max_total_tokens=50), tools=[_make_simple_tool()])
    responses = _tool_call_responses(4)
    responses[0].usage = None
    mock = AsyncMock(side_effect=responses)

    with patch("litellm.acompletion", new=mock):
        result = agent.run_with_response("loop")

    assert result.stop_reason == "budget"
    assert result.usage is None
    assert mock.call_count == 3


def test_cost_budget_counts_steps_after_one_without_pricing():
    """A completion LiteLLM can't price doesn't switch the cost limit off."""
    agent = Agent(config=_make_config(max_cost=0.002), tools=[_make_simple_tool()])
    cost = MagicMock(side_effect=[Exception("no pricing"), 0.001, 0.001, 0.001])
    mock = AsyncMock(side_effect=_tool_call_responses(4))

    with (
        patch("litellm.acompletion", new=mock),
        patch("litellm.completion_cost", new=cost),
    ):
        result = agent.run_with_response("loop")

    assert result.stop_reason == "budget"
    assert result.cost is None
    assert mock.call_count == 3


@pytest.mark.asyncio
async def test_token_budget_ends_a_streamed_run():
    session = Session()
    agent = Agent(config=_make_config(max_total_tokens=30), tools=[_make_simple_tool()])
    first = _make_completion_response(
        content=None, tool_calls=[_make_tool_call("tc_1", "add", '{"a": 1, "b": 1}')]
    )
    chunks = [_tool_call_chunk(0, '{"a": 1, "b": 1}', id="tc_1", name="add")]
    mock = AsyncMock(side_effect=[_stream(chunks), _stream([_text_chunk("2")])])

    with (
        patch("litellm.acompletion", new=mock),
        patch("litellm.stream_chunk_builder", return_value=first),
    ):
        text = [chunk async for chunk in agent.astream("1 + 1?", session=session)]

    assert text == []
    assert mock.call_count == 1
    assert session.messages[-1]["role"] == "tool"


def test_token_budget_ends_a_naive_run():
    agent = Agent(
        config=_make_config(tool_mode="naive", max_total_tokens=30),
        tools=[_make_simple_tool()],
    )
    responses = [
        _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 1}}'),
        _make_completion_response("2"),
    ]
    mock = AsyncMock(side_effect=responses)

    with patch("litellm.acompletion", new=mock):
        result = agent.run_with_response("1 + 1?")

    assert result.stop_reason == "budget"
    assert result.steps == 1
    assert mock.call_count == 1


def test_former_token_fields_are_rejected():
    from pydantic import ValidationError

    from cyclops import RunResult

    with pytest.raises(ValidationError):
        RunResult(content="x", model="m", tokens_used=99, cost=1.5)


def test_failed_run_returns_error_result_with_usage_so_far():
    agent = Agent(config=_make_config(num_retries=0), tools=[_make_simple_tool()])
    error = litellm.APIConnectionError(
        message="down", llm_provider="openai", model="gpt-3.5-turbo"
    )
    responses = [*_tool_call_responses(1), error]

    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        result = agent.run_with_response("1 + 1?")

    assert result.stop_reason == "error"
    assert result.error is error
    assert result.steps == 1
    assert result.tokens_used == 30
    assert "error" not in result.model_dump()


def test_error_result_follows_on_run_error():
    hooks = MagicMock(spec=AgentHooks)
    agent = Agent(config=_make_config(hooks=hooks, num_retries=0))
    error = litellm.APIConnectionError(
        message="down", llm_provider="openai", model="gpt-3.5-turbo"
    )

    with patch("litellm.acompletion", new=AsyncMock(side_effect=error)):
        result = agent.run_with_response("hi")

    assert result.error is error
    hooks.on_run_error.assert_called_once_with(error)
    hooks.on_run_end.assert_not_called()


@pytest.mark.asyncio
async def test_arun_raises_the_run_error():
    import asyncio

    agent = Agent(config=_make_config(num_retries=0))
    error = litellm.APIConnectionError(
        message="down", llm_provider="openai", model="gpt-3.5-turbo"
    )

    with patch("litellm.acompletion", new=AsyncMock(side_effect=error)):
        with pytest.raises(litellm.APIConnectionError):
            await agent.arun("hi")
        with pytest.raises(litellm.APIConnectionError):
            await asyncio.to_thread(agent.run, "hi")


@pytest.mark.asyncio
async def test_cancelled_run_closes_pending_tool_calls():
    """Cancelling during tool execution leaves a result for every tool call."""
    session = Session()
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
        task = asyncio.create_task(agent.arun("go", session=session))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    tool_messages = [m for m in session.messages if m["role"] == "tool"]
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
    session = Session()
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
        chunks = [c async for c in agent.astream("2 + 3?", session=session)]

    assert chunks == ["Adding. ", "The answer ", "is 5."]
    assert mock.call_count == 2
    assert all("tools" in call.kwargs for call in mock.call_args_list)
    roles = [m["role"] for m in session.messages]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert session.messages[1]["tool_calls"][0]["function"]["arguments"] == (
        '{"a": 2, "b": 3}'
    )
    assert session.messages[2]["content"] == "5"
    assert session.messages[3]["content"] == "The answer is 5."
    assert hooks.events == [
        ("llm_end", None),
        ("llm_end", None),
        ("run_end", "The answer is 5."),
    ]


def test_stream_with_tools_sync():
    """The sync stream() wrapper drives the same loop."""
    session = Session()
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    first = [_tool_call_chunk(0, '{"a": 1, "b": 1}', id="tc_1", name="add")]
    second = [_text_chunk("2")]

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[_stream(first), _stream(second)]),
    ):
        assert list(agent.stream("1 + 1?", session=session)) == ["2"]

    assert session.messages[2]["content"] == "2"


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


def test_litellm_callbacks_fire_when_sync_and_async_runs_alternate():
    """Alternating styles moves LiteLLM's logging between loops; no callback is lost."""
    import asyncio
    import functools

    from litellm.integrations.custom_logger import CustomLogger

    events = []

    # LiteLLM skips a CustomLogger whose class name it has already seen.
    class AlternationRecorder(CustomLogger):
        async def async_log_success_event(self, *args, **kwargs):
            events.append("success")

    agent = Agent(config=_make_config())

    async def alternate():
        for _ in range(5):
            agent.run("hi")
            await agent.arun("hi")
        for _ in range(100):
            if len(events) == 10:
                break
            await asyncio.sleep(0.05)

    acompletion = functools.partial(litellm.acompletion, mock_response="hi")
    with (
        patch.object(litellm, "callbacks", [AlternationRecorder()]),
        patch("litellm.acompletion", new=acompletion),
    ):
        asyncio.run(alternate())

    assert events == ["success"] * 10


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
    session = Session()
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
        assert agent.run("go", session=session) == "ok"

    assert "Use the async Agent methods" in session.messages[2]["content"]


@pytest.mark.asyncio
async def test_tool_raising_cancelled_stops_sibling_calls():
    """A tool that raises CancelledError also stops the calls still running."""
    session = Session()
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
        tools=[
            Tool("cancels", "Cancels", cancels, READ_ONLY),
            Tool("slow", "Slow", slow, READ_ONLY),
        ],
    )
    calls = [
        _make_tool_call("tc_1", "cancels", "{}"),
        _make_tool_call("tc_2", "slow", "{}"),
    ]
    response = _make_completion_response(content=None, tool_calls=calls)

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        with pytest.raises(asyncio.CancelledError):
            await agent.arun("go", session=session)

    assert sibling_cancelled.is_set()
    tool_messages = [m for m in session.messages if m["role"] == "tool"]
    assert [m["content"] for m in tool_messages] == ["Tool call cancelled."] * 2


def test_interrupted_sync_run_cancels_the_run():
    """An interrupt while run() waits cancels the run and waits for it to unwind."""
    session = Session()
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
    # Start the shared loop first, so its thread's start-up isn't the wait
    # that gets interrupted.
    _run_coroutine_sync(asyncio.sleep, 0)

    with (
        patch("litellm.acompletion", new=AsyncMock(return_value=response)),
        patch.object(threading.Event, "wait", interrupted),
    ):
        with pytest.raises(KeyboardInterrupt):
            agent.run("go", session=session)

    assert cleaned.is_set()
    assert session.messages[-1]["content"] == "Tool call cancelled."


def test_max_steps_must_allow_a_completion():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        _make_config(max_steps=0)


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
    session = Session()
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
            next(agent.stream("go", session=session))

    assert cleaned.is_set()
    assert session.messages[-1]["content"] == "Tool call cancelled."


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

        return Tool(name, name, tool, READ_ONLY)

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
    session = Session()
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
        assert agent.run("Ask about France", session=session) == "It is Paris."

    assert session.messages[2]["content"] == "Paris"


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
    """Run one round of native tool calls and a final answer; return the transcript."""
    session = Session()
    tool_calls = [
        _make_tool_call(f"tc_{i}", name, "{}") for i, name in enumerate(calls)
    ]
    responses = [
        _make_completion_response(content=None, tool_calls=tool_calls),
        _make_completion_response(content="done"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        assert agent.run("go", session=session) == "done"
    return session.messages


def test_a_failed_run_is_freed_without_the_cycle_collector():
    """Its error keeps the message and traceback lines but holds no cycle."""

    async def fail(**kwargs):
        # Its frame keeps a copy of the run's context, as LiteLLM's frames do.
        kwargs["context"] = contextvars.copy_context()
        raise RuntimeError("boom")

    agent = Agent(config=_make_config())
    gc.disable()
    try:
        with patch("litellm.acompletion", new=fail):
            result = agent.run_with_response("hi")
        assert str(result.error) == "boom"
        assert "in fail" in "".join(traceback.format_exception(result.error))
        freed = weakref.ref(result)
        del result
        assert freed() is None
    finally:
        gc.enable()


@pytest.mark.asyncio
async def test_a_reused_error_leaves_frames_of_its_earlier_raise_alone():
    """Frames past the run's own, as of a caller that caught it, keep running."""
    import asyncio

    error = RuntimeError("down")

    async def fail(**kwargs):
        raise error

    async def caught_then_waits(release):
        try:
            raise error
        except RuntimeError:
            pass
        await release
        return "running"

    release = asyncio.get_running_loop().create_future()
    waiter = asyncio.create_task(caught_then_waits(release))
    await asyncio.sleep(0)
    with patch("litellm.acompletion", new=fail):
        result = await Agent(config=_make_config()).arun_with_response("hi")
    assert result.error is error
    await asyncio.sleep(0)
    release.set_result(None)
    assert await waiter == "running"


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
    messages = _run_tool_calls(agent, "data", "text", "fail")

    tool_messages = [m["content"] for m in messages if m["role"] == "tool"]
    assert tool_messages == ['{"rows": 2}', "two rows", "Error: no access"]
    # Parallel tools finish in any order.
    assert sorted(c.args[2] for c in hooks.on_tool_end.call_args_list) == sorted(
        tool_messages
    )


def test_raising_tool_is_marked_as_error():
    from cyclops.toolkit import tool

    @tool
    def boom() -> str:
        """Raise."""
        raise RuntimeError("disk full")

    agent = Agent(config=_make_config(), tools=[boom])
    messages = _run_tool_calls(agent, "boom")
    assert messages[2]["content"] == "Error: boom failed: disk full"


def test_tool_images_stay_in_tool_message_when_provider_accepts_them():
    agent = Agent(
        config=_make_config(model="anthropic/claude-3-5-sonnet-latest"),
        tools=[_chart_tool()],
    )
    messages = _run_tool_calls(agent, "chart")

    assert [m["role"] for m in messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert messages[2]["content"] == [
        {"type": "text", "text": "a chart"},
        _IMAGE_PART,
    ]


def test_tool_images_move_to_user_message_for_text_only_tool_messages():
    session = Session()
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
        agent.run("go", session=session)

    # The images follow every tool message of the round
    assert [m["role"] for m in session.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
        "assistant",
    ]
    assert session.messages[2]["content"] == "a chart"
    assert session.messages[3]["content"] == "3"
    assert session.messages[4]["content"] == [
        {"type": "text", "text": "Images returned by chart:"},
        _IMAGE_PART,
    ]
    sent = mock.call_args_list[1].kwargs["messages"]
    assert sent[-1] == session.messages[4]


def test_tool_images_in_naive_mode():
    session = Session()
    agent = Agent(config=_make_config(tool_mode="naive"), tools=[_chart_tool()])
    responses = [
        _make_completion_response('{"tool": "chart", "args": {}}'),
        _make_completion_response("done"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        agent.run("go", session=session)

    assert session.messages[2] == {
        "role": "user",
        "content": [{"type": "text", "text": "Tool result: a chart"}, _IMAGE_PART],
    }


def test_streamed_run_renders_tool_results():
    session = Session()
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
        assert list(agent.stream("go", session=session)) == ["sorry"]

    assert session.messages[2]["content"] == "Error: no access"


def test_denied_and_unknown_tools_are_reported_as_errors():
    class Deny(AgentHooks):
        def on_tool_start(self, tool_name, args):
            return "deny"

    agent = Agent(config=_make_config(hooks=Deny()), tools=[_make_simple_tool()])
    messages = _run_tool_calls(agent, "add", "missing")

    tool_messages = [m["content"] for m in messages if m["role"] == "tool"]
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
    session = Session()
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
    agent.run("go", session=session)

    roles = [m["role"] for m in session.messages]
    if kept:
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert session.messages[2]["content"][1] == _IMAGE_PART
    else:
        assert roles == ["user", "assistant", "tool", "user", "assistant"]
        assert session.messages[2]["content"] == "a chart"


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _event_summary(events):
    from cyclops.core.events import (
        ReasoningDelta,
        RunFinished,
        StepFinished,
        TextDelta,
        ToolCallFinished,
        ToolCallStarted,
    )

    summary = []
    for e in events:
        assert e.agent_path == ()
        if isinstance(e, TextDelta):
            summary.append(("text", e.text))
        elif isinstance(e, ReasoningDelta):
            summary.append(("reasoning", e.text))
        elif isinstance(e, ToolCallStarted):
            summary.append(("tool_start", e.call_id, e.name, e.arguments))
        elif isinstance(e, ToolCallFinished):
            summary.append(("tool_end", e.call_id, e.result, e.is_error))
        elif isinstance(e, StepFinished):
            summary.append(("step", e.step))
        elif isinstance(e, RunFinished):
            summary.append(("run", e.result.stop_reason, e.result.content))
    return summary


@pytest.mark.asyncio
async def test_astream_events_reports_text_tools_steps_and_result():
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    first = [
        ModelResponseStream(
            model="gpt-3.5-turbo",
            choices=[StreamingChoices(delta=Delta(reasoning_content="Sum it."))],
        ),
        _text_chunk("Adding. "),
        _tool_call_chunk(0, '{"a": 2, "b": 3}', id="tc_1", name="add"),
    ]
    second = [_text_chunk("The answer "), _text_chunk("is 5.")]
    mock = AsyncMock(side_effect=[_stream(first), _stream(second)])

    with patch("litellm.acompletion", new=mock):
        events = [e async for e in agent.astream_events("2 + 3?")]

    assert _event_summary(events) == [
        ("reasoning", "Sum it."),
        ("text", "Adding. "),
        ("tool_start", "tc_1", "add", {"a": 2, "b": 3}),
        ("tool_end", "tc_1", "5", False),
        ("step", 1),
        ("text", "The answer "),
        ("text", "is 5."),
        ("step", 2),
        ("run", "completed", "The answer is 5."),
    ]
    assert events[-1].result.steps == 2


def test_stream_events_carries_usage_of_every_step():
    """A streamed run reports its totals through RunFinished and StepFinished."""
    from cyclops.core.events import RunFinished, StepFinished

    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    tc = _make_tool_call("tc_1", "add", '{"a": 1, "b": 1}')
    built = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response(content="2"),
    ]
    mock = AsyncMock(side_effect=[_stream([]), _stream([_text_chunk("2")])])

    with (
        patch("litellm.acompletion", new=mock),
        patch("litellm.stream_chunk_builder", side_effect=built),
        patch("litellm.completion_cost", return_value=0.001),
    ):
        events = list(agent.stream_events("1 + 1?"))

    steps = [e for e in events if isinstance(e, StepFinished)]
    assert [s.usage.total_tokens for s in steps] == [30, 30]
    result = events[-1]
    assert isinstance(result, RunFinished)
    assert result.result.tokens_used == 60
    assert result.result.cost == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_failed_run_ends_events_with_error_and_astream_raises():
    agent = Agent(config=_make_config(num_retries=0))
    error = litellm.APIConnectionError(
        message="down", llm_provider="openai", model="gpt-3.5-turbo"
    )

    with patch("litellm.acompletion", new=AsyncMock(side_effect=error)):
        events = [e async for e in agent.astream_events("hi")]
        with pytest.raises(litellm.APIConnectionError):
            async for _ in agent.astream("hi"):
                pass

    assert _event_summary(events) == [("run", "error", "")]
    assert events[-1].result.error is error


@pytest.mark.asyncio
async def test_invalid_arguments_and_naive_calls_have_tool_events():
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    responses = [
        _stream([_tool_call_chunk(0, "{not json", id="tc_1", name="add")]),
        _stream([_text_chunk("sorry")]),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        events = [e async for e in agent.astream_events("go")]
    start, end = _event_summary(events)[:2]
    assert start == ("tool_start", "tc_1", "add", {})
    assert end[0] == "tool_end" and end[3] is True

    naive = Agent(config=_make_config(tool_mode="naive"), tools=[_make_simple_tool()])
    responses = [
        _make_completion_response('{"tool": "add", "args": {"a": 1, "b": 2}}'),
        _make_completion_response("3"),
    ]
    with patch("litellm.acompletion", new=AsyncMock(side_effect=responses)):
        events = [e async for e in naive.astream_events("1 + 2?")]
    assert _event_summary(events) == [
        ("tool_start", None, "add", {"a": 1, "b": 2}),
        ("tool_end", None, "3", False),
        ("step", 1),
        ("text", "3"),
        ("step", 2),
        ("run", "completed", "3"),
    ]


@pytest.mark.asyncio
async def test_closing_events_early_cancels_the_run():
    session = Session()
    import asyncio

    from cyclops.core.events import ToolCallStarted
    from cyclops.toolkit.tool import Tool

    async def slow() -> str:
        await asyncio.sleep(10)
        return "done"

    agent = Agent(config=_make_config(), tools=[Tool("slow", "Slow tool", slow)])
    chunks = [_tool_call_chunk(0, "{}", id="tc_1", name="slow")]

    with patch("litellm.acompletion", new=AsyncMock(return_value=_stream(chunks))):
        events = agent.astream_events("go", session=session)
        async for event in events:
            if isinstance(event, ToolCallStarted):
                break
        await events.aclose()

    assert session.messages[-1] == {
        "role": "tool",
        "tool_call_id": "tc_1",
        "content": "Tool call cancelled.",
        "name": "slow",
    }


@pytest.mark.asyncio
async def test_run_ending_without_run_finished_raises_from_the_stream():
    """A tool raising CancelledError ends the stream with it instead of hanging."""
    import asyncio

    from cyclops.core.events import ToolCallFinished, ToolCallStarted
    from cyclops.toolkit.tool import Tool

    async def bad() -> str:
        await asyncio.sleep(0)
        raise asyncio.CancelledError()

    async def slow() -> str:
        await asyncio.sleep(10)
        return "done"

    agent = Agent(
        config=_make_config(),
        tools=[
            Tool("bad", "Bad tool", bad, READ_ONLY),
            Tool("slow", "Slow tool", slow, READ_ONLY),
        ],
    )
    chunks = [
        _tool_call_chunk(0, "{}", id="tc_1", name="bad"),
        _tool_call_chunk(1, "{}", id="tc_2", name="slow"),
    ]
    events = []

    async def consume():
        async for event in agent.astream_events("go"):
            events.append(event)

    with patch("litellm.acompletion", new=AsyncMock(return_value=_stream(chunks))):
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(consume(), 5)

    started = {e.call_id for e in events if isinstance(e, ToolCallStarted)}
    finished = {
        e.call_id: (e.result, e.is_error)
        for e in events
        if isinstance(e, ToolCallFinished)
    }
    assert started == {"tc_1", "tc_2"}
    assert finished == {
        "tc_1": ("Tool call cancelled.", True),
        "tc_2": ("Tool call cancelled.", True),
    }


@pytest.mark.asyncio
async def test_dropped_event_stream_leaves_no_thread_or_task():
    """A stream dropped without aclose() is closed by the loop and cleans up."""
    import asyncio
    import functools
    import gc
    import threading

    from cyclops.core.events import TextDelta

    def workers():
        return {t for t in threading.enumerate() if t.name == "AnyIO worker thread"}

    before = workers()
    acompletion = functools.partial(litellm.acompletion, mock_response="a b c d")
    with patch("litellm.acompletion", new=acompletion):
        for _ in range(20):
            events = Agent(config=_make_config()).astream_events("hi")
            async for event in events:
                if isinstance(event, TextDelta):
                    break
            del events
            gc.collect()
            for _ in range(10):
                await asyncio.sleep(0)

    assert asyncio.all_tasks() == {asyncio.current_task()}
    assert not workers() - before


@pytest.mark.asyncio
async def test_run_waits_for_the_stream_consumer():
    """A consumer that stops reading without closing holds the run where it is."""
    import asyncio

    from cyclops.toolkit.tool import Tool

    calls = []

    async def record() -> str:
        calls.append(1)
        return "done"

    agent = Agent(config=_make_config(), tools=[Tool("record", "Records", record)])
    chunks = [
        _text_chunk("Calling."),
        _tool_call_chunk(0, "{}", id="tc_1", name="record"),
    ]

    with patch("litellm.acompletion", new=AsyncMock(return_value=_stream(chunks))):
        events = agent.astream_events("go")
        await events.__anext__()
        await asyncio.sleep(0.05)
        assert calls == []
        await events.aclose()

    assert calls == []


# ---------------------------------------------------------------------------
# Run handle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_is_awaitable_and_iterable():
    from cyclops import RunHandle

    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    first = [_tool_call_chunk(0, '{"a": 2, "b": 3}', id="tc_1", name="add")]
    second = [_text_chunk("5")]
    mock = AsyncMock(side_effect=[_stream(first), _stream(second)])

    with patch("litellm.acompletion", new=mock):
        handle = agent.start("2 + 3?")
        assert isinstance(handle, RunHandle)
        events = [e async for e in handle]
        result = await handle

    assert events[-1].result is result
    assert result.content == "5"
    assert _event_summary(events)[:2] == [
        ("tool_start", "tc_1", "add", {"a": 2, "b": 3}),
        ("tool_end", "tc_1", "5", False),
    ]


@pytest.mark.asyncio
async def test_steer_adds_a_user_message_after_the_tool_calls():
    import asyncio

    from cyclops.toolkit.tool import Tool

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow() -> str:
        started.set()
        await release.wait()
        return "done"

    agent = Agent(config=_make_config(), tools=[Tool("slow", "Slow tool", slow)])
    tc = _make_tool_call("tc_1", "slow", "{}")
    responses = [
        _make_completion_response(content=None, tool_calls=[tc]),
        _make_completion_response("ok, in French"),
    ]
    mock = AsyncMock(side_effect=responses)

    with patch("litellm.acompletion", new=mock):
        handle = agent.start("go", stream=False)
        await started.wait()
        handle.steer("Answer in French.")
        release.set()
        result = await handle

    assert result.content == "ok, in French"
    sent = mock.call_args_list[1].kwargs["messages"]
    assert [m["role"] for m in sent] == ["user", "assistant", "tool", "user"]
    assert sent[-1]["content"] == "Answer in French."


@pytest.mark.asyncio
async def test_steer_during_the_final_answer_gets_its_own_step():
    session = Session()
    import asyncio

    agent = Agent(config=_make_config())
    asked = asyncio.Event()
    answers = iter(["Hi.", "Bonjour."])

    async def complete(**kwargs):
        asked.set()
        await asyncio.sleep(0)
        return _make_completion_response(next(answers))

    with patch("litellm.acompletion", new=complete):
        handle = agent.start("hello", stream=False, session=session)
        await asked.wait()
        handle.steer("In French, please.")
        result = await handle

    assert result.content == "Bonjour."
    assert result.steps == 2
    assert [m["content"] for m in session.messages] == [
        "hello",
        "Hi.",
        "In French, please.",
        "Bonjour.",
    ]
    with pytest.raises(RuntimeError):
        handle.steer("too late")


@pytest.mark.asyncio
async def test_cancel_closes_tool_calls_and_resolves_cancelled():
    """A call cut short gets a cancelled result; a finished one keeps its own."""
    session = Session()
    import asyncio

    from cyclops.core.hooks import RUN_STOPPED
    from cyclops.toolkit.tool import Tool

    ran = []
    handles = []

    async def one() -> str:
        ran.append("one")
        handles[0].cancel()
        await asyncio.sleep(10)
        return "done"

    async def two() -> str:
        ran.append("two")
        return "done"

    hooks = MagicMock(spec=AgentHooks)
    hooks.on_tool_start.return_value = "allow"
    agent = Agent(
        config=_make_config(hooks=hooks),
        tools=[Tool("one", "One", one, READ_ONLY), Tool("two", "Two", two, READ_ONLY)],
    )
    calls = [_make_tool_call("tc_1", "one", "{}"), _make_tool_call("tc_2", "two", "{}")]
    response = _make_completion_response(content=None, tool_calls=calls)

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        handles.append(agent.start("go", stream=False, session=session))
        events = [e async for e in handles[0]]
        result = await handles[0]

    assert ran == ["one", "two"]
    assert result.stop_reason == "cancelled"
    assert _event_summary(events) == [
        ("tool_start", "tc_1", "one", {}),
        ("tool_start", "tc_2", "two", {}),
        ("tool_end", "tc_2", "done", False),
        ("tool_end", "tc_1", "Tool call cancelled.", True),
        ("run", "cancelled", ""),
    ]
    assert [m["content"] for m in session.messages if m["role"] == "tool"] == [
        "Tool call cancelled.",
        "done",
    ]
    (error,) = hooks.on_run_error.call_args.args
    assert error.args == (RUN_STOPPED,)


@pytest.mark.asyncio
async def test_cancel_before_the_run_starts():
    agent = Agent(config=_make_config())
    mock = AsyncMock(return_value=_make_completion_response("hi"))

    with patch("litellm.acompletion", new=mock):
        handle = agent.start("hello")
        handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"
    assert result.steps == 0
    mock.assert_not_called()


@pytest.mark.asyncio
async def test_cancel_during_a_streamed_completion():
    session = Session()
    import asyncio

    from cyclops.core.events import TextDelta

    async def stalled():
        yield _text_chunk("Hel")
        await asyncio.sleep(10)
        yield _text_chunk("lo")

    agent = Agent(config=_make_config())
    with patch("litellm.acompletion", new=AsyncMock(return_value=stalled())):
        handle = agent.start("hello", session=session)
        async for event in handle:
            if isinstance(event, TextDelta):
                handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"
    assert result.content == ""
    assert session.messages == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_handle_iterates_once():
    agent = Agent(config=_make_config())
    with patch(
        "litellm.acompletion",
        new=AsyncMock(return_value=_make_completion_response("hi")),
    ):
        handle = agent.start("hello", stream=False)
        assert [type(e).__name__ for e in [e async for e in handle]][-1] == (
            "RunFinished"
        )
        with pytest.raises(RuntimeError):
            handle.__aiter__()
        await handle


@pytest.mark.asyncio
async def test_handle_iterated_from_the_start_gets_every_delta():
    agent = Agent(config=_make_config())
    chunks = [_text_chunk(t) for t in "abc"]
    with patch("litellm.acompletion", new=AsyncMock(return_value=_stream(chunks))):
        handle = agent.start("hi")
        events = [e async for e in handle]
    assert _event_summary(events) == [
        ("text", "a"),
        ("text", "b"),
        ("text", "c"),
        ("step", 1),
        ("run", "completed", "abc"),
    ]


@pytest.mark.asyncio
async def test_handle_awaited_before_iterating_keeps_one_delta_per_step():
    from cyclops.core.events import ReasoningDelta, TextDelta

    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    first = [
        _text_chunk("Let"),
        _text_chunk(" me add."),
        _tool_call_chunk(0, '{"a": 2, "b": 3}', id="tc_1", name="add"),
    ]
    second = [_text_chunk(t) for t in ["It", " is", " 5."]]
    mock = AsyncMock(side_effect=[_stream(first), _stream(second)])
    with patch("litellm.acompletion", new=mock):
        handle = agent.start("2 + 3?")
        await handle
        # Seven events and the finished task.
        assert handle._run.events.qsize() == 8
        events = [e async for e in handle]
    assert _event_summary(events) == [
        ("text", "Let me add."),
        ("tool_start", "tc_1", "add", {"a": 2, "b": 3}),
        ("tool_end", "tc_1", "5", False),
        ("step", 1),
        ("text", "It is 5."),
        ("step", 2),
        ("run", "completed", "It is 5."),
    ]

    # Only deltas of one kind merge.
    from cyclops.core.run_state import _Events

    queue = _Events()
    for event in [
        TextDelta(text="a"),
        TextDelta(text="b"),
        ReasoningDelta(text="c"),
        ReasoningDelta(text="d"),
        TextDelta(text="e"),
    ]:
        queue.put_nowait(event)
    merged = [queue.get_nowait() for _ in range(queue.qsize())]
    assert [(type(e).__name__, e.text) for e in merged] == [
        ("TextDelta", "ab"),
        ("ReasoningDelta", "cd"),
        ("TextDelta", "e"),
    ]


@pytest.mark.asyncio
async def test_paced_run_reads_a_bounded_number_of_chunks_ahead():
    """A consumer that stops reading holds the stream within _READ_AHEAD chunks."""
    import asyncio

    from cyclops.core.completion import _READ_AHEAD

    read = []

    async def completion_stream():
        for i in range(100):
            read.append(i)
            yield _text_chunk(str(i))

    agent = Agent(config=_make_config())
    with patch("litellm.acompletion", new=AsyncMock(return_value=completion_stream())):
        events = agent.astream_events("go")
        await events.__anext__()
        await asyncio.sleep(0.05)
        assert len(read) == _READ_AHEAD + 1
        texts = [e.text async for e in events if hasattr(e, "text")]
    assert texts == [str(i) for i in range(1, 100)]


def test_sync_events_caller_that_stops_reading_holds_the_run():
    """A sync caller holding the iterator after the first event stops tools from running."""
    import time

    from cyclops.toolkit.tool import Tool

    calls = []

    def record() -> str:
        calls.append(1)
        return "done"

    agent = Agent(config=_make_config(), tools=[Tool("record", "Records", record)])
    chunks = [
        _text_chunk("Calling."),
        _tool_call_chunk(0, "{}", id="tc_1", name="record"),
    ]

    with patch("litellm.acompletion", new=AsyncMock(return_value=_stream(chunks))):
        events = agent.stream_events("go")
        next(events)
        time.sleep(0.05)
        assert calls == []
        events.close()

    assert calls == []


@pytest.mark.asyncio
async def test_second_cancel_leaves_the_tool_cleanup_alone():
    session = Session()
    import asyncio

    from cyclops.toolkit.tool import Tool

    started = asyncio.Event()
    cleaned = []

    async def slow() -> str:
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            await asyncio.sleep(0.01)
            cleaned.append(1)
            raise
        return "done"

    agent = Agent(config=_make_config(), tools=[Tool("slow", "Slow tool", slow)])
    tc = _make_tool_call("tc_1", "slow", "{}")
    response = _make_completion_response(content=None, tool_calls=[tc])

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        handle = agent.start("go", stream=False, session=session)
        await started.wait()
        handle.cancel()
        await asyncio.sleep(0)
        handle.cancel()
        result = await handle

    assert result.stop_reason == "cancelled"
    assert cleaned == [1]
    assert session.messages[-1]["content"] == "Tool call cancelled."


@pytest.mark.asyncio
async def test_steer_before_the_first_completion_is_in_it():
    agent = Agent(config=_make_config())
    mock = AsyncMock(return_value=_make_completion_response("Bonjour."))

    with patch("litellm.acompletion", new=mock):
        handle = agent.start("hello", stream=False)
        handle.steer("In French, please.")
        result = await handle

    assert result.steps == 1
    sent = mock.call_args.kwargs["messages"]
    assert [m["content"] for m in sent] == ["hello", "In French, please."]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits", [{"max_steps": 1}, {"max_steps": 5, "max_total_tokens": 30}]
)
async def test_steer_without_a_step_left_stays_for_the_next_run(limits):
    session = Session()
    import asyncio

    agent = Agent(config=_make_config(**limits))
    asked = asyncio.Event()

    async def complete(**kwargs):
        asked.set()
        await asyncio.sleep(0)
        return _make_completion_response("Hi.")

    with patch("litellm.acompletion", new=complete):
        handle = agent.start("hello", stream=False, session=session)
        await asked.wait()
        handle.steer("In French, please.")
        result = await handle

    assert result.stop_reason == "completed"
    assert result.content == "Hi."
    assert session.messages[-1] == {"role": "user", "content": "In French, please."}


@pytest.mark.asyncio
async def test_guards_judge_the_answer_after_a_steer():
    import asyncio

    judged = []

    async def guard(context):
        judged.append(context.content)
        return None

    agent = Agent(config=_make_config(guards=[guard]))
    asked = asyncio.Event()
    answers = iter(["Hi.", "Bonjour."])

    async def complete(**kwargs):
        asked.set()
        await asyncio.sleep(0)
        return _make_completion_response(next(answers))

    with patch("litellm.acompletion", new=complete):
        handle = agent.start("hello", stream=False)
        await asked.wait()
        handle.steer("In French, please.")
        result = await handle

    assert result.content == "Bonjour."
    assert judged == ["Bonjour."]


@pytest.mark.asyncio
async def test_cancel_during_tool_calls_logs_nothing():
    import asyncio

    from cyclops.toolkit.tool import Tool

    started = asyncio.Event()

    async def slow() -> str:
        started.set()
        await asyncio.sleep(10)
        return "done"

    agent = Agent(
        config=_make_config(),
        tools=[Tool("slow", "Slow tool", slow), Tool("other", "Other", slow)],
    )
    calls = [
        _make_tool_call("tc_1", "slow", "{}"),
        _make_tool_call("tc_2", "other", "{}"),
    ]
    response = _make_completion_response(content=None, tool_calls=calls)
    loop = asyncio.get_running_loop()
    reported = []
    loop.set_exception_handler(lambda _, context: reported.append(context))

    try:
        with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
            handle = agent.start("go", stream=False)
            await started.wait()
            handle.cancel()
            result = await handle
        # Let any callbacks the cancellation scheduled run.
        for _ in range(3):
            await asyncio.sleep(0)
    finally:
        loop.set_exception_handler(None)

    assert result.stop_reason == "cancelled"
    assert reported == []


def test_first_failing_tool_call_ends_the_step():
    import asyncio

    from cyclops.core.middleware import ToolMiddleware
    from cyclops.toolkit.tool import Tool

    class Broken(ToolMiddleware):
        async def before_tool(self, call):
            # The second call fails first.
            await asyncio.sleep(0.02 if call.tool_name == "one" else 0)
            raise RuntimeError(f"{call.tool_name} broke")

    async def tool() -> str:
        return "done"

    agent = Agent(
        config=_make_config(middleware=[Broken()]),
        tools=[
            Tool("one", "One", tool, READ_ONLY),
            Tool("two", "Two", tool, READ_ONLY),
        ],
    )
    calls = [_make_tool_call("tc_1", "one", "{}"), _make_tool_call("tc_2", "two", "{}")]
    response = _make_completion_response(content=None, tool_calls=calls)

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = agent.run_with_response("go")

    assert str(result.error) == "two broke"


# ---------------------------------------------------------------------------
# Finish reasons
# ---------------------------------------------------------------------------


def _finish_chunk(finish_reason: str):
    return ModelResponseStream(
        model="gpt-3.5-turbo",
        choices=[StreamingChoices(delta=Delta(), finish_reason=finish_reason)],
    )


def test_finish_reason_is_recorded_on_each_step_and_the_result():
    from cyclops.core.events import StepFinished

    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    first = [
        _tool_call_chunk(0, '{"a": 1, "b": 2}', id="tc_1", name="add"),
        _finish_chunk("tool_calls"),
    ]
    second = [_text_chunk("3"), _finish_chunk("stop")]
    mock = AsyncMock(side_effect=[_stream(first), _stream(second)])

    with patch("litellm.acompletion", new=mock):
        events = list(agent.stream_events("1 + 2?"))

    steps = [e for e in events if isinstance(e, StepFinished)]
    assert [s.finish_reason for s in steps] == ["tool_calls", "stop"]
    result = events[-1].result
    assert (result.stop_reason, result.finish_reason) == ("completed", "stop")


def test_text_cut_by_the_length_limit_ends_with_max_tokens():
    hooks = _RecordingHooks()
    agent = Agent(config=_make_config(hooks=hooks))
    session = Session()
    response = _make_completion_response("The answer is", finish_reason="length")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = agent.run_with_response("Explain.", session=session)

    assert result.stop_reason == "max_tokens"
    assert result.finish_reason == "length"
    assert result.content == ""
    assert session.messages[-1] == {"role": "assistant", "content": "The answer is"}


def test_guards_do_not_judge_a_cut_answer():
    judged = []

    async def guard(context):
        judged.append(context.content)
        return "Try again."

    agent = Agent(config=_make_config(guards=[guard]))
    response = _make_completion_response("The answer is", finish_reason="length")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = agent.run_with_response("Explain.")

    assert result.stop_reason == "max_tokens"
    assert judged == []


def test_a_tool_call_cut_by_the_length_limit_is_not_run():
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    call = _make_tool_call("tc_1", "add", '{"a": 1, "b')
    response = _make_completion_response(
        None, tool_calls=[call], finish_reason="length"
    )
    mock = AsyncMock(return_value=response)
    session = Session()

    with patch("litellm.acompletion", new=mock):
        result = agent.run_with_response("1 + 2?", session=session)

    assert result.stop_reason == "max_tokens"
    assert result.error is None
    assert result.tool_calls == []
    assert mock.call_count == 1
    # An empty cut turn is left out, so the conversation needs no tool result.
    assert session.messages == [{"role": "user", "content": "1 + 2?"}]


def test_a_complete_call_before_a_cut_one_is_not_run_either():
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    calls = [
        _make_tool_call("tc_1", "add", '{"a": 1, "b": 2}'),
        _make_tool_call("tc_2", "add", '{"a": 3, "b'),
    ]
    response = _make_completion_response(
        "Adding.", tool_calls=calls, finish_reason="length"
    )

    session = Session()

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        result = agent.run_with_response("1 + 2 and 3 + 4?", session=session)

    assert result.stop_reason == "max_tokens"
    assert result.tool_calls == []
    assert session.messages[-1] == {"role": "assistant", "content": "Adding."}


@pytest.mark.asyncio
async def test_a_streamed_tool_call_cut_by_the_length_limit_is_not_run():
    agent = Agent(config=_make_config(), tools=[_make_simple_tool()])
    chunks = [
        _text_chunk("Adding. "),
        _tool_call_chunk(0, '{"a": 1, "b', id="tc_1", name="add"),
        _finish_chunk("length"),
    ]
    session = Session()

    with patch("litellm.acompletion", new=AsyncMock(return_value=_stream(chunks))):
        events = [e async for e in agent.astream_events("1 + 2?", session=session)]

    assert _event_summary(events) == [
        ("text", "Adding. "),
        ("step", 1),
        ("run", "max_tokens", ""),
    ]
    assert events[-1].result.finish_reason == "length"
    assert session.messages[-1] == {"role": "assistant", "content": "Adding. "}


def test_a_naive_tool_call_cut_by_the_length_limit_ends_with_max_tokens():
    agent = Agent(config=_make_config(tool_mode="naive"), tools=[_make_simple_tool()])
    response = _make_completion_response(
        '{"tool": "add", "args": {"a": 1', finish_reason="length"
    )

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        events = list(agent.stream_events("1 + 2?"))

    # The cut text may be a call, so it is not shown as answer text.
    assert _event_summary(events) == [("step", 1), ("run", "max_tokens", "")]
    assert events[-1].result.finish_reason == "length"


@pytest.mark.parametrize("stream", [False, True])
def test_filtered_output_ends_with_content_filter(stream):
    agent = Agent(config=_make_config())
    if stream:
        response = _stream([_text_chunk("I can"), _finish_chunk("content_filter")])
    else:
        response = _make_completion_response("I can", finish_reason="content_filter")

    with patch("litellm.acompletion", new=AsyncMock(return_value=response)):
        if stream:
            result = list(agent.stream_events("Do it."))[-1].result
        else:
            result = agent.run_with_response("Do it.")

    assert result.stop_reason == "content_filter"
    assert result.finish_reason == "content_filter"
    assert result.content == ""
