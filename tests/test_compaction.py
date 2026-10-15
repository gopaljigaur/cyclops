"""Tests for compaction and the tool output limit"""

from base64 import b64encode
from typing import List
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from pydantic import ValidationError

from cyclops import (
    Agent,
    AgentConfig,
    AgentHooks,
    Compaction,
    InMemorySessionStore,
    Session,
    ToolMiddleware,
    ToolResult,
    tool,
)
from cyclops.toolkit.tool import Tool
from cyclops.toolkit.types import ImageContent, TextContent
from cyclops.core.compaction import _SUMMARY_PREFIX, _window
from cyclops.core.events import Compacted, StepFinished, ToolCallFinished

# A model LiteLLM knows with a 400 token input limit, so a few short messages
# reach the default threshold of 320 tokens and keep 80.
MODEL = "cyclops-tiny-window"
litellm.register_model(
    {
        MODEL: {
            "max_input_tokens": 400,
            "max_output_tokens": 100,
            "litellm_provider": "openai",
            "mode": "chat",
        }
    }
)


def _words(n: int) -> str:
    """About n tokens of text."""
    return "word " * n


def _replies(*replies):
    """Patch litellm.acompletion to answer each call with the next reply.

    A reply is the text of a mock_response, a dict of mock arguments, or an
    exception to raise.
    """
    real = litellm.acompletion
    queue = list(replies)

    async def complete(**kwargs):
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            reply = {"mock_response": reply}
        return await real(**kwargs, **reply)

    return patch("litellm.acompletion", new=AsyncMock(side_effect=complete))


def _call(name: str, arguments: str = "{}", id: str = "call_1"):
    return {
        "mock_response": "",
        "mock_tool_calls": [
            {
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ],
    }


def _long_session() -> Session:
    """A session of about 440 tokens, each message past what compaction keeps."""
    messages = []
    for _ in range(3):
        messages.append({"role": "user", "content": _words(70)})
        messages.append({"role": "assistant", "content": _words(70)})
    return Session(messages=messages)


def _is_summary_request(kwargs) -> bool:
    messages = kwargs["messages"]
    return len(messages) == 1 and "<conversation>" in messages[0]["content"]


class _Recorder(AgentHooks):
    def __init__(self) -> None:
        self.compactions: List[tuple] = []

    def on_compaction(self, summary, tokens_before, tokens_after):
        self.compactions.append((summary, tokens_before, tokens_after))


def test_a_request_past_the_threshold_is_compacted_first():
    hooks = _Recorder()
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction(), hooks=hooks))
    session = _long_session()

    with _replies("They talked about words.", "Done.") as mock:
        events = list(agent.stream_events("Go on.", session=session))

    summary_call, step_call = mock.call_args_list
    assert _is_summary_request(summary_call.kwargs)
    # The summary is written from the replaced messages only.
    assert "Go on." not in summary_call.kwargs["messages"][0]["content"]
    sent = step_call.kwargs["messages"]
    assert sent[0]["content"] == _SUMMARY_PREFIX + "They talked about words."
    assert sent[-1] == {"role": "user", "content": "Go on."}

    compacted = [e for e in events if isinstance(e, Compacted)]
    assert len(compacted) == 1
    event = compacted[0]
    assert event.summary == "They talked about words."
    assert event.replaced_messages == 6
    assert event.tokens_after < 320 <= event.tokens_before
    assert hooks.compactions == [
        (event.summary, event.tokens_before, event.tokens_after)
    ]
    # The summary's completion counts in the run's usage but is not a step.
    step = next(e for e in events if isinstance(e, StepFinished))
    result = events[-1].result
    assert result.steps == 1
    assert result.usage.total_tokens == step.usage.total_tokens + 30


def test_the_summary_stays_in_the_saved_session():
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction()))
    session = _long_session()

    with _replies("Earlier: words.", "Done."):
        agent.run("Go on.", session=session)

    saved = Session.model_validate_json(session.model_dump_json())
    assert [m["role"] for m in saved.messages] == ["user", "user", "assistant"]
    assert saved.messages[0]["content"] == _SUMMARY_PREFIX + "Earlier: words."
    assert saved.usage.total_tokens == 60

    with _replies("Again.") as mock:
        agent.run("And now?", session=saved)
    assert mock.call_count == 1
    assert mock.call_args.kwargs["messages"][0] == saved.messages[0]


def test_a_request_below_the_threshold_is_sent_as_it_is():
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction()))
    session = Session(messages=[{"role": "user", "content": _words(20)}])

    with _replies("Done.") as mock:
        agent.run("Go on.", session=session)

    assert mock.call_count == 1
    assert len(session.messages) == 3


def _audio(seconds: int) -> dict:
    """An input_audio part of about 32 tokens per second."""
    data = b64encode(b"\0" * 16000 * seconds).decode()
    return {"type": "input_audio", "input_audio": {"data": data, "format": "mp3"}}


def test_the_text_next_to_audio_is_counted():
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction()))
    text = {"type": "text", "text": _words(330)}
    session = Session(messages=[{"role": "user", "content": [text, _audio(1)]}])

    with _replies("They sent a voice note.", "Done.") as mock:
        agent.run("Go on.", session=session)

    assert _is_summary_request(mock.call_args_list[0].kwargs)


def test_audio_is_counted_by_its_size():
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction()))
    text = {"type": "text", "text": "Listen."}
    session = Session(messages=[{"role": "user", "content": [text, _audio(12)]}])
    small = Session(messages=[{"role": "user", "content": [text, _audio(1)]}])

    with _replies("A short note.", "Done.", "Done.") as mock:
        agent.run("Go on.", session=session)
        agent.run("Go on.", session=small)

    summary, *steps = mock.call_args_list
    assert _is_summary_request(summary.kwargs)
    assert not any(_is_summary_request(c.kwargs) for c in steps)


def test_without_compaction_nothing_is_summarized():
    agent = Agent(AgentConfig(model=MODEL))
    session = _long_session()

    with _replies("Done.") as mock:
        agent.run("Go on.", session=session)

    assert mock.call_count == 1
    assert len(session.messages) == 8


def test_a_tool_call_and_its_results_stay_together():
    @tool
    def look(page: int) -> str:
        """Read a page."""
        return _words(30)

    agent = Agent(AgentConfig(model=MODEL), tools=[look])
    session = Session(
        messages=[
            {"role": "user", "content": _words(120)},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": "look", "arguments": "{}"},
                    }
                    for i in (1, 2)
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "look",
                "content": _words(60),
            },
            {
                "role": "tool",
                "tool_call_id": "call_2",
                "name": "look",
                "content": _words(60),
            },
            {"role": "assistant", "content": "Both read."},
        ]
    )

    with _replies("Read two pages.") as mock:
        assert agent.compact(session)

    # Keeping 80 tokens reaches into the second tool result, so the cut moves
    # back to the call and keeps both results with it.
    assert [m["role"] for m in session.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assert session.messages[0]["content"] == _SUMMARY_PREFIX + "Read two pages."
    transcript = mock.call_args.kwargs["messages"][0]["content"]
    assert "called look" not in transcript
    assert "tool result" not in transcript


def test_manual_compaction_summarizes_a_session_on_demand():
    hooks = _Recorder()
    agent = Agent(AgentConfig(model=MODEL, hooks=hooks))
    session = _long_session()
    session.messages.append({"role": "user", "content": "Latest."})

    with _replies("Summary.") as mock:
        assert agent.compact(session)

    assert mock.call_args.kwargs["model"] == MODEL
    assert session.messages[0]["content"] == _SUMMARY_PREFIX + "Summary."
    assert session.messages[-1] == {"role": "user", "content": "Latest."}
    assert session.usage.total_tokens == 30
    assert len(hooks.compactions) == 1


def test_manual_compaction_of_kept_messages_changes_nothing():
    agent = Agent(AgentConfig(model=MODEL))
    session = Session(messages=[{"role": "user", "content": "Hi."}])

    with _replies() as mock:
        assert not agent.compact(session)

    mock.assert_not_called()
    assert session.messages == [{"role": "user", "content": "Hi."}]


def test_compaction_needs_known_model_limits():
    agent = Agent(AgentConfig(model="openai/cyclops-unmapped", compaction=Compaction()))
    session = _long_session()

    with _replies() as mock:
        result = agent.run_with_response("Go on.", session=session)
        with pytest.raises(ValueError, match="register_model"):
            agent.compact(session)

    mock.assert_not_called()
    assert isinstance(result.error, ValueError)


def test_the_summary_model_can_differ():
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction(model="gpt-4o-mini")))

    with _replies("Summary.", "Done.") as mock:
        agent.run("Go on.", session=_long_session())

    assert [c.kwargs["model"] for c in mock.call_args_list] == ["gpt-4o-mini", MODEL]


def test_a_router_uses_its_smallest_deployment_window():
    def router(*models):
        return litellm.Router(
            model_list=[
                {"model_name": "smart", "litellm_params": {"model": m}} for m in models
            ]
        )

    config = AgentConfig(model="smart", router=router("gpt-4o-mini", MODEL))
    assert _window(config) == (MODEL, 400, 100)
    unmapped = router("gpt-4o-mini", "openai/cyclops-unmapped")
    assert _window(AgentConfig(model="smart", router=unmapped)) is None


def test_the_summary_counts_toward_the_budget():
    @tool
    def look() -> str:
        """Look around."""
        return "Nothing."

    agent = Agent(
        AgentConfig(model=MODEL, compaction=Compaction(), max_total_tokens=50),
        tools=[look],
    )

    # The summary and the first step use 60 tokens, so no second step starts.
    with _replies("Summary.", _call("look")) as mock:
        result = agent.run_with_response("Go on.", session=_long_session())

    assert mock.call_count == 2
    assert result.stop_reason == "budget"


def test_keep_must_be_below_the_threshold():
    with pytest.raises(ValidationError):
        Compaction(threshold=0.5, keep=0.5)


async def test_long_tool_output_is_cut_in_the_middle():
    @tool
    def dump() -> str:
        """Dump a log."""
        return "a" * 50 + "b" * 50

    agent = Agent(AgentConfig(model=MODEL, max_tool_output_chars=20), tools=[dump])
    session = Session()

    with _replies(_call("dump"), "Done."):
        events = [
            e async for e in agent.start("Dump it.", stream=False, session=session)
        ]

    text = "a" * 10 + "\n[80 characters omitted]\n" + "b" * 10
    assert session.messages[2]["content"] == text
    finished = [e for e in events if isinstance(e, ToolCallFinished)]
    assert finished[0].result == text
    assert events[-1].result.tool_calls[0].result == text


def test_short_tool_output_is_sent_whole():
    @tool
    def dump() -> str:
        """Dump a log."""
        return "short"

    agent = Agent(AgentConfig(model=MODEL, max_tool_output_chars=20), tools=[dump])
    session = Session()

    with _replies(_call("dump"), "Done."):
        agent.run("Dump it.", session=session)

    assert session.messages[2]["content"] == "short"


def _summary_response(content, finish_reason="stop"):
    return {
        "mock_response": litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        )
    }


def test_the_newest_message_is_always_kept():
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction()))
    question = "Please answer this. " + _words(100)

    with _replies("Summary.", "Done.") as mock:
        agent.run(question, session=_long_session())

    summary, step = mock.call_args_list
    assert "Please answer this." not in summary.kwargs["messages"][0]["content"]
    assert step.kwargs["messages"][-1] == {"role": "user", "content": question}


async def test_a_second_step_after_a_compaction_is_not_compacted_again():
    @tool
    def look() -> str:
        """Look around."""
        return "Nothing."

    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction()), tools=[look])
    session = _long_session()

    with _replies("Summary.", _call("look"), "Done.") as mock:
        handle = agent.start("Go on.", stream=False, session=session)
        events = [e async for e in handle]

    assert mock.call_count == 3
    assert len([e for e in events if isinstance(e, Compacted)]) == 1
    second = mock.call_args_list[2].kwargs["messages"]
    assert second[0]["content"] == _SUMMARY_PREFIX + "Summary."
    assert [m["role"] for m in session.messages] == [
        "user",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


def test_the_previous_summary_alone_is_not_summarized_again():
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction()))
    summary = {"role": "user", "content": _SUMMARY_PREFIX + _words(330)}
    session = Session(messages=[summary])

    with _replies("Done.") as mock:
        agent.run("Go on.", session=session)
        assert not agent.compact(
            Session(messages=[summary, {"role": "user", "content": "Hi."}])
        )

    assert mock.call_count == 1
    assert session.messages[0] == summary


def test_a_compaction_that_leaves_the_request_past_the_threshold_warns_once(caplog):
    @tool
    def look() -> str:
        """Look around."""
        return "Nothing."

    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction()), tools=[look])

    with caplog.at_level("WARNING", logger="cyclops.core.compaction"):
        with _replies(_words(330), _call("look"), "Done.") as mock:
            agent.run("Go on.", session=_long_session())

    # No second summary for the steps that follow.
    assert mock.call_count == 3
    warnings = [r for r in caplog.records if "not compacted again" in r.message]
    assert len(warnings) == 1


def test_the_summary_input_fits_the_summarizing_model():
    litellm.register_model(
        {
            "cyclops-tinier-window": {
                "max_input_tokens": 200,
                "max_output_tokens": 50,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )
    config = AgentConfig(
        model=MODEL, compaction=Compaction(model="cyclops-tinier-window")
    )

    with _replies("Summary.", "Done.") as mock:
        Agent(config).run("Go on.", session=_long_session())

    sent = mock.call_args_list[0].kwargs["messages"]
    assert "earlier messages omitted]" in sent[0]["content"]
    assert litellm.token_counter(model="cyclops-tinier-window", messages=sent) <= 150


def test_a_model_whose_output_limit_equals_its_input_limit_gets_the_transcript():
    model = "cyclops-even-window"
    litellm.register_model(
        {
            model: {
                "max_input_tokens": 400,
                "max_output_tokens": 400,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )
    agent = Agent(AgentConfig(model=model, compaction=Compaction()))

    with _replies("Summary.", "Done.") as mock:
        agent.run("Go on.", session=_long_session())

    summary = mock.call_args_list[0].kwargs
    # The summary gets keep of the window, and the transcript the rest.
    assert summary["max_tokens"] == 80
    assert "user: word word" in summary["messages"][0]["content"]


def test_a_summarizing_window_without_room_for_the_conversation_raises():
    litellm.register_model(
        {
            "cyclops-no-room": {
                "max_input_tokens": 60,
                "max_output_tokens": 10,
                "litellm_provider": "openai",
                "mode": "chat",
            }
        }
    )
    agent = Agent(
        AgentConfig(model=MODEL, compaction=Compaction(model="cyclops-no-room"))
    )
    session = _long_session()
    earlier = list(session.messages)

    with _replies() as mock:
        result = agent.run_with_response("Go on.", session=session)

    mock.assert_not_called()
    assert isinstance(result.error, RuntimeError)
    assert session.messages == earlier + [{"role": "user", "content": "Go on."}]


@pytest.mark.parametrize(
    "reasoning", [{"thinking_budget": 95}, {"reasoning_effort": "low"}]
)
def test_the_summary_is_asked_without_reasoning(reasoning):
    litellm.register_model(
        {
            "anthropic/cyclops-tiny-thinker": {
                "max_input_tokens": 400,
                "max_output_tokens": 100,
                "litellm_provider": "anthropic",
                "mode": "chat",
                "supports_reasoning": True,
            }
        }
    )
    config = AgentConfig(
        model="anthropic/cyclops-tiny-thinker", compaction=Compaction(), **reasoning
    )

    with _replies("Summary.", "Done.") as mock:
        Agent(config).run("Go on.", session=_long_session())

    summary, step = (c.kwargs for c in mock.call_args_list)
    # Its max_tokens of 80 would be under the thinking budget of 95.
    assert summary["max_tokens"] == 80
    assert not {"thinking", "reasoning_effort"} & summary.keys()
    assert {"thinking", "reasoning_effort"} & step.keys()


def test_a_failing_summary_leaves_the_transcript_as_it_was():
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction(), num_retries=0))
    session = _long_session()
    earlier = list(session.messages)
    error = litellm.APIConnectionError(
        message="down", llm_provider="openai", model=MODEL
    )

    with _replies(error) as mock:
        result = agent.run_with_response("Go on.", session=session)

    assert mock.call_count == 1
    assert result.error is error
    assert session.messages == earlier + [{"role": "user", "content": "Go on."}]


@pytest.mark.parametrize(
    "reply", [_summary_response(None), _summary_response("Half a", "length")]
)
def test_an_empty_or_cut_summary_raises(reply):
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction()))
    session = _long_session()
    earlier = list(session.messages)

    with _replies(reply, reply):
        result = agent.run_with_response("Go on.", session=session)
        with pytest.raises(RuntimeError, match="no usable summary"):
            agent.compact(Session(messages=list(earlier)))

    assert isinstance(result.error, RuntimeError)
    assert session.messages == earlier + [{"role": "user", "content": "Go on."}]
    # The summary's completion still counts.
    assert result.usage.total_tokens == 10


def _big_tool(deferred: bool, words: int = 300) -> Tool:
    """A tool whose schema alone is about words tokens."""

    def big(text: str) -> str:
        return text

    big.__doc__ = f"Big.\n\nArgs:\n    text: {_words(words)}\n"
    return Tool("big", "Big tool.", big, defer_loading=deferred)


def test_tools_loaded_during_a_run_are_counted():
    @tool
    def look() -> str:
        """Look around."""
        return "Nothing."

    agent = Agent(
        AgentConfig(model=MODEL, compaction=Compaction()),
        tools=[look, _big_tool(deferred=True)],
    )
    session = Session(messages=[{"role": "user", "content": _words(150)}])

    replies = [_call("search_tools", '{"query": "big"}'), "Summary.", "Done."]
    with _replies(*replies) as mock:
        agent.run("Go on.", session=session)

    # The first step's request is under the threshold; once the search loads
    # the big tool, its schema puts the second past it.
    assert _is_summary_request(mock.call_args_list[1].kwargs)


def test_manual_compaction_counts_only_the_tools_a_run_starts_with():
    def tokens_before(tool: Tool) -> int:
        hooks = _Recorder()
        agent = Agent(AgentConfig(model=MODEL, hooks=hooks), tools=[tool])
        with _replies("Summary."):
            agent.compact(_long_session())
        return hooks.compactions[0][1]

    assert tokens_before(_big_tool(True, 300)) == tokens_before(_big_tool(True, 10))
    assert tokens_before(_big_tool(False, 300)) > tokens_before(_big_tool(False, 10))


def test_the_naive_fallback_restores_the_transcript_before_compaction():
    model = "ollama/cyclops-tiny"
    litellm.register_model(
        {
            model: {
                "max_input_tokens": 400,
                "max_output_tokens": 100,
                "litellm_provider": "ollama",
                "mode": "chat",
            }
        }
    )

    @tool
    def look() -> str:
        """Look around."""
        return "Nothing."

    agent = Agent(AgentConfig(model=model, compaction=Compaction()), tools=[look])
    session = _long_session()
    unsupported = litellm.UnsupportedParamsError(
        message="does not support parameters: ['tools']",
        llm_provider="ollama",
        model=model,
    )

    with _replies("First.", unsupported, "Second.", "Done.") as mock:
        agent.run("Go on.", session=session)

    assert mock.call_count == 4
    # The native attempt's compaction is undone, and the naive one compacts
    # the transcript as it was, so the request appears once.
    assert session.messages == [
        {"role": "user", "content": _SUMMARY_PREFIX + "Second."},
        {"role": "user", "content": "Go on."},
        {"role": "assistant", "content": "Done."},
    ]


def test_autosave_stores_the_compacted_session():
    store = InMemorySessionStore()

    @tool
    def look() -> str:
        """Look around."""
        return _words(100)

    agent = Agent(
        AgentConfig(model=MODEL, compaction=Compaction()),
        tools=[look],
        session_store=store,
        durability="sync",
    )
    session = Session(
        messages=[
            {"role": "user", "content": _words(20)},
            {"role": "assistant", "content": "ok"},
        ]
    )
    replies = [
        _call("look", id="c1"),
        _call("look", id="c2"),
        _call("look", id="c3"),
        "Summary.",
        _call("look", id="c4"),
        "Done.",
    ]

    with _replies(*replies):
        agent.run("Go.", session=session)

    assert session.messages[0]["content"] == _SUMMARY_PREFIX + "Summary."
    stored = _run(store.load(session.id))
    assert stored.messages == session.messages


def _run(coroutine):
    import asyncio

    return asyncio.run(coroutine)


class _Seen(ToolMiddleware):
    def __init__(self) -> None:
        self.results: List[str] = []

    async def after_tool(self, call, result, error):
        self.results.append(ToolResult.message_text(result.to_message_content()))


def test_hooks_and_middleware_see_the_whole_tool_output():
    class Hooks(AgentHooks):
        def __init__(self) -> None:
            self.results: List[str] = []

        def on_tool_end(self, tool_name, args, result):
            self.results.append(result)

    @tool
    def dump() -> str:
        """Dump a log."""
        return "a" * 100

    hooks, seen = Hooks(), _Seen()
    agent = Agent(
        AgentConfig(
            model=MODEL, max_tool_output_chars=20, hooks=hooks, middleware=[seen]
        ),
        tools=[dump],
    )
    session = Session()

    with _replies(_call("dump"), "Done."):
        agent.run("Dump it.", session=session)

    assert hooks.results == ["a" * 100]
    assert seen.results == ["a" * 100]
    assert session.messages[2]["content"].startswith("a" * 10 + "\n[80 characters")


def test_images_stay_in_a_cut_tool_result():
    image = ImageContent(data="aGk=", mime_type="image/png")

    @tool
    def shot() -> ToolResult:
        """Take a screenshot."""
        return ToolResult(content=[TextContent(text="a" * 100), image])

    agent = Agent(AgentConfig(model=MODEL, max_tool_output_chars=20), tools=[shot])
    session = Session()

    with _replies(_call("shot"), "Done."):
        agent.run("Look.", session=session)

    text = "a" * 10 + "\n[80 characters omitted]\n" + "a" * 10
    assert session.messages[2]["content"] == text
    # The model takes no images in a tool message, so they follow in a user one.
    images = session.messages[3]["content"]
    assert images[1] == {
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,aGk="},
    }


OVERFLOW = "litellm.ContextWindowExceededError"

# Like MODEL, for a provider that takes a thinking budget.
THINKING_MODEL = "anthropic/cyclops-tiny-window"
litellm.register_model(
    {
        THINKING_MODEL: {
            "max_input_tokens": 400,
            "max_output_tokens": 100,
            "litellm_provider": "anthropic",
            "mode": "chat",
            "supports_reasoning": True,
        }
    }
)


def _session(tokens: int) -> Session:
    """A session of about tokens tokens: past the room for 100 output tokens."""
    return Session(messages=[{"role": "user", "content": _words(tokens)}])


@pytest.mark.parametrize("stream", [False, True])
async def test_an_overflow_is_retried_with_a_max_tokens_that_fits(stream):
    agent = Agent(AgentConfig(model=MODEL))

    with _replies(OVERFLOW, "Done.") as mock:
        result = await agent.start("Go on.", stream=stream, session=_session(300))

    assert result.stop_reason == "completed"
    first, retry = mock.call_args_list
    assert "max_tokens" not in first.kwargs
    # 400 input tokens less the request's own: below the model's output limit.
    assert 0 < retry.kwargs["max_tokens"] < 100
    assert retry.kwargs["messages"] == first.kwargs["messages"]


def test_later_steps_keep_the_lowered_max_tokens():
    @tool
    def look() -> str:
        """Look around."""
        return "Nothing."

    agent = Agent(AgentConfig(model=MODEL), tools=[look])

    with _replies(OVERFLOW, _call("look"), "Done.") as mock:
        result = agent.run_with_response("Go on.", session=_session(300))

    assert result.stop_reason == "completed"
    # The second step is sent within the window at once, without an overflow.
    assert mock.call_count == 3
    retry, second = mock.call_args_list[1:]
    assert 0 < second.kwargs["max_tokens"] < retry.kwargs["max_tokens"]


def test_with_compaction_max_tokens_is_lowered_before_any_overflow():
    agent = Agent(AgentConfig(model=MODEL, compaction=Compaction(threshold=0.9)))

    with _replies("Done.") as mock:
        agent.run("Go on.", session=_session(300))

    assert 0 < mock.call_args.kwargs["max_tokens"] < 100


def test_an_overflow_after_lowering_is_compacted():
    hooks = _Recorder()
    agent = Agent(
        AgentConfig(
            model=MODEL, compaction=Compaction(threshold=0.9, keep=0.1), hooks=hooks
        )
    )

    with _replies(OVERFLOW, "Summary.", "Done.") as mock:
        events = list(agent.stream_events("Go on.", session=_session(300)))

    lowered, summary, retry = mock.call_args_list
    assert 0 < lowered.kwargs["max_tokens"] < 100
    assert _is_summary_request(summary.kwargs)
    assert retry.kwargs["messages"][0]["content"] == _SUMMARY_PREFIX + "Summary."
    assert len(hooks.compactions) == 1
    assert any(isinstance(e, Compacted) for e in events)
    assert events[-1].result.stop_reason == "completed"


async def test_an_overflow_with_no_room_to_lower_is_compacted():
    agent = Agent(
        AgentConfig(
            model=MODEL, max_tokens=50, compaction=Compaction(threshold=0.9, keep=0.1)
        )
    )

    with _replies(OVERFLOW, "Summary.", "Done.") as mock:
        result = await agent.start("Go on.", stream=False, session=_session(300))

    first, summary, retry = mock.call_args_list
    assert first.kwargs["max_tokens"] == retry.kwargs["max_tokens"] == 50
    assert _is_summary_request(summary.kwargs)
    # The summary's completion counts with the step's.
    assert result.usage.total_tokens == 60


@pytest.mark.parametrize(
    "config, replies",
    [
        # Lowered max_tokens, then a second overflow, with no compaction.
        (AgentConfig(model=MODEL), [OVERFLOW, OVERFLOW]),
        # Lowered before the request, then compacted, then a third overflow.
        (
            AgentConfig(model=MODEL, compaction=Compaction(threshold=0.9, keep=0.1)),
            [OVERFLOW, "Summary.", OVERFLOW],
        ),
        # Lowered after the first overflow, compacted after the second.
        (
            AgentConfig(model=MODEL, compaction=Compaction(threshold=0.99, keep=0.1)),
            [OVERFLOW, "Summary.", OVERFLOW],
        ),
    ],
)
def test_each_remedy_is_tried_once(config, replies):
    with _replies(*replies) as mock:
        result = Agent(config).run_with_response("Go on.", session=_session(300))

    assert mock.call_count == len(replies)
    assert isinstance(result.error, litellm.ContextWindowExceededError)


@pytest.mark.parametrize(
    "config",
    [
        # LiteLLM knows no limits for the model.
        AgentConfig(model="openai/cyclops-unmapped"),
        # max_tokens already fits, and there is no compaction.
        AgentConfig(model=MODEL, max_tokens=50),
        # The room is within the thinking budget, which max_tokens must exceed.
        AgentConfig(model=THINKING_MODEL, thinking_budget=95),
        # Or within the largest budget LiteLLM makes of a reasoning effort.
        AgentConfig(model=THINKING_MODEL, reasoning_effort="low"),
    ],
)
def test_an_overflow_without_a_remedy_ends_the_run(config):
    with _replies(OVERFLOW) as mock:
        result = Agent(config).run_with_response("Go on.", session=_session(300))

    assert mock.call_count == 1
    assert isinstance(result.error, litellm.ContextWindowExceededError)


@pytest.mark.parametrize(
    "reasoning, sent",
    [
        ({"thinking_budget": 95}, "thinking"),
        ({"reasoning_effort": "low"}, "reasoning_effort"),
    ],
)
def test_a_thinking_budget_sends_an_overflow_to_compaction(reasoning, sent):
    config = AgentConfig(
        model=THINKING_MODEL,
        compaction=Compaction(threshold=0.9, keep=0.1),
        **reasoning,
    )

    with _replies(OVERFLOW, "Summary.", "Done.") as mock:
        result = Agent(config).run_with_response("Go on.", session=_session(300))

    assert result.stop_reason == "completed"
    first = mock.call_args_list[0].kwargs
    assert sent in first
    assert "max_tokens" not in first
    assert _is_summary_request(mock.call_args_list[1].kwargs)
