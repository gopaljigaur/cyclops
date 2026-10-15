"""Tests for compaction and the tool output limit"""

from typing import List
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from pydantic import ValidationError

from cyclops import Agent, AgentConfig, AgentHooks, Compaction, Session, tool
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

    A reply is the text of a mock_response, or a dict of mock arguments.
    """
    real = litellm.acompletion
    queue = list(replies)

    async def complete(**kwargs):
        reply = queue.pop(0)
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

    # Keeping 80 tokens reaches into the second tool result, so the whole
    # call goes into the summary.
    assert session.messages == [
        {"role": "user", "content": _SUMMARY_PREFIX + "Read two pages."},
        {"role": "assistant", "content": "Both read."},
    ]
    transcript = mock.call_args.kwargs["messages"][0]["content"]
    assert "called look({})" in transcript
    assert "tool result (look)" in transcript


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


def test_a_model_without_known_limits_is_never_compacted():
    agent = Agent(AgentConfig(model="openai/cyclops-unmapped", compaction=Compaction()))
    session = _long_session()

    with _replies("Done.") as mock:
        agent.run("Go on.", session=session)
        assert not agent.compact(session)

    assert mock.call_count == 1
    assert len(session.messages) == 8


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
    assert _window(config) == (MODEL, 400)
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
