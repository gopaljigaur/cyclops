"""Tests for response_model validation and output retries inside the run loop."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from cyclops.core.agent import Agent
from cyclops.core.events import OutputRejected, RunFinished, StepFinished, TextDelta
from cyclops.core.session import Session
from cyclops.core.types import AgentConfig
from tests.test_agent import _stream, _text_chunk
from tests.test_provider_options import _add_call, _response, add


class Answer(BaseModel):
    value: int


FEEDBACK = """1 validation error:
```json
[
  {
    "type": "int_parsing",
    "loc": [
      "value"
    ],
    "msg": "Input should be a valid integer, unable to parse string as an integer"
  }
]
```

Fix the errors and try again."""

INVALID = '{"value": "three"}'
VALID = '{"value": 3}'


async def _start(agent, responses, session=None, stream=False):
    """Run with Answer as response_model; the result and the completion mock."""
    mock = AsyncMock(side_effect=responses)
    with patch("litellm.acompletion", new=mock):
        handle = agent.start(
            "1 + 2?", stream=stream, response_model=Answer, session=session
        )
        events = [event async for event in handle]
    return events[-1].result, mock, events


async def test_an_invalid_answer_is_sent_back_with_its_errors():
    agent = Agent(AgentConfig(model="gpt-4o-mini"))
    session = Session()
    mock = AsyncMock(side_effect=[_response(INVALID), _response(VALID)])
    with patch("litellm.acompletion", new=mock):
        result = await agent.arun("1 + 2?", response_model=Answer, session=session)

    assert result == Answer(value=3)
    assert mock.call_count == 2
    # Fed back as a user message, as guard feedback is.
    assert [(m["role"], m["content"]) for m in session.messages] == [
        ("user", "1 + 2?"),
        ("assistant", INVALID),
        ("user", FEEDBACK),
        ("assistant", VALID),
    ]


async def test_exhausted_retries_end_the_run_with_the_last_answer():
    agent = Agent(AgentConfig(model="gpt-4o-mini", output_retries=2))
    answers = ['{"value": "one"}', '{"value": "two"}', "three"]
    result, mock, events = await _start(agent, [_response(a) for a in answers])

    assert mock.call_count == 3
    assert result.stop_reason == "output_validation"
    assert result.content == "three"
    assert result.steps == 3
    assert len([e for e in events if isinstance(e, OutputRejected)]) == 2


async def test_arun_raises_the_validation_error_once_retries_are_exhausted():
    agent = Agent(AgentConfig(model="gpt-4o-mini"))
    mock = AsyncMock(side_effect=[_response(INVALID), _response("three")])
    with patch("litellm.acompletion", new=mock):
        with pytest.raises(ValidationError, match="json_invalid|Invalid JSON"):
            await agent.arun("1 + 2?", response_model=Answer)

    assert mock.call_count == 2


async def test_zero_retries_take_no_extra_completion():
    agent = Agent(AgentConfig(model="gpt-4o-mini", output_retries=0))
    session = Session()
    mock = AsyncMock(side_effect=[_response(INVALID)])
    with patch("litellm.acompletion", new=mock):
        with pytest.raises(ValidationError):
            await agent.arun("1 + 2?", response_model=Answer, session=session)

    assert mock.call_count == 1
    assert [m["role"] for m in session.messages] == ["user", "assistant"]


async def test_a_streamed_invalid_answer_is_marked_and_the_retry_streams():
    agent = Agent(AgentConfig(model="gpt-4o-mini"))
    streams = [
        _stream([_text_chunk('{"value": '), _text_chunk('"three"}')]),
        _stream([_text_chunk('{"value": '), _text_chunk("3}")]),
    ]
    result, _, events = await _start(agent, streams, stream=True)

    assert result.stop_reason == "completed"
    assert result.content == VALID
    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "TextDelta",
        "TextDelta",
        "OutputRejected",
        "StepFinished",
        "TextDelta",
        "TextDelta",
        "StepFinished",
        "RunFinished",
    ]
    assert events[2] == OutputRejected(feedback=FEEDBACK)
    assert "".join(e.text for e in events[4:6] if isinstance(e, TextDelta)) == VALID


async def test_tool_calls_and_output_retries_share_the_run():
    agent = Agent(AgentConfig(model="gpt-4o-mini"), tools=[add])
    session = Session()
    responses = [
        _response(tool_calls=[_add_call()]),
        _response(INVALID),
        _response(VALID),
    ]
    result, mock, events = await _start(agent, responses, session=session)

    assert result.stop_reason == "completed"
    assert result.content == VALID
    assert result.steps == 3
    assert [tc.name for tc in result.tool_calls] == ["add"]
    assert [m["role"] for m in session.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
    ]


@pytest.mark.parametrize(
    "model, native",
    [("gpt-4o-mini", True), ("gpt-3.5-turbo", False)],
    ids=["native-schema", "validated-only"],
)
async def test_retries_work_with_and_without_a_schema_sent(model, native):
    agent = Agent(AgentConfig(model=model))
    result, mock, _ = await _start(agent, [_response(INVALID), _response(VALID)])

    assert result.content == VALID
    for call in mock.call_args_list:
        assert ("response_format" in call.kwargs) is native


async def test_a_budget_spent_by_an_invalid_answer_stops_the_retry():
    # Each completion uses 2 tokens.
    agent = Agent(AgentConfig(model="gpt-4o-mini", max_total_tokens=2))
    session = Session()
    result, mock, _ = await _start(
        agent, [_response(INVALID), _response(VALID)], session=session
    )

    assert mock.call_count == 1
    assert result.stop_reason == "budget"
    assert result.content == ""
    # The feedback is not added once no step is left for it.
    assert [m["role"] for m in session.messages] == ["user", "assistant"]


async def test_output_retries_count_as_steps():
    agent = Agent(AgentConfig(model="gpt-4o-mini", max_steps=2, output_retries=5))
    result, mock, events = await _start(agent, [_response(INVALID)] * 2)

    assert mock.call_count == 2
    assert result.steps == 2
    assert result.stop_reason == "max_steps"
    assert [e.step for e in events if isinstance(e, StepFinished)] == [1, 2]
    assert isinstance(events[-1], RunFinished)


async def test_guards_judge_only_a_valid_answer():
    judged = []

    async def guard(context):
        judged.append(context.content)
        return None

    agent = Agent(AgentConfig(model="gpt-4o-mini", guards=[guard]))
    result, _, _ = await _start(agent, [_response(INVALID), _response(VALID)])

    assert result.content == VALID
    assert judged == [VALID]
