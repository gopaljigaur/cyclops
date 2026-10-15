"""Tests for content parts as run input and steering."""

import asyncio
from base64 import b64encode
from unittest.mock import AsyncMock, patch

import httpx
import litellm
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from cyclops.core.agent import Agent
from cyclops.core.content import _user_content
from cyclops.core.hooks import AgentHooks
from cyclops.core.middleware import GuardContext
from cyclops.core.session import FileSessionStore, Session
from cyclops.core.types import AgentConfig, Message
from cyclops.observability import TelemetryHooks
from tests.test_agent import _make_completion_response
from tests.test_provider_options import PNG, _request_body

PNG_B64 = b64encode(PNG).decode()
PDF_B64 = b64encode(b"%PDF-1.4\n%%EOF\n").decode()

TEXT = {"type": "text", "text": "What is this?"}
IMAGE_URL = {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}
IMAGE_DATA = {
    "type": "image_url",
    "image_url": {"url": f"data:image/png;base64,{PNG_B64}"},
}
PDF = {
    "type": "file",
    "file": {
        "file_data": f"data:application/pdf;base64,{PDF_B64}",
        "filename": "report.pdf",
    },
}
AUDIO = {"type": "input_audio", "input_audio": {"data": "UklGRg==", "format": "wav"}}

ANTHROPIC = "anthropic/claude-sonnet-4-5"
GEMINI = "gemini/gemini-2.5-flash"
BEDROCK = "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0"


# ---------------------------------------------------------------------------
# Request bodies, through LiteLLM's own transforms
# ---------------------------------------------------------------------------


def _sent_parts(monkeypatch, model, part):
    # Without cache markers, which LiteLLM adds to the last block.
    message = [TEXT, part]
    body = _request_body(monkeypatch, model, message=message, prompt_caching=False)
    if model == GEMINI:
        (message,) = body["contents"]
        return message["parts"]
    (message,) = body["messages"]
    return message["content"]


@pytest.mark.parametrize("part", [IMAGE_URL, IMAGE_DATA, PDF])
def test_openai_request_sends_the_parts_as_given(monkeypatch, part):
    assert _sent_parts(monkeypatch, "gpt-4o", part) == [TEXT, part]


@pytest.mark.parametrize(
    "part, block",
    [
        (IMAGE_URL, {"type": "url", "url": "https://example.com/cat.png"}),
        (IMAGE_DATA, {"type": "base64", "media_type": "image/png", "data": PNG_B64}),
    ],
)
def test_anthropic_request_has_an_image_block(monkeypatch, part, block):
    sent = _sent_parts(monkeypatch, ANTHROPIC, part)
    assert sent[1] == {"type": "image", "source": block}


def test_anthropic_request_has_a_document_block(monkeypatch):
    sent = _sent_parts(monkeypatch, ANTHROPIC, PDF)
    source = {"type": "base64", "media_type": "application/pdf", "data": PDF_B64}
    assert sent[1] == {"type": "document", "source": source}


@pytest.mark.parametrize(
    "part, data, mime_type",
    [
        (IMAGE_URL, PNG_B64, "image/png"),
        (IMAGE_DATA, PNG_B64, "image/png"),
        (PDF, PDF_B64, "application/pdf"),
    ],
)
def test_gemini_request_inlines_the_data(monkeypatch, part, data, mime_type):
    sent = _sent_parts(monkeypatch, GEMINI, part)
    assert sent == [
        {"text": "What is this?"},
        {"inline_data": {"data": data, "mime_type": mime_type}},
    ]


@pytest.mark.parametrize("part", [IMAGE_URL, IMAGE_DATA])
def test_bedrock_request_has_image_bytes(monkeypatch, part):
    sent = _sent_parts(monkeypatch, BEDROCK, part)
    assert sent[1] == {"image": {"source": {"bytes": PNG_B64}, "format": "png"}}


def test_bedrock_request_has_document_bytes(monkeypatch):
    sent = _sent_parts(monkeypatch, BEDROCK, PDF)
    document = sent[1]["document"]
    assert document["source"] == {"bytes": PDF_B64}
    assert document["format"] == "pdf"


# ---------------------------------------------------------------------------
# Checks before anything is sent
# ---------------------------------------------------------------------------


def _agent(model="gpt-4o", **config):
    return Agent(AgentConfig(model=model, **config))


# LiteLLM's model map sets these flags to false.
NO_VISION = "o3-mini"
NO_PDF = "vertex_ai/gemini-live-2.5-flash-native-audio"
NO_AUDIO = "bedrock/moonshotai.kimi-k2.5"


def _file(**file):
    return {"type": "file", "file": file}


@pytest.mark.parametrize(
    "model, part, kind",
    [
        (NO_VISION, IMAGE_DATA, "image"),
        (NO_VISION, IMAGE_URL, "image"),
        (NO_VISION, _file(file_data=f"data:image/png;base64,{PNG_B64}"), "image"),
        (NO_PDF, PDF, "pdf"),
        (NO_PDF, _file(file_id="file-123", format="application/pdf"), "pdf"),
        (NO_AUDIO, AUDIO, "audio"),
        (NO_AUDIO, _file(file_data="data:audio/wav;base64,UklGRg=="), "audio"),
        # A route LiteLLM's lookup strips to find the entry.
        ("bedrock/converse/moonshotai.kimi-k2.5", AUDIO, "audio"),
    ],
)
def test_unsupported_input_raises_before_any_completion(model, part, kind):
    mock = AsyncMock()
    with patch("litellm.acompletion", new=mock), pytest.raises(ValueError) as error:
        _agent(model).run([TEXT, part])
    assert str(error.value).startswith(
        f"{model}, one of its deployments or one of its fallback models "
        f"can't take {kind} input"
    )
    assert "litellm.register_model at startup" in str(error.value)
    mock.assert_not_called()


def test_unsupported_input_checks_every_router_deployment():
    router = litellm.Router(
        model_list=[
            {"model_name": "chat", "litellm_params": {"model": "gpt-4o"}},
            {"model_name": "chat", "litellm_params": {"model": NO_VISION}},
        ]
    )
    with pytest.raises(
        ValueError,
        match="^chat, one of its deployments or one of its fallback models can't take image input",
    ):
        _agent("chat", router=router).run([TEXT, IMAGE_DATA])


def test_unsupported_input_checks_every_fallback_model():
    mock = AsyncMock()
    agent = _agent("gpt-4o", fallback_models=[NO_VISION])
    with patch("litellm.acompletion", new=mock):
        with pytest.raises(
            ValueError,
            match="^gpt-4o, one of its deployments or one of its fallback models can't take image input",
        ):
            agent.run([TEXT, IMAGE_DATA])
    mock.assert_not_called()


def test_steer_checks_every_fallback_model():
    async def main():
        mock = AsyncMock(return_value=_make_completion_response("Hi."))
        with patch("litellm.acompletion", new=mock):
            handle = _agent("gpt-4o", fallback_models=[NO_VISION]).start(
                "hello", stream=False
            )
            with pytest.raises(
                ValueError,
                match="^gpt-4o, one of its deployments or one of its fallback models can't take image input",
            ):
                handle.steer([IMAGE_DATA])
            return await handle

    assert asyncio.run(main()).steps == 1


@pytest.mark.parametrize(
    "model, part",
    [
        # Mapped without the flag.
        ("gpt-3.5-turbo", IMAGE_DATA),
        ("gpt-4o", AUDIO),
        # Not mapped.
        ("openai/my-endpoint-model", IMAGE_DATA),
        # A file of unknown type.
        (NO_VISION, _file(file_id="file-123")),
    ],
)
def test_input_not_marked_unsupported_is_sent(model, part):
    mock = AsyncMock(return_value=_make_completion_response("Done."))
    with patch("litellm.acompletion", new=mock):
        assert _agent(model).run([TEXT, part]) == "Done."
    assert mock.call_args.kwargs["messages"][-1]["content"] == [TEXT, part]


def test_input_check_makes_no_network_call():
    # LiteLLM's get_model_info asks an Ollama server for its model info, and
    # logs a failed request instead of raising it.
    requests = []

    def send(self, request, **kwargs):
        requests.append(request.url)
        raise httpx.ConnectError("no network in tests", request=request)

    config = AgentConfig(model="ollama/llava")
    with patch.object(httpx.Client, "send", send):
        assert _user_content(config, [TEXT, IMAGE_DATA]) == [TEXT, IMAGE_DATA]
    assert requests == []


@pytest.mark.parametrize(
    "content",
    [
        [],
        ["What is this?"],
        [{"type": "text"}],
        [{"type": "image", "source": {"type": "base64", "data": PNG_B64}}],
        [{"type": "input_image", "image_url": f"data:image/png;base64,{PNG_B64}"}],
        [{"type": "image_url", "image_url": {"detail": "low"}}],
    ],
)
def test_parts_that_are_not_chat_content_parts_raise(content):
    mock = AsyncMock()
    with patch("litellm.acompletion", new=mock), pytest.raises(ValueError) as error:
        _agent().run(content)
    assert PNG_B64 not in str(error.value)
    mock.assert_not_called()


def test_start_raises_before_the_run_starts():
    async def main():
        with pytest.raises(ValueError):
            _agent(NO_VISION).start([TEXT, IMAGE_DATA])

    asyncio.run(main())


# ---------------------------------------------------------------------------
# Sessions, steering, hooks and guards
# ---------------------------------------------------------------------------


def test_session_keeps_the_parts_through_a_store(tmp_path):
    store = FileSessionStore(tmp_path)
    agent = Agent(AgentConfig(model="gpt-4o"), session_store=store)
    session = Session()
    mock = AsyncMock(
        side_effect=[
            _make_completion_response("A cat."),
            _make_completion_response("Grey."),
        ]
    )

    with patch("litellm.acompletion", new=mock):
        agent.run([TEXT, IMAGE_DATA], session=session)
        loaded = asyncio.run(store.load(session.id))
        assert loaded == session
        assert loaded.messages[0] == {"role": "user", "content": [TEXT, IMAGE_DATA]}
        agent.run("What colour is it?", session=loaded)

    sent = mock.call_args.kwargs["messages"]
    assert sent[0]["content"] == [TEXT, IMAGE_DATA]
    assert Message.from_dict(sent[0]).to_dict() == sent[0]


def test_steer_with_an_image():
    async def main():
        mock = AsyncMock(return_value=_make_completion_response("Both are cats."))
        with patch("litellm.acompletion", new=mock):
            handle = _agent().start([TEXT, IMAGE_URL], stream=False)
            handle.steer([{"type": "text", "text": "And this one?"}, IMAGE_DATA])
            await handle
        return mock.call_args.kwargs["messages"]

    sent = asyncio.run(main())
    assert [m["content"] for m in sent] == [
        [TEXT, IMAGE_URL],
        [{"type": "text", "text": "And this one?"}, IMAGE_DATA],
    ]


def test_steer_with_an_image_the_model_cannot_take_raises():
    async def main():
        mock = AsyncMock(return_value=_make_completion_response("Hi."))
        with patch("litellm.acompletion", new=mock):
            handle = _agent(NO_VISION).start("hello", stream=False)
            with pytest.raises(ValueError, match="can't take image input"):
                handle.steer([IMAGE_DATA])
            return await handle

    assert asyncio.run(main()).steps == 1


def test_hooks_and_guards_see_the_parts():
    started = []
    guarded = []

    class Hooks(AgentHooks):
        def on_run_start(self, input_message):
            started.append(input_message)

    async def guard(context: GuardContext):
        guarded.append(context.messages[0]["content"])
        return None

    agent = _agent(hooks=Hooks(), guards=[guard])
    mock = AsyncMock(return_value=_make_completion_response("A cat."))
    with patch("litellm.acompletion", new=mock):
        agent.run([TEXT, IMAGE_DATA])

    assert started == guarded == [[TEXT, IMAGE_DATA]]


def test_telemetry_has_no_base64_data():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    agent = _agent(hooks=TelemetryHooks(tracer_provider=provider))
    mock = AsyncMock(return_value=_make_completion_response("A cat."))

    with patch("litellm.acompletion", new=mock):
        agent.run([TEXT, IMAGE_DATA, PDF])

    spans = exporter.get_finished_spans()
    (run,) = [s for s in spans if s.name == "invoke_agent"]
    assert run.attributes["cyclops.input.length"] == len(TEXT["text"])
    values = [str(v) for s in spans for v in s.attributes.values()]
    values += [str(v) for s in spans for e in s.events for v in e.attributes.values()]
    assert not [v for v in values if PNG_B64 in v or PDF_B64 in v]
