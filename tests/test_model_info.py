"""Tests that model info lookups never ask a provider's server."""

from unittest.mock import AsyncMock, patch
from uuid import uuid4

import httpx
import litellm
import pytest
from litellm import Choices, Message, ModelResponse, Usage
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from pydantic import BaseModel

from cyclops import Agent, AgentConfig, Compaction, Session, tool
from cyclops.core.compaction import _window
from cyclops.core.providers import _holds, _model_info

# LiteLLM's lookups ask these providers' servers (Ollama's for unmapped models).
SERVER_PROVIDERS = ["ollama", "ollama_chat", "lemonade", "huggingface"]
REGISTERED = "ollama/cyclops-registered"
litellm.register_model(
    {
        REGISTERED: {
            "max_input_tokens": 400,
            "max_output_tokens": 100,
            "litellm_provider": "ollama",
            "mode": "chat",
        }
    }
)


class Answer(BaseModel):
    value: int


@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


def _response(content=None, tool_calls=None):
    response = ModelResponse(
        choices=[Choices(message=Message(content=content, tool_calls=tool_calls))],
        usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    # As LiteLLM prices a response, so the run asks for no cost.
    response._hidden_params["response_cost"] = 0.0
    return response


def _add_call():
    return ChatCompletionMessageToolCall(
        id="call_1", function=Function(name="add", arguments='{"a": 1, "b": 2}')
    )


def _overflow(model):
    return litellm.ContextWindowExceededError(
        message="too long", model=model, llm_provider="ollama"
    )


def _unmapped(provider):
    """A new model name each time, since LiteLLM caches its lookups."""
    return f"{provider}/cyclops-{uuid4().hex}"


def _router(model, **model_info):
    return litellm.Router(
        model_list=[
            {
                "model_name": "local",
                "litellm_params": {"model": model, "api_base": "http://gpu:11434"},
                "model_info": model_info,
            }
        ]
    )


@pytest.fixture
def requests():
    """Fail every HTTP request LiteLLM sends, and record its URL."""
    urls = []

    def send(self, request, **kwargs):
        urls.append(str(request.url))
        raise httpx.ConnectError("no network in tests", request=request)

    async def asend(self, request, **kwargs):
        return send(self, request)

    _holds.cache_clear()
    with (
        patch.object(httpx.Client, "send", send),
        patch.object(httpx.AsyncClient, "send", asend),
    ):
        yield urls


@pytest.fixture(params=[False, True], ids=["model", "router"])
def routed(request):
    return request.param


@pytest.fixture(params=SERVER_PROVIDERS)
def config(request, routed, requests):
    """A config for an unmapped model of the provider, or a router of one."""
    model = _unmapped(request.param)
    if not routed:
        return AgentConfig(model=model)
    config = AgentConfig(model="local", router=_router(model))
    # LiteLLM's Router asks the server as it is built.
    requests.clear()
    return config


def _complete(config, *replies):
    target = config.router or litellm
    return patch.object(target, "acompletion", new=AsyncMock(side_effect=replies))


@pytest.mark.parametrize("tools", [[add], []])
def test_the_capability_checks_ask_no_server(config, requests, tools):
    config = config.model_copy(
        update={"reasoning_effort": "low", "prompt_caching": True}
    )
    replies = [_response(tool_calls=[_add_call()])] if tools else []
    with _complete(config, *replies, _response('{"value": 3}')):
        answer = Agent(config, tools=tools).run("1 + 2?", response_model=Answer)

    assert answer == Answer(value=3)
    assert requests == []


def test_the_input_check_asks_no_server(config, requests):
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
    content = [{"type": "text", "text": "What is this?"}, image]

    with _complete(config, _response("A cat.")) as mock:
        assert Agent(config).run(content) == "A cat."

    assert mock.call_args.kwargs["messages"][-1]["content"] == content
    assert requests == []


def test_an_overflow_without_known_limits_ends_the_run(config, requests):
    with _complete(config, _overflow(config.model)) as mock:
        result = Agent(config).run_with_response("Go on.")

    assert mock.call_count == 1
    assert isinstance(result.error, litellm.ContextWindowExceededError)
    assert requests == []


def test_compaction_without_known_limits_raises(config, requests):
    config = config.model_copy(update={"compaction": Compaction()})
    session = Session(messages=[{"role": "user", "content": "Hi."}])

    with _complete(config) as mock:
        result = Agent(config).run_with_response("Go on.", session=session)
        with pytest.raises(ValueError, match="register_model"):
            Agent(config).compact(session)

    mock.assert_not_called()
    assert isinstance(result.error, ValueError)
    assert "register_model" in str(result.error)
    assert requests == []


def test_a_summarizing_model_without_known_limits_raises(requests):
    config = AgentConfig(
        model=REGISTERED, compaction=Compaction(model=_unmapped("ollama"))
    )

    with _complete(config) as mock:
        result = Agent(config).run_with_response("Go on.")

    mock.assert_not_called()
    assert isinstance(result.error, ValueError)
    assert requests == []


def test_context_usage_asks_no_server(config, requests):
    session = Session(messages=[{"role": "user", "content": "Hi."}])

    assert Agent(config).context_usage(session) is None
    assert requests == []


def test_a_registered_models_context_usage_asks_no_server(requests):
    session = Session(messages=[{"role": "user", "content": "Hi."}])

    usage = Agent(AgentConfig(model=REGISTERED)).context_usage(session)

    assert usage.max_input_tokens == 400
    assert requests == []


def test_another_providers_entry_is_not_read(requests):
    # The bare name has an OpenAI entry.
    assert _model_info("ollama/gpt-4o") == {}
    assert _model_info("ollama/llama2")["max_input_tokens"]
    assert requests == []


def test_a_registered_model_gets_its_window(requests):
    assert _window(AgentConfig(model=REGISTERED)) == (REGISTERED, 400, 100)
    assert requests == []


def test_a_registered_model_is_compacted(requests):
    config = AgentConfig(model=REGISTERED, compaction=Compaction())
    session = Session(messages=[{"role": "user", "content": "word " * 350}])

    with _complete(config, _response("Summary."), _response("Done.")) as mock:
        result = Agent(config).run_with_response("Go on.", session=session)

    assert result.content == "Done."
    assert mock.call_count == 2
    assert requests == []


def test_a_mapped_lemonade_model_gets_its_window(requests):
    model = next(
        k for k, v in litellm.model_cost.items() if v["litellm_provider"] == "lemonade"
    )
    info = litellm.model_cost[model]

    limits = _window(AgentConfig(model=model))

    assert limits == (model, info["max_input_tokens"], info["max_output_tokens"])
    assert requests == []


@pytest.mark.parametrize("provider", SERVER_PROVIDERS)
def test_a_router_deployment_gets_the_window_of_its_model_info(provider, requests):
    model = _unmapped(provider)
    router = _router(model, max_input_tokens=9000, max_output_tokens=900)
    requests.clear()

    assert _window(AgentConfig(model="local", router=router)) == (model, 9000, 900)
    assert requests == []
