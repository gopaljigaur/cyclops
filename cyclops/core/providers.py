"""Resolve a model's deployments and check what their providers support."""

from functools import lru_cache
from typing import Any, Callable, Dict, Hashable, List, Optional

import litellm
from litellm.utils import supports_prompt_caching

from cyclops.core.types import AgentConfig

# LiteLLM turns image parts of a tool message into the provider's own image
# blocks for these providers (on vertex_ai only for Gemini models); others
# accept only text in a tool message.
_TOOL_MESSAGE_IMAGE_PROVIDERS = {"anthropic", "bedrock", "gemini"}
# The LiteLLM model info flag for each kind of input a content part may need.
_INPUT_SUPPORT = {
    "image": "supports_vision",
    "pdf": "supports_pdf_input",
    "audio": "supports_audio_input",
}
# Providers that cache a prompt prefix at cache_control markers (on vertex_ai
# only for Anthropic models). Gemini turns them into billed cachedContents
# resources instead, and OpenAI caches without them.
_CACHE_MARKER_PROVIDERS = {"anthropic", "bedrock"}

# A check of one deployment: its model, its custom_llm_provider, then any arguments.
_Check = Callable[..., bool]


def _takes_tool_message_images(model: str, custom_llm_provider: Optional[str]) -> bool:
    model, provider = litellm.get_llm_provider(model, custom_llm_provider)[:2]
    if provider == "vertex_ai":
        return model.startswith("gemini")
    return provider in _TOOL_MESSAGE_IMAGE_PROVIDERS


def _caches_at_markers(model: str, custom_llm_provider: Optional[str]) -> bool:
    model, provider = litellm.get_llm_provider(model, custom_llm_provider)[:2]
    if provider == "vertex_ai":
        marker_provider = model.startswith("claude")
    else:
        marker_provider = provider in _CACHE_MARKER_PROVIDERS
    # LiteLLM adds Bedrock cache points for every model.
    return marker_provider and supports_prompt_caching(
        model=model, custom_llm_provider=provider
    )


def _native_schema(model: str, custom_llm_provider: Optional[str] = None) -> bool:
    """True if LiteLLM's own entry for this provider lists native structured output.

    Not `supports_native_structured_output`: it falls back to another provider's
    entry, while the Vertex AI builder reads only its own.
    """
    try:
        info = litellm.get_model_info(model, custom_llm_provider)
    except Exception:  # not mapped
        return False
    return info.get("supports_native_structured_output") is True


def _takes_param(model: str, custom_llm_provider: Optional[str], param: str) -> bool:
    """True if LiteLLM lists `param` as supported."""
    supported = litellm.get_supported_openai_params(
        model=model, custom_llm_provider=custom_llm_provider
    )
    return param in (supported or ())


def _lacks_tools(model: str, custom_llm_provider: Optional[str]) -> bool:
    """True if LiteLLM knows the model's parameters and `tools` is not one."""
    supported = litellm.get_supported_openai_params(
        model=model, custom_llm_provider=custom_llm_provider
    )
    return supported is not None and "tools" not in supported


def _input_unsupported(config: AgentConfig, kind: str) -> Optional[str]:
    """The first model LiteLLM's model map marks as not taking this input.

    Checks every deployment of the model and every fallback model. Only an explicit false counts: most entries leave the flags out. The entry
    is read from litellm.model_cost under provider/model, else model, and not
    from get_model_info, which asks some providers (Ollama) over the network.
    """
    flag = _INPUT_SUPPORT[kind]
    # Fallback models read the same history.
    for p in _deployments(config) + [{"model": m} for m in config.fallback_models]:
        model, provider = litellm.get_llm_provider(
            p["model"], custom_llm_provider=p.get("custom_llm_provider")
        )[:2]
        costs = litellm.model_cost
        info = costs.get(f"{provider}/{model}") or costs.get(model) or {}
        if info.get(flag) is False:
            return p["model"]
    return None


def _deployments(config: AgentConfig) -> List[Dict[str, Any]]:
    """LiteLLM params of every deployment the model resolves to."""
    if config.router:
        return [
            d["litellm_params"]
            for d in config.router.get_model_list(config.model) or []
        ]
    return [{"model": config.model}]


# LiteLLM's lookups read and normalise its model map on every call.
@lru_cache(maxsize=1024)
def _holds(
    check: _Check, model: str, custom_llm_provider: Optional[str], *args: Hashable
) -> bool:
    return check(model, custom_llm_provider, *args)


def _every_deployment(
    config: AgentConfig, check: _Check, *args: Hashable, fallbacks: bool = False
) -> bool:
    """True if the model has deployments and `check` holds for every one.

    With `fallbacks`, the fallback models count as deployments too.
    """
    deployments = _deployments(config)
    if fallbacks:
        deployments += [{"model": m} for m in config.fallback_models]
    return bool(deployments) and all(
        _holds(check, p["model"], p.get("custom_llm_provider"), *args)
        for p in deployments
    )
