"""Resolve a model's deployments and check what their providers support."""

from typing import Any, Callable, Dict, List, Optional

import litellm
from litellm.utils import supports_prompt_caching

from cyclops.core.types import AgentConfig

# LiteLLM turns image parts of a tool message into the provider's own image
# blocks for these providers (on vertex_ai only for Gemini models); others
# accept only text in a tool message.
_TOOL_MESSAGE_IMAGE_PROVIDERS = {"anthropic", "bedrock", "gemini"}
# Providers that cache a prompt prefix at cache_control markers (on vertex_ai
# only for Anthropic models). Gemini turns them into billed cachedContents
# resources instead, and OpenAI caches without them.
_CACHE_MARKER_PROVIDERS = {"anthropic", "bedrock"}


def _takes_tool_message_images(model: str, provider: str) -> bool:
    if provider == "vertex_ai":
        return model.startswith("gemini")
    return provider in _TOOL_MESSAGE_IMAGE_PROVIDERS


def _caches_at_markers(model: str, provider: str) -> bool:
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


def _deployments(config: AgentConfig) -> List[Dict[str, Any]]:
    """LiteLLM params of every deployment the model resolves to."""
    if config.router:
        return [
            d["litellm_params"]
            for d in config.router.get_model_list(config.model) or []
        ]
    return [{"model": config.model}]


def _every_deployment(config: AgentConfig, supports: Callable[..., bool]) -> bool:
    """True if a LiteLLM `supports_*` check holds for every deployment."""
    return all(
        supports(model=p["model"], custom_llm_provider=p.get("custom_llm_provider"))
        for p in _deployments(config)
    )


def _every_deployment_takes(config: AgentConfig, param: str) -> bool:
    """True if LiteLLM lists `param` as supported for every deployment of the model."""
    return all(
        param
        in (
            litellm.get_supported_openai_params(
                model=p["model"], custom_llm_provider=p.get("custom_llm_provider")
            )
            or ()
        )
        for p in _deployments(config)
    )


def _tools_unsupported(config: AgentConfig) -> bool:
    """True if LiteLLM lists no `tools` support for every deployment of the model."""
    params = _deployments(config)
    for p in params:
        supported = litellm.get_supported_openai_params(
            model=p["model"], custom_llm_provider=p.get("custom_llm_provider")
        )
        if supported is None or "tools" in supported:
            return False
    return bool(params)


def _tool_message_images_supported(config: AgentConfig) -> bool:
    """True if every deployment of the model takes images in a tool message."""
    params = _deployments(config)
    return bool(params) and all(
        _takes_tool_message_images(
            *litellm.get_llm_provider(
                p["model"], custom_llm_provider=p.get("custom_llm_provider")
            )[:2]
        )
        for p in params
    )


def _marker_caching_supported(config: AgentConfig) -> bool:
    """True if every deployment of the model caches at cache_control markers."""
    return all(
        _caches_at_markers(
            *litellm.get_llm_provider(
                p["model"], custom_llm_provider=p.get("custom_llm_provider")
            )[:2]
        )
        for p in _deployments(config)
    )
