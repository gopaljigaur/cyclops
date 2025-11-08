"""LiteLLM provider implementation"""

from typing import Any, Dict, List, Optional
import litellm
from cyclops.providers.base import LLMProvider


class LiteLLMProvider(LLMProvider):
    """LiteLLM provider supporting direct models and Router"""

    def __init__(self, model: Optional[str] = None, router: Any = None):
        """Initialize LiteLLM provider

        Args:
            model: Model name (e.g., "gpt-4o-mini", "ollama/qwen3:4b")
            router: Optional litellm.Router instance for fallbacks/load balancing

        Note: If router is provided, model should be the model_name from router config
        """
        if not model and not router:
            raise ValueError("Either model or router must be provided")

        self.model = model
        self.router = router

    def completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous completion using litellm or router"""
        completion_kwargs = {
            "messages": messages,
            **kwargs,
        }

        if tools:
            completion_kwargs["tools"] = tools

        if self.router:
            # Use router for fallbacks/load balancing
            completion_kwargs["model"] = self.model
            return self.router.completion(**completion_kwargs)
        else:
            # Direct litellm call
            completion_kwargs["model"] = self.model
            return litellm.completion(**completion_kwargs)

    async def acompletion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchronous completion using litellm or router"""
        completion_kwargs = {
            "messages": messages,
            **kwargs,
        }

        if tools:
            completion_kwargs["tools"] = tools

        if self.router:
            # Use router for fallbacks/load balancing
            completion_kwargs["model"] = self.model
            return await self.router.acompletion(**completion_kwargs)
        else:
            # Direct litellm call
            completion_kwargs["model"] = self.model
            return await litellm.acompletion(**completion_kwargs)
