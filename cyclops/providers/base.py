"""Base LLM provider abstraction"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Synchronous completion call

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            Provider-specific response object
        """
        pass

    @abstractmethod
    async def acompletion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Any:
        """Asynchronous completion call

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            **kwargs: Provider-specific parameters (temperature, max_tokens, etc.)

        Returns:
            Provider-specific response object
        """
        pass
