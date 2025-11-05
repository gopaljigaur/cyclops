"""Memory management for agents"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class MemoryItem(BaseModel):
    """Single memory item"""

    key: str
    value: Any
    metadata: Dict[str, Any] = {}


class Memory(ABC):
    """Abstract memory interface"""

    @abstractmethod
    async def store(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a value in memory"""
        pass

    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory"""
        pass

    @abstractmethod
    async def list_keys(self) -> List[str]:
        """List all keys in memory"""
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear all memory"""
        pass


class InMemoryStorage(Memory):
    """Simple in-memory storage implementation"""

    def __init__(self):
        self._storage: Dict[str, MemoryItem] = {}

    async def store(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a value in memory"""
        self._storage[key] = MemoryItem(key=key, value=value, metadata=metadata or {})

    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value from memory"""
        item = self._storage.get(key)
        return item.value if item else None

    async def list_keys(self) -> List[str]:
        """List all keys in memory"""
        return list(self._storage.keys())

    async def clear(self) -> None:
        """Clear all memory"""
        self._storage.clear()
