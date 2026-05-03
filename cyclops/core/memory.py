"""Memory management for agents"""

import json
import os
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
        self._storage[key] = MemoryItem(key=key, value=value, metadata=metadata or {})

    async def retrieve(self, key: str) -> Optional[Any]:
        item = self._storage.get(key)
        return item.value if item else None

    async def list_keys(self) -> List[str]:
        return list(self._storage.keys())

    async def clear(self) -> None:
        self._storage.clear()


class FileStorage(Memory):
    """File-backed persistent memory storage using JSON."""

    def __init__(self, path: str):
        self.path = path
        self._storage: Dict[str, MemoryItem] = {}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                raw: Dict[str, Any] = json.load(fh)
            for key, item_data in raw.items():
                self._storage[key] = MemoryItem(**item_data)
        except FileNotFoundError:
            return
        except (json.JSONDecodeError, OSError):
            self._storage = {}

    def _save(self) -> None:
        data = {key: item.model_dump() for key, item in self._storage.items()}
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)

    async def store(
        self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        self._storage[key] = MemoryItem(key=key, value=value, metadata=metadata or {})
        self._save()

    async def retrieve(self, key: str) -> Optional[Any]:
        item = self._storage.get(key)
        return item.value if item else None

    async def list_keys(self) -> List[str]:
        return list(self._storage.keys())

    async def clear(self) -> None:
        self._storage.clear()
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass
