"""Tests for cyclops memory backends: InMemoryStorage and FileStorage."""

import json
import os

import pytest

from cyclops.core.memory import FileStorage, InMemoryStorage


# ---------------------------------------------------------------------------
# InMemoryStorage
# ---------------------------------------------------------------------------


class TestInMemoryStorage:
    """Tests for the in-memory storage backend."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self):
        mem = InMemoryStorage()
        await mem.store("name", "Alice")
        result = await mem.retrieve("name")
        assert result == "Alice"

    @pytest.mark.asyncio
    async def test_retrieve_missing_key_returns_none(self):
        mem = InMemoryStorage()
        result = await mem.retrieve("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_keys_empty(self):
        mem = InMemoryStorage()
        keys = await mem.list_keys()
        assert keys == []

    @pytest.mark.asyncio
    async def test_list_keys_after_store(self):
        mem = InMemoryStorage()
        await mem.store("a", 1)
        await mem.store("b", 2)
        keys = await mem.list_keys()
        assert sorted(keys) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_clear_removes_all(self):
        mem = InMemoryStorage()
        await mem.store("x", 10)
        await mem.store("y", 20)
        await mem.clear()
        keys = await mem.list_keys()
        assert keys == []

    @pytest.mark.asyncio
    async def test_clear_then_retrieve_returns_none(self):
        mem = InMemoryStorage()
        await mem.store("key", "value")
        await mem.clear()
        result = await mem.retrieve("key")
        assert result is None

    @pytest.mark.asyncio
    async def test_overwrite_existing_key(self):
        mem = InMemoryStorage()
        await mem.store("counter", 1)
        await mem.store("counter", 2)
        result = await mem.retrieve("counter")
        assert result == 2

    @pytest.mark.asyncio
    async def test_store_with_metadata(self):
        mem = InMemoryStorage()
        await mem.store("item", "value", metadata={"source": "test"})
        result = await mem.retrieve("item")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_store_various_value_types(self):
        mem = InMemoryStorage()
        await mem.store("string", "hello")
        await mem.store("integer", 42)
        await mem.store("float", 3.14)
        await mem.store("list", [1, 2, 3])
        await mem.store("dict", {"key": "val"})
        await mem.store("none", None)

        assert await mem.retrieve("string") == "hello"
        assert await mem.retrieve("integer") == 42
        assert abs((await mem.retrieve("float")) - 3.14) < 1e-9
        assert await mem.retrieve("list") == [1, 2, 3]
        assert await mem.retrieve("dict") == {"key": "val"}
        assert await mem.retrieve("none") is None

    @pytest.mark.asyncio
    async def test_list_keys_count(self):
        mem = InMemoryStorage()
        for i in range(5):
            await mem.store(f"key_{i}", i)
        keys = await mem.list_keys()
        assert len(keys) == 5


# ---------------------------------------------------------------------------
# FileStorage
# ---------------------------------------------------------------------------


class TestFileStorage:
    """Tests for the file-backed persistent storage backend."""

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("greeting", "hello")
        result = await fs.retrieve("greeting")
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_persists_to_disk(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("city", "London")

        # File should now exist
        assert os.path.exists(path)

        # Load JSON directly and verify structure
        with open(path, "r") as fh:
            data = json.load(fh)
        assert "city" in data
        assert data["city"]["value"] == "London"

    @pytest.mark.asyncio
    async def test_reloads_from_disk(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs1 = FileStorage(path)
        await fs1.store("language", "Python")

        # New instance should load from the same file
        fs2 = FileStorage(path)
        result = await fs2.retrieve("language")
        assert result == "Python"

    @pytest.mark.asyncio
    async def test_missing_file_starts_empty(self, tmp_path):
        path = str(tmp_path / "does_not_exist.json")
        fs = FileStorage(path)
        keys = await fs.list_keys()
        assert keys == []

    @pytest.mark.asyncio
    async def test_retrieve_missing_key_returns_none(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        result = await fs.retrieve("missing")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_keys(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("a", 1)
        await fs.store("b", 2)
        keys = await fs.list_keys()
        assert sorted(keys) == ["a", "b"]

    @pytest.mark.asyncio
    async def test_clear_removes_all_entries(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("x", 10)
        await fs.store("y", 20)
        await fs.clear()
        keys = await fs.list_keys()
        assert keys == []

    @pytest.mark.asyncio
    async def test_clear_removes_file(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("temp", "data")
        assert os.path.exists(path)
        await fs.clear()
        assert not os.path.exists(path)

    @pytest.mark.asyncio
    async def test_overwrite_existing_key(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("counter", 1)
        await fs.store("counter", 99)
        result = await fs.retrieve("counter")
        assert result == 99

    @pytest.mark.asyncio
    async def test_store_with_metadata(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("key", "value", metadata={"tag": "test"})
        result = await fs.retrieve("key")
        assert result == "value"

    @pytest.mark.asyncio
    async def test_persists_metadata_to_disk(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("key", "value", metadata={"env": "prod"})

        with open(path, "r") as fh:
            data = json.load(fh)
        assert data["key"]["metadata"]["env"] == "prod"

    @pytest.mark.asyncio
    async def test_reload_preserves_all_entries(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs1 = FileStorage(path)
        entries = {"alpha": "a", "beta": "b", "gamma": "c"}
        for k, v in entries.items():
            await fs1.store(k, v)

        fs2 = FileStorage(path)
        for k, v in entries.items():
            assert await fs2.retrieve(k) == v

    @pytest.mark.asyncio
    async def test_corrupt_file_starts_fresh(self, tmp_path):
        path = str(tmp_path / "memory.json")
        # Write invalid JSON
        with open(path, "w") as fh:
            fh.write("not valid json {{")

        fs = FileStorage(path)
        keys = await fs.list_keys()
        assert keys == []

    @pytest.mark.asyncio
    async def test_store_numeric_values(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("pi", 3.14159)

        # Reload and verify
        fs2 = FileStorage(path)
        result = await fs2.retrieve("pi")
        assert abs(result - 3.14159) < 1e-5

    @pytest.mark.asyncio
    async def test_store_dict_value(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("config", {"host": "localhost", "port": 8080})

        fs2 = FileStorage(path)
        result = await fs2.retrieve("config")
        assert result == {"host": "localhost", "port": 8080}

    @pytest.mark.asyncio
    async def test_store_list_value(self, tmp_path):
        path = str(tmp_path / "memory.json")
        fs = FileStorage(path)
        await fs.store("tags", ["python", "ai", "agent"])

        fs2 = FileStorage(path)
        result = await fs2.retrieve("tags")
        assert result == ["python", "ai", "agent"]
