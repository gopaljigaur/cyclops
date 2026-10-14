# Memory

## InMemoryStorage

`InMemoryStorage` keeps key-value pairs in a Python dictionary. Data is lost when the process exits. It is the right choice for single-session state, caching intermediate results, or testing.

```python
import asyncio
from cyclops import InMemoryStorage


async def main():
    memory = InMemoryStorage()

    await memory.store("user_name", "Alice")
    await memory.store("preferences", {"language": "Python", "level": "intermediate"})

    name = await memory.retrieve("user_name")       # "Alice"
    prefs = await memory.retrieve("preferences")    # {"language": "Python", ...}
    missing = await memory.retrieve("unknown_key")  # None

    keys = await memory.list_keys()  # ["user_name", "preferences"]
    await memory.clear()             # wipes all data


asyncio.run(main())
```

## FileStorage

`FileStorage` persists data to a JSON file on disk. It loads existing data at construction time and writes to disk after every `store()` call. It is suitable for cross-session context, user profiles, or any state that must survive restarts.

```python
from cyclops import FileStorage

memory = FileStorage(path="./data/agent_memory.json")
```

The parent directory is created automatically if it does not exist. If the file is corrupt or unreadable at startup, `FileStorage` starts with an empty store rather than raising an error.

```python
import asyncio


async def main():
    memory = FileStorage("./data/session.json")

    # Store context that will survive a process restart.
    await memory.store("last_query", "what is the weather in Tokyo")
    await memory.store("session_count", 42)

    # Retrieve on the next run:
    last_query = await memory.retrieve("last_query")
    print(last_query)  # "what is the weather in Tokyo"

    # Remove the file and clear in-memory state:
    await memory.clear()


asyncio.run(main())
```

## Memory abstract base

Both storage classes implement the `Memory` abstract base class. You can write your own backend (Redis, SQLite, a remote API) by subclassing `Memory` and implementing all four async methods.

```python
from cyclops.core.memory import Memory
from typing import Any, Dict, List, Optional


class RedisMemory(Memory):
    def __init__(self, url: str):
        import redis.asyncio as aioredis
        self.client = aioredis.from_url(url)

    async def store(self, key: str, value: Any, metadata: Optional[Dict] = None) -> None:
        import json
        await self.client.set(key, json.dumps(value))

    async def retrieve(self, key: str) -> Optional[Any]:
        import json
        raw = await self.client.get(key)
        return json.loads(raw) if raw else None

    async def list_keys(self) -> List[str]:
        keys = await self.client.keys("*")
        return [k.decode() for k in keys]

    async def clear(self) -> None:
        await self.client.flushdb()
```

## Using memory with Agent

The agent does not read or write memory itself (it no longer takes a `memory` argument): memory is a side channel for your application logic to pass context in and out of agent runs. To keep a conversation's history, use a [session](sessions.md) instead.

```python
import asyncio
from cyclops import Agent, AgentConfig, InMemoryStorage


async def main():
    memory = InMemoryStorage()
    await memory.store("user_name", "Alice")
    await memory.store("last_topic", "machine learning")

    config = AgentConfig(model="groq/llama-3.1-8b-instant")
    agent = Agent(config)

    # Build a context-aware prompt from memory.
    user_name = await memory.retrieve("user_name")
    last_topic = await memory.retrieve("last_topic")

    prompt = (
        f"The user's name is {user_name}. "
        f"They last asked about {last_topic}. "
        "Suggest a good follow-up resource for them."
    )

    response = await agent.arun(prompt)
    print(response)

    # Update memory with new session data.
    await memory.store("last_topic", "neural networks")


asyncio.run(main())
```

## Conversation history

To carry a conversation across runs or processes, keep it in a [session](sessions.md) and save it with a `SessionStore`, instead of rebuilding context from memory by hand.
