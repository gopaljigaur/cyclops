---
title: Memory
description: Persist state across agent calls and sessions.
---

## InMemoryStorage

`InMemoryStorage` keeps key-value pairs in a Python dictionary. Data is lost when the process exits. Use it for single-session state, caching intermediate results, or testing.

```python
import asyncio
from cyclops import InMemoryStorage


async def main():
    memory = InMemoryStorage()

    await memory.store("user_name", "Alice")
    await memory.store("preferences", {"language": "Python", "level": "intermediate"})

    name = await memory.retrieve("user_name")      # "Alice"
    prefs = await memory.retrieve("preferences")   # {"language": "Python", ...}
    missing = await memory.retrieve("unknown_key") # None

    keys = await memory.list_keys()  # ["user_name", "preferences"]
    await memory.clear()             # wipes all data


asyncio.run(main())
```

## FileStorage

`FileStorage` persists data to a JSON file on disk. It loads existing data at construction time and writes to disk after every `store()` call. Use it for cross-session context, user profiles, or any state that must survive restarts.

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

## The four methods

All memory backends implement the same async interface:

| Method | Signature | Description |
|---|---|---|
| `store` | `store(key, value, metadata=None)` | Write a value. Overwrites any existing entry for that key. |
| `retrieve` | `retrieve(key)` | Read a value. Returns `None` if the key does not exist. |
| `list_keys` | `list_keys()` | Return a list of all stored keys. |
| `clear` | `clear()` | Delete all entries. `FileStorage` also removes the backing file. |

## Using memory with Agent

Pass any `Memory` instance as the `memory` argument to `Agent`. The agent does not read or write memory automatically. Memory is a side channel for your application logic to pass context in and out of agent runs.

```python
import asyncio
from cyclops import Agent, AgentConfig, InMemoryStorage


async def main():
    memory = InMemoryStorage()
    await memory.store("user_name", "Alice")
    await memory.store("last_topic", "machine learning")

    config = AgentConfig(model="groq/llama-3.1-8b-instant")
    agent = Agent(config, memory=memory)

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

## Cross-session example

Load previous conversation history from a file, inject it into the system prompt, then save the updated history:

```python
import asyncio
import json
from cyclops import Agent, AgentConfig, FileStorage


async def chat_session(user_id: str, message: str) -> str:
    memory = FileStorage(f"./data/users/{user_id}.json")

    # Load history from the previous session.
    history_raw = await memory.retrieve("conversation_history") or []

    config = AgentConfig(
        model="groq/llama-3.1-8b-instant",
        system_prompt=(
            "You are a helpful assistant. "
            f"Previous context: {json.dumps(history_raw[-5:])}"  # last 5 turns
        ),
    )
    agent = Agent(config)
    response = await agent.arun(message)

    # Persist the updated history.
    history_raw.append({"user": message, "assistant": response})
    await memory.store("conversation_history", history_raw)

    return response


asyncio.run(chat_session("user_123", "What did we talk about last time?"))
```

## Custom backend

Implement the `Memory` abstract base class to use any storage backend (Redis, SQLite, a remote API):

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
