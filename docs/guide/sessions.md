# Sessions

A `Session` holds one conversation: its transcript (user and assistant messages, tool calls, tool results and reasoning blocks), its `id`, and the usage and cost of every run in it. Pass it to any run with `session=`; the run then reads and extends the session's transcript, so one agent can serve many conversations, also at the same time. A run without a session starts from a fresh transcript and keeps nothing once it ends: sessions are how a conversation continues.

```python
from cyclops import Agent, AgentConfig, Session

agent = Agent(AgentConfig(model="gpt-4o-mini"))
session = Session()

agent.run("My name is Alice.", session=session)
print(agent.run("What is my name?", session=session))

print(session.id, len(session.messages), session.usage)
```

Every entry point takes `session=`: `run`, `arun`, `run_with_response`, `arun_with_response`, `stream`, `astream`, `stream_events`, `astream_events` and `start`.

`session.usage` sums the usage of every run, including runs that fail or that `handle.cancel()` stops. Like a run's own `usage`, it becomes `None` once a run reports no usage.

A run that fails or is cancelled leaves what it got to in the transcript: the user message, any completed steps, and an answer for every tool call it made (a cancelled call gets a cancelled result, a call whose middleware raised gets an error), so the session can go on. A completion in progress when the run stopped leaves nothing.

Don't run the same session twice at once: both runs would append to one transcript.

## Saving and loading

A session is a Pydantic model with a stable JSON format that carries a `version` field (currently `1`):

```python
data = session.model_dump_json()
session = Session.model_validate_json(data)
```

A `SessionStore` keeps sessions between runs or processes. It has four async methods: `load(session_id)` (returns `None` when there is no such session), `save(session)`, which replaces the stored session, `add_messages(session, messages)`, which adds messages to the end of a session saved before and stores its usage, and `delete(session_id)`. Cyclops ships two:

- `InMemorySessionStore()` keeps sessions in the process.
- `FileSessionStore(directory)` writes each session to `<directory>/<id>.jsonl`: the saved session on the first line, then one line per `add_messages`. Ids must be letters, digits, `_` or `-`.

Keep one writer per session: a store doesn't lock, so two processes saving the same session can interleave one's `save` with the other's `add_messages` and store a mix of both transcripts.

```python
from cyclops import FileSessionStore, Session

store = FileSessionStore("./sessions")


async def reply(user_id: str, message: str) -> str:
    session = await store.load(user_id) or Session(id=user_id)
    answer = await agent.arun(message, session=session)
    await store.save(session)
    return answer
```

To keep sessions in a database, write a class with the same four methods; `SessionStore` is a `Protocol`, so there is nothing to inherit.

## Saving automatically

Give the agent a store and every run with a session saves it there; `load` stays yours to call:

```python
store = FileSessionStore("./sessions")
agent = Agent(config, session_store=store)


async def reply(user_id: str, message: str) -> str:
    session = await store.load(user_id) or Session(id=user_id)
    return await agent.arun(message, session=session)
```

`durability` says when, with LangGraph's names:

- `"async"` (the default) saves after each step while the next one runs, and once more when the run ends.
- `"sync"` saves after each step before the next one starts, and when the run ends.
- `"exit"` saves once, when the run ends.

A run saves when it ends however it ends, also when it fails or is cancelled. Its first save stores the whole session with `save`, since the session may have changed since the store last saw it; each later save of the run passes only the new messages to `add_messages`.

A save that fails fails the run, like any other error: `stop_reason` is `"error"`, `error` is what the store raised, and `arun` raises it. With `"async"` the error of a step's save comes with the next save. The session in memory still has the whole transcript, so the next run, or your own `save`, can store it.

## Forking and going back

`session.fork()` returns a copy with a new id. The copy and the original go on independently, so one conversation can try two directions:

```python
draft = session.fork()
agent.run("Make it shorter.", session=draft)
```

`session.truncate(index)` keeps the messages before `index` (a negative index counts from the end), to retry from an earlier point. An index that would keep a tool call without its results raises `ValueError`; truncate before the assistant message with the tool calls instead.

```python
session.truncate(-2)  # drop the last question and its answer
agent.run("Ask it differently.", session=session)
```

Both change the session in memory only; the next run with a store, or your own `save`, stores the result.

## Moving from Memory

Earlier versions shipped `Memory`, `InMemoryStorage` and `FileStorage`, key value stores the agent never read. They are gone. To carry a conversation across runs or processes, keep it in a session and save it with a `SessionStore` as above. For other application state, use the store your application already has.
