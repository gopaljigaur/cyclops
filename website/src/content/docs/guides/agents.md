---
title: Agents
description: Configure and run agents.
---

## AgentConfig

`AgentConfig` is a Pydantic model that holds all runtime settings. Every field has a sensible default; you only need to set what you care about.

```python
from cyclops import AgentConfig

config = AgentConfig(
    model="groq/llama-3.1-8b-instant",
    temperature=0.3,
    max_tokens=1024,
    system_prompt="You are a concise technical assistant.",
    tool_mode="auto",
    max_iterations=10,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | LiteLLM model string, e.g. `"gpt-4o-mini"`, `"groq/llama-3.1-8b-instant"`, `"ollama/qwen3:4b"`. |
| `temperature` | `float` | `0.1` | Sampling temperature. Lower is more deterministic. |
| `max_tokens` | `int` or `None` | `None` | Maximum tokens in the response. `None` uses the model default. |
| `system_prompt` | `str` or `None` | `None` | System instruction prepended to every conversation turn. |
| `tool_mode` | `str` | `"auto"` | `"auto"` detects native function-calling support; `"native"` forces it; `"naive"` uses prompt-based tool parsing (works with any model). |
| `router` | `Router` or `None` | `None` | Optional LiteLLM `Router` for fallback and load balancing. |
| `max_iterations` | `int` | `10` | Maximum number of tool-call rounds before the agent stops. |

## Constructing an Agent

Pass a config, an optional list of tools, and optional memory storage.

```python
from cyclops import Agent, AgentConfig, InMemoryStorage
from cyclops.toolkit import tool


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


memory = InMemoryStorage()
config = AgentConfig(model="groq/llama-3.1-8b-instant")
agent = Agent(config, tools=[greet], memory=memory)
```

## run() and arun()

`run()` is the simplest entry point. It sends a message, handles any tool calls, and returns the final text response as a string.

```python
response = agent.run("Who invented the telephone?")
print(response)
```

`arun()` is the async equivalent. Use it inside coroutines or when running multiple agents concurrently.

```python
import asyncio


async def main():
    response = await agent.arun("Who invented the telephone?")
    print(response)


asyncio.run(main())
```

Both methods accept an optional `response_model` parameter for structured output. See [Structured Output](/guides/structured-output/) for details.

## Multi-turn conversations

The agent accumulates every user and assistant message in its internal history. Each call appends a user message and an assistant reply, so follow-up questions automatically have full context.

```python
agent.run("My name is Alice.")
agent.run("What is my name?")  # correctly answers "Alice"
```

Inspect the current history at any time with the `messages` property. It returns a shallow copy, so the internal state is not accidentally modified.

```python
agent.run("My name is Alice.")
agent.run("What is my name?")

for msg in agent.messages:
    print(msg["role"], ":", msg["content"][:60])
```

Each message in `messages` is a plain dict with at least `"role"` and `"content"` keys. Tool messages also include `"tool_call_id"` and `"name"`.

## reset()

Call `reset()` to clear the conversation history and start fresh without creating a new agent.

```python
agent.run("Remember: the code word is banana.")
agent.reset()

# History is now empty.
response = agent.run("What was the code word?")
```

## run_with_response() and arun_with_response()

When you need token counts or cost estimates alongside the text reply, use `run_with_response()`. It returns an `AgentResponse` object instead of a plain string.

```python
result = agent.run_with_response("Summarize quantum computing in two sentences.")

print(result.content)
print(f"Tokens: {result.tokens_used}  Cost: ${result.cost:.6f}")
print(f"Tool calls made: {len(result.tool_calls)}")
```

See [Cost Tracking](/guides/cost-tracking/) for the full field reference.

## stream() and astream()

`stream()` yields text tokens as they arrive. For agents without tools this is true token-by-token streaming from the LLM. For agents with tools, the tool loop runs first, then the final answer streams.

```python
for token in agent.stream("Explain black holes briefly."):
    print(token, end="", flush=True)
print()
```

See [Streaming](/guides/streaming/) for detailed examples.

## max_iterations

When tools are in use, the agent can call tools multiple times before producing the final answer. `max_iterations` caps how many tool-call rounds are allowed. If the cap is reached, the agent returns the string `"Reached maximum tool call iterations."`.

```python
config = AgentConfig(
    model="groq/llama-3.1-8b-instant",
    max_iterations=5,  # allow up to 5 rounds of tool calls
)
```

The default of `10` is enough for almost every task. Lowering it prevents runaway loops in production.
