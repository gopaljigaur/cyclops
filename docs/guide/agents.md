# Agents

## AgentConfig

`AgentConfig` is a Pydantic model that holds all runtime settings for an agent. Every field has a sensible default, so you only need to set what you care about.

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
| `temperature` | `float` | `0.1` | Sampling temperature. Lower values are more deterministic. |
| `max_tokens` | `int \| None` | `None` | Maximum tokens in the response. `None` uses the model's default. |
| `system_prompt` | `str \| None` | `None` | System instruction prepended to every conversation. |
| `tool_mode` | `str` | `"auto"` | `"auto"` detects native function-calling support; `"native"` forces it; `"naive"` uses prompt-based tool parsing (works with any model). |
| `router` | `Router \| None` | `None` | Optional LiteLLM `Router` for fallback and load balancing. |
| `max_iterations` | `int` | `10` | Maximum number of tool-call rounds before the agent stops. |
| `hooks` | `AgentHooks \| None` | `None` | Lifecycle callbacks for observability and tool approval. See [Hooks](hooks.md). |

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

`run()` is the simplest entry point. It sends a message, handles any tool calls, and returns the final text response.

```python
response = agent.run("Who invented the telephone?")
print(response)
```

`arun()` is the async equivalent, useful inside coroutines or when running multiple agents concurrently.

```python
import asyncio

async def main():
    response = await agent.arun("Who invented the telephone?")
    print(response)

asyncio.run(main())
```

Both methods accept an optional `response_model` parameter for structured output. See [Structured Output](structured-output.md).

## run_with_response() and arun_with_response()

When you need token counts or cost estimates alongside the text reply, use `run_with_response()`. It returns an `AgentResponse` object instead of a bare string.

```python
from cyclops import AgentResponse

result: AgentResponse = agent.run_with_response("Summarize quantum computing in two sentences.")

print(result.content)
print(f"Tokens: {result.tokens_used}  Cost: ${result.cost:.6f}")
print(f"Tool calls made: {len(result.tool_calls)}")
```

See [Cost Tracking](cost-tracking.md) for the full field reference.

## stream() and astream()

`stream()` yields text tokens as they are produced. For agents without tools this is true streaming straight from the LLM. For agents with tools, the tool loop runs first, then the final answer is streamed.

```python
for token in agent.stream("Explain black holes briefly."):
    print(token, end="", flush=True)
print()  # newline after stream ends
```

Async streaming works the same way:

```python
async def main():
    async for token in agent.astream("Explain black holes briefly."):
        print(token, end="", flush=True)
    print()
```

See [Streaming](streaming.md) for detailed examples.

## Conversation history

The agent accumulates every user and assistant message in `_history`. You can inspect the current history at any time via the `messages` property, which returns a shallow copy so the internal state is not accidentally mutated.

```python
agent.run("My name is Alice.")
agent.run("What is my name?")

for msg in agent.messages:
    print(msg["role"], ":", msg["content"][:60])
```

The history persists across calls as long as the same `Agent` instance is used. Each call appends a user message and an assistant reply (and any tool turns) to the history, so follow-up questions automatically have full context.

## reset()

Call `reset()` to clear the conversation history and start fresh without creating a new agent object.

```python
agent.run("Remember: the code word is banana.")
agent.reset()

# History is now empty; the agent has forgotten the code word.
response = agent.run("What was the code word?")
```

## max_iterations

When tools are in use the agent can call tools multiple times in a row before producing the final answer. `max_iterations` caps how many tool-call rounds are allowed. If the cap is hit the agent returns the string `"Reached maximum tool call iterations."`.

```python
config = AgentConfig(
    model="groq/llama-3.1-8b-instant",
    max_iterations=5,   # Allow up to 5 rounds of tool calls
)
```

For most tasks the default of `10` is more than enough. Lowering it prevents runaway loops in production.
