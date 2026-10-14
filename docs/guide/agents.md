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
    max_steps=10,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | LiteLLM model string, e.g. `"gpt-4o-mini"`, `"groq/llama-3.1-8b-instant"`, `"ollama/qwen3:4b"`. |
| `temperature` | `float` | `0.1` | Sampling temperature. Lower values are more deterministic. |
| `max_tokens` | `int \| None` | `None` | Maximum tokens in the response. `None` uses the model's default. |
| `system_prompt` | `str \| None` | `None` | System instruction prepended to every conversation. |
| `tool_mode` | `str` | `"auto"` | `"auto"` uses native function calling and switches this agent to `"naive"` if LiteLLM reports the model does not support tools; `"native"` forces it; `"naive"` uses prompt-based tool parsing (works with any model). |
| `router` | `Router \| None` | `None` | Optional LiteLLM `Router` for fallback and load balancing. |
| `max_steps` | `int` | `10` | Most LLM completions in one run, guard rounds included; at least 1. |
| `max_total_tokens` | `int \| None` | `None` | Most tokens one run may use, checked between steps. |
| `max_cost` | `float \| None` | `None` | Most USD one run may cost, checked between steps. |
| `hooks` | `AgentHooks \| None` | `None` | Lifecycle callbacks for observability and tool approval. See [Hooks](hooks.md). |
| `completion_kwargs` | `dict` | `{}` | Extra arguments for every LiteLLM completion call, also through a `router`. See [Provider options](#provider-options). |
| `prompt_caching` | `bool` | `False` | Mark cache breakpoints for models with prompt caching on Anthropic, Bedrock and Vertex AI (Anthropic models). See [Prompt caching](#prompt-caching). |

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

When you need token counts or cost estimates alongside the text reply, use `run_with_response()`. It returns a `RunResult` instead of a bare string.

```python
from cyclops import RunResult

result: RunResult = agent.run_with_response("Summarize quantum computing in two sentences.")

print(result.content)
print(f"Tokens: {result.tokens_used}  Cost: ${result.cost:.6f}")
print(f"Tool calls made: {len(result.tool_calls)}")
```

See [Cost Tracking](cost-tracking.md) for the full field reference.

## stream() and astream()

`stream()` yields text tokens as they are produced. Every completion in the run is streamed, including the steps that call tools.

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

## start(): steer and cancel a run

`start()` begins a run in a task on the running event loop and returns a `RunHandle`. Await the handle for the `RunResult`, iterate it (once) for the run's [events](streaming.md#run-events), and use it to steer or stop the run while it works.

```python
from cyclops.core.events import TextDelta, ToolCallStarted

handle = agent.start("Research the three largest moons of Jupiter.")

async for event in handle:
    if isinstance(event, ToolCallStarted) and event.name == "web_search":
        handle.steer("Only use sources from the last five years.")
    elif isinstance(event, TextDelta):
        print(event.text, end="", flush=True)

result = await handle
```

- `handle.steer(message)` adds a user message that the model sees before its next completion. It is added once the current step's tool calls finish; if the model already gave its final answer in that step, the run takes one more step to answer the message. Steering a run that has finished raises `RuntimeError`.
- `handle.cancel()` stops the run. Tool calls in progress are cancelled and get a cancelled result, so the conversation stays valid, and awaiting the handle gives `stop_reason="cancelled"`. A completion in progress is dropped.

`start(input, stream=False)` makes the completions without streaming, so each step's text arrives as one `TextDelta`. `arun()`, `arun_with_response()` and `astream()` are shortcuts over `start()`. Iterating a handle doesn't hold the run back the way `astream_events()` does: events wait in the handle until you read them.

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

## Provider options

`completion_kwargs` is passed to every LiteLLM completion call the agent makes, so any argument LiteLLM accepts works per agent: credentials, endpoints, headers, retries, timeouts and reasoning settings. Keys here win over `temperature` and `max_tokens`.

```python
import os

config = AgentConfig(
    model="anthropic/claude-sonnet-4-5",
    temperature=1,  # Anthropic requires 1 with extended thinking
    completion_kwargs={
        "api_key": os.environ["MY_ANTHROPIC_KEY"],
        "extra_headers": {"X-Request-Source": "docs"},
        "num_retries": 3,
        "timeout": 60,
        "max_tokens": 8192,
        "thinking": {"type": "enabled", "budget_tokens": 4096},
    },
)
```

For OpenAI style reasoning models use `"reasoning_effort": "low"` and so on. See the [LiteLLM input docs](https://docs.litellm.ai/docs/completion/input) for every argument.

## Reasoning in history

When a model returns reasoning, the assistant messages in `agent.messages` keep it as `reasoning_content` and, for Anthropic extended thinking, `thinking_blocks` with their signatures. They are sent back on the next call, which Anthropic requires when thinking is combined with tool use: the signed thinking block has to precede the tool call it led to.

```python
agent.run("What is 17 * 23? Use the calculator.")
for msg in agent.messages:
    if "thinking_blocks" in msg:
        print(msg["thinking_blocks"][0]["thinking"])
```

## Prompt caching

`prompt_caching=True` asks LiteLLM to mark two cache breakpoints on every call: the system prompt, which caches the tool definitions before it too, and the latest message, which caches the conversation so far. Each tool round then reads the earlier prefix from the cache.

The markers are sent only to providers that cache at them (Anthropic, Bedrock, and Anthropic models on Vertex AI), and only for models that LiteLLM lists with prompt caching (`supports_prompt_caching`). Through a `router`, every deployment of the model has to qualify. Every other model gets no markers: OpenAI caches long prefixes without them, Gemini would turn them into billed explicit cache resources, and Bedrock models that LiteLLM lists without prompt caching (such as Llama) get none. A model missing from LiteLLM's model map gets no markers either.

```python
config = AgentConfig(
    model="anthropic/claude-sonnet-4-5",
    system_prompt=LONG_INSTRUCTIONS,
    prompt_caching=True,
)
agent = Agent(config)
response = agent.run_with_response("Summarise the report.")
print(response.cached_tokens)
```

Providers only cache a prefix above a minimum length (1024 tokens for most Anthropic models). To place breakpoints yourself, leave `prompt_caching` off and pass LiteLLM's `cache_control_injection_points` in `completion_kwargs`.

## Step and budget limits

When tools are in use the agent can call tools several times before producing the final answer. Three limits bound one run, each checked between steps (a step is one LLM completion and the tool calls it asks for):

- `max_steps` caps the number of completions (default `10`).
- `max_total_tokens` caps the tokens summed over the run's completions.
- `max_cost` caps the run's estimated USD cost.

Reaching a limit is not an error. The run ends with `stop_reason="max_steps"` or `stop_reason="budget"` on the `RunResult` from `run_with_response()`, and `content` is empty. A step in progress always finishes, so a run can exceed a budget by its last step. A completion that reports no usage, or a cost LiteLLM can't price, doesn't count against a budget.

```python
config = AgentConfig(
    model="groq/llama-3.1-8b-instant",
    max_steps=5,
    max_total_tokens=20_000,
    max_cost=0.05,  # USD
)
```
