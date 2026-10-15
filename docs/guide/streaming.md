# Streaming

## stream(): no tools

When an agent has no tools, `stream()` provides true token-by-token streaming directly from the LLM. Each iteration of the loop yields a small string chunk as it arrives.

```python
from cyclops import Agent, AgentConfig

config = AgentConfig(model="groq/llama-3.1-8b-instant")
agent = Agent(config)

for token in agent.stream("Write a haiku about the ocean."):
    print(token, end="", flush=True)
print()  # final newline
```

With a [session](sessions.md), the streamed reply is added to its transcript whole, so the next run on that session sees it.

## astream(): async streaming

`astream()` is the async version. It returns an `AsyncIterator[str]` and must be consumed inside an `async for` loop.

```python
import asyncio
from cyclops import Agent, AgentConfig


async def main():
    config = AgentConfig(model="groq/llama-3.1-8b-instant")
    agent = Agent(config)

    async for token in agent.astream("Explain how rainbows form."):
        print(token, end="", flush=True)
    print()


asyncio.run(main())
```

Use `astream()` whenever you are already in an async context, such as inside a FastAPI endpoint or an async CLI.

## stream(): with tools

When the agent has tools configured, `stream()` streams every completion of the run. Text the model writes before a tool call arrives as it is generated; the tool calls are assembled from the stream, the tools run, and the next completion streams in turn. From the caller's perspective the API is identical: you just iterate the generator:

```python
from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool
from datetime import datetime


@tool
def current_time() -> str:
    """Return the current UTC time."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


config = AgentConfig(model="groq/llama-3.1-8b-instant")
agent = Agent(config, tools=[current_time])

for token in agent.stream("What time is it, and what day of the week is today?"):
    print(token, end="", flush=True)
print()
```

/// note
Tokens pause while tools run. In naive tool mode the model's text may itself be a tool call, so the final answer arrives as one chunk.
///

## astream(): async with tools

The async path works identically to the sync path, just awaited:

```python
import asyncio
from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool


@tool
async def lookup_price(ticker: str) -> str:
    """Return a mock stock price for a ticker symbol."""
    prices = {"AAPL": 189.50, "GOOGL": 172.30, "MSFT": 415.00}
    return str(prices.get(ticker.upper(), "unknown"))


async def main():
    config = AgentConfig(model="groq/llama-3.1-8b-instant")
    agent = Agent(config, tools=[lookup_price])

    async for token in agent.astream("What is the price of Apple stock?"):
        print(token, end="", flush=True)
    print()


asyncio.run(main())
```

## Run events

`astream_events()` (and its sync twin `stream_events()`) streams everything a run does as typed events from `cyclops.core.events`, ending with `RunFinished`:

```python
from cyclops.core.events import (
    ReasoningDelta,
    RunFinished,
    StepFinished,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)

async for event in agent.astream_events("What is 2 + 3?"):
    if isinstance(event, TextDelta):
        print(event.text, end="", flush=True)
    elif isinstance(event, ToolCallStarted):
        print(f"\n-> {event.name}({event.arguments})")
    elif isinstance(event, ToolCallFinished):
        print(f"<- {event.result}{' (error)' if event.is_error else ''}")
    elif isinstance(event, RunFinished):
        result = event.result
        print(f"\n{result.stop_reason}: {result.tokens_used} tokens, ${result.cost}")
```

| Event | Fields | When |
|---|---|---|
| `TextDelta` | `text` | A piece of the answer text. |
| `ReasoningDelta` | `text` | A piece of the model's reasoning, for models that return it. |
| `ToolCallStarted` | `call_id`, `name`, `arguments` | Before a tool runs, with the arguments the model sent. `call_id` is `None` in naive tool mode. |
| `ToolCallFinished` | `call_id`, `name`, `result`, `is_error` | After a tool returns, fails, is cancelled or is denied by middleware. `result` is the text sent to the model. |
| `GuardRejected` | `feedback` | After a rejected answer's text, when a [guard](hooks.md#end-of-run-guards) sends the model back. |
| `OutputRejected` | `feedback` | After an answer's text, when it failed `response_model` validation and the model is [sent back](structured-output.md#retries) with the errors. |
| `CompletionRetry` | `model`, `delay`, `status_code`, `headers`, `rate_limit`, `error` | Before a completion that failed with a transient error or stalled is [retried](agents.md#retries-and-timeouts). See [Rate limits](#rate-limits). |
| `ModelFallback` | `model`, `fallback_model`, `error`, `usage`, `status_code`, `headers`, `rate_limit` | When a model failed or refused and the step's completion goes to the next of the [fallback models](agents.md#fallback-models). `error` is `None` for a refusal, whose usage is in `usage`. |
| `StepFinished` | `step`, `model`, `usage`, `finish_reason`, `headers`, `rate_limit` | After one completion and the tool calls it asked for. `model` is the model that answered. |
| `Compacted` | `summary`, `replaced_messages`, `tokens_before`, `tokens_after` | Before a completion, when a summary replaced older messages. See [Context window](agents.md#context-window). |
| `RunFinished` | `result` | Last event, with the run's `RunResult`, including usage and cost. |

Every event has `agent_path`, the names of the nested agents it came from, outermost first. It is empty for events of the agent you ran; events of a [subagent](subagents.md) carry its tool name, and its `RunFinished` comes before the outer run's.

### Rate limits

`StepFinished`, `CompletionRetry` and `ModelFallback` carry the provider's HTTP response `headers`, as LiteLLM passes them on, and `rate_limit`, the limits read from them. `CompletionRetry` and `ModelFallback` also carry the error's HTTP `status_code`; for a refusal, `ModelFallback` has the refused response's headers and no status.

```python
from cyclops.core.events import CompletionRetry, StepFinished

async for event in agent.astream_events("Hi"):
    if isinstance(event, CompletionRetry):
        print(f"{event.status_code}: retrying in {event.delay:.1f}s")
    if isinstance(event, (StepFinished, CompletionRetry)) and event.rate_limit:
        print(f"{event.rate_limit.remaining_tokens} tokens left")
```

`RateLimit` has `remaining_requests`, `reset_requests`, `remaining_tokens` and `reset_tokens`, each `None` when the headers lack it; a reset is the seconds from the response until that limit resets. It reads OpenAI's `x-ratelimit-*` headers, which many OpenAI-compatible providers send too, and Anthropic's `anthropic-ratelimit-*` headers. Gemini and Bedrock send no rate limit headers, so their `rate_limit` is `None`, and LiteLLM passes on no headers of a Gemini response.

An error that ends the run is in `RunResult.error`. LiteLLM's exceptions have the `status_code` and the provider's headers in `litellm_response_headers`, so `RateLimit.from_headers(error.litellm_response_headers)` reads its limits.

A failed run doesn't raise from `astream_events()`: it ends with `RunFinished` whose result has `stop_reason="error"`. `astream()` and `stream()` yield only the text of the agent you ran and raise its error. The run waits for you to take its events: within a completion it gets at most 16 text or reasoning deltas ahead of you, and it starts no tool call or completion until you have taken every event before it, so a loop you `break` out of holds the run. Closing the iterator (or leaving an `aclosing()` block) cancels the run; tool calls in progress get a cancelled result so the conversation stays valid. If a tool or hook raises `CancelledError`, the iterator raises it too.

## Building a streaming HTTP endpoint

A common pattern is to expose the stream over HTTP using Server-Sent Events:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from cyclops import Agent, AgentConfig

app = FastAPI()
config = AgentConfig(model="groq/llama-3.1-8b-instant")


@app.get("/chat")
async def chat(message: str):
    agent = Agent(config)

    async def token_generator():
        async for token in agent.astream(message):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")
```
