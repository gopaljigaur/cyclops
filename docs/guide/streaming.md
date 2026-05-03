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

The streamed tokens are also accumulated into the conversation history, so the next call to `run()` or `stream()` correctly sees the full previous assistant reply.

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

When the agent has tools configured, `stream()` works in two phases:

1. **Tool loop** (non-streaming): the agent sends the message, detects tool calls, executes them, and repeats until there are no more tool calls.
2. **Final answer** (streaming): once all tool work is done, the final response is streamed token by token.

From the caller's perspective the API is identical: you just iterate the generator:

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

# Tokens start arriving only after the tool call completes.
for token in agent.stream("What time is it, and what day of the week is today?"):
    print(token, end="", flush=True)
print()
```

/// note
There will be a brief pause before tokens begin for tool-enabled agents: this is expected. The tool execution must finish before the final streaming call is made.
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
