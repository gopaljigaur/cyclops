---
title: Getting Started
description: Install cyclops and run your first agent.
---

## Installation

```bash
pip install cyclops-ai
```

Or with uv (recommended):

```bash
uv add cyclops-ai
```

Python 3.10 or later is required.

## Set your API key

Cyclops uses [LiteLLM](https://github.com/BerriAI/litellm) under the hood, so credentials are just environment variables. Set the one for the provider you want.

```bash
# Groq: free tier, fast (good for experimenting)
export GROQ_API_KEY="gsk_..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Together AI
export TOGETHERAI_API_KEY="..."

# Ollama: no key needed; install from https://ollama.ai
```

You can put these in a `.env` file and load them with `python-dotenv`.

## First agent

Three lines to get a response:

```python
from cyclops import Agent, AgentConfig

agent = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"))
print(agent.run("What is the capital of France?"))
```

The agent keeps conversation history automatically. The next call sees the full prior context:

```python
agent.run("What is the capital of France?")
agent.run("And what is its population?")  # knows "it" means Paris
```

## First agent with tools

Decorate any function with `@tool`. The function name becomes the tool name; the docstring becomes the description; type annotations become the JSON schema.

```python
from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool
from datetime import datetime


@tool
def get_time() -> str:
    """Return the current time as HH:MM:SS."""
    return datetime.now().strftime("%H:%M:%S")


@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the result."""
    return a + b


agent = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"), tools=[get_time, add])

print(agent.run("What time is it right now?"))
print(agent.run("What is 42 plus 58?"))
```

## Async usage

Every sync method has an async counterpart. Use `arun`, `astream`, and `arun_with_response` inside any async context.

```python
import asyncio
from cyclops import Agent, AgentConfig


async def main():
    agent = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"))

    # Sequential calls share history.
    await agent.arun("Name three planets.")
    await agent.arun("Which is largest?")

    # Run two independent agents concurrently.
    a = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"))
    b = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"))
    results = await asyncio.gather(
        a.arun("Capital of Japan?"),
        b.arun("Capital of Brazil?"),
    )
    print(results)


asyncio.run(main())
```

## Switch providers

Change the model string. Nothing else changes.

```python
# Ollama (local, no API key)
Agent(AgentConfig(model="ollama/qwen3:4b"))

# OpenAI
Agent(AgentConfig(model="gpt-4o-mini"))

# Anthropic
Agent(AgentConfig(model="claude-haiku-4-5-20251001"))

# Together AI
Agent(AgentConfig(model="together_ai/meta-llama/Llama-3-8b-chat-hf"))
```

Ollama runs entirely on your machine. Pull a model with `ollama pull qwen3:4b` and use `"ollama/<model-name>"` with no API key.
