# Getting Started

## Installation

Install Cyclops with pip:

```bash
pip install cyclops-ai
```

Or with uv (faster, recommended for new projects):

```bash
uv add cyclops-ai
```

Python 3.10 or later is required.

## Environment variables

Cyclops uses LiteLLM under the hood, so provider credentials are set as environment variables. You only need the key for the provider you choose.

```bash
# Groq: free tier, fast inference (recommended for experimenting)
export GROQ_API_KEY="gsk_..."

# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Together AI: free tier available
export TOGETHERAI_API_KEY="..."

# Ollama: no key needed; just install and run locally
# https://ollama.ai
```

You can also place these in a `.env` file and load them with `python-dotenv` or any equivalent tool.

## First agent (no tools)

The minimum working example creates an `AgentConfig`, passes it to `Agent`, and calls `run()`.

```python
from cyclops import Agent, AgentConfig

config = AgentConfig(
    model="groq/llama-3.1-8b-instant",
    system_prompt="You are a helpful assistant.",
)
agent = Agent(config)

response = agent.run("What is the capital of France?")
print(response)  # "The capital of France is Paris."

# The agent keeps conversation history automatically.
follow_up = agent.run("And what is its population?")
print(follow_up)
```

## First agent with tools

Decorate any Python function with `@tool` and pass the list to `Agent`. The function's docstring becomes the tool description; type annotations become the JSON schema.

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


config = AgentConfig(model="groq/llama-3.1-8b-instant")
agent = Agent(config, tools=[get_time, add])

print(agent.run("What time is it right now?"))
print(agent.run("What is 42 plus 58?"))
```

## Async usage

Every synchronous method has an async counterpart (`arun`, `astream`, `arun_with_response`). Use them in any async context.

```python
import asyncio
from cyclops import Agent, AgentConfig


async def main():
    config = AgentConfig(model="groq/llama-3.1-8b-instant")
    agent = Agent(config)

    # Sequential async calls share conversation history.
    answer1 = await agent.arun("Name three planets.")
    answer2 = await agent.arun("Which of those is largest?")
    print(answer1)
    print(answer2)

    # Or run independent questions concurrently (separate agents).
    agent_a = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"))
    agent_b = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"))
    results = await asyncio.gather(
        agent_a.arun("Capital of Japan?"),
        agent_b.arun("Capital of Brazil?"),
    )
    print(results)


asyncio.run(main())
```

## Multiple providers

Cyclops supports every provider LiteLLM knows about. Swap the model string; no other changes needed.

```python
from cyclops import Agent, AgentConfig

# Groq (free, fast)
agent = Agent(AgentConfig(model="groq/llama-3.1-8b-instant"))

# Ollama (local, no API key)
agent = Agent(AgentConfig(model="ollama/qwen3:4b"))

# OpenAI
agent = Agent(AgentConfig(model="gpt-4o-mini"))

# Anthropic
agent = Agent(AgentConfig(model="claude-haiku-4-5-20251001"))

# Together AI (free tier)
agent = Agent(AgentConfig(model="together_ai/meta-llama/Llama-3-8b-chat-hf"))
```

/// note
Ollama runs entirely on your machine. Install it from [ollama.ai](https://ollama.ai), pull a model (`ollama pull qwen3:4b`), and use `"ollama/<model-name>"` (no API key required).
///
