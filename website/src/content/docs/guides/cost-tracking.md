---
title: Cost Tracking
description: Track tokens and estimated cost for every agent run.
---

## run_with_response() vs run()

`run()` returns a plain string. `run_with_response()` returns an `AgentResponse` that includes the text reply, every tool call made, and token/cost metadata from the LiteLLM response.

```python
from cyclops import Agent, AgentConfig

config = AgentConfig(model="gpt-4o-mini")
agent = Agent(config)

result = agent.run_with_response("Summarise the theory of relativity in two sentences.")

print(result.content)
print(f"Model      : {result.model}")
print(f"Prompt     : {result.prompt_tokens} tokens")
print(f"Completion : {result.completion_tokens} tokens")
print(f"Total      : {result.tokens_used} tokens")
print(f"Cost       : ${result.cost:.6f}")
```

The async version works identically:

```python
import asyncio


async def main():
    result = await agent.arun_with_response("Explain neural networks briefly.")
    print(result.content)
    print(f"Cost: ${result.cost:.6f}")


asyncio.run(main())
```

## AgentResponse fields

| Field | Type | Description |
|---|---|---|
| `content` | `str` | The final text response from the agent. |
| `tool_calls` | `List[ToolCall]` | Every tool call made during the run, in order. |
| `model` | `str` | The model identifier used (from `AgentConfig.model`). |
| `tokens_used` | `int` or `None` | Total tokens consumed (prompt + completion). `None` if unavailable. |
| `prompt_tokens` | `int` or `None` | Tokens in the prompt/context sent to the model. |
| `completion_tokens` | `int` or `None` | Tokens in the generated response. |
| `cost` | `float` or `None` | Estimated USD cost from LiteLLM's pricing table. `None` for self-hosted models. |

Each `ToolCall` in `tool_calls` has:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique call identifier assigned by the LLM. |
| `name` | `str` | Name of the tool that was called. |
| `arguments` | `dict` | Arguments the model passed to the tool. |
| `result` | `Any` or `None` | The value returned by the tool. |

## Accumulating cost across multiple calls

Track totals across many runs with a simple dataclass:

```python
import asyncio
from dataclasses import dataclass, field
from typing import List
from cyclops import Agent, AgentConfig
from cyclops.core.types import AgentResponse


@dataclass
class UsageTracker:
    total_cost: float = 0.0
    total_tokens: int = 0
    runs: List[AgentResponse] = field(default_factory=list)

    def record(self, response: AgentResponse) -> None:
        self.runs.append(response)
        self.total_cost += response.cost or 0.0
        self.total_tokens += response.tokens_used or 0

    def report(self) -> None:
        print(f"Runs         : {len(self.runs)}")
        print(f"Total cost   : ${self.total_cost:.4f}")
        print(f"Total tokens : {self.total_tokens:,}")


tracker = UsageTracker()
config = AgentConfig(model="gpt-4o-mini")
agent = Agent(config)

questions = [
    "What is photosynthesis?",
    "Explain DNS in one sentence.",
    "What is the boiling point of water in Celsius?",
]

for q in questions:
    result = agent.run_with_response(q)
    tracker.record(result)
    print(f"Q: {q}")
    print(f"A: {result.content}")
    print(f"   cost={result.cost:.6f}  tokens={result.tokens_used}\n")

tracker.report()
```

## Inspecting tool calls

When tools are used, each invocation is recorded in `result.tool_calls`:

```python
from cyclops.toolkit import tool


@tool
def lookup(symbol: str) -> str:
    """Look up a stock price."""
    return "189.50"


config = AgentConfig(model="gpt-4o-mini")
agent = Agent(config, tools=[lookup])

result = agent.run_with_response("What is the current price of AAPL?")

print(result.content)
for tc in result.tool_calls:
    print(f"  Called: {tc.name}({tc.arguments}) -> {tc.result}")
```

:::note
Cost tracking depends on LiteLLM's internal pricing table. Ollama and other self-hosted models will have `cost=None` since no billing data is available. Token counts are still reported if the server returns usage information.
:::
