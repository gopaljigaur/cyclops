---
title: Hooks
description: Observe and control agent execution with lifecycle callbacks.
---

`AgentHooks` is a lifecycle callback system that lets you observe and control every stage of an agent run — from the initial call, through each LLM request, to every tool execution. Subclass `AgentHooks`, override the methods you need, and pass an instance to `AgentConfig`.

## Quick example

```python
from cyclops import Agent, AgentConfig, AgentHooks


class LoggingHooks(AgentHooks):
    def on_run_start(self, input_message: str) -> None:
        print(f"[run] {input_message!r}")

    def on_llm_end(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            print(f"[llm] tokens={usage.total_tokens}")

    def on_tool_start(self, tool_name: str, args: dict):
        print(f"[tool] {tool_name}({args})")


config = AgentConfig(
    model="groq/llama-3.1-8b-instant",
    hooks=LoggingHooks(),
)
agent = Agent(config)
agent.run("What is the weather in London?")
```

## Available hooks

| Method | When it fires | Return value |
|---|---|---|
| `on_run_start(input_message)` | Once at the start of `run()`, `arun()`, `stream()`, or `astream()` | `None` |
| `on_run_end(content)` | After `run()` or `arun()` returns (not called for `stream()`/`astream()`) | `None` |
| `on_llm_start(messages)` | Before each LiteLLM completion call | `None` |
| `on_llm_end(response)` | After each non-streaming LiteLLM completion call | `None` |
| `on_llm_error(error)` | When a LiteLLM call raises an exception | `None` |
| `on_tool_start(tool_name, args)` | Before each tool execution | `"deny"` to block, anything else to allow |
| `on_tool_end(tool_name, args, result)` | After a tool executes successfully | `None` |
| `on_tool_error(tool_name, args, error)` | When a tool raises an exception | `None` |

All methods are no-ops by default. Override only the ones you need.

## Tool approval with on_tool_start

`on_tool_start` is the only hook with side effects: returning the string `"deny"` blocks the tool call. Any other return value (including `None`) allows it. The agent receives `"[Tool '<name>' was not approved]"` as the tool result and continues.

```python
from cyclops import Agent, AgentConfig, AgentHooks


ALLOWED_TOOLS = {"search", "calculator"}


class AllowListHooks(AgentHooks):
    def on_tool_start(self, tool_name: str, args: dict):
        if tool_name not in ALLOWED_TOOLS:
            print(f"Blocked: {tool_name}")
            return "deny"


config = AgentConfig(model="gpt-4o-mini", hooks=AllowListHooks())
```

## Observability example

Accumulate token usage and cost across multiple runs:

```python
from cyclops import Agent, AgentConfig, AgentHooks


class UsageHooks(AgentHooks):
    def __init__(self):
        self.total_tokens = 0
        self.total_cost = 0.0

    def on_llm_end(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def on_run_end(self, content: str) -> None:
        print(f"Run complete. Total tokens so far: {self.total_tokens}")


hooks = UsageHooks()
config = AgentConfig(model="gpt-4o-mini", hooks=hooks)
agent = Agent(config)

agent.run("Summarise the Pythagorean theorem.")
agent.run("What is the derivative of x squared?")

print(f"Session total: {hooks.total_tokens} tokens")
```

## Error handling example

Log errors without crashing the agent run:

```python
import logging
from cyclops import Agent, AgentConfig, AgentHooks

logger = logging.getLogger(__name__)


class ErrorHooks(AgentHooks):
    def on_llm_error(self, error: Exception) -> None:
        logger.error("LLM call failed: %s", error)

    def on_tool_error(self, tool_name: str, args: dict, error: Exception) -> None:
        logger.error("Tool %r failed with args %r: %s", tool_name, args, error)
```

## Default behaviour

The default `AgentHooks` base class implements `on_tool_start` to return `"allow"` and all other methods as no-ops. You do not need to call `super()` in your overrides.
