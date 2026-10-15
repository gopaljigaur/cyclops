# Hooks

`AgentHooks` is a lifecycle callback system that lets you observe and control every stage of an agent run: from the initial call, through each LLM request, to every tool execution. Subclass `AgentHooks`, override the methods you need, and pass an instance to `AgentConfig`.

## Quick example

```python
from cyclops import Agent, AgentConfig, AgentHooks


class LoggingHooks(AgentHooks):
    def on_run_start(self, input_message) -> None:
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
| `on_run_start(input_message)` | Once at the start of `run()`, `arun()`, `stream()`, or `astream()`, with the input text or its list of content parts | `None` |
| `on_run_end(content)` | When a run finishes, streamed or not | `None` |
| `on_run_error(error)` | Instead of `on_run_end` when a run raises, is cancelled, or its stream is closed early. A run its caller stopped (`handle.cancel()`, or a stream closed early) gets a `CancelledError` whose `args` are `(RUN_STOPPED,)`, from `cyclops.core.hooks`. A handle cancelled before its run started fires no hooks at all; its events still end with `RunFinished`. | `None` |
| `on_llm_start(messages)` | Before each LiteLLM completion call | `None` |
| `on_llm_end(response)` | After each LiteLLM completion; for a stream, with the assembled response | `None` |
| `on_llm_error(error)` | When a LiteLLM call raises an exception | `None` |
| `on_compaction(summary, tokens_before, tokens_after)` | After a summary replaced older messages, in a run or through `agent.compact()`. See [Context window](agents.md#context-window). | `None` |
| `on_tool_start(tool_name, args)` | Before each tool execution | `"deny"` to block, anything else to allow |
| `on_tool_end(tool_name, args, result)` | After a tool executes successfully; `result` is the tool's result as text | `None` |
| `on_tool_error(tool_name, args, error)` | When a tool raises an exception or is cancelled | `None` |

All methods are no-ops by default. Override only the ones you need.

In the tool hooks, `cyclops.core.hooks.current_tool_call_id()` returns the id of the native tool call being run (`None` in naive mode). Each native call runs in its own asyncio task, so the id is exact when calls run in parallel.

## Tool approval with on_tool_start

`on_tool_start` is the only hook with side effects: returning the string `"deny"` blocks the tool call. Any other return value (including `None`) allows it. The agent receives `"Error: Tool '<name>' was not approved"` as the tool result and continues.

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

## Tool middleware

For more control over tool calls, pass a list of `ToolMiddleware` to `AgentConfig.middleware`. Both methods are async and optional:

- `before_tool(call)` returns `None` to allow the call, `Deny(reason)` to block it, or a dict to replace the arguments. The model sees a denied call as an error result with your reason.
- `after_tool(call, result, error)` sees the `ToolResult` and, if the tool raised, the exception. Return a `ToolResult` to replace what the model sees, or `None` to keep it.

`call` is a `ToolCallContext` with `tool_name`, `arguments`, `call_id` (`None` in naive tool mode), `run_id`, `annotations` (the tool's `ToolAnnotations`) and `agent_path`.

```python
from cyclops import Agent, AgentConfig, Deny, ToolMiddleware, ToolResult


class ReadOnly(ToolMiddleware):
    async def before_tool(self, call):
        if not call.annotations.read_only:
            return Deny(f"{call.tool_name} may change data; ask the user first.")
        return None


class LimitRows(ToolMiddleware):
    async def before_tool(self, call):
        if call.tool_name == "search_rows":
            return {**call.arguments, "limit": min(call.arguments.get("limit", 50), 50)}
        return None


class Redact(ToolMiddleware):
    async def after_tool(self, call, result, error):
        if call.tool_name == "get_user":
            return ToolResult.success_result("[redacted]")
        return None


config = AgentConfig(model="gpt-4o-mini", middleware=[ReadOnly(), LimitRows(), Redact()])
```

Middleware run in list order before the call and in reverse order after it. A denied call skips the tool and the remaining middleware; `after_tool` still runs for the middleware that allowed it. `AgentHooks` tool callbacks run as the innermost middleware, so `on_tool_start` sees the final arguments and returning `"deny"` keeps working. For `TelemetryHooks` this means a call denied by middleware gets no tool span, and middleware work is not nested under the tool span.

If a tool call is cancelled, `after_tool` runs with the `CancelledError` and its return value is ignored. If middleware raises, the run fails with that exception and the other calls of the same step are cancelled.

## Tool permissions

`ToolPermissions` is a ready-made middleware for allow, ask and deny rules. Each rule is a tool name or an [fnmatch](https://docs.python.org/3/library/fnmatch.html) pattern such as `"github_*"`, matched case-sensitively.

```python
import asyncio

from cyclops import AgentConfig, Deny, ToolPermissions


async def approve(call):
    answer = await asyncio.to_thread(input, f"Run {call.tool_name}({call.arguments})? [y/N] ")
    return answer.lower() == "y" or Deny("The user declined this call.")


permissions = ToolPermissions(
    allow=["search_*", "get_*"],
    ask=["send_email"],
    deny=["*_delete_*"],
    approve=approve,
    default="deny",
)
config = AgentConfig(model="gpt-4o-mini", middleware=[permissions])
```

A call takes the strictest rule it matches: deny, then ask, then allow. A call that matches no rule gets `default` (`"allow"` unless you set it), so allow rules matter when the default is `"ask"` or `"deny"`.

An "ask" call awaits `approve(call)` with the `ToolCallContext`. Return `True` to run the call, `False` to deny it, or `Deny(reason)` to deny it with your own reason. `call.annotations` carries the tool's hints, such as `destructive`, if the callback wants them. `ask` rules or `default="ask"` without `approve` raise `ValueError`. If `approve` raises, the run fails as it does when any middleware raises, and the tool does not run.

Approvals happen where the call would run, so they follow the same parallelism. Calls to tools not marked `read_only` are asked about one at a time, in the model's order, each after the previous one has run. `read_only` calls are asked about concurrently, so a callback that prompts a person, like the one above, should expect overlapping questions for them, or allow read-only tools outright. A call waiting on `approve` holds one of the `max_parallel_tool_calls` slots, and `tool_timeout` does not count the wait.

A denied call reaches the model as an error result, like any `Deny`. `RunResult.denied_tool_calls` lists every call middleware denied in the run, `ToolPermissions` or otherwise, with its `id` (`None` in naive tool mode), `name`, `arguments` and `reason`:

```python
result = agent.run_with_response("Clean up old branches")
for denied in result.denied_tool_calls:
    print(f"{denied.name}: {denied.reason}")
```

## End-of-run guards

A guard runs when the model ends its turn without calling a tool. It is an async function that takes a `GuardContext` (`content`, `messages`, `run_id`, `agent_path`) and returns feedback or `None`. Feedback is added as a user message and the model runs again. Guards run in order and the first one with feedback wins. `max_guard_rounds` (default `3`) caps how many times per run guards may send the model back. If a guard still has feedback after that, the run ends with `stop_reason="guard"` and empty `content`. Each guard round is a step, so it counts against `max_steps` and the token and cost budgets; if a limit stops the run first, the feedback is not added to the conversation.

```python
from cyclops import Agent, AgentConfig


async def cites_sources(context):
    if "[1]" not in context.content:
        return "Cite at least one source as [1]."
    return None


agent = Agent(AgentConfig(model="gpt-4o-mini", guards=[cites_sources], max_guard_rounds=2))
```

Guards also run in streamed runs: the rejected answer has already been streamed, then a `GuardRejected` [event](streaming.md) marks it, and the next answer streams after it. A rejected answer is never the run's `content` or the value `on_run_end` gets.

## Observability example

Accumulate token usage and cost across multiple runs:

```python
from cyclops import Agent, AgentConfig, AgentHooks


class UsageHooks(AgentHooks):
    def __init__(self):
        self.total_tokens = 0

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

    def on_tool_error(self, tool_name: str, args: dict, error: BaseException) -> None:
        logger.error("Tool %r failed with args %r: %s", tool_name, args, error)
```

## Default behaviour

The default `AgentHooks` base class implements `on_tool_start` to return `"allow"` and all other methods as no-ops. You do not need to call `super()` in your overrides.
