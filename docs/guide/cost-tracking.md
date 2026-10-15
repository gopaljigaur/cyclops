# Cost Tracking

## run_with_response() and arun_with_response()

Instead of returning a plain string, these methods return a `RunResult` that includes the text reply, why the run stopped, every tool call made, and token/cost totals summed over every LiteLLM completion in the run.

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

## RunResult fields

| Field | Type | Description |
|---|---|---|
| `content` | `str` | The final answer. Empty unless the run completed. |
| `tool_calls` | `List[ToolCall]` | Every tool call made during the run, in order. |
| `denied_tool_calls` | `List[DeniedToolCall]` | Tool calls that middleware denied, each with `id` (`None` in naive tool mode), `name`, `arguments` and the `reason` the model saw. |
| `model` | `str` | The model identifier used (from `AgentConfig.model`). |
| `run_id` | `str` | Unique id of the run, also given to tool middleware and guards. |
| `stop_reason` | `str` | `"completed"` for a final answer, `"max_tokens"` when the output limit cut the last completion, `"content_filter"` when the provider filtered or refused it ([output limit and filtering](agents.md#output-limit-and-filtering)), `"max_steps"` or `"budget"` when the run reached a [limit](agents.md#step-and-budget-limits), `"guard"` when a [guard](hooks.md#end-of-run-guards) still rejected the answer after `max_guard_rounds` or with no step or budget left for another round, `"output_validation"` when the answer still failed `response_model` validation after `output_retries` ([retries](structured-output.md#retries)), `"cancelled"` when a run handle was cancelled, `"error"` when it failed. |
| `finish_reason` | `str \| None` | The last completion's finish reason as LiteLLM normalizes it: `"stop"`, `"length"`, `"tool_calls"` or `"content_filter"`. `None` if the run made none. |
| `steps` | `int` | LLM completions made in the run. |
| `usage` | `Usage \| None` | Tokens and cost summed over every completion in the run, including those of [subagents](subagents.md) it called. `None` if any completion reported no usage. |
| `usage_by_model` | `Dict[str, Usage \| None]` | `usage` split by the model, as configured, that made each completion: [fallback models](agents.md#fallback-models), [subagents](subagents.md) on other models and a compaction model each get their own entry. A model's entry is `None` if one of its completions reported no usage. |
| `duration` | `float` | Wall-clock seconds the run took, tool calls and waits included. |
| `error` | `Exception \| None` | The exception that ended the run with `"error"`. Left out of `model_dump()`. |
| `tokens_used` | `int \| None` | Total tokens (prompt + completion), from `usage`. |
| `prompt_tokens` | `int \| None` | Tokens in the prompt/context sent to the model, from `usage`. |
| `completion_tokens` | `int \| None` | Tokens in the generated response, from `usage`. |
| `cached_tokens` | `int \| None` | Prompt tokens the provider served from its cache, from `usage`. |
| `cost` | `float \| None` | Estimated USD cost LiteLLM computed for each completion, with the pricing of the deployment that served it. `None` if LiteLLM can't price a completion of the run. |

`tokens_used`, `prompt_tokens`, `completion_tokens`, `cached_tokens` and `cost` are read-only properties over `usage`.

## Usage fields

`RunResult.usage`, each entry of `usage_by_model`, `StepFinished.usage` and `session.usage` are `Usage` objects, and two of them add up with `+`. The token counts are those LiteLLM normalizes from each provider's response.

| Field | Type | Description |
|---|---|---|
| `prompt_tokens` | `int` | Input tokens, the cached and cache-written ones included. |
| `completion_tokens` | `int` | Output tokens, the reasoning ones included. |
| `cached_tokens` | `int` | Prompt tokens read from the provider's cache. |
| `cache_write_tokens` | `int` | Prompt tokens written to the provider's cache. Anthropic and Bedrock report them; OpenAI and Gemini cache without reporting writes. |
| `reasoning_tokens` | `int` | Completion tokens the model spent reasoning, for models that report them. |
| `total_tokens` | `int` | Prompt plus completion tokens. |
| `cost` | `float \| None` | Estimated USD cost. `None` if LiteLLM can't price a completion. |
| `api_duration` | `float` | Seconds from each completion's request to its whole response, retries included. A paced stream (`stream()`, `astream()`, `stream_events()` or `astream_events()`) reads only a few chunks ahead of the caller, so the time the caller takes to read it counts too. Completions of subagents run in parallel add up, so this can exceed the run's `duration`. |

```python
result = agent.run_with_response("Summarise the report.")
for model, usage in result.usage_by_model.items():
    print(f"{model}: {usage.total_tokens} tokens, ${usage.cost:.6f}")
print(f"{result.duration:.1f}s in all, {result.usage.api_duration:.1f}s in completions")
```

Each `ToolCall` in `tool_calls` has:

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique call identifier assigned by the LLM. |
| `name` | `str` | Name of the tool that was called. |
| `arguments` | `dict` | Arguments the model passed to the tool. |
| `result` | `Any \| None` | The value returned by the tool. |

## Failed runs

`run_with_response()` and `arun_with_response()` don't raise when the run fails. They return a `RunResult` with `stop_reason="error"`, the exception in `error`, and the usage of the steps that finished, so a failed run's cost is still known. `run()` and `arun()` raise the exception.

```python
result = agent.run_with_response("Summarise this report.")
if result.stop_reason == "error":
    print(f"Failed after {result.steps} steps: {result.error}, cost ${result.cost}")
```

## Logging costs over time

A common production pattern is to accumulate cost and token data across many agent runs:

```python
import asyncio
from dataclasses import dataclass, field
from typing import List
from cyclops import Agent, AgentConfig
from cyclops import RunResult


@dataclass
class UsageTracker:
    total_cost: float = 0.0
    total_tokens: int = 0
    runs: List[RunResult] = field(default_factory=list)

    def record(self, response: RunResult) -> None:
        self.runs.append(response)
        self.total_cost += response.cost or 0.0
        self.total_tokens += response.tokens_used or 0

    def report(self) -> None:
        print(f"Runs       : {len(self.runs)}")
        print(f"Total cost : ${self.total_cost:.4f}")
        print(f"Total tokens: {self.total_tokens:,}")


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

## Tool calls in the response

When tools are used, each tool invocation is recorded in `result.tool_calls`:

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
    print(f"  Called: {tc.name}({tc.arguments}) → {tc.result}")
```

/// note
Cost tracking depends on LiteLLM's internal pricing table. A `router` deployment's own prices (`input_cost_per_token`, `output_cost_per_token`) or its `model_info.base_model` take precedence, as does a `base_model` in `completion_kwargs`. Ollama and other self-hosted models will have `cost=None` since no billing data is available. Token counts are still reported if the server returns usage information.
///
