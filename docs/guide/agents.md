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
| `temperature` | `float \| None` | `None` | Sampling temperature. Lower values are more deterministic. `None` uses the model's default. |
| `max_tokens` | `int \| None` | `None` | Maximum tokens in the response. `None` uses the model's default. |
| `system_prompt` | `str \| None` | `None` | System instruction prepended to every conversation. |
| `tool_mode` | `str` | `"auto"` | `"auto"` uses native function calling and switches this agent to `"naive"` if LiteLLM reports the model does not support tools; `"native"` forces it; `"naive"` uses prompt-based tool parsing (works with any model). |
| `router` | `Router \| None` | `None` | Optional LiteLLM `Router` for fallback and load balancing. |
| `max_steps` | `int` | `10` | Most LLM completions in one run, guard rounds included; at least 1. |
| `max_total_tokens` | `int \| None` | `None` | Most tokens one run may use, checked between steps. |
| `max_cost` | `float \| None` | `None` | Most USD one run may cost, checked between steps. |
| `max_subagent_depth` | `int` | `5` | Most levels of [subagent](subagents.md) runs below one run. |
| `max_concurrent_subagents` | `int` | `10` | Most subagent runs one run has going at once; more calls wait. |
| `max_parallel_tool_calls` | `int` | `10` | Most tool calls of one step that run at the same time; at least 1. See [Parallel calls and timeouts](tools.md#parallel-calls-and-timeouts). |
| `tool_timeout` | `float \| None` | `None` | Seconds a tool call may run before it is cancelled and returns an error. A tool's own `timeout` wins. |
| `hooks` | `AgentHooks \| None` | `None` | Lifecycle callbacks for observability and tool approval. See [Hooks](hooks.md). |
| `num_retries` | `int` | `2` | Retries of a completion after a rate limit, overload, 5xx, timeout or connection error. See [Retries and timeouts](#retries-and-timeouts). |
| `timeout` | `float` | `600` | Seconds LiteLLM's HTTP client waits to connect and for each read or write. |
| `stream_idle_timeout` | `float` | `120` | Seconds a stream may go without a chunk after its first. |
| `completion_kwargs` | `dict` | `{}` | Extra arguments for every LiteLLM completion call, also through a `router`. See [Provider options](#provider-options). |
| `reasoning_effort` | `str \| None` | `None` | `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` or `"max"`, sent to models that take it. See [Reasoning](#reasoning). |
| `thinking_budget` | `int \| None` | `None` | Tokens the model may think, sent to models that take a thinking budget; not with `reasoning_effort`. See [Reasoning](#reasoning). |
| `prompt_caching` | `bool \| "5m" \| "1h"` | `True` | Mark cache breakpoints for models with prompt caching on Anthropic, Bedrock and Vertex AI (Anthropic models); `"5m"` or `"1h"` sets the time to live, `False` turns it off. See [Prompt caching](#prompt-caching). |

## Constructing an Agent

Pass a config and an optional list of tools.

```python
from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool


@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"


config = AgentConfig(model="groq/llama-3.1-8b-instant")
agent = Agent(config, tools=[greet])
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

### One event loop per process

LiteLLM runs async success callbacks on one background worker per process, and that worker serves one event loop at a time. When a call comes from another loop, LiteLLM moves the worker, logging `LoggingWorker: event loop changed; carried N pending and revived N dequeued logging task(s) onto the new loop` at WARNING when tasks were waiting, and asyncio logs "Task exception was never retrieved" and "Task was destroyed but it is pending!" for the tasks left on the old loop. Some callbacks run only when the process exits, and with loops in several threads some never run at all.

Any process with more than one event loop calling LiteLLM hits this, async-only code included, for example `asyncio.run` in two threads. The sync methods bring a loop of their own: they run on one shared loop in a background thread, while the async methods run on the caller's loop, so a process that calls both has two.

Keep every call on one loop. An async application whose sync code needs an agent can hand the run to its own loop from another thread:

```python
loop = asyncio.get_running_loop()  # in async code


def answer(question: str) -> str:  # sync code on another thread, e.g. asyncio.to_thread
    return asyncio.run_coroutine_threadsafe(agent.arun(question), loop).result()
```

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

- `handle.steer(message)` adds a user message that the model sees in its next completion. A message that arrives during a completion goes into the one after, once the step's tool calls finish. If that completion was the final answer, the run takes another step to answer the message (guards then judge the new answer); when `max_steps` or a budget leaves no step for it, the message stays at the end of the history for the next run (a run that already answered completes with that answer). Steering a run that has finished raises `RuntimeError`.
- `handle.cancel()` stops the run. Tool calls in progress are cancelled and get a cancelled result, so the conversation stays valid, and awaiting the handle gives `stop_reason="cancelled"`. A completion in progress is dropped. Calling it again does nothing.

Cancelling the task that awaits a handle (or `arun()`) also cancels the run, but as a plain `asyncio.CancelledError`: the await raises it and there is no `"cancelled"` result.

`start(input, stream=False)` makes the completions without streaming, so each step's text arrives as one `TextDelta`. `arun()`, `arun_with_response()` and `astream()` are shortcuts over `start()`. Iterating a handle doesn't hold the run back the way `astream_events()` does: events wait in the handle until you read them. Until you start iterating, the handle keeps them in short form: consecutive text or reasoning deltas as one event, and no subagent events. A handle you only await holds a few events per step, not one per token.

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

`completion_kwargs` is passed to every LiteLLM completion call the agent makes, so any argument LiteLLM accepts works per agent: credentials, endpoints, headers and reasoning settings. Keys here win over `temperature`, `max_tokens` and `timeout`. Set retries with `num_retries` (see [Retries and timeouts](#retries-and-timeouts)): LiteLLM's own `num_retries` here would retry on top of it.

```python
import os

config = AgentConfig(
    model="anthropic/claude-sonnet-4-5",
    completion_kwargs={
        "api_key": os.environ["MY_ANTHROPIC_KEY"],
        "extra_headers": {"X-Request-Source": "docs"},
        "max_tokens": 8192,
    },
)
```

For reasoning, use the typed options in [Reasoning](#reasoning). See the [LiteLLM input docs](https://docs.litellm.ai/docs/completion/input) for every argument.

## Retries and timeouts

A completion that fails with a rate limit, an overloaded or 5xx response, a timeout or a connection error is retried up to `num_retries` times (default `2`) with exponential backoff and jitter: about 0.5s, then 1s, doubling up to 8s, each plus up to 0.75s. When the provider sends a `retry-after` of 60 seconds or less, the retry waits that long instead; a longer `retry-after` falls back to the backoff. These delays are LiteLLM's, set by its `INITIAL_RETRY_DELAY`, `MAX_RETRY_DELAY` and `JITTER` environment variables. Other errors, such as authentication, permission or invalid request errors, are never retried, nor is a spent quota or budget. A streamed completion is retried only while none of its text or reasoning has been emitted; after that the error ends the run. A failed attempt adds nothing to the run's usage, though the provider may still bill for it.

`timeout` (default `600` seconds) is the request timeout LiteLLM gives its HTTP client, which applies it to connecting and to each read and write, not to the response as a whole. A timeout is retried like the errors above, so one completion can take up to `(num_retries + 1) * timeout`. `stream_idle_timeout` (default `120` seconds) is how long a stream may go without a chunk after its first. Only the chunks LiteLLM yields reset it; it drops keep-alive events such as Anthropic's pings. A stalled stream is aborted with `StreamIdleTimeout`, a subclass of `litellm.Timeout`, and retried once if none of it has been emitted. A pause longer than `stream_idle_timeout` after the stream has emitted content ends the run.

```python
config = AgentConfig(
    model="anthropic/claude-sonnet-4-5",
    num_retries=4,
    timeout=120,
    stream_idle_timeout=60,
)
```

Each attempt fires `on_llm_start`, and `on_llm_error` if it fails. `num_retries=0` turns off the retries of failed requests; a stalled stream is still retried once. With a `router`, the request's retries and timeout are the Router's own (`num_retries`, `retry_policy`, `timeout`, `stream_timeout`), and the stream idle timeout and its one retry still apply.

## Reasoning in history

When a model returns reasoning, the assistant messages in `agent.messages` keep it as `reasoning_content` and, for Anthropic extended thinking, `thinking_blocks` with their signatures. They are sent back on the next call, which Anthropic requires when thinking is combined with tool use: the signed thinking block has to precede the tool call it led to.

```python
agent.run("What is 17 * 23? Use the calculator.")
for msg in agent.messages:
    if "thinking_blocks" in msg:
        print(msg["thinking_blocks"][0]["thinking"])
```

## Reasoning

`reasoning_effort` sets how much a reasoning model thinks before it answers, and `thinking_budget` sets its thinking in tokens. Set one or the other; a config with both is rejected. Both are off by default, and the model keeps its own default behaviour.

LiteLLM maps them onto each provider's own setting, and the mapping changes from model to model. Some notable cases with LiteLLM 1.105.1:

- OpenAI models get `reasoning_effort` as is and take no `thinking_budget`.
- Older Anthropic models, on the Anthropic API and Bedrock, get a thinking budget: `thinking_budget` as given, or one LiteLLM picks for the effort (1024 tokens for `"low"`). Newer ones such as `claude-opus-4-6` get adaptive thinking with `output_config.effort` for an effort, and on models that take only adaptive thinking (`claude-opus-4-7` and later) LiteLLM turns a budget into an effort too, so it caps nothing.
- Gemini 2.5 gets `thinkingBudget`. Gemini 3 gets `thinkingLevel` for an effort and drops `thinking_budget`.

Each option goes only to models where LiteLLM lists the matching parameter (`reasoning_effort` or `thinking` in `get_supported_openai_params`), and through a `router` every deployment has to list it; other models get nothing. The check is per parameter, not per level: the levels a model accepts vary, and LiteLLM or the provider rejects one it doesn't: LiteLLM raises for `gpt-5` with `"xhigh"`, while `o3-mini` passes `"none"`, `"minimal"` and `"max"` through to OpenAI. A listed parameter is also no promise that the model reasons: LiteLLM lists `reasoning_effort` for every Ollama model, so a model that cannot think rejects it. `"none"` turns reasoning off where the provider allows it. A `reasoning_effort` or `thinking` in `completion_kwargs` replaces both options.

```python
config = AgentConfig(
    model="anthropic/claude-sonnet-4-5",
    reasoning_effort="medium",
)
```

Leave `temperature` unset with these options: Anthropic models accept thinking only at temperature 1, and OpenAI reasoning models reject any other value. When `max_tokens` is unset and the request carries a thinking budget, LiteLLM sends the budget plus 4096 tokens (its `DEFAULT_MAX_TOKENS`) as the Anthropic output limit; when it is set, it has to exceed the budget.

## Prompt caching

Prompt caching is on by default. Every call marks two cache breakpoints: the system prompt, which caches the tool definitions before it too, and the latest message, which caches the conversation so far. Each step then reads the earlier prefix from the cache. LiteLLM places the breakpoints on a copy of the messages, so the history never keeps them, and it skips any past the four a request may carry, counting breakpoints already in the history.

The markers are sent only to providers that cache at them (Anthropic, Bedrock, and Anthropic models on Vertex AI), and only for models that LiteLLM lists with prompt caching (`supports_prompt_caching`). Through a `router`, every deployment of the model has to qualify. Every other model gets no markers: OpenAI caches long prefixes without them, Gemini would turn them into billed explicit cache resources, and Bedrock models that LiteLLM lists without prompt caching (such as Llama) get none. A model missing from LiteLLM's model map gets no markers either.

`True` caches for the provider's default time, five minutes on Anthropic models. `"5m"` or `"1h"` sets the time to live; a one hour cache costs more to write. Bedrock honours `"1h"` only for the models that support it.

```python
config = AgentConfig(
    model="anthropic/claude-sonnet-4-5",
    system_prompt=LONG_INSTRUCTIONS,
    prompt_caching="1h",
)
agent = Agent(config)
response = agent.run_with_response("Summarise the report.")
print(response.cached_tokens)
```

A provider caches a prefix only above a minimum length that depends on the model, from 512 to 4096 tokens on Anthropic models; below it the marker is ignored and costs nothing (see Anthropic's [prompt caching docs](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)). Writing to the cache costs more than a plain input token, and later reads cost much less, so a run of several steps gains while a single call that is not repeated within the time to live pays the write for nothing. `prompt_caching=False` sends no markers. To place breakpoints yourself, pass LiteLLM's `cache_control_injection_points` in `completion_kwargs`; they replace the default ones.

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

### Output limit and filtering

The run reads each completion's `finish_reason`, as LiteLLM normalizes it across providers (`stop`, `length`, `tool_calls`, `content_filter`). Output cut by the output limit (`max_tokens`, or the model's own) ends the run with `stop_reason="max_tokens"`, and output the provider filtered or refused (LiteLLM maps Anthropic's `refusal` and Bedrock's guardrail to `content_filter`) ends it with `stop_reason="content_filter"`. Either way `content` is empty, guards don't run, and any partial text stays in the conversation as the assistant's turn. Tool calls in such a completion are not run, since a cut one may be incomplete. The run doesn't continue on its own: raise `max_tokens` or ask the model to go on in the next run. `StepFinished.finish_reason` gives the value of each step, and `RunResult.finish_reason` that of the last.
