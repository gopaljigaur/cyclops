<div align="center">
  <img src="https://raw.githubusercontent.com/gopaljigaur/cyclops/main/docs/assets/banner.svg" alt="Cyclops" width="900"/>

  <br/>
  <br/>

  **Build LLM agents in Python. Any model. Any tool. No magic.**

  <br/>

  [Quick Start](#quick-start) | [Examples](./examples) | [Docs](https://cyclops.gopalji.me) | [Why Cyclops](#why-cyclops)

  <br/>

  [![PyPI](https://img.shields.io/pypi/v/cyclops-ai?color=58a6ff&label=PyPI)](https://pypi.org/project/cyclops-ai)
  [![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/cyclops-ai)
  [![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
  [![Tests](https://img.shields.io/github/actions/workflow/status/gopaljigaur/cyclops/ci.yml?label=tests)](https://github.com/gopaljigaur/cyclops/actions)
  [![Stars](https://img.shields.io/github/stars/gopaljigaur/cyclops?style=social)](https://github.com/gopaljigaur/cyclops)

</div>

---

Cyclops is a thin wrapper around [LiteLLM](https://github.com/BerriAI/litellm) that gives you agents, tool calling, streaming, structured output, memory, and MCP in one clean API. It works with every model LiteLLM supports. It doesn't hide the underlying calls from you.

## Install

```bash
pip install cyclops-ai
```

```bash
uv add cyclops-ai
```

## Quick Start

```python
from cyclops import Agent, AgentConfig

agent = Agent(AgentConfig(model="gpt-4o-mini"))
print(agent.run("What is the capital of Japan?"))
```

Add tools:

```python
from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool

@tool
def get_price(ticker: str) -> str:
    """Get stock price for a ticker symbol"""
    return f"{ticker}: $142.50"

agent = Agent(AgentConfig(model="gpt-4o-mini"), tools=[get_price])
print(agent.run("What is Apple's stock price?"))
# The agent calls get_price("AAPL") automatically
```

## What's included

| | |
|---|---|
| **Tool loop** | Calls tools in a loop until the model stops asking for them, not just one round |
| **Streaming** | `agent.stream()` and `agent.astream()` for token-by-token output |
| **Structured output** | `agent.run(..., response_model=MyModel)` returns a Pydantic instance |
| **Cost tracking** | `agent.run_with_response()` returns tokens used and estimated cost |
| **Skills** | Load instructions from `SKILL.md` folders only when the task needs them |
| **Sessions** | `Session` transcripts with usage, saved and loaded through a `SessionStore` (in memory or files) |
| **Memory** | `InMemoryStorage` and `FileStorage` for persistent key-value context |
| **MCP** | Connect to any MCP server as a tool source, or expose your tools as an MCP server |
| **Plugins** | Install `cyclops-toolkit-*` packages and tools are discovered automatically |
| **Any LLM** | OpenAI, Anthropic, Groq, Gemini, Ollama, Together AI, Bedrock, and 100+ more via LiteLLM |
| **Provider options** | `completion_kwargs` passes any LiteLLM argument (keys, headers, retries, thinking); reasoning stays in history; opt-in `prompt_caching` |
| **Fallback routing** | Pass a LiteLLM Router for automatic failover and load balancing |
| **Naive mode** | Prompt-based tool calling for models without native function calling support |

## Streaming

```python
agent = Agent(AgentConfig(model="gpt-4o-mini"))

for chunk in agent.stream("Explain the water cycle"):
    print(chunk, end="", flush=True)
```

Async:

```python
async for chunk in agent.astream("Write a haiku about Python"):
    print(chunk, end="", flush=True)
```

## Structured Output

```python
from pydantic import BaseModel
from cyclops import Agent, AgentConfig

class Summary(BaseModel):
    title: str
    key_points: list[str]
    sentiment: str

agent = Agent(AgentConfig(model="gpt-4o-mini"))
result = agent.run("Summarize the French Revolution", response_model=Summary)

print(result.title)       # "The French Revolution"
print(result.key_points)  # ["Storming of the Bastille", ...]
```

## Cost and Token Tracking

```python
response = agent.run_with_response("Write a product description for noise-cancelling headphones")

print(f"Model:      {response.model}")
print(f"Tokens:     {response.tokens_used}")
print(f"Cost:       ${response.cost:.6f}")
print(f"Answer:     {response.content}")
```

## Multi-Turn Conversations

History is kept automatically. Call `reset()` to start fresh.

```python
agent = Agent(AgentConfig(model="gpt-4o-mini"))

agent.run("My name is Alice and I work in Tokyo")
print(agent.run("Where do I work?"))  # "You work in Tokyo"

agent.reset()
print(agent.run("What's my name?"))   # no memory of Alice
```

To keep conversations apart, or save one and continue it later, pass a `Session`:

```python
from cyclops import FileSessionStore, Session

store = FileSessionStore("./sessions")
session = await store.load("alice") or Session(id="alice")
await agent.arun("Where do I work?", session=session)
await store.save(session)
```

## Memory

```python
import asyncio
from cyclops import Agent, AgentConfig, FileStorage

memory = FileStorage("./memory.json")  # persists across restarts

async def main():
    await memory.store("user_name", "Alice")
    await memory.store("language", "Python")

    name = await memory.retrieve("user_name")
    print(f"Hello, {name}")

asyncio.run(main())
```

## MCP

Connect to MCP servers over stdio, streamable HTTP or SSE and give their tools to an agent:

```python
from cyclops.mcp import MCPClient

async with MCPClient() as mcp:
    await mcp.attach("files", ["npx", "-y", "@modelcontextprotocol/server-filesystem", "."])
    await mcp.attach("docs", url="https://example.com/mcp", headers={"Authorization": "Bearer ..."})

    agent = Agent(config, tools=mcp.tools())  # files__read_file, docs__search, ...
    response = await agent.arun("Summarize README.md")
```

Expose your tools as an MCP server:

```python
from cyclops.mcp import MCPServer
from cyclops.toolkit import tool

@tool
def lookup(id: str) -> str:
    """Look up a record by ID"""
    return f"Record {id}: active"

server = MCPServer("my-server")
server.add_tool(lookup)
await server.run_stdio()
```

## Observability

Wire `TelemetryHooks` to get OpenTelemetry spans for every LLM call and tool execution. Works with any OTLP backend — Jaeger, Honeycomb, Grafana Tempo, Datadog, or the console.

```python
from cyclops import Agent, AgentConfig, TelemetryHooks

agent = Agent(AgentConfig(model="groq/llama-3.1-8b-instant", hooks=TelemetryHooks.console()))
agent.run("What files are in the current directory?")
```

Send to Jaeger / Tempo / Datadog instead:

```python
# uv add opentelemetry-exporter-otlp-proto-grpc
agent = Agent(AgentConfig(model="groq/llama-3.1-8b-instant", hooks=TelemetryHooks.otlp("http://localhost:4317")))
agent.run("What files are in the current directory?")
```

Span hierarchy per run: `invoke_agent` with `chat` (token counts) and `execute_tool <name>` children, using the OpenTelemetry GenAI semantic conventions. The factories build their own tracer provider; pass `TelemetryHooks(tracer_provider=...)` to use yours. One instance is safe to share across concurrent runs. See [Observability guide](https://cyclops.gopalji.me/guides/observability/) for OTLP/Jaeger setup and full attribute reference.

## Providers

Switch models by changing one string. Set the matching API key as an environment variable.

```python
# OpenAI
Agent(AgentConfig(model="gpt-4o-mini"))                        # OPENAI_API_KEY

# Anthropic
Agent(AgentConfig(model="claude-haiku-4-5-20251001"))          # ANTHROPIC_API_KEY

# Groq (fast, free tier)
Agent(AgentConfig(model="groq/llama-3.1-8b-instant"))          # GROQ_API_KEY

# Ollama (local, no API key)
Agent(AgentConfig(model="ollama/qwen3:4b"))

# Google
Agent(AgentConfig(model="gemini/gemini-1.5-flash"))            # GEMINI_API_KEY
```

## Fallback Routing

```python
from litellm import Router
from cyclops import Agent, AgentConfig

router = Router(
    model_list=[
        {"model_name": "primary", "litellm_params": {"model": "gpt-4o-mini"}},
        {"model_name": "primary", "litellm_params": {"model": "groq/llama-3.1-8b-instant"}},
    ],
    fallbacks=[{"primary": ["groq/llama-3.1-8b-instant"]}],
    num_retries=2,
)

agent = Agent(AgentConfig(model="primary", router=router))
```

## Why Cyclops?

Most agent frameworks add heavy abstractions, require specific clouds, or make it hard to see what's actually being sent to the LLM. Cyclops doesn't. The `Agent` class is ~400 lines. `_history` is a plain list of dicts in LiteLLM format. Every call goes straight to LiteLLM.

|  | Cyclops | LangChain | smolagents |
|---|---|---|---|
| Lines to first agent | 3 | 15+ | 8 |
| Any LiteLLM model | yes | partial | yes |
| MCP native | yes | plugin only | no |
| Streaming | yes | yes | no |
| Structured output | yes | yes | no |
| Cost tracking | yes | partial | no |
| Abstraction | thin | thick | medium |

## Examples

See the [`examples/`](./examples) directory:

| File | What it shows |
|---|---|
| [`basic_agent.py`](examples/basic_agent.py) | Hello world, multi-turn |
| [`agent_with_tools.py`](examples/agent_with_tools.py) | Tool calling |
| [`streaming_example.py`](examples/streaming_example.py) | `stream()` and `astream()` |
| [`structured_output.py`](examples/structured_output.py) | `response_model` with Pydantic |
| [`cost_tracking.py`](examples/cost_tracking.py) | Tokens and cost |
| [`multi_turn_agent.py`](examples/multi_turn_agent.py) | Conversation history |
| [`tool_loop_demo.py`](examples/tool_loop_demo.py) | Multi-step tool chains |
| [`skills_example.py`](examples/skills_example.py) | Skills from `SKILL.md` folders |
| [`memory_persistence.py`](examples/memory_persistence.py) | FileStorage |
| [`async_agent.py`](examples/async_agent.py) | `arun()` and concurrent calls |
| [`different_llms.py`](examples/different_llms.py) | Groq, Ollama, OpenAI, Together AI |
| [`router_fallback.py`](examples/router_fallback.py) | Automatic failover |
| [`mcp_server.py`](examples/mcp_server.py) | Expose tools via MCP |
| [`plugin_system.py`](examples/plugin_system.py) | Auto-discover toolkit packages |
| [`otel_example.py`](examples/otel_example.py) | OpenTelemetry tracing |

## Development

```bash
git clone https://github.com/gopaljigaur/cyclops
cd cyclops
uv sync
uv run pre-commit install
uv run pytest
```

## Contributing

Bug reports, feature requests, and pull requests are welcome. Open an issue first for large changes.

## License

MIT
