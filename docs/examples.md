# Examples

All example scripts live in the `examples/` directory of the repository. Each one is self-contained and runnable. Most use `ollama/qwen3:4b` as the default model so they work without any API key — just install [Ollama](https://ollama.ai) and pull the model with `ollama pull qwen3:4b`.

---

## basic_agent.py

**Path:** `examples/basic_agent.py`

The simplest possible agent: create an `AgentConfig`, build an `Agent`, and call `run()` twice. Shows that the agent automatically maintains conversation history, so the second question ("What's the population?") is answered in context of the first ("What is the capital of France?").

---

## agent_with_tools.py

**Path:** `examples/agent_with_tools.py`

Demonstrates the `@tool` decorator with three tools: `get_time` (no parameters), `get_weather` (one string parameter), and `calculate` (three parameters of mixed types). Each tool has typed parameters so Cyclops auto-generates the JSON schema. The agent is asked questions that require each tool in turn.

---

## async_agent.py

**Path:** `examples/async_agent.py`

Shows concurrent async usage: four independent questions are dispatched with `asyncio.gather()`, each in its own `Agent` instance. All four questions run in parallel and the results are printed in order after all complete.

---

## memory_persistence.py

**Path:** `examples/memory_persistence.py`

Uses `InMemoryStorage` to store user context (name, preferences, last topic) and then retrieves it to build a context-aware prompt. Illustrates the store/retrieve/list_keys lifecycle and how to inject memory into an agent's first message.

---

## different_llms.py

**Path:** `examples/different_llms.py`

A quick tour of five providers — Groq (free), Together AI (free tier), Ollama (local), and OpenAI — all running the same prompt with only the `model=` string changed. Includes the environment variable each provider requires.

---

## ollama_local.py

**Path:** `examples/ollama_local.py`

Focused example for fully local, zero-cost operation with Ollama. Includes a `get_system_info` tool that returns OS and Python version, demonstrating that the no-API-key workflow still supports native tool use.

---

## custom_tool_class.py

**Path:** `examples/custom_tool_class.py`

Shows how to subclass `BaseTool` for stateful tools. `DatabaseTool` holds an in-memory record set and exposes optional filter parameters. `CalculatorTool` accumulates a `history` list across calls. Both tools are used together in a single agent and tool state persists between calls.

---

## plugin_system.py

**Path:** `examples/plugin_system.py`

End-to-end walkthrough of the plugin system: manually instantiates and registers a `WeatherToolkit` (from `examples/example_toolkit_plugin/`), calls `register_tools()` to populate a `ToolRegistry`, pulls tool instances out of the registry, and passes them to an agent. Mirrors what `load_plugins()` does automatically for installed packages.

---

## mcp_server.py

**Path:** `examples/mcp_server.py`

Creates a minimal MCP server with three tools (`add`, `multiply`, `greet`), registers them with `server.add_tool()`, and starts the server with `run_stdio()`. Run this script as a background process and connect to it from any MCP client.

---

## test_mcp.py

**Path:** `examples/test_mcp.py`

Client-side counterpart to `mcp_server.py`. Connects to the server via `connect_stdio`, lists the available tools, calls `add` with `a=5, b=3`, and calls `greet` with `name="World"`. A good smoke test to verify your MCP server is working.

---

## router_fallback.py

**Path:** `examples/router_fallback.py`

Configures a LiteLLM `Router` with a primary model (`ollama/qwen3:4b`) and a fallback (`ollama/llama3.2:1b`). If the primary fails or is rate-limited, the router automatically retries with the fallback. Passes the router through `AgentConfig.router` with no other code changes.

---

## router_load_balancing.py

**Path:** `examples/router_load_balancing.py`

Uses `routing_strategy="simple-shuffle"` to distribute requests across two backend entries. Also includes a `calculate` tool to show that routers and tools work together. Prints `router.deployment_stats` at the end to show per-backend usage.
