"""OpenTelemetry observability example.

Demonstrates wiring TelemetryHooks to an agent and emitting spans to the console.
Use TelemetryHooks.otlp("http://localhost:4317") to send to Jaeger, Honeycomb,
Grafana Tempo, Datadog, or any OTLP-compatible backend.

Span hierarchy per agent.run(), following the OpenTelemetry GenAI conventions:
    invoke_agent
    ├── chat                   (gen_ai.response.model, gen_ai.usage.*)
    ├── execute_tool <name>    (gen_ai.tool.name)
    └── ...
"""

from cyclops import Agent, AgentConfig, TelemetryHooks
from cyclops.toolkit import tool

MODEL = "ollama/qwen3:4b"

# Alternatives:
#   MODEL = "groq/llama-3.1-8b-instant"              # GROQ_API_KEY
#   MODEL = "anthropic/claude-haiku-4-5-20251001"    # ANTHROPIC_API_KEY
#   MODEL = "gpt-4o-mini"                            # OPENAI_API_KEY


# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Sunny, 22C in {location}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    try:
        return str(eval(expression))  # noqa: S307 (demo only)
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# 3. Run agent with TelemetryHooks
# ---------------------------------------------------------------------------


def main() -> None:
    # TelemetryHooks.console() builds its own TracerProvider with a ConsoleSpanExporter.
    # Pass TelemetryHooks(tracer_provider=...) to use the provider your app configures.
    # Swap for TelemetryHooks.otlp("http://localhost:4317") to send to Jaeger/Tempo/etc.
    hooks = TelemetryHooks.console()

    agent = Agent(
        AgentConfig(model=MODEL, hooks=hooks),
        tools=[get_weather, calculate],
    )

    print("Running agent...\n")
    agent.run("What is the weather in Tokyo, and what is 42 multiplied by 17?")
    hooks.flush()
    print("\nDone. Spans printed above.")


if __name__ == "__main__":
    main()
