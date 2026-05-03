"""OpenTelemetry observability example.

Demonstrates wiring OTelHooks to an agent and emitting spans to the console.
Replace ConsoleSpanExporter with OTLPSpanExporter to send to Jaeger, Honeycomb,
Grafana Tempo, Datadog, or any OTLP-compatible backend.

Span hierarchy per agent.run():
    agent.run
    ├── llm.completion   (attributes: model, tokens, latency_ms)
    ├── tool.<name>      (attributes: name, args_count, result.length)
    └── ...
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

from cyclops import Agent, AgentConfig
from cyclops.observability import OTelHooks
from cyclops.toolkit import tool

MODEL = "ollama/qwen3:4b"

# Alternatives:
#   MODEL = "groq/llama-3.1-8b-instant"              # GROQ_API_KEY
#   MODEL = "anthropic/claude-haiku-4-5-20251001"    # ANTHROPIC_API_KEY
#   MODEL = "gpt-4o-mini"                            # OPENAI_API_KEY

# ---------------------------------------------------------------------------
# 1. Configure an OTel TracerProvider (console output for this demo)
# ---------------------------------------------------------------------------

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(provider)


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
# 3. Run agent with OTelHooks
# ---------------------------------------------------------------------------


def main() -> None:
    agent = Agent(
        AgentConfig(model=MODEL, hooks=OTelHooks()),
        tools=[get_weather, calculate],
    )

    print("Running agent (spans will print after BatchSpanProcessor flushes)...\n")
    agent.run("What is the weather in Tokyo, and what is 42 multiplied by 17?")

    provider.force_flush()
    print("\nDone. See span output above.")


if __name__ == "__main__":
    main()
