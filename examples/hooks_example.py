"""Hooks example: observe and control agent execution with AgentHooks

Demonstrates:
  1. Logging hooks: print every run start, LLM call, and tool invocation
  2. Tool approval: block specific tools at runtime with on_tool_start
  3. Token accumulation: track total tokens across multiple runs
  4. Error hooks: log LLM and tool failures without crashing the agent
"""

import logging

from cyclops import Agent, AgentConfig, AgentHooks
from cyclops.toolkit import tool

MODEL = "ollama/qwen3:4b"

# Alternatives:
#   MODEL = "gpt-4o-mini"                      # OPENAI_API_KEY
#   MODEL = "groq/llama-3.1-8b-instant"        # GROQ_API_KEY

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared tools for the demos
# ---------------------------------------------------------------------------


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    return f"Sunny, 72F in {location}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple arithmetic expression."""
    try:
        return str(eval(expression))  # noqa: S307 (demo only)
    except Exception as exc:
        return f"Error: {exc}"


# ---------------------------------------------------------------------------
# 1. Logging hooks
# ---------------------------------------------------------------------------


class LoggingHooks(AgentHooks):
    def on_run_start(self, input_message: str) -> None:
        print(f"[run:start] {input_message!r}")

    def on_llm_end(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            print(f"[llm:end]  tokens={usage.total_tokens}")

    def on_tool_start(self, tool_name: str, args: dict):
        print(f"[tool:start] {tool_name}({args})")

    def on_tool_end(self, tool_name: str, args: dict, result: str) -> None:
        print(f"[tool:end]   {tool_name} -> {result!r}")

    def on_run_end(self, content: str) -> None:
        print(f"[run:end]  {content[:80]!r}")
        print()


def demo_logging_hooks() -> None:
    print("=" * 60)
    print("1. Logging hooks")
    print("=" * 60)

    config = AgentConfig(model=MODEL, hooks=LoggingHooks())
    agent = Agent(config, tools=[get_weather, calculate])

    agent.run("What is the weather in Paris?")
    agent.run("What is 144 divided by 12?")


# ---------------------------------------------------------------------------
# 2. Tool approval: block specific tools
# ---------------------------------------------------------------------------


ALLOWED_TOOLS = {"get_weather"}


class AllowListHooks(AgentHooks):
    def on_tool_start(self, tool_name: str, args: dict):
        if tool_name not in ALLOWED_TOOLS:
            print(f"[approval] BLOCKED: {tool_name}")
            return "deny"
        print(f"[approval] ALLOWED: {tool_name}")


def demo_tool_approval() -> None:
    print("=" * 60)
    print("2. Tool approval")
    print("=" * 60)

    config = AgentConfig(model=MODEL, hooks=AllowListHooks())
    agent = Agent(config, tools=[get_weather, calculate])

    # calculate is blocked; agent receives "[Tool 'calculate' was not approved]"
    response = agent.run("What is the weather in Tokyo, and also what is 7 times 8?")
    print(f"Response: {response}\n")


# ---------------------------------------------------------------------------
# 3. Token accumulation across multiple runs
# ---------------------------------------------------------------------------


class UsageHooks(AgentHooks):
    def __init__(self):
        self.total_tokens = 0

    def on_llm_end(self, response) -> None:
        usage = getattr(response, "usage", None)
        if usage:
            self.total_tokens += getattr(usage, "total_tokens", 0) or 0

    def on_run_end(self, content: str) -> None:
        print(f"  (cumulative tokens: {self.total_tokens})")


def demo_token_accumulation() -> None:
    print("=" * 60)
    print("3. Token accumulation across multiple runs")
    print("=" * 60)

    hooks = UsageHooks()
    config = AgentConfig(model=MODEL, hooks=hooks)
    agent = Agent(config)

    agent.run("Summarise the Pythagorean theorem in one sentence.")
    agent.run("What is the derivative of x squared?")
    agent.run("Name three sorting algorithms.")

    print(f"\nSession total: {hooks.total_tokens} tokens\n")


# ---------------------------------------------------------------------------
# 4. Error hooks: log failures without crashing
# ---------------------------------------------------------------------------


class ErrorHooks(AgentHooks):
    def on_llm_error(self, error: Exception) -> None:
        logger.error("LLM call failed: %s", error)

    def on_tool_error(self, tool_name: str, args: dict, error: Exception) -> None:
        logger.error("Tool %r failed with args %r: %s", tool_name, args, error)


def demo_error_hooks() -> None:
    print("=" * 60)
    print("4. Error hooks (config only, no error expected here)")
    print("=" * 60)

    config = AgentConfig(model=MODEL, hooks=ErrorHooks())
    agent = Agent(config)

    response = agent.run("What is the capital of Germany?")
    print(f"Response: {response}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_logging_hooks()
    demo_tool_approval()
    demo_token_accumulation()
    demo_error_hooks()
