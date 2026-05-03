"""Streaming example — token-by-token output with agent.stream() and agent.astream()

Demonstrates:
  1. Synchronous streaming with no tools (token-by-token)
  2. Asynchronous streaming with no tools (astream)
  3. Streaming with tools — tool calls are resolved first, then the final
     answer is streamed token-by-token
"""

import asyncio

from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Default: Ollama (free, runs locally — install at https://ollama.ai)
#   ollama pull qwen3:4b
MODEL = "ollama/qwen3:4b"

# Alternatives — set the right env-var first, then swap MODEL:
#   OpenAI  : MODEL = "gpt-4o-mini"          (OPENAI_API_KEY)
#   Groq    : MODEL = "groq/llama-3.1-8b-instant"  (GROQ_API_KEY)
#   Anthropic: MODEL = "claude-3-haiku-20240307"    (ANTHROPIC_API_KEY)


# ---------------------------------------------------------------------------
# 1. Synchronous streaming — no tools
# ---------------------------------------------------------------------------


def demo_sync_stream() -> None:
    print("=" * 60)
    print("1. Synchronous streaming (no tools)")
    print("=" * 60)

    config = AgentConfig(
        model=MODEL,
        system_prompt="You are a concise assistant.",
        temperature=0.7,
    )
    agent = Agent(config)

    prompt = "Explain what a Python generator is in 2-3 sentences."
    print(f"\nPrompt: {prompt}\n")
    print("Response (streaming): ", end="", flush=True)

    # agent.stream() returns Iterator[str] — each chunk is a small string
    # (usually a word or a few characters, depending on the model)
    for chunk in agent.stream(prompt):
        print(chunk, end="", flush=True)

    print("\n")


# ---------------------------------------------------------------------------
# 2. Asynchronous streaming — no tools
# ---------------------------------------------------------------------------


async def demo_async_stream() -> None:
    print("=" * 60)
    print("2. Asynchronous streaming (no tools)")
    print("=" * 60)

    config = AgentConfig(
        model=MODEL,
        system_prompt="You are a concise assistant.",
        temperature=0.7,
    )
    agent = Agent(config)

    prompt = "What are three benefits of async programming?"
    print(f"\nPrompt: {prompt}\n")
    print("Response (astream): ", end="", flush=True)

    # agent.astream() is an AsyncIterator[str] — use `async for`
    async for chunk in agent.astream(prompt):
        print(chunk, end="", flush=True)

    print("\n")


# ---------------------------------------------------------------------------
# Tool definitions for demo 3
# ---------------------------------------------------------------------------


@tool
def get_stock_price(ticker: str) -> str:
    """Look up the current stock price for a ticker symbol"""
    # Simulated prices — in production this would call a real API
    mock_prices = {
        "AAPL": 189.30,
        "GOOGL": 175.12,
        "MSFT": 415.85,
        "NVDA": 875.43,
    }
    price = mock_prices.get(ticker.upper(), 100.00)
    return f"{ticker.upper()} is trading at ${price:.2f}"


@tool
def calculate_portfolio_value(ticker: str, shares: float) -> str:
    """Calculate the total value of a stock position"""
    mock_prices = {
        "AAPL": 189.30,
        "GOOGL": 175.12,
        "MSFT": 415.85,
        "NVDA": 875.43,
    }
    price = mock_prices.get(ticker.upper(), 100.00)
    value = price * shares
    return f"{shares} shares of {ticker.upper()} at ${price:.2f} = ${value:,.2f}"


# ---------------------------------------------------------------------------
# 3. Streaming with tools — tool calls resolved first, then answer streamed
# ---------------------------------------------------------------------------


def demo_stream_with_tools() -> None:
    print("=" * 60)
    print("3. Streaming with tools")
    print("=" * 60)
    print("(Tool calls are processed first, then the final answer streams)\n")

    config = AgentConfig(
        model=MODEL,
        system_prompt="You are a helpful financial assistant.",
        temperature=0.3,
        max_iterations=5,
    )
    agent = Agent(config, tools=[get_stock_price, calculate_portfolio_value])

    prompt = (
        "I own 50 shares of AAPL and 20 shares of MSFT. "
        "What are their current prices and what is my total portfolio value?"
    )
    print(f"Prompt: {prompt}\n")
    print("Response (streaming after tool resolution): ", end="", flush=True)

    # The agent will call tools internally (non-streaming), then stream the
    # final answer token-by-token.  The caller sees the same Iterator[str]
    # interface regardless of whether tools were used.
    for chunk in agent.stream(prompt):
        print(chunk, end="", flush=True)

    print("\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_sync_stream()
    asyncio.run(demo_async_stream())
    demo_stream_with_tools()
