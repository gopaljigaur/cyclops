"""Cost and token tracking example — run_with_response() / arun_with_response()

Demonstrates:
  1. Inspecting AgentResponse metadata (tokens_used, cost, prompt_tokens,
     completion_tokens) from a single synchronous run
  2. Async variant with arun_with_response()
  3. Accumulating cost and tokens across multiple calls to track a session budget

Cost data is powered by LiteLLM's built-in pricing table.  The numbers are
accurate for cloud APIs (OpenAI, Anthropic, etc.) but will be None for
local/free models like Ollama since they have no billing cost.

Default model is gpt-4o-mini so the cost fields are populated with real values.
Swap MODEL below if you only have a local Ollama instance available.
"""

import asyncio

from cyclops import Agent, AgentConfig, AgentResponse

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# gpt-4o-mini gives real cost data (requires OPENAI_API_KEY)
MODEL = "gpt-4o-mini"

# For a free/local alternative, swap to:
#   MODEL = "ollama/qwen3:4b"
# Note: cost will be None for Ollama — token counts still work.

# Other cloud alternatives with cost data:
#   MODEL = "claude-3-haiku-20240307"    # ANTHROPIC_API_KEY
#   MODEL = "groq/llama-3.1-8b-instant"  # GROQ_API_KEY (cost is ~$0 but tracked)


# ---------------------------------------------------------------------------
# Helper: pretty-print an AgentResponse
# ---------------------------------------------------------------------------

def print_response(label: str, response: AgentResponse) -> None:
    print(f"\n--- {label} ---")
    print(f"Content          : {response.content[:120]}{'...' if len(response.content) > 120 else ''}")
    print(f"Model            : {response.model}")
    print(f"Prompt tokens    : {response.prompt_tokens}")
    print(f"Completion tokens: {response.completion_tokens}")
    print(f"Total tokens     : {response.tokens_used}")
    if response.cost is not None:
        print(f"Cost             : ${response.cost:.6f}")
    else:
        print(f"Cost             : N/A (local model or no pricing data)")
    if response.tool_calls:
        print(f"Tool calls       : {[tc.name for tc in response.tool_calls]}")


# ---------------------------------------------------------------------------
# 1. Single synchronous call
# ---------------------------------------------------------------------------

def demo_single_run() -> AgentResponse:
    print("=" * 60)
    print("1. Single run_with_response() call")
    print("=" * 60)

    config = AgentConfig(
        model=MODEL,
        system_prompt="You are a concise assistant. Keep answers brief.",
        temperature=0.3,
    )
    agent = Agent(config)

    response = agent.run_with_response("What is the difference between RAM and ROM?")
    print_response("run_with_response", response)
    return response


# ---------------------------------------------------------------------------
# 2. Single async call
# ---------------------------------------------------------------------------

async def demo_async_run() -> AgentResponse:
    print("\n" + "=" * 60)
    print("2. Single arun_with_response() call (async)")
    print("=" * 60)

    config = AgentConfig(
        model=MODEL,
        system_prompt="You are a concise assistant. Keep answers brief.",
        temperature=0.3,
    )
    agent = Agent(config)

    response = await agent.arun_with_response(
        "Explain TCP vs UDP in one sentence each."
    )
    print_response("arun_with_response", response)
    return response


# ---------------------------------------------------------------------------
# 3. Accumulate cost across multiple calls in a session
# ---------------------------------------------------------------------------

def demo_cost_accumulation() -> None:
    print("\n" + "=" * 60)
    print("3. Accumulating cost across multiple runs")
    print("=" * 60)

    config = AgentConfig(
        model=MODEL,
        system_prompt="You are a helpful coding assistant. Be concise.",
        temperature=0.2,
    )
    agent = Agent(config)

    questions = [
        "What does the 'yield' keyword do in Python?",
        "When should I use a list comprehension vs a generator expression?",
        "Give me a one-line example of a generator expression.",
    ]

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cost = 0.0
    cost_available = False

    print(f"\nAsking {len(questions)} questions to the same agent (history accumulates):\n")

    for i, question in enumerate(questions, start=1):
        response = agent.run_with_response(question)
        print_response(f"Q{i}: {question[:50]}...", response)

        # Accumulate — guard against None (local models)
        if response.prompt_tokens is not None:
            total_prompt_tokens += response.prompt_tokens
        if response.completion_tokens is not None:
            total_completion_tokens += response.completion_tokens
        if response.tokens_used is not None:
            total_tokens += response.tokens_used
        if response.cost is not None:
            total_cost += response.cost
            cost_available = True

    print("\n" + "-" * 40)
    print("SESSION TOTALS")
    print("-" * 40)
    print(f"Prompt tokens     : {total_prompt_tokens}")
    print(f"Completion tokens : {total_completion_tokens}")
    print(f"Total tokens      : {total_tokens}")
    if cost_available:
        print(f"Total cost        : ${total_cost:.6f}")
        print(f"Average per call  : ${total_cost / len(questions):.6f}")
    else:
        print("Total cost        : N/A (local model)")


# ---------------------------------------------------------------------------
# 4. Demonstrate accessing the raw AgentResponse fields directly
# ---------------------------------------------------------------------------

def demo_field_access() -> None:
    print("\n" + "=" * 60)
    print("4. Direct field access on AgentResponse")
    print("=" * 60)

    config = AgentConfig(model=MODEL, temperature=0.1)
    agent = Agent(config)

    response: AgentResponse = agent.run_with_response("Name three sorting algorithms.")

    # AgentResponse is a Pydantic model — all fields are typed attributes
    assert isinstance(response.content, str)
    assert response.model == MODEL

    # Token fields are Optional[int] — check before using
    if response.tokens_used is not None:
        print(f"Used {response.tokens_used} tokens total.")

    # Cost is Optional[float] — may be None for local/unknown models
    if response.cost is not None:
        budget = 0.01  # $0.01 example budget
        remaining = budget - response.cost
        print(f"Cost ${response.cost:.6f} of ${budget:.2f} budget. Remaining: ${remaining:.4f}")
    else:
        print("Cost not available for this model.")

    print(f"\nFull response content:\n{response.content}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_single_run()
    asyncio.run(demo_async_run())
    demo_cost_accumulation()
    demo_field_access()
