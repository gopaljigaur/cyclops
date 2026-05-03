"""Multi-turn conversation example — history, reset(), and messages inspection

Demonstrates:
  1. A realistic multi-turn conversation (planning a trip to Japan)
  2. Inspecting agent.messages to see the full accumulated history
  3. Resetting the conversation with agent.reset()
  4. Starting a completely unrelated new conversation after reset
  5. Showing how history length grows with each turn
"""

from cyclops import Agent, AgentConfig

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Default: Ollama (free, local — https://ollama.ai, then `ollama pull qwen3:4b`)
MODEL = "ollama/qwen3:4b"

# Alternatives:
#   MODEL = "gpt-4o-mini"                      # OPENAI_API_KEY
#   MODEL = "groq/llama-3.1-8b-instant"        # GROQ_API_KEY
#   MODEL = "claude-3-haiku-20240307"          # ANTHROPIC_API_KEY


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def print_turn(turn: int, question: str, answer: str, history_len: int) -> None:
    print(f"\n[Turn {turn}] History length: {history_len} messages")
    print(f"  User : {question}")
    # Trim long answers for readability
    short = answer[:300] + "..." if len(answer) > 300 else answer
    print(f"  Agent: {short}")


# ---------------------------------------------------------------------------
# 1. Multi-turn trip planning conversation
# ---------------------------------------------------------------------------

def demo_trip_planning() -> Agent:
    print("=" * 60)
    print("1. Multi-turn trip planning conversation (Japan)")
    print("=" * 60)

    config = AgentConfig(
        model=MODEL,
        system_prompt=(
            "You are a knowledgeable travel assistant specialising in Japan. "
            "Give concise, practical advice."
        ),
        temperature=0.7,
    )
    agent = Agent(config)

    # The agent remembers context from every previous turn in _history
    conversation = [
        "I'm planning a two-week trip to Japan in April. Where should I start?",
        "We're a couple who love food, temples, and nature. Any specific cities?",
        "How many days would you suggest spending in Kyoto?",
        "What's the best way to travel between Tokyo and Kyoto?",
        "Can you summarise our itinerary so far in bullet points?",
    ]

    for i, question in enumerate(conversation, start=1):
        answer = agent.run(question)
        print_turn(i, question, answer, len(agent.messages))

    return agent


# ---------------------------------------------------------------------------
# 2. Inspect the accumulated message history
# ---------------------------------------------------------------------------

def demo_inspect_history(agent: Agent) -> None:
    print("\n" + "=" * 60)
    print("2. Inspecting agent.messages")
    print("=" * 60)

    messages = agent.messages  # returns a shallow copy of _history
    print(f"\nTotal messages in history: {len(messages)}")
    print(f"Expected: {len(messages) // 2} user + {len(messages) // 2} assistant turns\n")

    for idx, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg.get("content") or ""
        preview = content[:80] + "..." if len(content) > 80 else content
        print(f"  [{idx:2d}] {role:9s}: {preview}")


# ---------------------------------------------------------------------------
# 3. Reset and start a new topic
# ---------------------------------------------------------------------------

def demo_reset(agent: Agent) -> None:
    print("\n" + "=" * 60)
    print("3. Resetting conversation with agent.reset()")
    print("=" * 60)

    print(f"\nBefore reset — messages in history: {len(agent.messages)}")

    agent.reset()

    print(f"After  reset — messages in history: {len(agent.messages)}")
    assert len(agent.messages) == 0, "History should be empty after reset"

    # Start a completely different conversation — the agent has no memory of Japan
    new_conversation = [
        "What is quantum entanglement in simple terms?",
        "Can it be used for faster-than-light communication?",
        "Summarise what we just discussed in one sentence.",
    ]

    print("\nNew topic after reset (quantum physics):\n")
    for i, question in enumerate(new_conversation, start=1):
        answer = agent.run(question)
        print_turn(i, question, answer, len(agent.messages))


# ---------------------------------------------------------------------------
# 4. Show that context is preserved (the agent recalls earlier turns)
# ---------------------------------------------------------------------------

def demo_context_recall() -> None:
    print("\n" + "=" * 60)
    print("4. Demonstrating context recall across turns")
    print("=" * 60)

    config = AgentConfig(
        model=MODEL,
        system_prompt="You are a helpful assistant with excellent memory.",
        temperature=0.3,
    )
    agent = Agent(config)

    # Establish a fact
    r1 = agent.run("My favourite programming language is Rust.")
    print(f"\nTurn 1 — establishing a fact")
    print(f"  User : My favourite programming language is Rust.")
    print(f"  Agent: {r1[:200]}")

    # Ask something unrelated
    r2 = agent.run("What is a closure?")
    print(f"\nTurn 2 — unrelated question")
    print(f"  User : What is a closure?")
    print(f"  Agent: {r2[:200]}")

    # Refer back to the earlier fact — the agent should recall it
    r3 = agent.run("Can you give me an example of a closure in my favourite language?")
    print(f"\nTurn 3 — refers back to earlier fact (should use Rust)")
    print(f"  User : Can you give me an example of a closure in my favourite language?")
    print(f"  Agent: {r3[:300]}")

    print(f"\nFinal history length: {len(agent.messages)} messages")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = demo_trip_planning()
    demo_inspect_history(agent)
    demo_reset(agent)
    demo_context_recall()
