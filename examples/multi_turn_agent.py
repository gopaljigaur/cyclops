"""Multi-turn conversation example: sessions and transcript inspection

Each run without a session starts fresh; a Session carries a conversation
across runs.

Demonstrates:
  1. A realistic multi-turn conversation (planning a trip to Japan)
  2. Inspecting session.messages to see the full transcript
  3. Starting a new conversation with a new Session
  4. Showing how the transcript grows with each turn
"""

from typing import Tuple

from cyclops import Agent, AgentConfig, Session

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Default: Ollama (free, local: https://ollama.ai, then `ollama pull qwen3:4b`)
MODEL = "ollama/qwen3:4b"

# Alternatives:
#   MODEL = "gpt-4o-mini"                      # OPENAI_API_KEY
#   MODEL = "groq/llama-3.1-8b-instant"        # GROQ_API_KEY
#   MODEL = "claude-3-haiku-20240307"          # ANTHROPIC_API_KEY


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def print_turn(turn: int, question: str, answer: str, history_len: int) -> None:
    print(f"\n[Turn {turn}] Transcript length: {history_len} messages")
    print(f"  User : {question}")
    # Trim long answers for readability
    short = answer[:300] + "..." if len(answer) > 300 else answer
    print(f"  Agent: {short}")


# ---------------------------------------------------------------------------
# 1. Multi-turn trip planning conversation
# ---------------------------------------------------------------------------


def demo_trip_planning() -> Tuple[Agent, Session]:
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
    # The session keeps every turn, so each run sees the ones before it
    session = Session()

    conversation = [
        "I'm planning a two-week trip to Japan in April. Where should I start?",
        "We're a couple who love food, temples, and nature. Any specific cities?",
        "How many days would you suggest spending in Kyoto?",
        "What's the best way to travel between Tokyo and Kyoto?",
        "Can you summarise our itinerary so far in bullet points?",
    ]

    for i, question in enumerate(conversation, start=1):
        answer = agent.run(question, session=session)
        print_turn(i, question, answer, len(session.messages))

    return agent, session


# ---------------------------------------------------------------------------
# 2. Inspect the accumulated message history
# ---------------------------------------------------------------------------


def demo_inspect_history(session: Session) -> None:
    print("\n" + "=" * 60)
    print("2. Inspecting session.messages")
    print("=" * 60)

    messages = session.messages
    print(f"\nTotal messages in history: {len(messages)}")
    print(
        f"Expected: {len(messages) // 2} user + {len(messages) // 2} assistant turns\n"
    )

    for idx, msg in enumerate(messages):
        role = msg["role"].upper()
        content = msg.get("content") or ""
        preview = content[:80] + "..." if len(content) > 80 else content
        print(f"  [{idx:2d}] {role:9s}: {preview}")


# ---------------------------------------------------------------------------
# 3. Start a new topic in a new session
# ---------------------------------------------------------------------------


def demo_new_session(agent: Agent, old: Session) -> None:
    print("\n" + "=" * 60)
    print("3. Starting a new conversation with a new Session")
    print("=" * 60)

    session = Session()
    print(f"\nOld session: {len(old.messages)} messages")
    print(f"New session: {len(session.messages)} messages")

    # A completely different conversation: this session knows nothing of Japan
    new_conversation = [
        "What is quantum entanglement in simple terms?",
        "Can it be used for faster-than-light communication?",
        "Summarise what we just discussed in one sentence.",
    ]

    print("\nNew topic in the new session (quantum physics):\n")
    for i, question in enumerate(new_conversation, start=1):
        answer = agent.run(question, session=session)
        print_turn(i, question, answer, len(session.messages))


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
    session = Session()

    # Establish a fact
    r1 = agent.run("My favourite programming language is Rust.", session=session)
    print("\nTurn 1: establishing a fact")
    print("  User : My favourite programming language is Rust.")
    print(f"  Agent: {r1[:200]}")

    # Ask something unrelated
    r2 = agent.run("What is a closure?", session=session)
    print("\nTurn 2: unrelated question")
    print("  User : What is a closure?")
    print(f"  Agent: {r2[:200]}")

    # Refer back to the earlier fact: the agent should recall it
    r3 = agent.run(
        "Can you give me an example of a closure in my favourite language?",
        session=session,
    )
    print("\nTurn 3: refers back to earlier fact (should use Rust)")
    print("  User : Can you give me an example of a closure in my favourite language?")
    print(f"  Agent: {r3[:300]}")

    print(f"\nFinal transcript length: {len(session.messages)} messages")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent, session = demo_trip_planning()
    demo_inspect_history(session)
    demo_new_session(agent, session)
    demo_context_recall()
