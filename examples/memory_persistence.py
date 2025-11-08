"""Memory persistence example - Agent with persistent state

This example shows how to use Memory to persist agent state across sessions.
Use case: Chatbot that remembers user preferences, multi-turn conversations.
"""

import asyncio
from cyclops import Agent, AgentConfig
from cyclops.core.memory import InMemoryStorage


async def main():
    # Create memory storage
    memory = InMemoryStorage()

    # Store some context
    await memory.store("user_name", "Alice")
    await memory.store("preferences", {"language": "Python", "framework": "FastAPI"})
    await memory.store("last_topic", "async programming")

    # Create agent
    config = AgentConfig(model="ollama/qwen3:4b")
    agent = Agent(config)

    # Retrieve context from memory and use in conversation
    user_name = await memory.retrieve("user_name")
    preferences = await memory.retrieve("preferences")
    last_topic = await memory.retrieve("last_topic")

    # Build context-aware prompt
    context = f"""
    User Info:
    - Name: {user_name}
    - Preferences: {preferences}
    - Last discussed: {last_topic}
    """

    print("Using persistent memory for context-aware conversation:\n")

    # Ask question with context
    response = agent.run(
        f"{context}\n\nBased on my preferences, suggest a good library for building APIs."
    )
    print(f"Agent: {response}\n")

    # Store the new topic
    await memory.store("last_topic", "API frameworks")

    # List all stored keys
    keys = await memory.list_keys()
    print(f"Memory contains: {keys}")


if __name__ == "__main__":
    asyncio.run(main())
