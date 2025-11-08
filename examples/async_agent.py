"""Async agent usage for concurrent operations"""

import asyncio
from cyclops import Agent, AgentConfig


async def ask_question(agent: Agent, question: str) -> str:
    """Ask a single question asynchronously"""
    response = await agent.arun(question)
    return f"Q: {question}\nA: {response}\n"


async def main():
    # Create agent
    config = AgentConfig(model="ollama/qwen3:4b")
    agent = Agent(config)

    # Ask multiple questions concurrently
    questions = [
        "What is 2+2?",
        "What is the capital of Japan?",
        "Name a primary color",
        "What year did World War 2 end?",
    ]

    print("Asking multiple questions concurrently...\n")

    # Run all questions in parallel
    tasks = [ask_question(agent, q) for q in questions]
    results = await asyncio.gather(*tasks)

    # Print results
    for result in results:
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
