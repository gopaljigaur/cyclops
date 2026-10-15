"""Subagents: a lead agent delegates to specialists through agent.as_tool()"""

import asyncio

from cyclops import Agent, AgentConfig
from cyclops.core.events import RunFinished, ToolCallStarted
from cyclops.toolkit import tool


@tool
def lookup_population(city: str) -> str:
    """Look up a city's population"""
    populations = {"paris": "2.1 million", "berlin": "3.7 million"}
    return populations.get(city.lower(), "unknown")


researcher = Agent(
    AgentConfig(
        model="ollama/qwen3:4b",
        system_prompt="You answer questions about cities. Use your tools.",
    ),
    tools=[lookup_population],
)
writer = Agent(
    AgentConfig(
        model="ollama/qwen3:4b",
        system_prompt="You turn notes into one short, friendly paragraph.",
    )
)

lead = Agent(
    AgentConfig(
        model="ollama/qwen3:4b",
        system_prompt=(
            "Delegate: ask the researcher for facts, one call per city, "
            "then ask the writer to write them up."
        ),
        # Counts every completion of the lead and the agents it calls.
        max_total_tokens=50_000,
    ),
    tools=[
        researcher.as_tool("researcher", "Answers factual questions about cities"),
        writer.as_tool("writer", "Writes notes up as a short paragraph"),
    ],
)


async def main():
    async for event in lead.astream_events(
        "Compare the populations of Paris and Berlin."
    ):
        who = "/".join(event.agent_path) or "lead"
        if isinstance(event, ToolCallStarted):
            print(f"[{who}] calls {event.name}: {event.arguments}")
        elif isinstance(event, RunFinished) and not event.agent_path:
            print(f"\n{event.result.content}")
            print(f"\nTokens, all agents: {event.result.tokens_used}")


if __name__ == "__main__":
    asyncio.run(main())
