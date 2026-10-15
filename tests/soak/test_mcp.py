"""Soak tests for MCP over stdio: tool calls and attach and detach cycles."""

import itertools
import sys
from pathlib import Path

import pytest

from cyclops import Agent, AgentConfig
from cyclops.mcp import MCPClient
from tests.soak.harness import soak

pytestmark = pytest.mark.soak

SERVER = [
    sys.executable,
    str(Path(__file__).parent.parent / "fixtures" / "mcp_demo_server.py"),
]


async def test_agent_runs_with_mcp_tools():
    async with MCPClient() as mcp:
        tools = [t for t in await mcp.attach("demo", SERVER) if t.name == "demo__add"]

        async def run():
            agent = Agent(AgentConfig(model="gpt-4o-mini"), tools=tools)
            result = await agent.arun_with_response("hi")
            assert result.tool_calls[0].result == "2"

        result = await soak(run)
    result.assert_steady()
    assert result.cyclic_garbage == 0, result


async def test_attach_detach_cycles():
    mcp = MCPClient()

    async def cycle():
        await mcp.attach("demo", SERVER)
        await mcp.detach("demo")

    # Each cycle starts a server process, so fewer runs.
    result = await soak(cycle, runs=50, warmup=5)
    await mcp.aclose()
    # Traced memory is left out: each cycle leaves a few hundred bytes in
    # anyio's memory streams that no reachable object holds.
    assert not result.grown_types, result
    assert result.after.within(result.before), result


async def test_detaching_unknown_names_keeps_nothing():
    mcp = MCPClient()
    names = (f"unknown{i}" for i in itertools.count())
    (await soak(lambda: mcp.detach(next(names)))).assert_steady()
