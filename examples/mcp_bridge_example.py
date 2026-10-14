"""Example: use MCPBridge + MCPClientTool to connect an MCP server to a cyclops Agent.

Run from the examples/ directory:

    python mcp_bridge_example.py

This spins up the bundled mcp_server.py as a subprocess MCP server, attaches it
through MCPBridge (a sync wrapper around the async MCPClient), and passes its
tools, named ``demo__<tool>``, to an Agent.
"""

import sys
from pathlib import Path

from cyclops import Agent, AgentConfig
from cyclops.mcp import MCPBridge

MCP_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")


def main() -> None:
    bridge = MCPBridge()
    try:
        print("Connecting to MCP server...")
        mcp_tools = bridge.attach("demo", [sys.executable, MCP_SERVER_SCRIPT])
        print(f"  {len(mcp_tools)} tools: {[t.name for t in mcp_tools]}")
        print()

        agent = Agent(
            AgentConfig(
                model="anthropic/claude-haiku-4-5-20251001",
                system_prompt="You are a helpful assistant. Use the available tools when asked.",
            ),
            tools=mcp_tools,
        )

        print("Asking agent: 'Add 7 and 13, then greet Alice'")
        result = agent.run("Add 7 and 13, then greet Alice")
        print()
        print("Agent response:")
        print(result)

    finally:
        bridge.stop()
        print("\nBridge stopped.")


if __name__ == "__main__":
    main()
