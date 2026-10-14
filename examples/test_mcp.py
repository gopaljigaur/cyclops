"""Test MCP server by connecting as a client"""

import asyncio
import sys
from pathlib import Path

from cyclops.mcp import MCPClient

MCP_SERVER_SCRIPT = str(Path(__file__).parent / "mcp_server.py")


async def test_mcp_server():
    """Test connecting to MCP server and calling tools"""
    async with MCPClient() as client:
        print("Connecting to MCP server...")
        tools = await client.attach("math", [sys.executable, MCP_SERVER_SCRIPT])
        print(f"Available tools: {[t.name for t in tools]}")
        by_name = {t.name: t for t in tools}

        print("\nCalling 'math__add' with a=5, b=3...")
        result = await by_name["math__add"].execute(a=5, b=3)
        print(f"Result: {result.result}")
        assert result.success

        print("\nCalling 'math__greet' with name='World'...")
        result = await by_name["math__greet"].execute(name="World")
        print(f"Result: {result.result}")
        assert result.success


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
