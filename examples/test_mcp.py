"""Test MCP server by connecting as a client"""

import asyncio
import sys
from cyclops.mcp import MCPClient


async def test_mcp_server():
    """Test connecting to MCP server and calling tools"""
    client = MCPClient()

    try:
        # Connect to the MCP server
        print("Connecting to MCP server...")
        await client.connect_stdio([sys.executable, "mcp_server.py"])

        # List available tools
        print("\nListing tools...")
        tools = await client.list_tools()
        print(f"Available tools: {[tool['name'] for tool in tools]}")

        # Test calling a tool
        print("\nCalling 'add' tool with a=5, b=3...")
        result = await client.call_tool("add", {"a": 5, "b": 3})
        print(f"Result: {result}")

        # Test another tool
        print("\nCalling 'greet' tool with name='World'...")
        result = await client.call_tool("greet", {"name": "World"})
        print(f"Result: {result}")

        print("\n[PASS] MCP server test passed!")

    except Exception as e:
        print(f"[FAIL] Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())
