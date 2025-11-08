"""MCP Server example - exposing tools via Model Context Protocol"""

import asyncio
from cyclops.mcp import MCPServer
from cyclops.toolkit import tool


@tool
def add(a: float, b: float) -> float:
    """Add two numbers"""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers"""
    return a * b


@tool
def greet(name: str) -> str:
    """Greet someone by name"""
    return f"Hello, {name}!"


async def main():
    # Create MCP server with tools
    server = MCPServer(name="math-toolkit", load_plugins=False)

    # Add tools manually
    server.add_tool(add)
    server.add_tool(multiply)
    server.add_tool(greet)

    print("Starting MCP server...")
    print("Tools available: add, multiply, greet")
    print("Connect via stdio transport\n")

    # Run server
    await server.run_stdio()


if __name__ == "__main__":
    asyncio.run(main())
