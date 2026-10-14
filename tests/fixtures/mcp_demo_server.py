"""Small FastMCP server for the client tests.

Usage: mcp_demo_server.py [stdio | streamable-http PORT | sse PORT]
"""

import asyncio
import base64
import os
import sys

from pydantic import BaseModel

from mcp.server.fastmcp import FastMCP, Image
from mcp.server.fastmcp.server import Context
from mcp.types import ToolAnnotations

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
mcp = FastMCP("demo", port=port)


class Point(BaseModel):
    x: int
    y: int


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True))
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def distance(points: list[Point]) -> float:
    """Distance from the first point to the last"""
    a, b = points[0], points[-1]
    return ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5


@mcp.tool()
def fail() -> str:
    """Always fails"""
    raise ValueError("boom")


@mcp.tool()
def picture() -> Image:
    """A one pixel PNG"""
    return Image(data=PNG, format="png")


@mcp.tool()
async def slow(seconds: float) -> str:
    """Sleep, then answer"""
    await asyncio.sleep(seconds)
    return "done"


@mcp.tool()
def pid() -> int:
    """The server's process id"""
    return os.getpid()


@mcp.tool()
def header(name: str, ctx: Context) -> str:
    """A request header sent by the client"""
    return ctx.request_context.request.headers.get(name, "")


@mcp.tool()
async def grow(name: str, ctx: Context) -> str:
    """Add a tool by that name and announce the change"""
    mcp.add_tool(lambda: name, name=name, description="Added at runtime")
    await ctx.session.send_tool_list_changed()
    return name


@mcp.tool()
def crash() -> str:
    """Exit the server process"""
    os._exit(1)


if __name__ == "__main__":
    mcp.run(sys.argv[1] if len(sys.argv) > 1 else "stdio")
