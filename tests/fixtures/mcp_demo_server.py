"""Small MCPServer for the client tests.

Usage: mcp_demo_server.py [stdio | streamable-http PORT | sse PORT | no-get PORT]

``no-get`` serves streamable HTTP but answers every GET with 404.
"""

import asyncio
import base64
import os
import sys

from pydantic import BaseModel

from mcp.server.mcpserver import Context, Image, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

mcp = MCPServer("demo")


class Point(BaseModel):
    x: int
    y: int


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
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
    raise ToolError("boom")


@mcp.tool()
def picture() -> Image:
    """A one pixel PNG"""
    return Image(data=PNG, format="png")


started = 0
finished = 0


@mcp.tool()
async def slow(seconds: float) -> str:
    """Sleep, then answer"""
    global started, finished
    started += 1
    await asyncio.sleep(seconds)
    finished += 1
    return "done"


@mcp.tool()
def slow_started() -> int:
    """How many slow calls started"""
    return started


@mcp.tool()
def slow_finished() -> int:
    """How many slow calls ran to the end"""
    return finished


def no_get(app):
    async def serve(scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "GET":
            await send({"type": "http.response.start", "status": 404, "headers": []})
            await send({"type": "http.response.body", "body": b""})
        else:
            await app(scope, receive, send)

    return serve


@mcp.tool()
def pid() -> int:
    """The server's process id"""
    return os.getpid()


@mcp.tool()
def header(name: str, ctx: Context) -> str:
    """A request header sent by the client"""
    return (ctx.headers or {}).get(name, "")


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
    if sys.argv[1:2] == ["no-get"]:
        import uvicorn

        uvicorn.run(no_get(mcp.streamable_http_app()), port=int(sys.argv[2]))
    elif len(sys.argv) > 2:
        mcp.run(sys.argv[1], port=int(sys.argv[2]))
    else:
        mcp.run()
