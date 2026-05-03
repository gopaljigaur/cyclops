"""MCP (Model Context Protocol) integration"""

from cyclops.mcp.server import MCPServer
from cyclops.mcp.client import MCPClient
from cyclops.mcp.bridge import MCPBridge
from cyclops.mcp.tools import MCPClientTool, tools_from_server

MCPTool = MCPClientTool

__all__ = [
    "MCPServer",
    "MCPClient",
    "MCPBridge",
    "MCPTool",
    "MCPClientTool",
    "tools_from_server",
]
