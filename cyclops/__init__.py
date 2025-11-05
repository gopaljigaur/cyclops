"""
Cyclops - Core agent framework with MCP toolkit support
"""

__version__ = "0.1.0"

from cyclops.core import Agent, BaseAgent
from cyclops.toolkit import ToolRegistry, Tool
from cyclops.mcp import MCPServer, MCPClient

__all__ = [
    "Agent",
    "BaseAgent",
    "ToolRegistry",
    "Tool",
    "MCPServer",
    "MCPClient",
]
