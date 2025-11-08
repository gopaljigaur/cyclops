"""
Cyclops - Core agent framework with MCP toolkit support
"""

__version__ = "0.1.0"

from cyclops.core import (
    Agent,
    AgentConfig,
    AgentResponse,
    Message,
    Memory,
    InMemoryStorage,
)
from cyclops.toolkit import (
    BaseTool,
    tool,
    ToolResult,
    ToolRegistry,
    PluginManager,
    Toolkit,
)
from cyclops.mcp import MCPServer, MCPClient

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentResponse",
    "Message",
    "Memory",
    "InMemoryStorage",
    "BaseTool",
    "tool",
    "ToolResult",
    "ToolRegistry",
    "PluginManager",
    "Toolkit",
    "MCPServer",
    "MCPClient",
]
