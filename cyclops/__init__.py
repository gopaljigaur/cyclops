"""
Cyclops - Core agent framework with MCP toolkit support
"""

__version__ = "0.3.0"

from cyclops.core import (
    Agent,
    AgentConfig,
    AgentHooks,
    AgentResponse,
    Message,
    Memory,
    InMemoryStorage,
    FileStorage,
)
from cyclops.toolkit import (
    BaseTool,
    tool,
    ToolResult,
    ToolRegistry,
    PluginManager,
    Toolkit,
)
from cyclops.mcp import (
    MCPServer,
    MCPClient,
    MCPBridge,
    MCPClientTool,
    tools_from_server,
)
from cyclops.observability import OTelHooks

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentHooks",
    "AgentResponse",
    "Message",
    "Memory",
    "InMemoryStorage",
    "FileStorage",
    "BaseTool",
    "tool",
    "ToolResult",
    "ToolRegistry",
    "PluginManager",
    "Toolkit",
    "MCPServer",
    "MCPClient",
    "MCPBridge",
    "MCPClientTool",
    "tools_from_server",
    "OTelHooks",
]
