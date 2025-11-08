"""
Cyclops - Core agent framework with MCP toolkit support
"""

__version__ = "0.1.0"

from cyclops.core import (
    Agent,
    BaseAgent,
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
from cyclops.providers import LLMProvider, LiteLLMProvider

__all__ = [
    "Agent",
    "BaseAgent",
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
    "LLMProvider",
    "LiteLLMProvider",
]
