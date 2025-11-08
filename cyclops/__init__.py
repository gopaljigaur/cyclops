"""
Cyclops - Core agent framework with MCP toolkit support
"""

__version__ = "0.1.0"

from cyclops.core import Agent, BaseAgent
from cyclops.core.agent import AgentConfig
from cyclops.toolkit import ToolRegistry, Tool
from cyclops.mcp import MCPServer, MCPClient
from cyclops.providers import LLMProvider, LiteLLMProvider

__all__ = [
    "Agent",
    "BaseAgent",
    "AgentConfig",
    "ToolRegistry",
    "Tool",
    "MCPServer",
    "MCPClient",
    "LLMProvider",
    "LiteLLMProvider",
]
