"""
Cyclops - Core agent framework with MCP toolkit support
"""

__version__ = "0.3.0"

from cyclops.core import (
    Agent,
    AgentConfig,
    AgentHooks,
    Compaction,
    DeniedToolCall,
    Deny,
    Guard,
    GuardContext,
    ToolCallContext,
    ToolMiddleware,
    ToolPermissions,
    Message,
    RunHandle,
    Session,
    SessionStore,
    StreamIdleTimeout,
    InMemorySessionStore,
    FileSessionStore,
    RunResult,
    Usage,
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
from cyclops.observability import TelemetryHooks
from cyclops.skills import Skill, Skills

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentHooks",
    "Compaction",
    "DeniedToolCall",
    "Deny",
    "Guard",
    "GuardContext",
    "ToolCallContext",
    "ToolMiddleware",
    "ToolPermissions",
    "Message",
    "RunHandle",
    "Session",
    "SessionStore",
    "StreamIdleTimeout",
    "InMemorySessionStore",
    "FileSessionStore",
    "RunResult",
    "Usage",
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
    "TelemetryHooks",
    "Skill",
    "Skills",
]
