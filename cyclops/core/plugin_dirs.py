"""Skills, subagents and MCP servers from plugin directories in Claude Code's layout"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Union

from cyclops.core.agent_files import Subagents
from cyclops.skills import Skills


def _expand(value: Any, root: Path) -> Any:
    """Expand ${CLAUDE_PLUGIN_ROOT} and environment variables in a config value."""
    if isinstance(value, str):
        return os.path.expandvars(value.replace("${CLAUDE_PLUGIN_ROOT}", str(root)))
    if isinstance(value, list):
        return [_expand(v, root) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v, root) for k, v in value.items()}
    return value


def _attach_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    """A .mcp.json server entry as keyword arguments for MCPClient.attach."""
    if "url" in config:
        return {
            "url": config["url"],
            "transport": "sse" if config.get("type") == "sse" else "http",
            "headers": config.get("headers"),
        }
    return {
        "command": [config["command"], *config.get("args", [])],
        "env": config.get("env"),
    }


class PluginDirs:
    """The skills/, agents/ and .mcp.json of one or more plugin directories."""

    def __init__(self, *dirs: Union[str, Path]):
        roots = [Path(d).expanduser().resolve() for d in dirs]
        for root in roots:
            if not root.is_dir():
                raise FileNotFoundError(f"Plugin directory not found: {root}")
        self.skills = Skills(*(r / "skills" for r in roots if (r / "skills").is_dir()))
        self.subagents = Subagents(
            *(r / "agents" for r in roots if (r / "agents").is_dir())
        )
        # Keyword arguments for MCPClient.attach, by server name.
        self.mcp_servers: Dict[str, Dict[str, Any]] = {}
        for root in roots:
            file = root / ".mcp.json"
            if not file.is_file():
                continue
            servers = json.loads(file.read_text())
            # Some plugins wrap the map in "mcpServers", as .claude.json does.
            servers = servers.get("mcpServers", servers)
            for name, config in servers.items():
                if name in self.mcp_servers:
                    raise ValueError(f"Duplicate MCP server '{name}' in {file}")
                self.mcp_servers[name] = _attach_kwargs(_expand(config, root))


__all__ = ["PluginDirs"]
