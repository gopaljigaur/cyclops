"""Simple pluggy-based toolkit plugin system"""

import pluggy
import importlib.metadata
from typing import List, Optional
from cyclops.toolkit.tool import BaseTool
from cyclops.toolkit.registry import ToolRegistry
from cyclops.utils.logging import get_logger

logger = get_logger(__name__)

# Plugin specification
hookspec = pluggy.HookspecMarker("cyclops")
hookimpl = pluggy.HookimplMarker("cyclops")


class ToolkitSpec:
    """Hook specifications for toolkit plugins"""

    @hookspec  # type: ignore[empty-body]
    def get_tools(self) -> List[BaseTool]:
        """Return list of tools provided by this toolkit"""
        ...


class PluginManager:
    """Manages toolkit plugins using pluggy"""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.pm = pluggy.PluginManager("cyclops")
        self.pm.add_hookspecs(ToolkitSpec)
        self.registry = registry or ToolRegistry()

    def load_plugins(self) -> None:
        """Load plugins from entry points"""
        try:
            # Discover entry points (Python 3.10+ API)
            entry_points = importlib.metadata.entry_points()
            toolkit_entries = entry_points.select(group="cyclops.toolkits")

            # Load and register plugins
            for entry_point in toolkit_entries:
                try:
                    plugin = entry_point.load()
                    self.pm.register(plugin, name=entry_point.name)
                    logger.info(f"Loaded toolkit plugin: {entry_point.name}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin {entry_point.name}: {e}")

        except Exception as e:
            logger.warning(f"Error loading plugins: {e}")

    def register_tools(self) -> None:
        """Register all tools from loaded plugins"""
        # Call hook to get tools from all plugins
        results = self.pm.hook.get_tools()

        for tool_list in results:
            if tool_list:
                for tool in tool_list:
                    self.registry.register(tool)
                    logger.info(f"Registered tool: {tool.name}")

    def get_plugin_names(self) -> List[str]:
        """Get names of loaded plugins"""
        return list(self.pm.list_name_plugin())


# Export the hook implementer decorator for toolkit authors
__all__ = ["PluginManager", "hookimpl", "BaseTool"]
