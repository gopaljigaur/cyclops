"""Simple pluggy-based toolkit plugin system"""

import pluggy
import importlib.metadata
from typing import List, Optional
from cyclops.toolkit.tool import BaseTool
from cyclops.toolkit.registry import ToolRegistry
from cyclops.utils.logging import get_logger

logger = get_logger(__name__)


class Toolkit:
    """Base class for toolkit plugins

    Define tool instances as class attributes and the framework
    will automatically discover them:

    class MyToolkit(Toolkit):
        weather = WeatherTool()
        forecast = ForecastTool()
    """

    def get_tools(self) -> List[BaseTool]:
        """Return list of tools provided by this toolkit"""
        tools = []
        for attr_name in dir(self):
            if not attr_name.startswith("_"):
                attr = getattr(self, attr_name)
                if isinstance(attr, BaseTool):
                    tools.append(attr)
        return tools


# Internal pluggy hooks
_hookspec = pluggy.HookspecMarker("cyclops")
_hookimpl = pluggy.HookimplMarker("cyclops")


class _ToolkitSpec:
    """Internal hook specifications"""

    @_hookspec  # type: ignore[empty-body]
    def get_tools(self) -> List[BaseTool]:
        """Return list of tools provided by this toolkit"""
        ...


class PluginManager:
    """Manages toolkit plugins using pluggy"""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.pm = pluggy.PluginManager("cyclops")
        self.pm.add_hookspecs(_ToolkitSpec)
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
        for name, plugin in self.pm.list_name_plugin():
            if isinstance(plugin, Toolkit):
                try:
                    tools = plugin.get_tools()
                    if tools:
                        for tool in tools:
                            self.registry.register(tool)
                            logger.info(f"Registered tool: {tool.name}")
                except Exception as e:
                    logger.warning(f"Failed to get tools from {name}: {e}")

    def get_plugin_names(self) -> List[str]:
        """Get names of loaded plugins"""
        return [name for name, _ in self.pm.list_name_plugin()]


__all__ = ["PluginManager", "Toolkit"]
