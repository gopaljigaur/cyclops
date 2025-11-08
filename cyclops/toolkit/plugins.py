"""Simple toolkit plugin system"""

import importlib.metadata
from typing import Dict, List, Optional
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


class PluginManager:
    """Manages toolkit plugins"""

    def __init__(self, registry: Optional[ToolRegistry] = None):
        self.plugins: Dict[str, Toolkit] = {}
        self.registry = registry or ToolRegistry()

    def register(self, plugin: Toolkit, name: Optional[str] = None) -> None:
        """Register a toolkit plugin"""
        plugin_name = name or plugin.__class__.__name__
        self.plugins[plugin_name] = plugin
        logger.info(f"Registered plugin: {plugin_name}")

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
                    self.register(plugin, name=entry_point.name)
                    logger.info(f"Loaded toolkit plugin: {entry_point.name}")
                except Exception as e:
                    logger.warning(f"Failed to load plugin {entry_point.name}: {e}")

        except Exception as e:
            logger.warning(f"Error loading plugins: {e}")

    def register_tools(self) -> None:
        """Register all tools from loaded plugins"""
        for name, plugin in self.plugins.items():
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
        return list(self.plugins.keys())


__all__ = ["PluginManager", "Toolkit"]
