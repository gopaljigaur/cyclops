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
        """Instantiate and register every toolkit in the cyclops.toolkits entry-point group

        Each entry point names a Toolkit subclass or a factory returning a
        Toolkit. A plugin that cannot be loaded raises instead of being skipped.
        """
        for entry_point in importlib.metadata.entry_points(group="cyclops.toolkits"):
            try:
                toolkit = entry_point.load()()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load toolkit plugin '{entry_point.name}' "
                    f"({entry_point.value})"
                ) from exc
            if not isinstance(toolkit, Toolkit):
                raise TypeError(
                    f"Toolkit plugin '{entry_point.name}' ({entry_point.value}) "
                    f"returned {type(toolkit).__name__}, not a Toolkit"
                )
            self.register(toolkit, name=entry_point.name)

    def register_tools(self) -> None:
        """Register all tools from loaded plugins"""
        for name, plugin in self.plugins.items():
            for tool in plugin.get_tools():
                try:
                    self.registry.register(tool)
                except ValueError as e:
                    logger.warning(f"Skipped tool from {name}: {e}")
                    continue
                logger.info(f"Registered tool: {tool.name}")

    def get_plugin_names(self) -> List[str]:
        """Get names of loaded plugins"""
        return list(self.plugins.keys())


__all__ = ["PluginManager", "Toolkit"]
