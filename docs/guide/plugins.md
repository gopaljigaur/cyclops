# Plugins

The plugin system lets you package tools as a `Toolkit` class, distribute it as a Python package, and have Cyclops (or an `MCPServer`) discover it automatically via entry points — no manual registration required.

## Toolkit base class

Subclass `Toolkit` and add `BaseTool` instances as class attributes. The `get_tools()` method scans the class for any attribute that is a `BaseTool` instance and returns them as a list.

```python
from cyclops.toolkit.plugins import Toolkit
from cyclops.toolkit.tool import BaseTool
import random


class WeatherTool(BaseTool):
    def __init__(self):
        super().__init__(name="get_weather", description="Get weather for a location.")

    async def execute(self, location: str) -> str:
        temps = {"london": 61, "tokyo": 74, "new york": 68}
        temp = temps.get(location.lower(), random.randint(55, 85))
        return f"{location}: {temp}°F"


class ForecastTool(BaseTool):
    def __init__(self):
        super().__init__(name="get_forecast", description="Get a 3-day forecast.")

    async def execute(self, location: str) -> str:
        return f"Forecast for {location}: Day 1: 70°F, Day 2: 68°F, Day 3: 72°F"


class WeatherToolkit(Toolkit):
    """Provides weather and forecast tools."""

    weather = WeatherTool()
    forecast = ForecastTool()
```

Tools are discovered by attribute inspection — there is no registration call needed inside the class.

## PluginManager

`PluginManager` orchestrates loading and registration. It holds a `ToolRegistry` and a dictionary of named `Toolkit` instances.

### register

Manually register a toolkit (useful in scripts and tests):

```python
from cyclops.toolkit.plugins import PluginManager
from cyclops.toolkit.registry import ToolRegistry

registry = ToolRegistry()
manager = PluginManager(registry)

manager.register(WeatherToolkit())
```

An optional `name` argument overrides the class name used as the plugin key.

### load_plugins

Discovers and loads all toolkits registered under the `"cyclops.toolkits"` entry-point group. Call this once at startup:

```python
manager.load_plugins()
```

### register_tools

After loading, push all tool instances into the registry:

```python
manager.register_tools()

print(registry.list_tools())  # ["get_weather", "get_forecast"]
```

### Full workflow

```python
from cyclops import Agent, AgentConfig
from cyclops.toolkit.plugins import PluginManager
from cyclops.toolkit.registry import ToolRegistry


def build_agent() -> Agent:
    registry = ToolRegistry()
    manager = PluginManager(registry)

    # Auto-discover all installed toolkit plugins.
    manager.load_plugins()
    manager.register_tools()

    tools = [registry.get_tool(name) for name in registry.list_tools()]
    config = AgentConfig(model="groq/llama-3.1-8b-instant")
    return Agent(config, tools=tools)


agent = build_agent()
print(agent.run("What is the weather in London?"))
```

## Defining tools as class attributes

You can also define tools inline using the `@tool` decorator with a registry, then attach them to a `Toolkit`:

```python
from cyclops.toolkit import tool
from cyclops.toolkit.plugins import Toolkit


class MathToolkit(Toolkit):
    @staticmethod
    def _make_tools():
        @tool
        def add(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        @tool
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b

        return add, multiply

    add, multiply = _make_tools.__func__(None)
```

For simpler cases, just instantiate `BaseTool` subclasses directly as class attributes (as shown in the `WeatherToolkit` example above).

## Entry-point registration in pyproject.toml

To make a toolkit auto-discoverable, add it to the `[project.entry-points."cyclops.toolkits"]` section of your package's `pyproject.toml`:

```toml
[project.entry-points."cyclops.toolkits"]
weather = "my_weather_package.plugin:WeatherToolkit"
```

The key (`weather`) becomes the plugin name reported by `get_plugin_names()`. The value is a dotted import path to the `Toolkit` subclass (or to a factory function that returns an instance).

After installing the package (`pip install my-weather-package` or `uv add my-weather-package`), calling `manager.load_plugins()` will find and instantiate `WeatherToolkit` automatically.

/// note
The entry point value should point to the **class** itself, not an instance. `PluginManager` calls `entry_point.load()` which returns the class or callable, then passes it directly to `register()`.
///

## Example toolkit project layout

```
my_weather_toolkit/
├── pyproject.toml
└── my_weather_toolkit/
    ├── __init__.py
    ├── plugin.py      ← WeatherToolkit class lives here
    └── tools.py       ← WeatherTool, ForecastTool classes
```

```toml
# pyproject.toml
[project]
name = "my-weather-toolkit"
version = "0.1.0"
dependencies = ["cyclops-ai>=0.1.0"]

[project.entry-points."cyclops.toolkits"]
weather = "my_weather_toolkit.plugin:WeatherToolkit"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```
