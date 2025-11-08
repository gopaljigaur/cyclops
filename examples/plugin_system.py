"""Plugin system example - Discover and load toolkit plugins

This example shows how to manually load a toolkit plugin.
In production, plugins would be installed via uv and auto-discovered.

To create a toolkit plugin:
1. Create a class that inherits from Toolkit
2. Implement get_tools() to return your tools
3. Register it via entry points or manually with PluginManager
"""

import sys
from pathlib import Path

# Add example_toolkit_plugin to path for this demo
sys.path.insert(0, str(Path(__file__).parent))

from cyclops import Agent, AgentConfig
from cyclops.toolkit.plugins import PluginManager
from cyclops.toolkit.registry import ToolRegistry
from example_toolkit_plugin import WeatherToolkit


def main():
    # Create plugin manager
    registry = ToolRegistry()
    plugin_manager = PluginManager(registry)

    # Manually register the toolkit plugin (in production, this would be via entry points)
    plugin_manager.pm.register(WeatherToolkit())

    # Load tools from plugins
    plugin_manager.register_tools()

    print(f"Loaded plugins: {plugin_manager.get_plugin_names()}")
    print(f"Available tools: {registry.list_tools()}\n")

    # Get tools from registry for Agent
    tools = [registry.get_tool(name) for name in registry.list_tools()]

    # Create agent with plugin-provided tools
    config = AgentConfig(model="ollama/qwen3:4b")
    agent = Agent(config, tools=tools)

    print("Agent with plugin-provided tools:\n")

    # Use weather tool
    response = agent.run("What's the weather in Seattle?")
    print("Q: What's the weather in Seattle?")
    print(f"A: {response}\n")

    # Use forecast tool
    response = agent.run("Give me a 3-day forecast for Portland")
    print("Q: Give me a 3-day forecast for Portland")
    print(f"A: {response}\n")


if __name__ == "__main__":
    main()
