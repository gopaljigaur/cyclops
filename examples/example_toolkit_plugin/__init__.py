"""Example toolkit plugin

This shows how to create a toolkit that can be discovered by the plugin system.
To make this installable, you would:
1. Create a separate package
2. Add entry point in pyproject.toml:
   [project.entry-points."cyclops.toolkits"]
   weather = "my_toolkit.plugin:WeatherToolkit"
3. Install with: uv add my-weather-toolkit
4. Tools auto-discovered by MCPServer or PluginManager
"""

from cyclops.toolkit.tool import BaseTool
from cyclops.toolkit.plugins import Toolkit
import random


class WeatherTool(BaseTool):
    """Get weather information"""

    def __init__(self):
        super().__init__(name="get_weather", description="Get weather for a location")

    async def execute(self, location: str) -> str:
        """Get weather for a location"""
        conditions = ["Sunny", "Cloudy", "Rainy", "Snowy"]
        temp = random.randint(50, 90)
        return f"Weather in {location}: {random.choice(conditions)}, {temp}°F"


class ForecastTool(BaseTool):
    """Get weather forecast"""

    def __init__(self):
        super().__init__(
            name="get_forecast", description="Get 3-day forecast for a location"
        )

    async def execute(self, location: str) -> str:
        """Get forecast for a location"""
        forecast = []
        for i in range(1, 4):
            temp = random.randint(50, 90)
            forecast.append(f"Day {i}: {temp}°F")
        return f"Forecast for {location}:\n" + "\n".join(forecast)


class WeatherToolkit(Toolkit):
    """Weather toolkit plugin

    Tools are automatically discovered from class attributes.
    No need to implement get_tools() unless you need custom logic.
    """

    weather = WeatherTool()
    forecast = ForecastTool()
