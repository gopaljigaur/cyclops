"""Agent with tools example"""

from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool
from datetime import datetime
import random


@tool
def get_time() -> str:
    """Get the current time"""
    return datetime.now().strftime("%H:%M:%S")


@tool
def get_weather(location: str) -> str:
    """Get weather for a location"""
    conditions = ["Sunny", "Cloudy", "Rainy", "Snowy"]
    temp = random.randint(50, 90)
    return f"Weather in {location}: {random.choice(conditions)}, {temp}°F"


@tool
def calculate(operation: str, a: float, b: float) -> str:
    """Perform basic math calculations"""
    ops = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero",
    }
    result = ops.get(operation, "Unknown operation")
    return f"{a} {operation} {b} = {result}"


# Create agent with tools
config = AgentConfig(model="ollama/qwen3:4b")
agent = Agent(config, tools=[get_time, get_weather, calculate])

# Ask questions that require tools
print("Agent with tools demo:\n")

response = agent.run("What time is it?")
print(f"Q: What time is it?\nA: {response}\n")

response = agent.run("What's the weather in New York?")
print(f"Q: What's the weather in New York?\nA: {response}\n")

response = agent.run("What's 15 multiplied by 7?")
print(f"Q: What's 15 multiplied by 7?\nA: {response}\n")
