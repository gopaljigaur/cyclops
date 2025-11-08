"""Ollama local model example (NO API KEY NEEDED)

Prerequisites:
1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh
2. Pull a model: ollama pull llama3
3. Run this script

No API keys or cloud services needed!
"""

from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool


@tool
def get_system_info() -> str:
    """Get system information"""
    import platform

    return f"OS: {platform.system()}, Python: {platform.python_version()}"


# Create agent using local Ollama model
config = AgentConfig(
    model="ollama/qwen3:4b",
    system_prompt="You are a helpful AI assistant running locally.",
)
agent = Agent(config, tools=[get_system_info])

print("Local Ollama Agent (No API key required)\n")

# Simple conversation
response = agent.run("Hello! Can you tell me what system I'm running on?")
print(f"Agent: {response}\n")

# Follow-up
response = agent.run("What's 25 * 4?")
print(f"Agent: {response}\n")
