"""Basic agent usage example"""

from cyclops import Agent, AgentConfig

# Create agent with default settings
config = AgentConfig(
    model="ollama/qwen3:4b", system_prompt="You are a helpful assistant."
)
agent = Agent(config)

# Run the agent
response = agent.run("What is the capital of France?")
print(response)

# Continue conversation (agent maintains history)
response = agent.run("What's the population?")
print(response)
