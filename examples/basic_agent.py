"""Basic agent usage example"""

from cyclops import Agent, AgentConfig, Session

# Create agent with default settings
config = AgentConfig(
    model="ollama/qwen3:4b", system_prompt="You are a helpful assistant."
)
agent = Agent(config)
session = Session()  # holds the conversation across runs

# Run the agent
response = agent.run("What is the capital of France?", session=session)
print(response)

# Continue the conversation in the same session
response = agent.run("What's the population?", session=session)
print(response)
