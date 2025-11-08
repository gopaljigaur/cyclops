"""Using different LLM providers example"""

from cyclops import Agent, AgentConfig

# Groq (FREE - very fast, generous free tier)
# Get free API key from: https://console.groq.com
print("=== Groq Llama 3.1 (FREE) ===")
config = AgentConfig(model="groq/llama-3.1-8b-instant")
agent = Agent(config)
response = agent.run("Say hello in one sentence")
print(response + "\n")

# Together AI (FREE tier available)
# Get free API key from: https://api.together.xyz
print("=== Together AI (FREE) ===")
config = AgentConfig(model="together_ai/meta-llama/Llama-3-8b-chat-hf")
agent = Agent(config)
response = agent.run("Say hello in one sentence")
print(response + "\n")

# Ollama (FREE - runs locally, no API key needed)
# Install: https://ollama.ai
print("=== Ollama Qwen3 (FREE, LOCAL) ===")
config = AgentConfig(model="ollama/qwen3:4b")
agent = Agent(config)
response = agent.run("Say hello in one sentence")
print(response + "\n")

# OpenAI (PAID - but most reliable)
print("=== OpenAI GPT-4o-mini ===")
config = AgentConfig(model="gpt-4o-mini")
agent = Agent(config)
response = agent.run("Say hello in one sentence")
print(response + "\n")

# Note: Set environment variables:
# - GROQ_API_KEY for Groq (free at console.groq.com)
# - TOGETHERAI_API_KEY for Together AI (free at api.together.xyz)
# - No key needed for Ollama (just install and run)
# - OPENAI_API_KEY for OpenAI (paid)
