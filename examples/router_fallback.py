"""Router with fallback example using LiteLLM Router

Demonstrates automatic fallback between models if one fails or is rate-limited.
"""

from cyclops import Agent, AgentConfig
from litellm import Router

# Configure router with fallback models
# This example uses Ollama models, but you can mix providers
router = Router(
    model_list=[
        {
            "model_name": "primary-model",
            "litellm_params": {"model": "ollama/qwen3:4b"},
        },
        {
            "model_name": "primary-model",  # Same name = fallback for same logical model
            "litellm_params": {"model": "ollama/llama3.2:1b"},
        },
    ],
    # Retry configuration
    num_retries=2,
    timeout=30,
    # Optional: set fallbacks explicitly
    fallbacks=[{"primary-model": ["ollama/llama3.2:1b"]}],
)

# Create agent with router in config
config = AgentConfig(model="primary-model", router=router)
agent = Agent(config)

print("Router with Fallback Example\n")
print("If qwen3:4b fails, automatically falls back to llama3.2:1b\n")

# Test query
response = agent.run("What is 2+2?")
print("Q: What is 2+2?")
print(f"A: {response}\n")

# Continue conversation
response = agent.run("Now multiply that by 3")
print("Q: Now multiply that by 3")
print(f"A: {response}\n")

print("\nNote: Check router.deployment_stats for usage statistics")
print(f"Stats: {router.deployment_stats}")
