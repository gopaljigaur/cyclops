"""Router with load balancing example

Demonstrates distributing requests across multiple providers/models
for better throughput and reliability.
"""

from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool
from litellm import Router


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


# Configure router with multiple backends for load balancing
router = Router(
    model_list=[
        # Multiple instances of the same model for load balancing
        {
            "model_name": "my-model",
            "litellm_params": {"model": "ollama/qwen3:4b"},
        },
        {
            "model_name": "my-model",
            "litellm_params": {"model": "ollama/qwen3:4b"},
        },
        # Can mix different providers too (if you have API keys)
        # {
        #     "model_name": "my-model",
        #     "litellm_params": {"model": "groq/llama-3.1-8b-instant"},
        # },
    ],
    # Load balancing strategy
    routing_strategy="simple-shuffle",  # Options: "simple-shuffle", "least-busy", "latency-based-routing"
    num_retries=2,
)

# Create agent with router in config
config = AgentConfig(model="my-model", router=router)
agent = Agent(config, tools=[calculate])

print("Router with Load Balancing Example\n")
print("Requests are distributed across multiple model instances\n")

# Run multiple queries - they'll be load-balanced
queries = [
    "What's 5 + 3?",
    "Calculate 10 multiplied by 7",
    "What is 100 divided by 4?",
]

for query in queries:
    response = agent.run(query)
    print(f"Q: {query}")
    print(f"A: {response}\n")

print("\nRouter deployment stats:")
print(router.deployment_stats)
