"""Skills example: the agent loads a skill's instructions only when it needs them"""

from pathlib import Path

from cyclops import Agent, AgentConfig, Skills

skills = Skills(Path(__file__).parent / "skills")
agent = Agent(AgentConfig(model="ollama/qwen3:4b"), skills=skills)

response = agent.run_with_response(
    "Write release notes for 1.2.0: added CSV export, fixed a crash on empty "
    "input, renamed the --out flag to --output."
)
print(response.content)
print("\nTool calls:", [(c.name, c.arguments) for c in response.tool_calls])
