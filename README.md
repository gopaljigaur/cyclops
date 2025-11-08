# Cyclops

A barebones core agent framework with MCP toolkit support.

*AI → A I → A-eye → One eye → Cyclops*

## Features

- **Core Agent Framework**: Simple, extensible agent architecture with LiteLLM integration
- **Tool System**: Flexible tool registry with MCP support
- **Memory Management**: Abstract memory interface with implementations

## Quick Start

```python
from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool

@tool
def get_time() -> str:
    """Get current time"""
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

config = AgentConfig(model="gpt-4o-mini")
agent = Agent(config, tools=[get_time])

response = agent.run("What time is it?")
print(response)
```

[//]: # (## Installation)

[//]: # ()
[//]: # (```bash)

[//]: # (# Install with uv &#40;recommended&#41;)

[//]: # (uv add cyclops)

[//]: # (```)

## Structure

```
cyclops/
├── core/           # Core agent framework
├── toolkit/        # Tools and utilities
├── mcp/           # MCP server/client
└── utils/         # Utilities and config
```

## Development

```bash
# Install dependencies
uv sync

# Setup pre-commit hooks
uv run pre-commit install

# Run checks manually
uv run pre-commit run --all-files
```