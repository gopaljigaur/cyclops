"""Subagents defined in markdown files, in the format of Claude Code and OpenCode"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator

from cyclops.core.agent import Agent
from cyclops.core.subagents import AgentTool
from cyclops.core.types import AgentConfig
from cyclops.skills.skills import _frontmatter


class SubagentFile(BaseModel):
    """One parsed subagent file: frontmatter fields, then the system prompt."""

    name: str = Field(max_length=64, pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    description: str = Field(min_length=1)
    tools: Optional[List[str]] = None
    model: Optional[str] = None
    prompt: str

    @field_validator("tools", mode="before")
    @classmethod
    def _split(cls, value: Any) -> Any:
        """Accept a comma-separated string, as Claude Code writes it."""
        if isinstance(value, str):
            return [t.strip() for t in value.split(",")]
        return value

    @classmethod
    def from_file(cls, file: Path) -> "SubagentFile":
        front, body = _frontmatter(file)
        return cls.model_validate({**front, "prompt": body})


class Subagents:
    """Subagents found in the .md files of one or more directories."""

    def __init__(self, *dirs: Union[str, Path]):
        self.agents: Dict[str, SubagentFile] = {}
        for d in dirs:
            root = Path(d).expanduser()
            if not root.is_dir():
                raise FileNotFoundError(f"Subagents directory not found: {root}")
            for file in sorted(root.glob("**/*.md")):
                agent = SubagentFile.from_file(file)
                if agent.name in self.agents:
                    raise ValueError(f"Duplicate subagent '{agent.name}' in {file}")
                self.agents[agent.name] = agent

    def tools(self, config: AgentConfig, tools: List) -> List[AgentTool]:
        """One AgentTool per subagent, built on the parent's config and tools.

        A subagent keeps config but for its own system prompt and model, and
        gets the tools it names, or all of them when it names none.
        """
        by_name = {t.name: t for t in tools}
        result = []
        for a in self.agents.values():
            if a.tools is None:
                chosen = list(tools)
            else:
                unknown = [n for n in a.tools if n not in by_name]
                if unknown:
                    raise ValueError(
                        f"Subagent '{a.name}' names unknown tools: {unknown}"
                    )
                chosen = [by_name[n] for n in a.tools]
            model = config.model if a.model in (None, "inherit") else a.model
            sub = config.model_copy(update={"model": model, "system_prompt": a.prompt})
            result.append(Agent(sub, tools=chosen).as_tool(a.name, a.description))
        return result


__all__ = ["SubagentFile", "Subagents"]
