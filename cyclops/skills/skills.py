"""Discover SKILL.md folders and load them on demand"""

import re
from pathlib import Path
from typing import Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field

from cyclops.toolkit.tool import Tool

_FRONTMATTER = re.compile(r"---\n(.*?)\n---\n?(.*)", re.DOTALL)

_TOOL_DESCRIPTION = (
    "Load a skill's instructions by name. Pass file to read a file the "
    "instructions refer to, as a path relative to the skill's folder."
)


class Skill(BaseModel):
    """One skill: a folder holding a SKILL.md and any files it refers to."""

    name: str = Field(max_length=64, pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")
    description: str = Field(min_length=1, max_length=1024)
    path: Path
    body: str

    @classmethod
    def from_file(cls, file: Path) -> "Skill":
        """Parse a SKILL.md: YAML frontmatter, then the markdown body."""
        match = _FRONTMATTER.match(file.read_text())
        if match is None:
            raise ValueError(f"{file} has no YAML frontmatter")
        front = yaml.safe_load(match.group(1))
        skill = cls.model_validate(
            {**front, "path": file.parent, "body": match.group(2)}
        )
        if skill.name != file.parent.name:
            raise ValueError(f"{file}: name '{skill.name}' does not match its folder")
        return skill


class Skills:
    """Skills found in the subfolders of one or more directories."""

    def __init__(self, *dirs: Union[str, Path]):
        self.skills: Dict[str, Skill] = {}
        for d in dirs:
            root = Path(d).expanduser()
            if not root.is_dir():
                raise FileNotFoundError(f"Skills directory not found: {root}")
            for file in sorted(root.glob("*/SKILL.md")):
                skill = Skill.from_file(file)
                if skill.name in self.skills:
                    raise ValueError(f"Duplicate skill '{skill.name}' in {file}")
                self.skills[skill.name] = skill
        self.tool = Tool("load_skill", _TOOL_DESCRIPTION, self.load)

    def prompt(self) -> str:
        """The skill listing appended to the system prompt."""
        listing = "".join(
            f"- {s.name}: {s.description}\n" for s in self.skills.values()
        )
        return (
            "\n\nThese skills hold instructions for specific tasks. Before such "
            "a task, call load_skill with the skill's name and follow what it "
            f"returns.\n\n{listing}"
        )

    def load(self, name: str, file: Optional[str] = None) -> str:
        """Return a skill's body, or one of its files when file is given."""
        if name not in self.skills:
            raise ValueError(f"Unknown skill '{name}'")
        skill = self.skills[name]
        if file is None:
            return skill.body
        root = skill.path.resolve()
        target = (root / file).resolve()
        # resolve() follows symlinks, so a link out of the folder is refused too.
        if not target.is_relative_to(root):
            raise ValueError(f"'{file}' is outside the skill '{name}'")
        return target.read_text()
