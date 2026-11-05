"""Tests for subagents loaded from markdown files"""

import pytest

from cyclops.core.agent_files import Subagents
from tests.test_agent import _make_config, _make_simple_tool


def _write(root, name, front="", body="You review code."):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.md").write_text(
        f"---\nname: {name}\ndescription: Reviews code.\n{front}---\n{body}\n"
    )


def test_builds_agent_tools_from_files(tmp_path):
    _write(tmp_path / "a", "reviewer", "tools: add, sub\nmodel: gpt-4o\ncolor: red\n")
    _write(tmp_path / "b" / "nested", "helper", "model: inherit\n", "You help.")
    parent = [_make_simple_tool("add"), _make_simple_tool("sub"), _make_simple_tool("mul")]

    reviewer, helper = Subagents(tmp_path / "a", tmp_path / "b").tools(
        _make_config(max_steps=4), parent
    )

    assert (reviewer.name, reviewer.description) == ("reviewer", "Reviews code.")
    assert reviewer.agent.config.model == "gpt-4o"
    assert reviewer.agent.config.system_prompt == "You review code.\n"
    assert reviewer.agent.config.max_steps == 4
    assert [t.name for t in reviewer.agent.tools] == ["add", "sub"]
    assert helper.agent.config.model == "gpt-3.5-turbo"


def test_absent_tools_gives_every_parent_tool(tmp_path):
    _write(tmp_path, "reviewer")
    parent = [_make_simple_tool("add"), _make_simple_tool("sub")]

    (reviewer,) = Subagents(tmp_path).tools(_make_config(), parent)

    assert [t.name for t in reviewer.agent.tools] == ["add", "sub"]


def test_unknown_tool_and_duplicate_name_raise(tmp_path):
    _write(tmp_path / "a", "reviewer", "tools: [add, grep]\n")
    with pytest.raises(ValueError, match="unknown tools: \\['grep'\\]"):
        Subagents(tmp_path / "a").tools(_make_config(), [_make_simple_tool("add")])

    _write(tmp_path / "b", "reviewer")
    with pytest.raises(ValueError, match="Duplicate subagent 'reviewer'"):
        Subagents(tmp_path / "a", tmp_path / "b")
