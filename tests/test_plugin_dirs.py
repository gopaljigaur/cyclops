"""Tests for skills, subagents and MCP servers loaded from plugin directories"""

import json
import shutil
import sys

import pytest

from cyclops.core.plugin_dirs import PluginDirs
from cyclops.mcp import MCPClient
from tests.test_mcp_client import SERVER


def _plugin(root, mcp=None):
    (root / ".claude-plugin").mkdir(parents=True)
    (root / ".claude-plugin" / "plugin.json").write_text(
        json.dumps({"name": root.name})
    )
    if mcp is not None:
        (root / ".mcp.json").write_text(json.dumps(mcp))
    return root


def test_reads_skills_agents_and_mcp_servers(tmp_path, monkeypatch):
    monkeypatch.setenv("DOCS_TOKEN", "secret")
    a = _plugin(
        tmp_path / "a",
        {"files": {"command": "node", "args": ["${CLAUDE_PLUGIN_ROOT}/server.js"]}},
    )
    (a / "skills" / "notes").mkdir(parents=True)
    (a / "skills" / "notes" / "SKILL.md").write_text(
        "---\nname: notes\ndescription: Notes.\n---\nWrite.\n"
    )
    (a / "agents").mkdir()
    (a / "agents" / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Reviews.\n---\nReview.\n"
    )
    _plugin(
        tmp_path / "b",
        {
            "mcpServers": {
                "docs": {
                    "type": "http",
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer ${DOCS_TOKEN}"},
                },
                "legacy": {"type": "sse", "url": "https://example.com/sse"},
            }
        },
    )
    _plugin(tmp_path / "c")

    plugins = PluginDirs(a, tmp_path / "b", tmp_path / "c")

    assert list(plugins.skills.skills) == ["notes"]
    assert list(plugins.subagents.agents) == ["reviewer"]
    assert plugins.mcp_servers == {
        "files": {"command": ["node", f"{a.resolve()}/server.js"], "env": None},
        "docs": {
            "url": "https://example.com/mcp",
            "transport": "http",
            "headers": {"Authorization": "Bearer secret"},
        },
        "legacy": {
            "url": "https://example.com/sse",
            "transport": "sse",
            "headers": None,
        },
    }


def test_duplicate_server_and_missing_dir_raise(tmp_path):
    _plugin(tmp_path / "a", {"files": {"command": "node"}})
    _plugin(tmp_path / "b", {"files": {"command": "node"}})
    with pytest.raises(ValueError, match="Duplicate MCP server 'files'"):
        PluginDirs(tmp_path / "a", tmp_path / "b")
    with pytest.raises(FileNotFoundError):
        PluginDirs(tmp_path / "missing")


async def test_server_attaches_with_its_kwargs(tmp_path):
    root = _plugin(
        tmp_path / "demo",
        {
            "demo": {
                "command": sys.executable,
                "args": ["${CLAUDE_PLUGIN_ROOT}/server.py"],
            }
        },
    )
    shutil.copy(SERVER, root / "server.py")

    async with MCPClient() as mcp:
        for name, kwargs in PluginDirs(root).mcp_servers.items():
            tools = await mcp.attach(name, **kwargs)

    assert tools and all(t.name.startswith("demo__") for t in tools)


def test_invalid_skill_and_agent_files_are_skipped(tmp_path, caplog):
    root = _plugin(tmp_path / "a")
    for folder, front in [
        ("notes", "name: notes\ndescription: Notes."),
        ("long", f"name: long\ndescription: {'x' * 1025}"),
        ("misnamed", "name: other\ndescription: Notes."),
    ]:
        (root / "skills" / folder).mkdir(parents=True)
        (root / "skills" / folder / "SKILL.md").write_text(
            f"---\n{front}\n---\nBody.\n"
        )
    (root / "agents").mkdir()
    (root / "agents" / "reviewer.md").write_text(
        "---\nname: reviewer\ndescription: Reviews.\n---\nReview.\n"
    )
    (root / "agents" / "nameless.md").write_text(
        "---\ndescription: Reviews.\n---\nReview.\n"
    )

    plugins = PluginDirs(root)

    assert list(plugins.skills.skills) == ["notes"]
    assert list(plugins.subagents.agents) == ["reviewer"]
    skipped = sorted(r.getMessage().split(":")[0] for r in caplog.records)
    assert skipped == [
        f"Skipping {root / 'agents' / 'nameless.md'}",
        f"Skipping {root / 'skills' / 'long' / 'SKILL.md'}",
        f"Skipping {root / 'skills' / 'misnamed' / 'SKILL.md'}",
    ]
