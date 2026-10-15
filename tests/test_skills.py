"""Tests for skills"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from cyclops.core.agent import Agent
from cyclops.core.types import AgentConfig
from cyclops.skills import Skills
from cyclops.toolkit.tool import Tool


def _write_skill(root, name, description="Does a thing.", body="Step one.", extra=""):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}\n"
    )
    return folder


def _response(content=None, tool_calls=None):
    msg = MagicMock(reasoning_content=None, thinking_blocks=None)
    msg.content = content
    msg.tool_calls = tool_calls or []
    response = MagicMock()
    finish_reason = "tool_calls" if tool_calls else "stop"
    response.choices = [MagicMock(message=msg, finish_reason=finish_reason)]
    response.usage = None
    return response


def _tool_call(name, arguments):
    tc = MagicMock()
    tc.id = "tc_1"
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


def test_discovers_skills_from_several_dirs(tmp_path):
    _write_skill(tmp_path / "a", "pdf-forms", "Fill PDF forms.")
    _write_skill(tmp_path / "b", "csv-report", "Summarize a CSV file.")
    (tmp_path / "b" / "notes").mkdir()

    skills = Skills(tmp_path / "a", str(tmp_path / "b"))

    assert sorted(skills.skills) == ["csv-report", "pdf-forms"]
    assert "- pdf-forms: Fill PDF forms.\n" in skills.prompt()
    assert "- csv-report: Summarize a CSV file.\n" in skills.prompt()


def test_extra_frontmatter_fields_are_ignored(tmp_path):
    _write_skill(tmp_path, "pdf-forms", extra="license: MIT\n")

    assert Skills(tmp_path).skills["pdf-forms"].description == "Does a thing."


def test_load_returns_body(tmp_path):
    _write_skill(tmp_path, "pdf-forms", body="# Filling forms\n\nUse the template.")

    assert (
        Skills(tmp_path).load("pdf-forms") == "# Filling forms\n\nUse the template.\n"
    )


def test_load_reads_a_file_in_the_skill(tmp_path):
    folder = _write_skill(tmp_path, "pdf-forms")
    (folder / "reference").mkdir()
    (folder / "reference" / "fields.md").write_text("Field list")

    assert Skills(tmp_path).load("pdf-forms", "reference/fields.md") == "Field list"


def test_load_refuses_paths_outside_the_skill(tmp_path):
    _write_skill(tmp_path, "pdf-forms")
    other = _write_skill(tmp_path, "secret")
    (tmp_path / "pdf-forms" / "link.md").symlink_to(other / "SKILL.md")
    skills = Skills(tmp_path)

    for path in ["../secret/SKILL.md", str(other / "SKILL.md"), "link.md"]:
        with pytest.raises(ValueError, match="outside"):
            skills.load("pdf-forms", path)


def test_load_unknown_skill(tmp_path):
    with pytest.raises(ValueError, match="Unknown skill"):
        Skills(tmp_path).load("missing")


@pytest.mark.parametrize(
    "front",
    [
        "name: Bad_Name\ndescription: d\n",
        "name: x\n",
        "name: x\ndescription: ''\n",
    ],
)
def test_invalid_frontmatter(tmp_path, front):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "SKILL.md").write_text(f"---\n{front}---\nbody\n")

    with pytest.raises(ValidationError):
        Skills(tmp_path)


def test_missing_frontmatter(tmp_path):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "SKILL.md").write_text("Just text\n")

    with pytest.raises(ValueError, match="no YAML frontmatter"):
        Skills(tmp_path)


@pytest.mark.parametrize(
    "front, error",
    [
        ("", "must be a YAML mapping"),
        ("- a\n- b", "must be a YAML mapping"),
        ("name: [unclosed", "invalid YAML frontmatter"),
    ],
)
def test_malformed_frontmatter_names_the_file(tmp_path, front, error):
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "SKILL.md").write_text(f"---\n{front}\n---\nbody\n")

    with pytest.raises(ValueError, match=error) as e:
        Skills(tmp_path)
    assert "SKILL.md" in str(e.value)


@pytest.mark.parametrize("path", ["missing.md", "reference", "."])
def test_load_missing_file_or_dir(tmp_path, path):
    (_write_skill(tmp_path, "pdf-forms") / "reference").mkdir()

    with pytest.raises(ValueError) as e:
        Skills(tmp_path).load("pdf-forms", path)
    assert str(e.value) == f"'{path}' not found in the skill 'pdf-forms'"


def test_tool_schema_and_annotations(tmp_path):
    definition = Skills(tmp_path).tool.definition

    assert definition.annotations.read_only is True
    assert definition.parameters["name"].description
    assert definition.parameters["file"].description
    assert definition.parameters["name"].required
    assert not definition.parameters["file"].required


def test_agent_rejects_a_tool_named_load_skill(tmp_path):
    clash = Tool("load_skill", "Mine.", lambda: "x")

    with pytest.raises(ValueError, match="Duplicate tool name: 'load_skill'"):
        Agent(AgentConfig(model="gpt-4o"), tools=[clash], skills=Skills(tmp_path))


def test_name_must_match_folder(tmp_path):
    folder = _write_skill(tmp_path, "pdf-forms")
    folder.rename(tmp_path / "forms")

    with pytest.raises(ValueError, match="does not match its folder"):
        Skills(tmp_path)


def test_duplicate_names_across_dirs(tmp_path):
    _write_skill(tmp_path / "a", "pdf-forms")
    _write_skill(tmp_path / "b", "pdf-forms")

    with pytest.raises(ValueError, match="Duplicate skill 'pdf-forms'"):
        Skills(tmp_path / "a", tmp_path / "b")


def test_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        Skills(tmp_path / "nope")


def test_agent_lists_skills_and_loads_one(tmp_path):
    _write_skill(tmp_path, "pdf-forms", "Fill PDF forms.", body="Use the template.")
    agent = Agent(
        AgentConfig(model="gpt-4o", system_prompt="Be brief."),
        skills=Skills(tmp_path),
    )
    first = _response(tool_calls=[_tool_call("load_skill", '{"name": "pdf-forms"}')])
    mock = AsyncMock(side_effect=[first, _response("Done.")])

    with patch("litellm.acompletion", new=mock):
        assert agent.run("Fill this form") == "Done."

    kwargs = mock.call_args_list[0].kwargs
    system = kwargs["messages"][0]["content"]
    assert system.startswith("Be brief.")
    assert "- pdf-forms: Fill PDF forms.\n" in system
    assert [t["function"]["name"] for t in kwargs["tools"]] == ["load_skill"]
    assert agent.messages[2]["content"] == "Use the template.\n"


def test_agent_without_skills_has_no_listing():
    agent = Agent(AgentConfig(model="gpt-4o", system_prompt="Be brief."))
    mock = AsyncMock(return_value=_response("Hi."))

    with patch("litellm.acompletion", new=mock):
        agent.run("Hi")

    assert mock.call_args.kwargs["messages"][0]["content"] == "Be brief."
