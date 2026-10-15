"""Toolkit plugins loaded through real cyclops.toolkits entry points"""

import logging
import sys
import textwrap

import pytest

from cyclops.toolkit.plugins import PluginManager

PLUGIN_MODULE = '''
from cyclops.toolkit import tool
from cyclops.toolkit.tool import Tool
from cyclops.toolkit.plugins import Toolkit


class MathToolkit(Toolkit):
    @tool
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b

    @tool
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b


class OtherMathToolkit(Toolkit):
    add = Tool("add", "Another add", lambda a, b: a + b)
    subtract = Tool("subtract", "Subtract", lambda a, b: a - b)


def make_toolkit():
    return MathToolkit()


def not_a_toolkit():
    return object()
'''


@pytest.fixture
def install_plugin(tmp_path, monkeypatch):
    """Install a distribution with the given cyclops.toolkits entry points"""

    def install(module: str, entry_points: str, source: str = PLUGIN_MODULE) -> None:
        (tmp_path / f"{module}.py").write_text(source)
        dist_info = tmp_path / f"{module}-0.1.dist-info"
        dist_info.mkdir()
        (dist_info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {module}\nVersion: 0.1\n"
        )
        (dist_info / "entry_points.txt").write_text(
            "[cyclops.toolkits]\n" + textwrap.dedent(entry_points)
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        monkeypatch.delitem(sys.modules, module, raising=False)

    return install


def test_class_entry_point_is_instantiated(install_plugin):
    install_plugin("math_plugin_cls", "math = math_plugin_cls:MathToolkit")
    manager = PluginManager()
    manager.load_plugins()
    manager.register_tools()

    assert manager.get_plugin_names() == ["math"]
    assert type(manager.plugins["math"]).__name__ == "MathToolkit"
    assert sorted(manager.registry.list_tools()) == ["add", "multiply"]


async def test_loaded_tools_execute(install_plugin):
    install_plugin("math_plugin_exec", "math = math_plugin_exec:MathToolkit")
    manager = PluginManager()
    manager.load_plugins()
    manager.register_tools()

    assert await manager.registry.execute_tool("multiply", a=3, b=4) == 12


def test_factory_entry_point_is_called(install_plugin):
    install_plugin("math_plugin_factory", "math = math_plugin_factory:make_toolkit")
    manager = PluginManager()
    manager.load_plugins()

    assert type(manager.plugins["math"]).__name__ == "MathToolkit"


def test_broken_plugin_raises(install_plugin):
    install_plugin(
        "broken_plugin", "broken = broken_plugin:Missing", source="import nowhere\n"
    )
    manager = PluginManager()

    with pytest.raises(RuntimeError, match="'broken' \\(broken_plugin:Missing\\)"):
        manager.load_plugins()
    assert manager.get_plugin_names() == []


def test_non_toolkit_plugin_raises(install_plugin):
    install_plugin("odd_plugin", "odd = odd_plugin:not_a_toolkit")
    manager = PluginManager()

    with pytest.raises(TypeError, match="returned object, not a Toolkit"):
        manager.load_plugins()


def test_duplicate_tool_names_keep_the_first(install_plugin, caplog):
    install_plugin(
        "dup_plugin",
        """
        math = dup_plugin:MathToolkit
        other = dup_plugin:OtherMathToolkit
        """,
    )
    manager = PluginManager()
    manager.load_plugins()
    with caplog.at_level(logging.WARNING):
        manager.register_tools()

    assert sorted(manager.registry.list_tools()) == ["add", "multiply", "subtract"]
    assert manager.registry.get_tool("add").description == "Add two numbers."
    assert "Skipped tool from other: Tool 'add' is already registered" in caplog.text
