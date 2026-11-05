"""Task tools: a task list the model keeps for multi-step work."""

from typing import List, Literal, Optional

from cyclops.core.events import TasksUpdated
from cyclops.core.run import _current_loop
from cyclops.core.run_state import _Run
from cyclops.core.session import Task
from cyclops.toolkit.tool import Tool
from cyclops.toolkit.types import ToolAnnotations


def _run() -> _Run:
    ref = _current_loop.get()
    loop = None if ref is None else ref()
    if loop is None:
        raise ValueError("Tasks can only be used during a run")
    return loop.run


def _find(run: _Run, id: str) -> int:
    for i, task in enumerate(run.tasks):
        if task.id == id:
            return i
    raise ValueError(f"No task has the id '{id}'")


def _changed(run: _Run) -> None:
    run.emit(TasksUpdated(tasks=list(run.tasks)))


async def _create(subject: str, description: str = "") -> str:
    """Add a task to the list, as pending.

    Args:
        subject: A short title, in the imperative.
        description: What needs to be done.
    """
    run = _run()
    task = Task(id=str(len(run.tasks) + 1), subject=subject, description=description)
    run.tasks.append(task)
    _changed(run)
    return f"Created task {task.id}."


async def _update(
    id: str,
    status: Optional[Literal["pending", "in_progress", "completed"]] = None,
    subject: Optional[str] = None,
    description: Optional[str] = None,
) -> str:
    """Change a task's status, subject or description.

    Args:
        id: The task's id.
        status: in_progress when you start it, completed once it is done.
        subject: A new subject.
        description: A new description.
    """
    run = _run()
    i = _find(run, id)
    fields = {"status": status, "subject": subject, "description": description}
    # Replaced, not changed in place, so events already emitted keep their list.
    run.tasks[i] = run.tasks[i].model_copy(
        update={k: v for k, v in fields.items() if v is not None}
    )
    _changed(run)
    return f"Updated task {id}."


async def _list() -> str:
    """List every task with its id, status and subject."""
    tasks = _run().tasks
    if not tasks:
        return "No tasks."
    return "\n".join(f"{t.id}. [{t.status}] {t.subject}" for t in tasks)


async def _get(id: str) -> str:
    """Get a task with its description.

    Args:
        id: The task's id.
    """
    run = _run()
    task = run.tasks[_find(run, id)]
    return f"{task.id}. [{task.status}] {task.subject}\n\n{task.description}"


# Built once, as the tool result reader is. Async, so events are emitted on the loop.
TASK_TOOLS: List[Tool] = [
    Tool(
        "create_task",
        "Add a task to your task list. Use the list to plan and track work "
        "of several steps.",
        _create,
    ),
    Tool(
        "update_task",
        "Change a task on your task list, as to mark it in_progress or " "completed.",
        _update,
    ),
    Tool(
        "list_tasks",
        "List the tasks on your task list.",
        _list,
        ToolAnnotations(read_only=True),
    ),
    Tool(
        "get_task",
        "Get one task on your task list, with its description.",
        _get,
        ToolAnnotations(read_only=True),
    ),
]


__all__ = ["TASK_TOOLS"]
