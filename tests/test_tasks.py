"""Tests for the task tools"""

from cyclops import TASK_TOOLS, Agent, AgentConfig, InMemorySessionStore, Session
from cyclops.core.events import TasksUpdated
from tests.test_compaction import MODEL
from tests.test_tool_results import _call, _replies, _tool_messages


async def _events(agent: Agent, session: Session) -> list:
    return [e async for e in agent.start("Go.", stream=False, session=session)]


async def test_tasks_are_kept_on_the_session_and_saved_with_it():
    store = InMemorySessionStore()
    agent = Agent(AgentConfig(model=MODEL), tools=TASK_TOOLS, session_store=store)
    session = Session()

    with _replies(
        _call("create_task", subject="Read the logs", description="All of them."),
        _call("update_task", id="1", status="completed"),
        _call("get_task", id="1"),
        _call("update_task", id="2", status="completed"),
        _call("list_tasks"),
        "Done.",
    ):
        events = await _events(agent, session)

    assert _tool_messages(session) == [
        "Created task 1.",
        "Updated task 1.",
        "1. [completed] Read the logs\n\nAll of them.",
        "Error: update_task failed: No task has the id '2'",
        "1. [completed] Read the logs",
    ]
    updates = [e.tasks for e in events if isinstance(e, TasksUpdated)]
    assert [[t.status for t in tasks] for tasks in updates] == [
        ["pending"],
        ["completed"],
    ]
    # The update came after the first save, so it was stored as an addition.
    saved = await store.load(session.id)
    assert saved is not None and saved.tasks == session.tasks
    assert session.tasks[0].status == "completed"


async def test_a_subagent_keeps_a_list_of_its_own():
    helper = Agent(AgentConfig(model=MODEL), tools=TASK_TOOLS)
    agent = Agent(
        AgentConfig(model=MODEL),
        tools=[*TASK_TOOLS, helper.as_tool("helper", "Helps.")],
    )
    session = Session()

    with _replies(
        _call("helper", input="Plan it."),
        _call("create_task", subject="Plan"),
        "Planned.",
        _call("list_tasks"),
        "Done.",
    ):
        events = await _events(agent, session)

    (update,) = [e for e in events if isinstance(e, TasksUpdated)]
    assert update.agent_path == ("helper",)
    assert [t.subject for t in update.tasks] == ["Plan"]
    assert session.tasks == []
    assert _tool_messages(session)[-1] == "No tasks."
