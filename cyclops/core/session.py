"""Sessions: a conversation's transcript and usage, and stores to keep them in."""

import asyncio
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
)

from pydantic import BaseModel, Field

from cyclops.core.types import Usage

# When a run saves its session to the agent's store, as LangGraph names it:
# after each step, before the next (sync) or while it runs (async), or once
# the run ends (exit).
Durability = Literal["sync", "async", "exit"]


class Session(BaseModel):
    """One conversation: its transcript and what its runs used.

    Pass it to a run with session=. Serialize it with model_dump_json() and
    read it back with Session.model_validate_json().
    """

    version: Literal[1] = Field(default=1, description="Format version of the JSON")
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="LiteLLM messages, with tool calls, tool results and reasoning",
    )
    usage: Optional[Usage] = Field(
        default_factory=Usage,
        description="Summed over every run; None once a run's usage was unknown",
    )

    def fork(self) -> "Session":
        """A copy of this session with a new id, which goes on independently."""
        return self.model_copy(deep=True, update={"id": uuid.uuid4().hex})

    def truncate(self, index: int) -> None:
        """Keep the messages before index, as to retry from an earlier point.

        Raises ValueError if that would keep tool calls without their results.
        """
        rest = self.messages[index:]
        if rest and rest[0]["role"] == "tool":
            raise ValueError(
                f"Message {index} is a tool result; truncate before its tool call"
            )
        del self.messages[index:]


class SessionStore(Protocol):
    """Where sessions are kept between runs."""

    async def load(self, session_id: str) -> Optional[Session]:
        """The session with this id, or None if there is none."""

    async def save(self, session: Session) -> None:
        """Store the session under its id, replacing any earlier version."""

    async def add_messages(
        self, session: Session, messages: List[Dict[str, Any]]
    ) -> None:
        """Add messages to the end of the stored session and store its usage.

        Called only for a session saved before.
        """

    async def delete(self, session_id: str) -> None:
        """Remove the session with this id, if there is one."""


class _Added(BaseModel):
    """Messages added to a stored session, and its usage after them."""

    messages: List[Dict[str, Any]]
    usage: Optional[Usage]


def _added_json(session: Session, messages: List[Dict[str, Any]]) -> str:
    return _Added(messages=messages, usage=session.usage).model_dump_json()


def _replay(lines: Iterable[str]) -> Session:
    """A session from its saved JSON followed by the JSON of each addition."""
    first, *added = lines
    session = Session.model_validate_json(first)
    for line in added:
        addition = _Added.model_validate_json(line)
        session.messages += addition.messages
        session.usage = addition.usage
    return session


class InMemorySessionStore:
    """Keeps sessions in this process, as JSON so each load is a fresh copy."""

    def __init__(self) -> None:
        # Per session, its saved JSON and then the JSON of each addition.
        self._sessions: Dict[str, List[str]] = {}

    async def load(self, session_id: str) -> Optional[Session]:
        lines = self._sessions.get(session_id)
        return None if lines is None else _replay(lines)

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = [session.model_dump_json()]

    async def add_messages(
        self, session: Session, messages: List[Dict[str, Any]]
    ) -> None:
        self._sessions[session.id].append(_added_json(session, messages))

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class FileSessionStore:
    """Keeps each session as <directory>/<id>.jsonl.

    The first line is the saved session, each line after it one addition.
    """

    def __init__(self, directory: Union[str, Path]) -> None:
        self.directory = Path(directory)

    def _path(self, session_id: str) -> Path:
        # An id is a file name, so it can't point outside the directory.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError(f"Session id {session_id!r} is not a valid file name")
        return self.directory / f"{session_id}.jsonl"

    async def load(self, session_id: str) -> Optional[Session]:
        path = self._path(session_id)
        try:
            data = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return None
        lines = data.splitlines(keepends=True)
        # A crash during an addition leaves its line without the newline.
        return _replay(line for line in lines if line.endswith("\n"))

    async def save(self, session: Session) -> None:
        await asyncio.to_thread(
            self._write, self._path(session.id), session.model_dump_json() + "\n"
        )

    async def add_messages(
        self, session: Session, messages: List[Dict[str, Any]]
    ) -> None:
        await asyncio.to_thread(
            self._append, self._path(session.id), _added_json(session, messages)
        )

    async def delete(self, session_id: str) -> None:
        await asyncio.to_thread(self._path(session_id).unlink, missing_ok=True)

    @staticmethod
    def _write(path: Path, data: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # A temporary file of its own per save, replaced in one step, so
        # concurrent saves don't collide and a crash leaves the old session.
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
        ) as tmp:
            tmp.write(data)
        os.replace(tmp.name, path)

    @staticmethod
    def _append(path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class _SessionSaver:
    """Saves one run's session to a store after each step or once at its end.

    A run's first save stores the whole session, since it may have changed
    since the store last saw it; later ones add the messages since.
    """

    def __init__(
        self, store: SessionStore, session: Session, durability: Durability
    ) -> None:
        self.store = store
        self.session = session
        self.durability = durability
        # How many messages the store has, once this run saved the session.
        self._saved: Optional[int] = None
        self._pending: Optional["asyncio.Task[None]"] = None

    async def step(self) -> None:
        """Save after a step, raising the error of an earlier save that failed."""
        if self.durability == "exit":
            return
        await self._wait()
        self._pending = self._save()
        if self.durability == "sync":
            await self._wait()

    async def end(self) -> None:
        """Save at the end of the run, once an earlier save is done."""
        await self._wait()
        await self._save()

    async def _wait(self) -> None:
        pending, self._pending = self._pending, None
        if pending is not None:
            await pending

    def _save(self) -> "asyncio.Task[None]":
        # What to store is taken now: the run goes on while the task saves.
        messages = self.session.messages
        if self._saved is None:
            copy = self.session.model_copy(update={"messages": list(messages)})
            write = self.store.save(copy)
        else:
            write = self.store.add_messages(self.session, messages[self._saved :])
        return asyncio.ensure_future(self._stored(write, len(messages)))

    async def _stored(self, write: Awaitable[None], count: int) -> None:
        await write
        self._saved = count


__all__ = [
    "Durability",
    "Session",
    "SessionStore",
    "InMemorySessionStore",
    "FileSessionStore",
]
