"""Sessions: a conversation's transcript and usage, and stores to keep them in."""

import asyncio
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Protocol, Union

from pydantic import BaseModel, Field

from cyclops.core.types import Usage


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


class SessionStore(Protocol):
    """Where sessions are kept between runs."""

    async def load(self, session_id: str) -> Optional[Session]:
        """The session with this id, or None if there is none."""

    async def save(self, session: Session) -> None:
        """Store the session under its id, replacing any earlier version."""

    async def delete(self, session_id: str) -> None:
        """Remove the session with this id, if there is one."""


class InMemorySessionStore:
    """Keeps sessions in this process, as JSON so each load is a fresh copy."""

    def __init__(self) -> None:
        self._sessions: Dict[str, str] = {}

    async def load(self, session_id: str) -> Optional[Session]:
        data = self._sessions.get(session_id)
        return None if data is None else Session.model_validate_json(data)

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session.model_dump_json()

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


class FileSessionStore:
    """Keeps each session as <directory>/<id>.json."""

    def __init__(self, directory: Union[str, Path]) -> None:
        self.directory = Path(directory)

    def _path(self, session_id: str) -> Path:
        # An id is a file name, so it can't point outside the directory.
        if not re.fullmatch(r"[A-Za-z0-9_-]+", session_id):
            raise ValueError(f"Session id {session_id!r} is not a valid file name")
        return self.directory / f"{session_id}.json"

    async def load(self, session_id: str) -> Optional[Session]:
        path = self._path(session_id)
        try:
            data = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return None
        return Session.model_validate_json(data)

    async def save(self, session: Session) -> None:
        await asyncio.to_thread(
            self._write, self._path(session.id), session.model_dump_json()
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


__all__ = ["Session", "SessionStore", "InMemorySessionStore", "FileSessionStore"]
