"""The Agent class and its public entry points."""

import asyncio
from contextlib import aclosing
from typing import Any, AsyncGenerator, Dict, Iterator, List, Optional, Tuple, Type

from cyclops.core.events import Event, RunFinished, TextDelta
from cyclops.core.run import _RunLoop
from cyclops.core.run_state import RunHandle, _Run
from cyclops.core.session import Session
from cyclops.core.subagents import AgentTool
from cyclops.core.sync import _iterate_sync, _run_coroutine_sync
from cyclops.core.types import AgentConfig, RunResult
from cyclops.skills import Skills
from cyclops.toolkit.types import ToolAnnotations


class Agent:
    """LLM agent implementation"""

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List] = None,
        skills: Optional[Skills] = None,
    ):
        self.config = config
        self.tools = (tools or []) + ([skills.tool] if skills else [])
        self.skills = skills
        self._tools_by_name: Dict[str, Any] = {}
        for t in self.tools:
            if t.name in self._tools_by_name:
                raise ValueError(f"Duplicate tool name: '{t.name}'")
            self._tools_by_name[t.name] = t
        # Set to "naive" once this agent's model rejects native tool calling.
        self._detected_tool_mode: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(
        self,
        input_message: str,
        *,
        stream: bool = True,
        response_model: Optional[Type] = None,
        session: Optional[Session] = None,
    ) -> RunHandle:
        """Start a run in a task on the running event loop and return its handle.

        With stream=False the completions aren't streamed, so each step's text
        arrives as one TextDelta. response_model asks the provider for that
        schema when it supports one; the result's content is the raw JSON.
        With a session, the run reads and extends the session's transcript and
        adds its usage to the session; without one, it starts a fresh transcript.
        """
        return RunHandle(
            *self._start(
                input_message,
                stream=stream,
                response_model=response_model,
                session=session,
            )
        )

    def run(
        self,
        input_message: str,
        response_model: Optional[Type] = None,
        *,
        session: Optional[Session] = None,
    ) -> Any:
        """Run the agent synchronously. Returns str or Pydantic model instance."""
        return _run_coroutine_sync(
            self.arun, input_message, response_model, session=session
        )

    async def arun(
        self,
        input_message: str,
        response_model: Optional[Type] = None,
        *,
        session: Optional[Session] = None,
    ) -> Any:
        """Run the agent asynchronously. Returns str or Pydantic model instance."""
        result = await self.start(
            input_message,
            stream=False,
            response_model=response_model,
            session=session,
        )
        if result.error is not None:
            raise result.error
        if response_model is not None:
            return response_model.model_validate_json(result.content)
        return result.content

    def run_with_response(
        self, input_message: str, *, session: Optional[Session] = None
    ) -> RunResult:
        """Run and return a RunResult with the stop reason, usage and cost."""
        return _run_coroutine_sync(
            self.arun_with_response, input_message, session=session
        )

    async def arun_with_response(
        self, input_message: str, *, session: Optional[Session] = None
    ) -> RunResult:
        """Run async and return a RunResult with the stop reason, usage and cost.

        An exception during the run ends it with stop_reason "error" instead of
        raising.
        """
        return await self.start(input_message, stream=False, session=session)

    def stream(
        self, input_message: str, *, session: Optional[Session] = None
    ) -> Iterator[str]:
        """Stream output tokens. Naive tool mode yields the final answer as one chunk."""
        return _iterate_sync(self.astream(input_message, session=session))

    async def astream(
        self, input_message: str, *, session: Optional[Session] = None
    ) -> AsyncGenerator[str, None]:
        """Async stream output tokens. Naive tool mode yields the final answer as one chunk."""
        async with aclosing(
            self.astream_events(input_message, session=session)
        ) as events:
            async for event in events:
                if event.agent_path:
                    continue
                if isinstance(event, TextDelta):
                    yield event.text
                elif isinstance(event, RunFinished) and event.result.error is not None:
                    raise event.result.error

    def stream_events(
        self, input_message: str, *, session: Optional[Session] = None
    ) -> Iterator[Event]:
        """Stream the run's events; the last one is RunFinished."""
        return _iterate_sync(self.astream_events(input_message, session=session))

    async def astream_events(
        self, input_message: str, *, session: Optional[Session] = None
    ) -> AsyncGenerator[Event, None]:
        """Async stream the run's events; the last one is RunFinished.

        A failed run ends with RunFinished and stop_reason "error" instead of
        raising. The run waits for the caller to take each event; closing the
        iterator early cancels it.
        """
        run, task = self._start(
            input_message,
            stream=True,
            response_model=None,
            session=session,
            paced=True,
        )
        handle = RunHandle(run, task)
        try:
            async with aclosing(handle.__aiter__()) as events:
                async for event in events:
                    yield event
        finally:
            handle.cancel()
            await asyncio.wait([task])

    def as_tool(
        self,
        name: str,
        description: str,
        *,
        annotations: Optional[ToolAnnotations] = None,
        timeout: Optional[float] = None,
    ) -> AgentTool:
        """This agent as a tool that runs it on an input and returns its answer.

        See AgentTool for how its runs nest in the calling agent's run. Calls
        run one at a time unless annotations mark the tool read_only.
        """
        return AgentTool(self, name, description, annotations, timeout)

    def _start(
        self,
        input_message: str,
        *,
        stream: bool,
        response_model: Optional[Type],
        session: Optional[Session],
        paced: bool = False,
    ) -> Tuple[_Run, "asyncio.Task[RunResult]"]:
        """Create a run and start its task on the running event loop."""
        history = [] if session is None else session.messages
        run = _Run(self.config, history, session, paced)
        task = asyncio.get_running_loop().create_task(
            _RunLoop(self, run, stream, response_model).execute(input_message)
        )
        return run, task

    def _get_tool_mode(self) -> str:
        if not self.tools:
            return "none"
        if self.config.tool_mode in ("native", "naive"):
            return self.config.tool_mode
        return self._detected_tool_mode or "native"
