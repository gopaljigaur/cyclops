"""Core agent implementation"""

import asyncio
from contextlib import aclosing
from typing import Any, AsyncGenerator, Dict, Generator, Iterator, List, Optional, Type

import litellm

from cyclops.core.completion import (
    _acompletion,
    _build_messages,
    _completion_kwargs,
    _streamed_completion,
    _system_prompt,
    _usage,
)
from cyclops.core.events import (
    Event,
    GuardRejected,
    ReasoningDelta,
    RunFinished,
    StepFinished,
    TextDelta,
)
from cyclops.core.hooks import RUN_STOPPED
from cyclops.core.middleware import GuardContext
from cyclops.core.providers import _tools_unsupported
from cyclops.core.run_state import _Run
from cyclops.core.session import Session
from cyclops.core.sync import _iterate_sync, _run_coroutine_sync
from cyclops.core.tool_execution import _parse_naive_tool_call, _ToolExecutor
from cyclops.core.types import AgentConfig, RunResult
from cyclops.skills import Skills


class RunHandle:
    """A run started with Agent.start().

    Await it for the RunResult, and iterate it (once) for the run's events.
    """

    def __init__(self, run: _Run, task: "asyncio.Task[RunResult]") -> None:
        self._run = run
        self._task = task
        self._iterated = False
        # Marks the end of a run that stopped without RunFinished.
        task.add_done_callback(run.events.put_nowait)

    def __await__(self) -> Generator[Any, None, RunResult]:
        return self._task.__await__()

    def __aiter__(self) -> AsyncGenerator[Event, None]:
        if self._iterated:
            raise RuntimeError("A run's events can be iterated only once")
        self._iterated = True
        return self._events()

    async def _events(self) -> AsyncGenerator[Event, None]:
        events = self._run.events
        while True:
            event = await events.get()
            if event is self._task:
                self._task.result()
            yield event
            events.task_done()
            if isinstance(event, RunFinished):
                return

    def steer(self, message: str) -> None:
        """Add a user message that the model sees in its next completion.

        A message that arrives during a completion goes into the one after,
        once the step's tool calls finish; if that completion was the final
        answer, the run takes another step to answer it when its limits allow,
        and otherwise leaves it in the history for the next run. Steering a
        finished run raises RuntimeError.
        """
        if self._task.done():
            raise RuntimeError("The run has finished")
        self._run.steering.append(message)

    def cancel(self) -> None:
        """Stop the run; awaiting it then gives stop_reason "cancelled".

        Tool calls in progress are cancelled and get a cancelled result. A
        second call does nothing.
        """
        if self._run.cancelled:
            return
        self._run.cancelled = True
        if self._run.started:
            self._task.cancel(RUN_STOPPED)


class Agent:
    """LLM agent implementation"""

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List] = None,
        skills: Optional[Skills] = None,
    ):
        self.config = config
        self._history: List[Dict[str, Any]] = []
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

    def reset(self) -> None:
        """Clear the history of runs without a session."""
        self._history = []

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """A copy of the history of runs without a session."""
        return list(self._history)

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
        adds its usage to the session, instead of using this agent's history.
        """
        history = self._history if session is None else session.messages
        run = _Run(self.config.model, history, session)
        task = asyncio.get_running_loop().create_task(
            self._execute(input_message, run, stream, response_model)
        )
        return RunHandle(run, task)

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
        handle = self.start(input_message, session=session)
        handle._run.paced = True
        try:
            async with aclosing(handle.__aiter__()) as events:
                async for event in events:
                    yield event
        finally:
            handle.cancel()
            await asyncio.wait([handle._task])

    # ------------------------------------------------------------------
    # The agent loop
    # ------------------------------------------------------------------

    async def _execute(
        self,
        input_message: str,
        run: _Run,
        stream: bool,
        response_model: Optional[Type] = None,
    ) -> RunResult:
        """Run one turn and emit its events, ending with RunFinished."""
        run.started = True
        try:
            if run.cancelled:
                # Cancelled before the task started: no hooks, no completion.
                raise asyncio.CancelledError(RUN_STOPPED)
            await self._loop(input_message, run, stream, response_model)
        except asyncio.CancelledError as e:
            if e.args != (RUN_STOPPED,):
                raise
            run.result.stop_reason = "cancelled"
        except Exception as e:
            run.result.stop_reason = "error"
            run.result.error = e
        session = run.session
        if session is not None:
            # Unknown once any run's usage is unknown, like a run's own total.
            if session.usage is None or run.result.usage is None:
                session.usage = None
            else:
                session.usage += run.result.usage
        run.emit(RunFinished(result=run.result))
        return run.result

    async def _loop(
        self,
        input_message: str,
        run: _Run,
        stream: bool,
        response_model: Optional[Type],
    ) -> None:
        if self.config.hooks:
            self.config.hooks.on_run_start(input_message)
        try:
            start = len(run.history)
            mode = self._get_tool_mode()
            try:
                await self._steps(input_message, run, stream, mode, response_model)
            except litellm.UnsupportedParamsError:
                if mode != "native" or self.config.tool_mode != "auto":
                    raise
                if not _tools_unsupported(self.config):
                    raise
                self._detected_tool_mode = "naive"
                del run.history[start:]
                await self._steps(input_message, run, stream, "naive", response_model)
        except BaseException as e:
            # Failures, cancellation and a stream closed early all end the run here.
            if self.config.hooks:
                self.config.hooks.on_run_error(e)
            raise
        # Steering the run had no step left for waits for the next run.
        run.history += [{"role": "user", "content": m} for m in run.steering]
        run.steering.clear()
        if self.config.hooks:
            self.config.hooks.on_run_end(run.result.content)

    async def _steps(
        self,
        input_message: str,
        run: _Run,
        stream: bool,
        mode: str,
        response_model: Optional[Type],
    ) -> None:
        run.history.append({"role": "user", "content": input_message})
        result = run.result
        naive = mode == "naive"
        # Naive mode never streams deltas: the text may be a tool call.
        streamed = stream and not naive
        kwargs = _completion_kwargs(self.config, self.tools, mode, response_model)
        system_prompt = _system_prompt(self.config, self.skills, self.tools, mode)
        executor = _ToolExecutor(self.config, self._tools_by_name, run)

        # Limits count what is known, even once the reported totals are not.
        spent_tokens, spent_cost = 0, 0.0
        guard_rounds = 0
        # Guard feedback waiting to be sent once another step is allowed.
        feedback: Optional[str] = None
        limit = self.config.max_steps
        for step in range(limit):
            await run.taken()
            if step and self._over_budget(spent_tokens, spent_cost):
                result.stop_reason = "budget"
                return
            if feedback is not None:
                run.history.append({"role": "user", "content": feedback})
                feedback = None
            # Cleared only once the completion returns, so a naive fallback
            # that drops this attempt sends them again.
            steered = len(run.steering)
            run.history += [
                {"role": "user", "content": m} for m in run.steering[:steered]
            ]
            messages = _build_messages(run.history, system_prompt)
            response: Any
            if streamed:
                response = await _streamed_completion(
                    self.config, messages, kwargs, run
                )
            else:
                response = await _acompletion(self.config, messages=messages, **kwargs)
            if self.config.hooks:
                self.config.hooks.on_llm_end(response)
            del run.steering[:steered]
            result.steps += 1
            step_usage = _usage(response)
            if step_usage is not None:
                spent_tokens += step_usage.total_tokens
                spent_cost += step_usage.cost or 0.0
            if result.usage is None or step_usage is None:
                result.usage = None
            else:
                result.usage += step_usage
            msg = response.choices[0].message
            content = msg.content or ""
            assistant = self._assistant_message(msg, msg.content)
            naive_call = _parse_naive_tool_call(content) if naive else None
            if not streamed:
                if "reasoning_content" in assistant:
                    run.emit(ReasoningDelta(text=assistant["reasoning_content"]))
                if content and not naive_call:
                    run.emit(TextDelta(text=content))

            done = False
            if not naive and msg.tool_calls:
                await executor.run_native_tool_calls(msg, assistant)
            elif naive_call:
                await executor.run_naive_tool_call(naive_call)
            elif (
                run.steering
                and step + 1 < limit
                and not self._over_budget(spent_tokens, spent_cost)
            ):
                # A steer came in while the model answered: answer it in the
                # next step. Guards judge the answer the run ends on.
                assistant["content"] = content
                run.history.append(assistant)
            else:
                assistant["content"] = content
                run.history.append(assistant)
                feedback = await self._guard_feedback(content, run)
                if feedback is None:
                    result.content = content
                    done = True
                elif guard_rounds == self.config.max_guard_rounds:
                    result.stop_reason = "guard"
                    done = True
                else:
                    guard_rounds += 1
                    run.emit(GuardRejected(feedback=feedback))
            run.emit(StepFinished(step=result.steps, usage=step_usage))
            if done:
                return

        result.stop_reason = "max_steps"

    def _over_budget(self, tokens: int, cost: float) -> bool:
        """True once tokens or cost reached max_total_tokens or max_cost."""
        max_tokens = self.config.max_total_tokens
        if max_tokens is not None and tokens >= max_tokens:
            return True
        max_cost = self.config.max_cost
        return max_cost is not None and cost >= max_cost

    async def _guard_feedback(self, content: str, run: _Run) -> Optional[str]:
        """Feedback from the first guard that has some, else None."""
        context = GuardContext(
            content=content, messages=list(run.history), run_id=run.result.run_id
        )
        for guard in self.config.guards:
            feedback = await guard(context)
            if feedback is not None:
                return feedback
        return None

    @staticmethod
    def _assistant_message(msg, content: Optional[str]) -> Dict[str, Any]:
        """An assistant turn, keeping any reasoning the provider needs sent back."""
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        for key in ("reasoning_content", "thinking_blocks"):
            value = getattr(msg, key, None)
            if value:
                message[key] = value
        return message

    # ------------------------------------------------------------------
    # Tool mode detection
    # ------------------------------------------------------------------

    def _get_tool_mode(self) -> str:
        if not self.tools:
            return "none"
        if self.config.tool_mode in ("native", "naive"):
            return self.config.tool_mode
        return self._detected_tool_mode or "native"
