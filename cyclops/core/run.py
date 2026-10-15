"""The step loop of a run: completions, tool calls, guards and limits."""

import asyncio
import time
import traceback
import weakref
from contextvars import ContextVar
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)

import litellm
from pydantic import TypeAdapter, ValidationError

from cyclops.core.compaction import _context_window, _ContextWindow
from cyclops.core.completion import _build_messages
from cyclops.core.events import (
    Compacted,
    GuardRejected,
    OutputRejected,
    ReasoningDelta,
    RunFinished,
    StepFinished,
    TextDelta,
)
from cyclops.core.fallback import _complete_with_fallbacks, _Request, _requests
from cyclops.core.hooks import RUN_STOPPED, _call_hook
from cyclops.core.middleware import GuardContext
from cyclops.core.providers import _every_deployment, _lacks_tools
from cyclops.core.content import _Content, _user_content
from cyclops.core.run_state import _Run
from cyclops.core.session import _SessionSaver
from cyclops.core.tool_execution import (
    _build_tools_prompt,
    _parse_naive_tool_call,
    _ToolExecutor,
)
from cyclops.core.tool_search import _ToolSearch
from cyclops.core.types import AgentConfig, RunResult, StopReason, Usage, _added
from cyclops.skills import Skills

if TYPE_CHECKING:
    from cyclops.core.agent import Agent

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

_ERRORS: TypeAdapter[List[Dict[str, Any]]] = TypeAdapter(List[Dict[str, Any]])

# Finish reasons, as LiteLLM normalizes them, that end a run without an answer.
# LiteLLM maps Anthropic's refusal and Bedrock's guardrail to content_filter,
# and Anthropic's compaction pause to length.
_FINISH_STOP_REASONS: Dict[str, StopReason] = {
    "length": "max_tokens",
    "content_filter": "content_filter",
}


def _system_prompt(
    config: AgentConfig, skills: Optional[Skills], tools: List[Any], mode: str
) -> Optional[str]:
    """The configured system prompt, plus the skills and, in naive mode, the tools."""
    system_prompt = config.system_prompt
    if skills:
        system_prompt = (system_prompt or _DEFAULT_SYSTEM_PROMPT) + skills.prompt()
    if mode == "naive":
        tools_prompt = _build_tools_prompt(tools)
        system_prompt = (system_prompt or _DEFAULT_SYSTEM_PROMPT) + tools_prompt
    return system_prompt


def _assistant_message(msg, content: Optional[str]) -> Dict[str, Any]:
    """An assistant turn, keeping any reasoning the provider needs sent back."""
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    for key in ("reasoning_content", "thinking_blocks"):
        value = getattr(msg, key, None)
        if value:
            message[key] = value
    return message


def _clear_frames(tb: Optional[TracebackType]) -> None:
    """Clear a failed run's own frames, which lead its error's traceback.

    The frames past them, of the code that raised the error and of any earlier
    raise of it, are left: before Python 3.13 clearing a suspended frame closes
    its coroutine.
    """
    for frame, _ in traceback.walk_tb(tb):
        if not frame.f_globals["__name__"].startswith("cyclops."):
            break
        frame.clear()


class _RunLoop:
    """The step loop of one run of an agent."""

    def __init__(
        self,
        agent: "Agent",
        run: _Run,
        stream: bool,
        response_model: Optional[Type] = None,
    ) -> None:
        self.agent = agent
        self.run = run
        self.stream = stream
        self.response_model = response_model
        store = agent.session_store
        self.saver = None
        if store is not None and run.session is not None:
            self.saver = _SessionSaver(store, run.session, agent.durability)

    @property
    def config(self) -> AgentConfig:
        return self.agent.config

    async def execute(self, input_message: _Content) -> RunResult:
        """Run one turn and emit its events, ending with RunFinished."""
        run = self.run
        run.started = True
        start = time.perf_counter()
        # Tools called in this run see it, so a subagent can nest its run in it.
        token = _current_loop.set(weakref.ref(self))
        interrupted: Optional[asyncio.CancelledError] = None
        try:
            if run.cancelled:
                # Cancelled before the task started: no hooks, no completion.
                raise asyncio.CancelledError(RUN_STOPPED)
            await self._loop(input_message)
        except asyncio.CancelledError as e:
            if e.args != (RUN_STOPPED,):
                if run.parent is None:
                    raise
                # A nested run cancelled otherwise, as by its call timing out,
                # still finishes before the cancellation goes on.
                interrupted = e
            run.result.stop_reason = "cancelled"
        except Exception as e:
            run.result.stop_reason = "error"
            run.result.error = e
            # Its frames hold the run in a cycle; clear them once this returns.
            asyncio.get_running_loop().call_soon(_clear_frames, e.__traceback__)
        finally:
            _current_loop.reset(token)
            run.result.duration = time.perf_counter() - start
        run.emit(RunFinished(result=run.result))
        if interrupted is not None:
            raise interrupted
        return run.result

    async def _end_session(self, failed: bool = False) -> None:
        """Add the run's usage to its session, and save the session if autosaving."""
        session, usage = self.run.session, self.run.result.usage
        if session is None:
            return
        # Unknown once any run's usage is unknown, like a run's own total.
        session.usage = _added(session.usage, usage)
        if self.saver is not None:
            await self.saver.end(failed)

    async def _loop(self, input_message: _Content) -> None:
        run = self.run
        try:
            replaced = await _call_hook(self.config, "on_run_start", input_message)
            if replaced is not None:
                input_message = _user_content(self.config, replaced)
            # Compaction may replace earlier messages too, so the transcript is
            # restored as a whole.
            before = list(run.history)
            mode = self.agent._get_tool_mode()
            try:
                await self._steps(input_message, mode)
            except litellm.UnsupportedParamsError:
                if mode != "native" or self.config.tool_mode != "auto":
                    raise
                if not _every_deployment(self.config, _lacks_tools):
                    raise
                self.agent._detected_tool_mode = "naive"
                run.history[:] = before
                if self.saver is not None:
                    self.saver.rewritten()
                await self._steps(input_message, "naive")
        except BaseException as e:
            # Failures, cancellation and a stream closed early all end the run here.
            await _call_hook(self.config, "on_run_error", e)
            await self._end_session(failed=True)
            raise
        # Steering the run had no step left for waits for the session's next run;
        # without a session it is dropped.
        run.history += [{"role": "user", "content": m} for m in run.steering]
        run.steering.clear()
        try:
            await self._end_session()
        except BaseException as e:
            await _call_hook(self.config, "on_run_error", e)
            raise
        await _call_hook(self.config, "on_run_end", run.result.content)

    async def _steps(self, input_message: _Content, mode: str) -> None:
        run, agent = self.run, self.agent
        run.history.append({"role": "user", "content": input_message})
        result = run.result
        naive = mode == "naive"
        # Naive mode never streams deltas: the text may be a tool call.
        streamed = self.stream and not naive
        tools = _ToolSearch(agent.tools, agent._tools_by_name)
        requests: List[_Request] = []
        system_prompt: Optional[str] = None
        executor = _ToolExecutor(self.config, tools.tools_by_name, run)
        compaction = self.config.compaction
        # Without compaction the transcript is counted once a request overflows.
        window = _context_window(self.config, compaction, run.spend)

        guard_rounds = output_retries = 0
        # Feedback waiting to be sent once another step is allowed.
        feedback: Optional[str] = None
        limit = self.config.max_steps
        for step in range(limit):
            await run.taken()
            if run.over_budget():
                result.stop_reason = "budget"
                return
            if tools.changed:
                tools.changed = False
                requests = _requests(
                    self.config, tools.loaded, mode, self.response_model
                )
                system_prompt = _system_prompt(
                    self.config, agent.skills, tools.loaded, mode
                )
                if window is not None:
                    window.request(system_prompt, requests[0][1].get("tools"))
            if feedback is not None:
                run.history.append({"role": "user", "content": feedback})
                feedback = None
            # Cleared only once the completion returns, so a naive fallback
            # that drops this attempt sends them again.
            steered = len(run.steering)
            run.history += [
                {"role": "user", "content": m} for m in run.steering[:steered]
            ]
            if window is not None and compaction is not None:
                self._compacted(await window.compact_if_needed(run.history, compaction))
            answer = await self._complete(window, system_prompt, requests, streamed)
            if answer is None:
                del run.steering[:steered]
                result.stop_reason = "budget"
                return
            model, response, step_usage = answer
            await _call_hook(self.config, "on_llm_end", response)
            del run.steering[:steered]
            result.steps += 1
            run.spend(model, step_usage)
            choice = response.choices[0]
            msg = choice.message
            result.finish_reason = choice.finish_reason
            content = msg.content or ""
            assistant = _assistant_message(msg, msg.content)
            naive_call = _parse_naive_tool_call(content) if naive else None
            stop_reason = _FINISH_STOP_REASONS.get(choice.finish_reason)
            if not streamed:
                if "reasoning_content" in assistant:
                    run.emit(ReasoningDelta(text=assistant["reasoning_content"]))
                # In naive mode cut text may be a call that no longer parses.
                if content and not naive_call and not (naive and stop_reason):
                    run.emit(TextDelta(text=content))

            done = False
            if stop_reason is not None:
                # Cut or filtered output is no answer, and a cut tool call
                # can't run, so the turn is kept as text only. An empty turn
                # is left out: providers rewrite, drop or pass on empty
                # assistant text differently.
                if content:
                    assistant["content"] = content
                    run.history.append(assistant)
                result.stop_reason = stop_reason
                done = True
            elif not naive and msg.tool_calls:
                await executor.run_native_tool_calls(msg, assistant)
            elif naive_call:
                await executor.run_naive_tool_call(naive_call)
            elif run.steering and step + 1 < limit and not run.over_budget():
                # A steer came in while the model answered: answer it in the
                # next step. Guards judge the answer the run ends on.
                assistant["content"] = content
                run.history.append(assistant)
            else:
                assistant["content"] = content
                run.history.append(assistant)
                # Whether a step is left to send the answer back in.
                retry = step + 1 < limit and not run.over_budget()
                feedback = self._validation_feedback(content)
                if feedback is not None:
                    if retry and output_retries < self.config.output_retries:
                        output_retries += 1
                        run.emit(OutputRejected(feedback=feedback))
                    else:
                        # The answer stays the content, for the caller to see.
                        result.content = content
                        result.stop_reason = "output_validation"
                        done = True
                elif (feedback := await self._guard_feedback(content)) is None:
                    result.content = content
                    done = True
                elif retry and guard_rounds < self.config.max_guard_rounds:
                    guard_rounds += 1
                    run.emit(GuardRejected(feedback=feedback))
                else:
                    result.stop_reason = "guard"
                    done = True
            run.emit(
                StepFinished(
                    step=result.steps,
                    model=model,
                    usage=step_usage,
                    finish_reason=choice.finish_reason,
                )
            )
            if done:
                return
            if self.saver is not None:
                await self.saver.step()

        result.stop_reason = "max_steps"

    async def _complete(
        self,
        window: Optional[_ContextWindow],
        system_prompt: Optional[str],
        requests: List[_Request],
        streamed: bool,
    ) -> Optional[Tuple[str, Any, Optional[Usage]]]:
        """The model that answered a step, its completion and usage, kept within the
        context window of the configured model; None if a refusal spent the
        run's budget before the next model.

        Once the transcript is counted, max_tokens is lowered to the room the
        request leaves. On an overflow each remedy is tried at most once: a
        lower max_tokens, then compaction when it is set.
        """
        history = self.run.history
        compaction = self.config.compaction
        kwargs = requests[0][1]
        fit = window is not None and window.counting
        compacted = False
        while True:
            fitted = requests
            max_tokens = None
            if window is not None and fit:
                max_tokens = window.fitting_max_tokens(history, kwargs)
            if max_tokens is not None:
                fitted = [(c, {**k, "max_tokens": max_tokens}) for c, k in requests]
            messages = _build_messages(history, system_prompt)
            try:
                return await _complete_with_fallbacks(
                    fitted, messages, self.run, streamed
                )
            except litellm.ContextWindowExceededError:
                if window is None:
                    raise
                if not fit and window.fitting_max_tokens(history, kwargs) is not None:
                    fit = True
                    continue
                if compaction is None or compacted:
                    raise
                compacted = fit = True
                if not self._compacted(await window.compact(history, compaction)):
                    raise

    def _compacted(self, event: Optional[Compacted]) -> bool:
        """Report a compaction, if there was one, and save the session whole next."""
        if event is None:
            return False
        self.run.emit(event)
        if self.saver is not None:
            self.saver.rewritten()
        return True

    def _validation_feedback(self, content: str) -> Optional[str]:
        """The answer's response_model validation errors as a retry prompt, else None."""
        if self.response_model is None:
            return None
        try:
            self.response_model.model_validate_json(content)
        except ValidationError as e:
            # As Pydantic AI does: the model has its answer, so only nested
            # errors keep the input that failed.
            errors = [
                {
                    k: v
                    for k, v in error.items()
                    if k != "input" or len(error["loc"]) > 1
                }
                for error in e.errors(include_url=False, include_context=False)
            ]
            plural = "s" if len(errors) != 1 else ""
            return (
                f"{len(errors)} validation error{plural}:\n"
                f"```json\n{_ERRORS.dump_json(errors, indent=2).decode()}\n```\n\n"
                "Fix the errors and try again."
            )
        return None

    async def _guard_feedback(self, content: str) -> Optional[str]:
        """Feedback from the first guard that has some, else None."""
        if not self.config.guards:
            return None
        run = self.run
        context = GuardContext(
            content=content,
            messages=list(run.history),
            run_id=run.result.run_id,
            agent_path=run.agent_path,
        )
        for guard in self.config.guards:
            feedback = await guard(context)
            if feedback is not None:
                return feedback
        return None


# The loop of the run in progress in this context. Weak, since contexts copied
# during the run (by tasks, or LiteLLM) can outlive it.
_current_loop: ContextVar[Optional["weakref.ref[_RunLoop]"]] = ContextVar(
    "cyclops_current_loop", default=None
)
