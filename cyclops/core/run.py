"""The step loop of a run: completions, tool calls, guards and limits."""

import asyncio
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
    Type,
)

import litellm

from cyclops.core.completion import (
    _acompletion,
    _build_messages,
    _completion_kwargs,
    _streamed_completion,
    _usage,
)
from cyclops.core.events import (
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
from cyclops.core.tool_execution import (
    _build_tools_prompt,
    _parse_naive_tool_call,
    _ToolExecutor,
)
from cyclops.core.types import AgentConfig, RunResult, StopReason
from cyclops.skills import Skills

if TYPE_CHECKING:
    from cyclops.core.agent import Agent

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."

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

    @property
    def config(self) -> AgentConfig:
        return self.agent.config

    async def execute(self, input_message: str) -> RunResult:
        """Run one turn and emit its events, ending with RunFinished."""
        run = self.run
        run.started = True
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
        session = run.session
        if session is not None:
            # Unknown once any run's usage is unknown, like a run's own total.
            if session.usage is None or run.result.usage is None:
                session.usage = None
            else:
                session.usage += run.result.usage
        run.emit(RunFinished(result=run.result))
        if interrupted is not None:
            raise interrupted
        return run.result

    async def _loop(self, input_message: str) -> None:
        run = self.run
        if self.config.hooks:
            self.config.hooks.on_run_start(input_message)
        try:
            start = len(run.history)
            mode = self.agent._get_tool_mode()
            try:
                await self._steps(input_message, mode)
            except litellm.UnsupportedParamsError:
                if mode != "native" or self.config.tool_mode != "auto":
                    raise
                if not _tools_unsupported(self.config):
                    raise
                self.agent._detected_tool_mode = "naive"
                del run.history[start:]
                await self._steps(input_message, "naive")
        except BaseException as e:
            # Failures, cancellation and a stream closed early all end the run here.
            if self.config.hooks:
                self.config.hooks.on_run_error(e)
            raise
        # Steering the run had no step left for waits for the session's next run;
        # without a session it is dropped.
        run.history += [{"role": "user", "content": m} for m in run.steering]
        run.steering.clear()
        if self.config.hooks:
            self.config.hooks.on_run_end(run.result.content)

    async def _steps(self, input_message: str, mode: str) -> None:
        run, agent = self.run, self.agent
        run.history.append({"role": "user", "content": input_message})
        result = run.result
        naive = mode == "naive"
        # Naive mode never streams deltas: the text may be a tool call.
        streamed = self.stream and not naive
        kwargs = _completion_kwargs(self.config, agent.tools, mode, self.response_model)
        system_prompt = _system_prompt(self.config, agent.skills, agent.tools, mode)
        executor = _ToolExecutor(self.config, agent._tools_by_name, run)

        guard_rounds = 0
        # Guard feedback waiting to be sent once another step is allowed.
        feedback: Optional[str] = None
        limit = self.config.max_steps
        for step in range(limit):
            await run.taken()
            if run.over_budget():
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
            run.spend(step_usage)
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
                feedback = await self._guard_feedback(content)
                if feedback is None:
                    result.content = content
                    done = True
                elif guard_rounds == self.config.max_guard_rounds:
                    result.stop_reason = "guard"
                    done = True
                else:
                    guard_rounds += 1
                    run.emit(GuardRejected(feedback=feedback))
            run.emit(
                StepFinished(
                    step=result.steps,
                    usage=step_usage,
                    finish_reason=choice.finish_reason,
                )
            )
            if done:
                return

        result.stop_reason = "max_steps"

    async def _guard_feedback(self, content: str) -> Optional[str]:
        """Feedback from the first guard that has some, else None."""
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
