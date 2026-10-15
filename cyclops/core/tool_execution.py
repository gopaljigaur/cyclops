"""Run a step's tool calls: parsing, middleware, parallel execution and cancellation.

Also holds the naive tool call format: the prompt that asks for it and its parser.
"""

import asyncio
import contextlib
import inspect
import json
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

from cyclops.core.events import ToolCallFinished, ToolCallStarted
from cyclops.core.hooks import AgentHooks, _tool_call_id
from cyclops.core.middleware import Deny, ToolCallContext, ToolMiddleware
from cyclops.core.providers import _tool_message_images_supported
from cyclops.core.run_state import _Run
from cyclops.core.types import AgentConfig, ToolCall
from cyclops.toolkit.types import ToolResult

_CANCELLED_RESULT = "Tool call cancelled."


class _Rendered(NamedTuple):
    """A tool result rendered once for the model, the hooks and the events."""

    content: Union[str, List[Dict[str, Any]]]
    text: str
    is_error: bool


def _render(result: ToolResult) -> _Rendered:
    content = result.to_message_content()
    return _Rendered(content, ToolResult.message_text(content), result.is_error)


def _images(content: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The image parts of rendered content; only content with images is a list."""
    if isinstance(content, str):
        return []
    return [p for p in content if p["type"] == "image_url"]


def _failed_text(error: BaseException) -> str:
    """What the model sees for a call that was cancelled or whose middleware raised."""
    if isinstance(error, asyncio.CancelledError):
        return _CANCELLED_RESULT
    return _render(ToolResult.error_result(str(error))).text


def _unfinished_text(task: asyncio.Future) -> str:
    """The tool message text for a call of a step that did not finish."""
    if task.cancelled():
        return _CANCELLED_RESULT
    error = task.exception()
    if error is not None:
        return _failed_text(error)
    return task.result()[1].text


def _build_tools_prompt(tools: List[Any]) -> str:
    if not tools:
        return ""

    tools_desc = "\n\nYou have access to the following tools:\n\n"
    for tool in tools:
        tools_desc += f"- {tool.name}: {tool.description}\n"
        if tool.definition.parameters:
            tools_desc += "  Parameters:\n"
            for param in tool.definition.parameters.values():
                tools_desc += f"    - {param.name} ({param.type}): {param.description or 'No description'}\n"

    tools_desc += (
        '\nTo use a tool, respond with JSON: {"tool": "tool_name", "args": {...}}\n'
    )
    tools_desc += "After using a tool, you'll receive the result and can provide a final answer.\n"
    return tools_desc


def _parse_naive_tool_call(content: str):
    try:
        parsed = json.loads(content.strip())
        if isinstance(parsed, dict) and "tool" in parsed and "args" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    if start == -1:
        return None

    brace_count = 0
    for i in range(start, len(content)):
        if content[i] == "{":
            brace_count += 1
        elif content[i] == "}":
            brace_count -= 1
            if brace_count == 0:
                try:
                    parsed = json.loads(content[start : i + 1])
                    if "tool" in parsed and "args" in parsed:
                        return parsed
                except json.JSONDecodeError:
                    pass
                break
    return None


def _append_tool_call_message(
    history: List[Dict[str, Any]], msg, message: Dict[str, Any]
) -> None:
    message["tool_calls"] = [
        {
            "id": tc.id,
            "type": "function",
            "function": {
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            },
        }
        for tc in msg.tool_calls
    ]
    history.append(message)


def _append_tool_result(history: List[Dict[str, Any]], tc, content: Any) -> None:
    history.append(
        {
            "role": "tool",
            "tool_call_id": tc.id,
            "content": content,
            "name": tc.function.name,
        }
    )


class _HooksMiddleware(ToolMiddleware):
    """Runs the AgentHooks tool callbacks as the innermost middleware."""

    def __init__(self, hooks: AgentHooks) -> None:
        self.hooks = hooks
        # The result on_tool_end saw; each call gets its own instance.
        self.rendered: Optional[_Rendered] = None

    async def before_tool(self, call):
        if self.hooks.on_tool_start(call.tool_name, call.arguments) == "deny":
            return Deny(f"Tool '{call.tool_name}' was not approved")
        return None

    async def after_tool(self, call, result, error):
        if error is not None:
            self.hooks.on_tool_error(call.tool_name, call.arguments, error)
            return None
        try:
            rendered = _render(result)
            self.hooks.on_tool_end(call.tool_name, call.arguments, rendered.text)
        except Exception as e:
            self.hooks.on_tool_error(call.tool_name, call.arguments, e)
            return ToolResult.error_result(f"{call.tool_name} failed: {e}")
        self.rendered = rendered
        return None


async def _after_tool(
    middleware: List[ToolMiddleware],
    call: ToolCallContext,
    result: ToolResult,
    error: Optional[BaseException],
) -> ToolResult:
    """Run after_tool in reverse order; each may replace the result."""
    for m in reversed(middleware):
        result = await m.after_tool(call, result, error) or result
    return result


class _ToolExecutor:
    """Runs a run's tool calls and records them in its transcript and result."""

    def __init__(
        self, config: AgentConfig, tools_by_name: Dict[str, Any], run: _Run
    ) -> None:
        self.config = config
        self.tools_by_name = tools_by_name
        self.run = run
        self._slots = asyncio.Semaphore(config.max_parallel_tool_calls)
        self._in_order = asyncio.Lock()

    async def run_naive_tool_call(self, call: Dict[str, Any]) -> None:
        name = call["tool"]
        _, rendered = await self._call_tool(None, name, call.get("args", {}))
        history = self.run.history
        history.append({"role": "assistant", "content": f"[Used tool: {name}]"})
        message: Any = f"Tool result: {rendered.text}"
        images = _images(rendered.content)
        if images:
            message = [{"type": "text", "text": message}, *images]
        history.append({"role": "user", "content": message})

    async def run_native_tool_calls(self, msg, assistant: Dict[str, Any]) -> None:
        run = self.run
        _append_tool_call_message(run.history, msg, assistant)
        tasks = [
            asyncio.ensure_future(self._execute_tool_call(tc)) for tc in msg.tool_calls
        ]
        try:
            # as_completed leaves the tools running when the run is cancelled,
            # so they are cancelled once, below, and can clean up. The first
            # call to fail ends the step.
            for finished in asyncio.as_completed(tasks):
                await finished
        except BaseException as e:
            # Cancelled, or middleware raised: stop the other calls too, with
            # the same reason, then answer every call to keep the transcript valid.
            reason = e.args if isinstance(e, asyncio.CancelledError) else ()
            for task in tasks:
                task.cancel(*reason)
            await asyncio.wait(tasks)
            for tc, task in zip(msg.tool_calls, tasks):
                _append_tool_result(run.history, tc, _unfinished_text(task))
            raise
        moved_images: List[Dict[str, Any]] = []
        for tc, task in zip(msg.tool_calls, tasks):
            args, rendered = task.result()
            run.result.tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                    result=rendered.text,
                )
            )
            content = rendered.content
            images = _images(content)
            if images and not _tool_message_images_supported(self.config):
                label = f"Images returned by {tc.function.name}:"
                moved_images += [{"type": "text", "text": label}, *images]
                content = rendered.text
            _append_tool_result(run.history, tc, content)
        # Must follow all tool messages of the round
        if moved_images:
            run.history.append({"role": "user", "content": moved_images})

    async def _execute_tool_call(self, tool_call) -> Tuple[Dict[str, Any], _Rendered]:
        """Parse a native tool call's arguments and run it once it may start.

        Returns (args, rendered result). At most max_parallel_tool_calls run at
        once, and tools not marked read_only run one at a time.
        """
        name = tool_call.function.name
        raw = tool_call.function.arguments
        error = None
        try:
            # Some providers send "" for a call without arguments.
            args = json.loads(raw) if raw else {}
        except json.JSONDecodeError as e:
            args, error = {}, f"arguments for {name} are not valid JSON: {e}"
        if not isinstance(args, dict):
            args, error = {}, f"arguments for {name} must be a JSON object"
        tool = self.tools_by_name.get(name)
        # Calls start in the model's order and nothing above awaits, so
        # they queue for the lock, and run, in that order.
        turn: Any = self._in_order
        if tool and tool.annotations.read_only:
            turn = contextlib.nullcontext()
        async with turn, self._slots:
            return await self._call_tool(tool_call.id, name, args, error)

    async def _call_tool(
        self,
        call_id: Optional[str],
        name: str,
        args: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], _Rendered]:
        """Run a tool, or report error instead, between its two events.

        Returns the arguments it ran with and its rendered result.
        current_tool_call_id() is call_id while the tool's middleware run.
        """
        run = self.run
        run.emit(ToolCallStarted(call_id=call_id, name=name, arguments=args))
        # Reset after, so a run nested in an outer tool leaves the outer id intact.
        token = _tool_call_id.set(call_id)
        try:
            if error:
                rendered = _render(ToolResult.error_result(error))
            else:
                args, rendered = await self._run_tool(call_id, name, args)
        except BaseException as e:
            # Middleware raised or the call was cancelled: it still finishes.
            run.emit(
                ToolCallFinished(
                    call_id=call_id, name=name, result=_failed_text(e), is_error=True
                )
            )
            raise
        finally:
            _tool_call_id.reset(token)
        run.emit(
            ToolCallFinished(
                call_id=call_id,
                name=name,
                result=rendered.text,
                is_error=rendered.is_error,
            )
        )
        return args, rendered

    async def _run_tool(
        self, call_id: Optional[str], name: str, args: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], _Rendered]:
        """Run a tool through the middleware. Sync tools run in a worker thread."""
        tool = self.tools_by_name.get(name)
        if not tool:
            return args, _render(ToolResult.error_result(f"Tool '{name}' not found"))
        call = ToolCallContext(
            tool_name=name,
            arguments=args,
            call_id=call_id,
            run_id=self.run.result.run_id,
            annotations=tool.annotations,
            agent_path=self.run.agent_path,
        )
        middleware = list(self.config.middleware)
        hooks = None
        if self.config.hooks:
            hooks = _HooksMiddleware(self.config.hooks)
            middleware.append(hooks)
        timeout = self.config.tool_timeout if tool.timeout is None else tool.timeout
        loop = asyncio.get_running_loop()
        allowed: List[ToolMiddleware] = []
        error: Optional[Exception] = None
        try:
            for m in middleware:
                decision = await m.before_tool(call)
                if isinstance(decision, Deny):
                    result = ToolResult.error_result(decision.reason)
                    break
                if decision is not None:
                    call.arguments = decision
                allowed.append(m)
            else:
                try:
                    started = loop.time()
                    if inspect.iscoroutinefunction(tool.execute):
                        execution = tool.execute(**call.arguments)
                    else:
                        execution = asyncio.to_thread(tool.execute, **call.arguments)
                    # On timeout wait_for cancels the call; a sync tool's thread
                    # runs on, since a thread cannot be stopped.
                    value = await asyncio.wait_for(execution, timeout)
                    result = (
                        value
                        if isinstance(value, ToolResult)
                        else ToolResult.success_result(value)
                    )
                except Exception as e:
                    error = e
                    # A TimeoutError before the deadline is the tool's own.
                    timed_out = (
                        isinstance(e, asyncio.TimeoutError)
                        and timeout is not None
                        and loop.time() - started >= timeout
                    )
                    outcome = (
                        f"timed out after {timeout} seconds"
                        if timed_out
                        else f"failed: {e}"
                    )
                    result = ToolResult.error_result(f"{name} {outcome}")
        except asyncio.CancelledError as e:
            cancelled = ToolResult.error_result(_CANCELLED_RESULT)
            await _after_tool(allowed, call, cancelled, e)
            raise
        result = await _after_tool(allowed, call, result, error)
        # Other middleware may change the result after the hooks rendered it.
        if hooks and hooks.rendered and not self.config.middleware:
            return call.arguments, hooks.rendered
        return call.arguments, _render(result)
