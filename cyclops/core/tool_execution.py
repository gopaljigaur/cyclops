"""Run a step's tool calls: parsing, middleware, parallel execution and cancellation.

Also holds the naive tool call format: the prompt that asks for it and its parser.
"""

import asyncio
import inspect
import json
from typing import Any, Dict, List, Optional, Tuple

from cyclops.core.events import ToolCallFinished, ToolCallStarted
from cyclops.core.hooks import AgentHooks, _tool_call_id
from cyclops.core.middleware import Deny, ToolCallContext, ToolMiddleware
from cyclops.core.providers import _tool_message_images_supported
from cyclops.core.run_state import _Run
from cyclops.core.types import AgentConfig, ToolCall
from cyclops.toolkit.types import ToolResult

_CANCELLED_RESULT = "Tool call cancelled."


def _split_images(content: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Split rendered tool message content into its text and its image parts."""
    if isinstance(content, str):
        return content, []
    text = "\n".join(p["text"] for p in content if p["type"] == "text")
    return text, [p for p in content if p["type"] == "image_url"]


def _failed_text(error: BaseException) -> str:
    """What the model sees for a call that was cancelled or whose middleware raised."""
    if isinstance(error, asyncio.CancelledError):
        return _CANCELLED_RESULT
    return _split_images(ToolResult.error_result(str(error)).to_message_content())[0]


def _unfinished_text(task: asyncio.Future) -> str:
    """The tool message text for a call of a step that did not finish."""
    if task.cancelled():
        return _CANCELLED_RESULT
    error = task.exception()
    if error is not None:
        return _failed_text(error)
    return _split_images(task.result()[1].to_message_content())[0]


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

    async def before_tool(self, call):
        if self.hooks.on_tool_start(call.tool_name, call.arguments) == "deny":
            return Deny(f"Tool '{call.tool_name}' was not approved")
        return None

    async def after_tool(self, call, result, error):
        if error is not None:
            self.hooks.on_tool_error(call.tool_name, call.arguments, error)
            return None
        try:
            text = _split_images(result.to_message_content())[0]
            self.hooks.on_tool_end(call.tool_name, call.arguments, text)
        except Exception as e:
            self.hooks.on_tool_error(call.tool_name, call.arguments, e)
            return ToolResult.error_result(f"{call.tool_name} failed: {e}")
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

    async def run_naive_tool_call(self, call: Dict[str, Any]) -> None:
        name = call["tool"]
        _, result = await self._call_tool(None, name, call.get("args", {}))
        text, images = _split_images(result.to_message_content())
        history = self.run.history
        history.append({"role": "assistant", "content": f"[Used tool: {name}]"})
        message: Any = f"Tool result: {text}"
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
            args, result = task.result()
            content: Any = result.to_message_content()
            text, images = _split_images(content)
            run.result.tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args, result=text)
            )
            if images and not _tool_message_images_supported(self.config):
                label = f"Images returned by {tc.function.name}:"
                moved_images += [{"type": "text", "text": label}, *images]
                content = text
            _append_tool_result(run.history, tc, content)
        # Must follow all tool messages of the round
        if moved_images:
            run.history.append({"role": "user", "content": moved_images})

    async def _execute_tool_call(self, tool_call) -> Tuple[Dict[str, Any], ToolResult]:
        """Parse a native tool call's arguments and run it. Returns (args, result)."""
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
        return await self._call_tool(tool_call.id, name, args, error)

    async def _call_tool(
        self,
        call_id: Optional[str],
        name: str,
        args: Dict[str, Any],
        error: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], ToolResult]:
        """Run a tool, or report error instead, between its two events.

        Returns the arguments it ran with and its result. current_tool_call_id()
        is call_id while the tool's middleware run.
        """
        run = self.run
        run.emit(ToolCallStarted(call_id=call_id, name=name, arguments=args))
        # Reset after, so a run nested in an outer tool leaves the outer id intact.
        token = _tool_call_id.set(call_id)
        try:
            if error:
                result = ToolResult.error_result(error)
            else:
                args, result = await self._run_tool(call_id, name, args)
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
                result=_split_images(result.to_message_content())[0],
                is_error=result.is_error,
            )
        )
        return args, result

    async def _run_tool(
        self, call_id: Optional[str], name: str, args: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], ToolResult]:
        """Run a tool through the middleware. Sync tools run in a worker thread."""
        tool = self.tools_by_name.get(name)
        if not tool:
            return args, ToolResult.error_result(f"Tool '{name}' not found")
        call = ToolCallContext(
            tool_name=name,
            arguments=args,
            call_id=call_id,
            run_id=self.run.result.run_id,
            annotations=tool.annotations,
            agent_path=self.run.agent_path,
        )
        middleware = list(self.config.middleware)
        if self.config.hooks:
            middleware.append(_HooksMiddleware(self.config.hooks))
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
                    if inspect.iscoroutinefunction(tool.execute):
                        value = await tool.execute(**call.arguments)
                    else:
                        value = await asyncio.to_thread(tool.execute, **call.arguments)
                    result = (
                        value
                        if isinstance(value, ToolResult)
                        else ToolResult.success_result(value)
                    )
                except Exception as e:
                    error = e
                    result = ToolResult.error_result(f"{name} failed: {e}")
        except asyncio.CancelledError as e:
            cancelled = ToolResult.error_result(_CANCELLED_RESULT)
            await _after_tool(allowed, call, cancelled, e)
            raise
        return call.arguments, await _after_tool(allowed, call, result, error)
