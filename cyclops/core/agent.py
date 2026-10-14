"""Core agent implementation"""

import asyncio
import inspect
import json
import os
import queue
import threading
from contextlib import aclosing
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import litellm
from litellm.utils import supports_prompt_caching

from cyclops.core.events import (
    Event,
    ReasoningDelta,
    RunFinished,
    StepFinished,
    TextDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from cyclops.core.hooks import RUN_STOPPED, _tool_call_id
from cyclops.core.types import AgentConfig, RunResult, ToolCall, Usage
from cyclops.skills import Skills
from cyclops.toolkit.types import ToolResult

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
_CANCELLED_RESULT = "Tool call cancelled."
# LiteLLM turns image parts of a tool message into the provider's own image
# blocks for these providers (on vertex_ai only for Gemini models); others
# accept only text in a tool message.
_TOOL_MESSAGE_IMAGE_PROVIDERS = {"anthropic", "bedrock", "gemini"}
_DONE = object()
T = TypeVar("T")
# Providers that cache a prompt prefix at cache_control markers (on vertex_ai
# only for Anthropic models). Gemini turns them into billed cachedContents
# resources instead, and OpenAI caches without them.
_CACHE_MARKER_PROVIDERS = {"anthropic", "bedrock"}
# Cache the system prompt (with the tools before it) and the conversation so far.
_CACHE_POINTS = [
    {"location": "message", "role": "system"},
    {"location": "message", "index": -1},
]


def _split_images(content: Any) -> Tuple[str, List[Dict[str, Any]]]:
    """Split rendered tool message content into its text and its image parts."""
    if isinstance(content, str):
        return content, []
    text = "\n".join(p["text"] for p in content if p["type"] == "text")
    return text, [p for p in content if p["type"] == "image_url"]


def _takes_tool_message_images(model: str, provider: str) -> bool:
    if provider == "vertex_ai":
        return model.startswith("gemini")
    return provider in _TOOL_MESSAGE_IMAGE_PROVIDERS


def _caches_at_markers(model: str, provider: str) -> bool:
    if provider == "vertex_ai":
        marker_provider = model.startswith("claude")
    else:
        marker_provider = provider in _CACHE_MARKER_PROVIDERS
    # LiteLLM adds Bedrock cache points for every model.
    return marker_provider and supports_prompt_caching(
        model=model, custom_llm_provider=provider
    )


_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_lock = threading.Lock()


def _reset_after_fork() -> None:
    global _loop, _loop_lock
    _loop = None
    _loop_lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    # A forked child inherits the loop but not the thread running it.
    os.register_at_fork(after_in_child=_reset_after_fork)


def _start_task(make: Callable[[], Awaitable], finished: threading.Event):
    """Start make() as a task on the shared loop; finished is set once it unwinds.

    One long-lived loop in a daemon thread serves every sync call, so LiteLLM's
    background logging can finish and a loop the caller runs is never blocked.
    """
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, daemon=True).start()
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    if running is _loop:
        # Blocking here would wait on the loop this call is running on.
        raise RuntimeError("Use the async Agent methods from async code")

    def on_done(_: asyncio.Future) -> None:
        finished.set()

    async def start() -> asyncio.Future:
        task = asyncio.ensure_future(make())
        task.add_done_callback(on_done)
        return task

    return asyncio.run_coroutine_threadsafe(start(), _loop).result()


def _wait(task: asyncio.Future, finished: threading.Event) -> None:
    """Wait for a task; if interrupted (e.g. Ctrl-C), cancel it and let it unwind."""
    try:
        finished.wait()
    finally:
        if not finished.is_set():
            task.get_loop().call_soon_threadsafe(task.cancel)
            finished.wait()


def _run_coroutine_sync(fn: Callable[..., Awaitable], *args: Any) -> Any:
    """Await fn(*args) from sync code on the shared loop."""
    finished = threading.Event()
    task = _start_task(lambda: fn(*args), finished)
    _wait(task, finished)
    return task.result()


def _iterate_sync(agen: AsyncGenerator[T, None]) -> Iterator[T]:
    """Drive an async generator from sync code in one task on the shared loop.

    One task keeps the generator's context across chunks. The task fetches a
    chunk only when the caller asks for one, like `async for` would.
    """
    chunks: queue.Queue = queue.Queue()
    demand = asyncio.Semaphore(0)
    finished = threading.Event()

    async def pump() -> None:
        try:
            async with aclosing(agen):
                while True:
                    await demand.acquire()
                    try:
                        chunks.put(await agen.__anext__())
                    except StopAsyncIteration:
                        return
        finally:
            chunks.put(_DONE)

    task = _start_task(pump, finished)
    loop = task.get_loop()
    try:
        while True:
            loop.call_soon_threadsafe(demand.release)
            chunk = chunks.get()
            if chunk is _DONE:
                break
            yield chunk
    finally:
        # Closing early cancels the task; this is a no-op once it is done.
        loop.call_soon_threadsafe(task.cancel)
        _wait(task, finished)
    task.result()


def _usage(response) -> Optional[Usage]:
    """Tokens and cost of one completion; None if it reports no usage."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    try:
        cost: Optional[float] = litellm.completion_cost(completion_response=response)
    except Exception:
        # LiteLLM has no pricing for this model.
        cost = None
    return Usage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
        cached_tokens=getattr(details, "cached_tokens", None) or 0,
        total_tokens=usage.total_tokens or 0,
        cost=cost,
    )


class _Run:
    """One run in progress: its result so far and, when streamed, its events."""

    def __init__(self, model: str, events: Optional[asyncio.Queue] = None) -> None:
        self.result = RunResult(model=model)
        self.events = events

    def emit(self, event: Event) -> None:
        if self.events is not None:
            self.events.put_nowait(event)

    async def taken(self) -> None:
        """Wait until the caller has taken every event emitted so far."""
        if self.events is not None:
            await self.events.join()


class Agent:
    """LLM agent implementation"""

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List] = None,
        memory=None,
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
        self.memory = memory
        # Set to "naive" once this agent's model rejects native tool calling.
        self._detected_tool_mode: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear conversation history."""
        self._history = []

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Return a copy of the current conversation history."""
        return list(self._history)

    def run(self, input_message: str, response_model: Optional[Type] = None) -> Any:
        """Run the agent synchronously. Returns str or Pydantic model instance."""
        return _run_coroutine_sync(self.arun, input_message, response_model)

    async def arun(
        self, input_message: str, response_model: Optional[Type] = None
    ) -> Any:
        """Run the agent asynchronously. Returns str or Pydantic model instance."""
        result = await self._execute(
            input_message, _Run(self.config.model), False, response_model
        )
        if result.error is not None:
            raise result.error
        if response_model is not None:
            return response_model.model_validate_json(result.content)
        return result.content

    def run_with_response(self, input_message: str) -> RunResult:
        """Run and return a RunResult with the stop reason, usage and cost."""
        return _run_coroutine_sync(self.arun_with_response, input_message)

    async def arun_with_response(self, input_message: str) -> RunResult:
        """Run async and return a RunResult with the stop reason, usage and cost.

        An exception during the run ends it with stop_reason "error" instead of
        raising.
        """
        return await self._execute(input_message, _Run(self.config.model), False)

    def stream(self, input_message: str) -> Iterator[str]:
        """Stream output tokens. Naive tool mode yields the final answer as one chunk."""
        return _iterate_sync(self.astream(input_message))

    async def astream(self, input_message: str) -> AsyncGenerator[str, None]:
        """Async stream output tokens. Naive tool mode yields the final answer as one chunk."""
        async with aclosing(self.astream_events(input_message)) as events:
            async for event in events:
                if isinstance(event, TextDelta):
                    yield event.text
                elif isinstance(event, RunFinished) and event.result.error is not None:
                    raise event.result.error

    def stream_events(self, input_message: str) -> Iterator[Event]:
        """Stream the run's events; the last one is RunFinished."""
        return _iterate_sync(self.astream_events(input_message))

    async def astream_events(self, input_message: str) -> AsyncGenerator[Event, None]:
        """Async stream the run's events; the last one is RunFinished.

        A failed run ends with RunFinished and stop_reason "error" instead of
        raising. The run waits for the caller to take each event; closing the
        iterator early cancels it.
        """
        events: asyncio.Queue = asyncio.Queue()
        run = _Run(self.config.model, events)
        task = asyncio.ensure_future(self._execute(input_message, run, True))
        # Marks the end of a run that stopped without RunFinished.
        task.add_done_callback(events.put_nowait)
        try:
            while True:
                event = await events.get()
                if event is task:
                    task.result()
                yield event
                events.task_done()
                if isinstance(event, RunFinished):
                    return
        finally:
            # A no-op once the run is done.
            task.cancel(RUN_STOPPED)
            await asyncio.wait([task])

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
        try:
            await self._loop(input_message, run, stream, response_model)
        except Exception as e:
            run.result.stop_reason = "error"
            run.result.error = e
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
            start = len(self._history)
            mode = self._get_tool_mode()
            try:
                await self._steps(input_message, run, stream, mode, response_model)
            except litellm.UnsupportedParamsError:
                if mode != "native" or self.config.tool_mode != "auto":
                    raise
                if not self._tools_unsupported():
                    raise
                self._detected_tool_mode = "naive"
                del self._history[start:]
                await self._steps(input_message, run, stream, "naive", response_model)
        except BaseException as e:
            # Failures, cancellation and a stream closed early all end the run here.
            if self.config.hooks:
                self.config.hooks.on_run_error(e)
            raise
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
        self._history.append({"role": "user", "content": input_message})
        result = run.result
        naive = mode == "naive"
        # Naive mode never streams deltas: the text may be a tool call.
        streamed = stream and not naive
        kwargs: Dict[str, Any] = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            **self.config.completion_kwargs,
        }
        if self.config.prompt_caching and self._caches_at_markers():
            kwargs["cache_control_injection_points"] = _CACHE_POINTS
        # Only without tools: some providers emulate the schema with a forced
        # tool call, which would stop the agent's own tools from being called.
        if (
            response_model is not None
            and mode == "none"
            and self._supports_response_schema()
        ):
            kwargs["response_format"] = response_model
        system_prompt = self.config.system_prompt
        if self.skills:
            system_prompt = (
                system_prompt or _DEFAULT_SYSTEM_PROMPT
            ) + self.skills.prompt()
        if mode == "native":
            kwargs["tools"] = self._get_tools_schema()
        if naive:
            system_prompt = (
                system_prompt or _DEFAULT_SYSTEM_PROMPT
            ) + self._build_tools_prompt()

        # Limits count what is known, even once the reported totals are not.
        spent_tokens, spent_cost = 0, 0.0
        for step in range(self.config.max_steps if mode != "none" else 1):
            await run.taken()
            if step and self._over_budget(spent_tokens, spent_cost):
                result.stop_reason = "budget"
                return
            messages = self._build_messages(system_prompt)
            response: Any
            if streamed:
                chunks = []
                async with aclosing(
                    self._astream_completion(messages, kwargs)
                ) as completion:
                    async for chunk in completion:
                        chunks.append(chunk)
                        delta = chunk.choices[0].delta if chunk.choices else None
                        # LiteLLM sets reasoning_content on a delta only when present.
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            run.emit(ReasoningDelta(text=reasoning))
                        if delta and delta.content:
                            run.emit(TextDelta(text=delta.content))
                        await run.taken()
                response = litellm.stream_chunk_builder(chunks, messages=messages)
            else:
                response = await self._acompletion(messages=messages, **kwargs)
            if self.config.hooks:
                self.config.hooks.on_llm_end(response)
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
            naive_call = self._parse_naive_tool_call(content) if naive else None
            if not streamed:
                if "reasoning_content" in assistant:
                    run.emit(ReasoningDelta(text=assistant["reasoning_content"]))
                if content and not naive_call:
                    run.emit(TextDelta(text=content))

            done = False
            if not naive and msg.tool_calls:
                await self._run_native_tool_calls(msg, assistant, run)
            elif naive_call:
                await self._run_naive_tool_call(naive_call, run)
            else:
                assistant["content"] = content
                self._history.append(assistant)
                result.content = content
                done = True
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

    async def _run_naive_tool_call(self, call: Dict[str, Any], run: _Run) -> None:
        name = call["tool"]
        result = await self._call_tool(run, None, name, call.get("args", {}))
        text, images = _split_images(result.to_message_content())
        self._history.append({"role": "assistant", "content": f"[Used tool: {name}]"})
        message: Any = f"Tool result: {text}"
        if images:
            message = [{"type": "text", "text": message}, *images]
        self._history.append({"role": "user", "content": message})

    async def _run_native_tool_calls(
        self, msg, assistant: Dict[str, Any], run: _Run
    ) -> None:
        self._append_tool_call_message(msg, assistant)
        tasks = [
            asyncio.ensure_future(self._execute_tool_call(tc, run))
            for tc in msg.tool_calls
        ]
        try:
            # Shielded so the tools are cancelled once, here, and can clean up.
            results = await asyncio.shield(asyncio.gather(*tasks))
        except asyncio.CancelledError as e:
            # Stop the other calls too, with the same reason, then answer every
            # call to keep the transcript valid.
            for task in tasks:
                task.cancel(*e.args)
            await asyncio.wait(tasks)
            for tc, task in zip(msg.tool_calls, tasks):
                if task.cancelled():
                    text = _CANCELLED_RESULT
                    run.emit(
                        ToolCallFinished(
                            call_id=tc.id,
                            name=tc.function.name,
                            result=text,
                            is_error=True,
                        )
                    )
                else:
                    text = _split_images(task.result()[1].to_message_content())[0]
                self._append_tool_result(tc, text)
            raise
        moved_images: List[Dict[str, Any]] = []
        for tc, (args, result) in zip(msg.tool_calls, results):
            content: Any = result.to_message_content()
            text, images = _split_images(content)
            run.result.tool_calls.append(
                ToolCall(id=tc.id, name=tc.function.name, arguments=args, result=text)
            )
            if images and not self._tool_message_images_supported():
                label = f"Images returned by {tc.function.name}:"
                moved_images += [{"type": "text", "text": label}, *images]
                content = text
            self._append_tool_result(tc, content)
        # Must follow all tool messages of the round
        if moved_images:
            self._history.append({"role": "user", "content": moved_images})

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    async def _execute_tool_call(
        self, tool_call, run: _Run
    ) -> Tuple[Dict[str, Any], ToolResult]:
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
        return args, await self._call_tool(run, tool_call.id, name, args, error)

    async def _call_tool(
        self,
        run: _Run,
        call_id: Optional[str],
        name: str,
        args: Dict[str, Any],
        error: Optional[str] = None,
    ) -> ToolResult:
        """Run a tool, or report error instead, between its two events.

        current_tool_call_id() is call_id while the tool's hooks run.
        """
        run.emit(ToolCallStarted(call_id=call_id, name=name, arguments=args))
        if error:
            result = ToolResult.error_result(error)
        else:
            # Reset after, so a run nested in an outer tool leaves the outer id intact.
            token = _tool_call_id.set(call_id)
            try:
                result = await self._run_tool(name, args)
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
        return result

    async def _run_tool(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Run a tool by name, applying hook gates. Sync tools run in a worker thread."""
        tool = self._tools_by_name.get(name)
        if not tool:
            return ToolResult.error_result(f"Tool '{name}' not found")
        hooks = self.config.hooks
        if hooks is not None and hooks.on_tool_start(name, args) == "deny":
            return ToolResult.error_result(f"Tool '{name}' was not approved")
        try:
            if inspect.iscoroutinefunction(tool.execute):
                result = await tool.execute(**args)
            else:
                result = await asyncio.to_thread(tool.execute, **args)
            if not isinstance(result, ToolResult):
                result = ToolResult.success_result(result)
            if hooks:
                hooks.on_tool_end(
                    name, args, _split_images(result.to_message_content())[0]
                )
            return result
        except asyncio.CancelledError as e:
            if hooks:
                hooks.on_tool_error(name, args, e)
            raise
        except Exception as e:
            if hooks:
                hooks.on_tool_error(name, args, e)
            return ToolResult.error_result(f"{name} failed: {e}")

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _append_tool_call_message(self, msg, message: Dict[str, Any]) -> None:
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
        self._history.append(message)

    @staticmethod
    def _assistant_message(msg, content: Optional[str]) -> Dict[str, Any]:
        """An assistant turn, keeping any reasoning the provider needs sent back."""
        message: Dict[str, Any] = {"role": "assistant", "content": content}
        for key in ("reasoning_content", "thinking_blocks"):
            value = getattr(msg, key, None)
            if value:
                message[key] = value
        return message

    def _append_tool_result(self, tc, content: Any) -> None:
        self._history.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content,
                "name": tc.function.name,
            }
        )

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------

    def _build_messages(
        self, system_prompt_override: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Build message list for LiteLLM, optionally prepending a system prompt."""
        messages = list(self._history)
        sys = system_prompt_override or self.config.system_prompt
        if sys:
            messages = [{"role": "system", "content": sys}] + messages
        return messages

    # ------------------------------------------------------------------
    # Tool schema helpers
    # ------------------------------------------------------------------

    def _get_tools_schema(self) -> List[Dict[str, Any]]:
        return [self._tool_to_openai_format(t) for t in self.tools]

    def _tool_to_openai_format(self, tool) -> Dict[str, Any]:
        """Convert a Tool to OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.definition.to_json_schema(),
            },
        }

    # ------------------------------------------------------------------
    # Tool mode detection
    # ------------------------------------------------------------------

    def _deployments(self) -> List[Dict[str, Any]]:
        """LiteLLM params of every deployment the model resolves to."""
        if self.config.router:
            return [
                d["litellm_params"]
                for d in self.config.router.get_model_list(self.config.model) or []
            ]
        return [{"model": self.config.model}]

    def _supports_response_schema(self) -> bool:
        """True if LiteLLM lists `response_format` schemas for every deployment."""
        return all(
            litellm.supports_response_schema(
                model=p["model"], custom_llm_provider=p.get("custom_llm_provider")
            )
            for p in self._deployments()
        )

    def _tools_unsupported(self) -> bool:
        """True if LiteLLM lists no `tools` support for every deployment of the model."""
        params = self._deployments()
        for p in params:
            supported = litellm.get_supported_openai_params(
                model=p["model"], custom_llm_provider=p.get("custom_llm_provider")
            )
            if supported is None or "tools" in supported:
                return False
        return bool(params)

    def _tool_message_images_supported(self) -> bool:
        """True if every deployment of the model takes images in a tool message."""
        params = self._deployments()
        return bool(params) and all(
            _takes_tool_message_images(
                *litellm.get_llm_provider(
                    p["model"], custom_llm_provider=p.get("custom_llm_provider")
                )[:2]
            )
            for p in params
        )

    def _caches_at_markers(self) -> bool:
        """True if every deployment of the model caches at cache_control markers."""
        return all(
            _caches_at_markers(
                *litellm.get_llm_provider(
                    p["model"], custom_llm_provider=p.get("custom_llm_provider")
                )[:2]
            )
            for p in self._deployments()
        )

    def _get_tool_mode(self) -> str:
        if not self.tools:
            return "none"
        if self.config.tool_mode in ("native", "naive"):
            return self.config.tool_mode
        return self._detected_tool_mode or "native"

    # ------------------------------------------------------------------
    # Naive tool prompt helpers
    # ------------------------------------------------------------------

    def _build_tools_prompt(self) -> str:
        if not self.tools:
            return ""

        tools_desc = "\n\nYou have access to the following tools:\n\n"
        for tool in self.tools:
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

    def _parse_naive_tool_call(self, content: str):
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

    # ------------------------------------------------------------------
    # LiteLLM wrappers
    # ------------------------------------------------------------------

    async def _astream_completion(self, messages, kwargs) -> AsyncGenerator[Any, None]:
        response = await self._acompletion(
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            **kwargs,
        )
        try:
            async for chunk in response:
                yield chunk
        except Exception as e:
            if self.config.hooks:
                self.config.hooks.on_llm_error(e)
            raise
        finally:
            await response.aclose()

    async def _acompletion(self, **kwargs):
        if self.config.hooks:
            self.config.hooks.on_llm_start(kwargs.get("messages", []))
        try:
            if self.config.router:
                return await self.config.router.acompletion(
                    model=self.config.model, **kwargs
                )
            return await litellm.acompletion(model=self.config.model, **kwargs)
        except Exception as e:
            if self.config.hooks:
                self.config.hooks.on_llm_error(e)
            raise
