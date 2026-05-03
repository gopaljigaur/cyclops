"""Core agent implementation"""

import asyncio
import concurrent.futures
import inspect
import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Type

import litellm

from cyclops.core.types import AgentConfig, AgentResponse, ToolCall


def _run_coroutine_sync(coro) -> Any:
    """Run a coroutine synchronously, even when an event loop is already running.

    Uses a ThreadPoolExecutor when a loop is active (e.g. Jupyter, nested async)
    so the caller's loop is never blocked or re-entered.
    """
    try:
        asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


class Agent:
    """LLM agent implementation"""

    # Class-level cache for detected tool modes per model
    _tool_mode_cache: Dict[str, str] = {}

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List] = None,
        memory=None,
    ):
        self.config = config
        self._history: List[Dict[str, Any]] = []
        self.tools = tools or []
        self.memory = memory

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
        if not self.tools:
            content = self._run_no_tools(input_message)
        else:
            tool_mode = self._get_tool_mode()
            if tool_mode == "naive":
                content = self._run_naive(input_message)
            else:
                try:
                    content = self._run_with_tools(input_message)
                except Exception as e:
                    error_str = str(e).lower()
                    if any(
                        kw in error_str for kw in ["tool", "function", "unsupported"]
                    ):
                        self._tool_mode_cache[self.config.model] = "naive"
                        content = self._run_naive(input_message)
                    else:
                        raise

        if response_model is not None:
            return response_model.model_validate_json(content)
        return content

    async def arun(
        self, input_message: str, response_model: Optional[Type] = None
    ) -> Any:
        """Run the agent asynchronously. Returns str or Pydantic model instance."""
        if not self.tools:
            content = await self._arun_no_tools(input_message)
        else:
            tool_mode = self._get_tool_mode()
            if tool_mode == "naive":
                content = await self._arun_naive(input_message)
            else:
                try:
                    content = await self._arun_with_tools(input_message)
                except Exception as e:
                    error_str = str(e).lower()
                    if any(
                        kw in error_str for kw in ["tool", "function", "unsupported"]
                    ):
                        self._tool_mode_cache[self.config.model] = "naive"
                        content = await self._arun_naive(input_message)
                    else:
                        raise

        if response_model is not None:
            return response_model.model_validate_json(content)
        return content

    def run_with_response(self, input_message: str) -> AgentResponse:
        """Run and return a full AgentResponse with cost/token metadata."""
        if not self.tools:
            content, raw_response = self._run_no_tools_tracked(input_message)
            tool_calls: List[ToolCall] = []
        else:
            tool_mode = self._get_tool_mode()
            if tool_mode == "naive":
                content = self._run_naive(input_message)
                raw_response = None
                tool_calls = []
            else:
                try:
                    content, raw_response, tool_calls = self._run_with_tools_tracked(
                        input_message
                    )
                except Exception as e:
                    error_str = str(e).lower()
                    if any(
                        kw in error_str for kw in ["tool", "function", "unsupported"]
                    ):
                        self._tool_mode_cache[self.config.model] = "naive"
                        content = self._run_naive(input_message)
                        raw_response = None
                        tool_calls = []
                    else:
                        raise

        return self._build_agent_response(content, raw_response, tool_calls)

    async def arun_with_response(self, input_message: str) -> AgentResponse:
        """Run async and return a full AgentResponse with cost/token metadata."""
        if not self.tools:
            content, raw_response = await self._arun_no_tools_tracked(input_message)
            tool_calls: List[ToolCall] = []
        else:
            tool_mode = self._get_tool_mode()
            if tool_mode == "naive":
                content = await self._arun_naive(input_message)
                raw_response = None
                tool_calls = []
            else:
                try:
                    (
                        content,
                        raw_response,
                        tool_calls,
                    ) = await self._arun_with_tools_tracked(input_message)
                except Exception as e:
                    error_str = str(e).lower()
                    if any(
                        kw in error_str for kw in ["tool", "function", "unsupported"]
                    ):
                        self._tool_mode_cache[self.config.model] = "naive"
                        content = await self._arun_naive(input_message)
                        raw_response = None
                        tool_calls = []
                    else:
                        raise

        return self._build_agent_response(content, raw_response, tool_calls)

    def stream(self, input_message: str) -> Iterator[str]:
        """Stream output tokens. True streaming for no-tools; tool calls processed first."""
        if not self.tools:
            yield from self._stream_no_tools(input_message)
        else:
            tool_mode = self._get_tool_mode()
            if tool_mode == "naive":
                yield self._run_naive(input_message)
            else:
                try:
                    self._run_with_tools_prepare(input_message)
                except Exception as e:
                    error_str = str(e).lower()
                    if any(
                        kw in error_str for kw in ["tool", "function", "unsupported"]
                    ):
                        self._tool_mode_cache[self.config.model] = "naive"
                        yield self._run_naive(input_message)
                        return
                    raise
                yield from self._stream_final_answer()

    async def astream(self, input_message: str) -> AsyncIterator[str]:
        """Async stream output tokens."""
        if not self.tools:
            async for chunk in self._astream_no_tools(input_message):
                yield chunk
        else:
            tool_mode = self._get_tool_mode()
            if tool_mode == "naive":
                yield await self._arun_naive(input_message)
            else:
                try:
                    await self._arun_with_tools_prepare(input_message)
                except Exception as e:
                    error_str = str(e).lower()
                    if any(
                        kw in error_str for kw in ["tool", "function", "unsupported"]
                    ):
                        self._tool_mode_cache[self.config.model] = "naive"
                        yield await self._arun_naive(input_message)
                        return
                    raise
                async for chunk in self._astream_final_answer():
                    yield chunk

    # ------------------------------------------------------------------
    # Sync internals — no tools
    # ------------------------------------------------------------------

    def _run_no_tools(self, input_message: str) -> str:
        self._history.append({"role": "user", "content": input_message})
        response = self._completion(
            messages=self._build_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": content})
        return content

    def _run_no_tools_tracked(self, input_message: str):
        self._history.append({"role": "user", "content": input_message})
        response = self._completion(
            messages=self._build_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": content})
        return content, response

    def _stream_no_tools(self, input_message: str) -> Iterator[str]:
        self._history.append({"role": "user", "content": input_message})
        response = self._completion(
            messages=self._build_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        collected = []
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                collected.append(delta)
                yield delta
        self._history.append({"role": "assistant", "content": "".join(collected)})

    # ------------------------------------------------------------------
    # Sync internals — with tools (native function calling)
    # ------------------------------------------------------------------

    def _run_with_tools(self, input_message: str) -> str:
        self._history.append({"role": "user", "content": input_message})
        tools_schema = self._get_tools_schema()

        for _ in range(self.config.max_iterations):
            response = self._completion(
                messages=self._build_messages(),
                tools=tools_schema,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content or ""
                self._history.append({"role": "assistant", "content": content})
                return content

            # Record assistant message with tool calls
            self._history.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            # Execute each tool
            for tc in msg.tool_calls:
                result = self._execute_tool_sync(tc)
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                        "name": tc.function.name,
                    }
                )

        return "Reached maximum tool call iterations."

    def _run_with_tools_tracked(self, input_message: str):
        """Like _run_with_tools but returns (content, last_response, tool_calls)."""
        self._history.append({"role": "user", "content": input_message})
        tools_schema = self._get_tools_schema()
        all_tool_calls: List[ToolCall] = []
        last_response = None

        for _ in range(self.config.max_iterations):
            response = self._completion(
                messages=self._build_messages(),
                tools=tools_schema,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            last_response = response
            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content or ""
                self._history.append({"role": "assistant", "content": content})
                return content, last_response, all_tool_calls

            self._history.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                result = self._execute_tool_sync(tc)
                args = (
                    json.loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments
                )
                all_tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                        result=result,
                    )
                )
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                        "name": tc.function.name,
                    }
                )

        return "Reached maximum tool call iterations.", last_response, all_tool_calls

    def _run_with_tools_prepare(self, input_message: str) -> None:
        """Run the tool loop without streaming; history is ready for final stream."""
        self._history.append({"role": "user", "content": input_message})
        tools_schema = self._get_tools_schema()

        for _ in range(self.config.max_iterations):
            response = self._completion(
                messages=self._build_messages(),
                tools=tools_schema,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content or ""
                self._history.append({"role": "assistant", "content": content})
                return

            self._history.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                result = self._execute_tool_sync(tc)
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                        "name": tc.function.name,
                    }
                )

        self._history.append(
            {"role": "assistant", "content": "Reached maximum tool call iterations."}
        )

    def _stream_final_answer(self) -> Iterator[str]:
        """Stream a fresh final answer, replacing the pre-computed one from the tool loop."""
        if self._history and self._history[-1].get("role") == "assistant":
            self._history.pop()

        response = self._completion(
            messages=self._build_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        collected = []
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                collected.append(delta)
                yield delta
        self._history.append({"role": "assistant", "content": "".join(collected)})

    # ------------------------------------------------------------------
    # Naive tool calling (prompt-based) — sync
    # ------------------------------------------------------------------

    def _run_naive(self, input_message: str) -> str:
        self._history.append({"role": "user", "content": input_message})
        system_prompt = (
            self.config.system_prompt or "You are a helpful assistant."
        ) + self._build_tools_prompt()

        for _ in range(self.config.max_iterations):
            response = self._completion(
                messages=self._build_messages(system_prompt_override=system_prompt),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = response.choices[0].message.content or ""
            tool_call = self._parse_naive_tool_call(content)

            if not tool_call:
                self._history.append({"role": "assistant", "content": content})
                return content

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})
            tool = next((t for t in self.tools if t.name == tool_name), None)
            result = self._execute_tool_by_dict(tool, tool_name, tool_args)

            self._history.append(
                {"role": "assistant", "content": f"[Used tool: {tool_name}]"}
            )
            self._history.append({"role": "user", "content": f"Tool result: {result}"})

        return "Reached maximum tool call iterations."

    # ------------------------------------------------------------------
    # Async internals — no tools
    # ------------------------------------------------------------------

    async def _arun_no_tools(self, input_message: str) -> str:
        self._history.append({"role": "user", "content": input_message})
        response = await self._acompletion(
            messages=self._build_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": content})
        return content

    async def _arun_no_tools_tracked(self, input_message: str):
        self._history.append({"role": "user", "content": input_message})
        response = await self._acompletion(
            messages=self._build_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        content = response.choices[0].message.content or ""
        self._history.append({"role": "assistant", "content": content})
        return content, response

    async def _astream_no_tools(self, input_message: str) -> AsyncIterator[str]:
        self._history.append({"role": "user", "content": input_message})
        response = await self._acompletion(
            messages=self._build_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        collected = []
        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                collected.append(delta)
                yield delta
        self._history.append({"role": "assistant", "content": "".join(collected)})

    # ------------------------------------------------------------------
    # Async internals — with tools (native function calling)
    # ------------------------------------------------------------------

    async def _arun_with_tools(self, input_message: str) -> str:
        self._history.append({"role": "user", "content": input_message})
        tools_schema = self._get_tools_schema()

        for _ in range(self.config.max_iterations):
            response = await self._acompletion(
                messages=self._build_messages(),
                tools=tools_schema,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content or ""
                self._history.append({"role": "assistant", "content": content})
                return content

            self._history.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                result = await self._execute_tool_async(tc)
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                        "name": tc.function.name,
                    }
                )

        return "Reached maximum tool call iterations."

    async def _arun_with_tools_tracked(self, input_message: str):
        self._history.append({"role": "user", "content": input_message})
        tools_schema = self._get_tools_schema()
        all_tool_calls: List[ToolCall] = []
        last_response = None

        for _ in range(self.config.max_iterations):
            response = await self._acompletion(
                messages=self._build_messages(),
                tools=tools_schema,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            last_response = response
            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content or ""
                self._history.append({"role": "assistant", "content": content})
                return content, last_response, all_tool_calls

            self._history.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                result = await self._execute_tool_async(tc)
                args = (
                    json.loads(tc.function.arguments)
                    if isinstance(tc.function.arguments, str)
                    else tc.function.arguments
                )
                all_tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                        result=result,
                    )
                )
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                        "name": tc.function.name,
                    }
                )

        return "Reached maximum tool call iterations.", last_response, all_tool_calls

    async def _arun_with_tools_prepare(self, input_message: str) -> None:
        self._history.append({"role": "user", "content": input_message})
        tools_schema = self._get_tools_schema()

        for _ in range(self.config.max_iterations):
            response = await self._acompletion(
                messages=self._build_messages(),
                tools=tools_schema,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            msg = response.choices[0].message

            if not msg.tool_calls:
                content = msg.content or ""
                self._history.append({"role": "assistant", "content": content})
                return

            self._history.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )

            for tc in msg.tool_calls:
                result = await self._execute_tool_async(tc)
                self._history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                        "name": tc.function.name,
                    }
                )

        self._history.append(
            {"role": "assistant", "content": "Reached maximum tool call iterations."}
        )

    async def _astream_final_answer(self) -> AsyncIterator[str]:
        """Async stream a fresh final answer, replacing the pre-computed one from the tool loop."""
        if self._history and self._history[-1].get("role") == "assistant":
            self._history.pop()

        response = await self._acompletion(
            messages=self._build_messages(),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,
        )
        collected = []
        async for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                collected.append(delta)
                yield delta
        self._history.append({"role": "assistant", "content": "".join(collected)})

    # ------------------------------------------------------------------
    # Naive tool calling (prompt-based) — async
    # ------------------------------------------------------------------

    async def _arun_naive(self, input_message: str) -> str:
        self._history.append({"role": "user", "content": input_message})
        system_prompt = (
            self.config.system_prompt or "You are a helpful assistant."
        ) + self._build_tools_prompt()

        for _ in range(self.config.max_iterations):
            response = await self._acompletion(
                messages=self._build_messages(system_prompt_override=system_prompt),
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = response.choices[0].message.content or ""
            tool_call = self._parse_naive_tool_call(content)

            if not tool_call:
                self._history.append({"role": "assistant", "content": content})
                return content

            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})
            tool = next((t for t in self.tools if t.name == tool_name), None)

            if not tool:
                result = f"Error: Tool '{tool_name}' not found"
            else:
                try:
                    result = str(await tool.execute(**tool_args))
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"

            self._history.append(
                {"role": "assistant", "content": f"[Used tool: {tool_name}]"}
            )
            self._history.append({"role": "user", "content": f"Tool result: {result}"})

        return "Reached maximum tool call iterations."

    # ------------------------------------------------------------------
    # Tool execution helpers
    # ------------------------------------------------------------------

    def _execute_tool_sync(self, tool_call) -> str:
        """Execute a tool call synchronously."""
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"

        try:
            args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            if inspect.iscoroutinefunction(tool.execute):
                result = _run_coroutine_sync(tool.execute(**args))
            else:
                result = tool.execute(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    async def _execute_tool_async(self, tool_call) -> str:
        """Execute a tool call asynchronously."""
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"

        try:
            args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            result = await tool.execute(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _execute_tool_by_dict(self, tool, tool_name: str, args: dict) -> str:
        """Execute a tool (possibly None) from dict args synchronously."""
        if not tool:
            return f"Error: Tool '{tool_name}' not found"
        try:
            if inspect.iscoroutinefunction(tool.execute):
                result = _run_coroutine_sync(tool.execute(**args))
            else:
                result = tool.execute(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

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

    def _build_messages_from(
        self,
        history: List[Dict[str, Any]],
        system_prompt_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        messages = list(history)
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
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "type": param.type,
                            "description": param.description or "",
                        }
                        for param in tool.definition.parameters.values()
                    },
                    "required": [
                        param.name
                        for param in tool.definition.parameters.values()
                        if param.required
                    ],
                },
            },
        }

    # ------------------------------------------------------------------
    # Tool mode detection
    # ------------------------------------------------------------------

    def _get_tool_mode(self) -> str:
        if self.config.tool_mode in ("native", "naive"):
            return self.config.tool_mode
        if self.config.model in self._tool_mode_cache:
            return self._tool_mode_cache[self.config.model]
        return "native"

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
            if "tool" in parsed and "args" in parsed:
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

    def _completion(self, **kwargs):
        if self.config.router:
            return self.config.router.completion(model=self.config.model, **kwargs)
        return litellm.completion(model=self.config.model, **kwargs)

    async def _acompletion(self, **kwargs):
        if self.config.router:
            return await self.config.router.acompletion(
                model=self.config.model, **kwargs
            )
        return await litellm.acompletion(model=self.config.model, **kwargs)

    # ------------------------------------------------------------------
    # AgentResponse builder
    # ------------------------------------------------------------------

    def _build_agent_response(
        self,
        content: str,
        raw_response,
        tool_calls: List[ToolCall],
    ) -> AgentResponse:
        prompt_tokens = None
        completion_tokens = None
        tokens_used = None
        cost = None

        if raw_response is not None:
            try:
                usage = raw_response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", None)
                completion_tokens = getattr(usage, "completion_tokens", None)
                tokens_used = getattr(usage, "total_tokens", None)
            except AttributeError:
                pass

            try:
                cost = litellm.completion_cost(completion_response=raw_response)
            except Exception:
                pass

        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            model=self.config.model,
            tokens_used=tokens_used,
            cost=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
