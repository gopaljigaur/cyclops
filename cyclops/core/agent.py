"""Core agent implementation"""

from typing import Any, Dict, List, Optional
import asyncio
import litellm
from cyclops.core.types import AgentConfig, Message


class Agent:
    """LLM agent implementation"""

    # Class-level cache for detected tool modes per model
    _tool_mode_cache: Dict[str, str] = {}

    def __init__(
        self,
        config: AgentConfig,
        tools: Optional[List] = None,
    ):
        self.config = config
        self.messages: List[Message] = []
        self.tools = tools or []

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the conversation"""
        self.messages.append(
            Message(role=role, content=content, metadata=metadata or {})
        )

    def run(self, input_message: str) -> str:
        """Run the agent with input using LiteLLM (sync)"""
        if not self.tools:
            return self._run_no_tools(input_message)

        tool_mode = self._get_tool_mode()

        if tool_mode == "naive":
            return self._run_naive_tools(input_message)

        # Try native first
        try:
            return self._run_native(input_message)
        except Exception as e:
            # Check if error is related to unsupported tools
            error_str = str(e).lower()
            if any(
                keyword in error_str for keyword in ["tool", "function", "unsupported"]
            ):
                # Cache this model as naive mode and retry
                self._tool_mode_cache[self.config.model] = "naive"
                return self._run_naive_tools(input_message)
            # Re-raise if not tool-related error
            raise

    def _run_no_tools(self, input_message: str) -> str:
        """Run without tools (sync)"""
        self.add_message("user", input_message)

        messages = [{"role": m.role, "content": m.content} for m in self.messages]
        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        response = self._completion(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        content = response.choices[0].message.content or ""
        self.add_message("assistant", content)
        return content

    def _run_native(self, input_message: str) -> str:
        """Run with native function calling support (sync)"""
        self.add_message("user", input_message)

        messages = [{"role": m.role, "content": m.content} for m in self.messages]
        if self.config.system_prompt:
            messages.insert(0, {"role": "system", "content": self.config.system_prompt})

        tools = [self._tool_to_openai_format(tool) for tool in self.tools]

        response = self._completion(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=tools,
        )

        assistant_message = response.choices[0].message

        if assistant_message.tool_calls:
            self.add_message(
                "assistant", "", {"tool_calls": assistant_message.tool_calls}
            )

            # Execute tools
            for tool_call in assistant_message.tool_calls:
                tool_result = self._execute_tool_sync(tool_call)
                self.add_message("tool", tool_result, {"tool_call_id": tool_call.id})

            # Get final response
            messages = [{"role": m.role, "content": m.content} for m in self.messages]
            if self.config.system_prompt:
                messages.insert(
                    0, {"role": "system", "content": self.config.system_prompt}
                )

            final_response = self._completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            final_content = final_response.choices[0].message.content
            self.add_message("assistant", final_content)
            return final_content
        else:
            content = assistant_message.content or ""
            self.add_message("assistant", content)
            return content

    def _run_naive_tools(self, input_message: str) -> str:
        """Run with naive tool calling (sync)"""
        self.add_message("user", input_message)

        system_prompt = self.config.system_prompt or "You are a helpful assistant."
        system_prompt += self._build_tools_prompt()

        messages = [{"role": m.role, "content": m.content} for m in self.messages]
        messages.insert(0, {"role": "system", "content": system_prompt})

        response = self._completion(
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        content = response.choices[0].message.content or ""
        tool_call = self._parse_naive_tool_call(content)

        if tool_call:
            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})

            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                result = f"Error: Tool '{tool_name}' not found"
            else:
                try:
                    result = self._execute_tool_sync_from_dict(tool, tool_args)
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"

            self.add_message("assistant", f"Using tool: {tool_name}")
            self.add_message("user", f"Tool result: {result}")

            messages = [{"role": m.role, "content": m.content} for m in self.messages]
            messages.insert(0, {"role": "system", "content": system_prompt})

            final_response = self._completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            final_content = final_response.choices[0].message.content or ""
            self.add_message("assistant", final_content)
            return final_content
        else:
            self.add_message("assistant", content)
            return content

    def _execute_tool_sync(self, tool_call):
        """Execute a tool call synchronously"""
        import json
        import inspect

        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"

        try:
            args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            # Check if tool.execute is async
            if inspect.iscoroutinefunction(tool.execute):
                result = asyncio.run(tool.execute(**args))
            else:
                result = tool.execute(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _execute_tool_sync_from_dict(self, tool, args: dict):
        """Execute a tool with dict args synchronously"""
        import inspect

        if inspect.iscoroutinefunction(tool.execute):
            result = asyncio.run(tool.execute(**args))
        else:
            result = tool.execute(**args)
        return str(result)

    def _get_tool_mode(self) -> str:
        """Determine which tool mode to use: 'native' or 'naive'"""
        # Explicit config overrides everything
        if self.config.tool_mode in ["native", "naive"]:
            return self.config.tool_mode

        # Check cache
        if self.config.model in self._tool_mode_cache:
            return self._tool_mode_cache[self.config.model]

        # Default to native, will fallback on error
        return "native"

    def _build_tools_prompt(self) -> str:
        """Build tools description for system prompt (naive mode)"""
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
        """Parse tool call from model response (naive mode)"""
        import json

        # Try parsing the entire content as JSON first
        try:
            tool_call = json.loads(content.strip())
            if "tool" in tool_call and "args" in tool_call:
                return tool_call
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from text (simple brace counting)
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
                        tool_call = json.loads(content[start : i + 1])
                        if "tool" in tool_call and "args" in tool_call:
                            return tool_call
                    except json.JSONDecodeError:
                        pass
                    break

        return None

    async def arun(self, input_message: str) -> str:
        """Run the agent with input using LiteLLM (async)"""
        if not self.tools:
            # No tools, simple completion
            return await self._arun_no_tools(input_message)

        tool_mode = self._get_tool_mode()

        if tool_mode == "naive":
            return await self._arun_naive_tools(input_message)

        # Try native first
        try:
            return await self._arun_native(input_message)
        except Exception as e:
            # Check if error is related to unsupported tools
            error_str = str(e).lower()
            if (
                "tool" in error_str
                or "function" in error_str
                or "unsupported" in error_str
            ):
                # Cache this model as naive mode and retry
                self._tool_mode_cache[self.config.model] = "naive"
                return await self._arun_naive_tools(input_message)
            # Re-raise if not tool-related error
            raise

    async def _arun_no_tools(self, input_message: str) -> str:
        """Run without tools"""
        self.add_message("user", input_message)

        llm_messages = [
            {"role": msg.role, "content": msg.content} for msg in self.messages
        ]
        if self.config.system_prompt:
            llm_messages.insert(
                0, {"role": "system", "content": self.config.system_prompt}
            )

        response = await self._acompletion(
            messages=llm_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        content = response.choices[0].message.content or ""
        self.add_message("assistant", content)
        return content

    async def _arun_native(self, input_message: str) -> str:
        """Run with native function calling support"""
        # Add user message
        self.add_message("user", input_message)

        # Prepare messages for LiteLLM
        llm_messages = [
            {"role": msg.role, "content": msg.content} for msg in self.messages
        ]

        # Add system prompt if configured
        if self.config.system_prompt:
            llm_messages.insert(
                0, {"role": "system", "content": self.config.system_prompt}
            )

        # Prepare tools for function calling
        tools = None
        if self.tools:
            tools = [self._tool_to_openai_format(tool) for tool in self.tools]

        # Call LiteLLM
        response = await self._acompletion(
            messages=llm_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            tools=tools,
        )

        assistant_message = response.choices[0].message

        # Handle tool calls
        if assistant_message.tool_calls:
            self.add_message(
                "assistant", "", {"tool_calls": assistant_message.tool_calls}
            )

            # Execute tools
            for tool_call in assistant_message.tool_calls:
                tool_result = await self._execute_tool(tool_call)
                self.add_message("tool", tool_result, {"tool_call_id": tool_call.id})

            # Get final response after tool execution
            llm_messages = [
                {"role": msg.role, "content": msg.content} for msg in self.messages
            ]
            if self.config.system_prompt:
                llm_messages.insert(
                    0, {"role": "system", "content": self.config.system_prompt}
                )

            final_response = await self._acompletion(
                messages=llm_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            final_content = final_response.choices[0].message.content
            self.add_message("assistant", final_content)
            return final_content
        else:
            # No tool calls, just return the content
            content = assistant_message.content or ""
            self.add_message("assistant", content)
            return content

    async def _arun_naive_tools(self, input_message: str) -> str:
        """Run with naive tool calling (prompt-based)"""
        self.add_message("user", input_message)

        # Build system prompt with tools
        system_prompt = self.config.system_prompt or "You are a helpful assistant."
        system_prompt += self._build_tools_prompt()

        llm_messages = [
            {"role": msg.role, "content": msg.content} for msg in self.messages
        ]
        llm_messages.insert(0, {"role": "system", "content": system_prompt})

        # First call - model decides if it needs a tool
        response = await self._acompletion(
            messages=llm_messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        content = response.choices[0].message.content or ""

        # Check if model wants to use a tool
        tool_call = self._parse_naive_tool_call(content)

        if tool_call:
            # Execute the tool
            tool_name = tool_call["tool"]
            tool_args = tool_call.get("args", {})

            tool = next((t for t in self.tools if t.name == tool_name), None)
            if not tool:
                result = f"Error: Tool '{tool_name}' not found"
            else:
                try:
                    result = await tool.execute(**tool_args)
                    result = str(result)
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"

            # Add tool usage to messages
            self.add_message("assistant", f"Using tool: {tool_name}")
            self.add_message("user", f"Tool result: {result}")

            # Get final response with tool result
            llm_messages = [
                {"role": msg.role, "content": msg.content} for msg in self.messages
            ]
            llm_messages.insert(0, {"role": "system", "content": system_prompt})

            final_response = await self._acompletion(
                messages=llm_messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )

            final_content = final_response.choices[0].message.content or ""
            self.add_message("assistant", final_content)
            return final_content
        else:
            # No tool call, return as-is
            self.add_message("assistant", content)
            return content

    def _tool_to_openai_format(self, tool):
        """Convert tool to OpenAI function format"""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "type": param.type.replace("<class '", "")
                            .replace("'>", "")
                            .replace("str", "string")
                            .replace("int", "integer"),
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

    async def _execute_tool(self, tool_call):
        """Execute a tool call"""
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        # Find the tool
        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return f"Error: Tool '{tool_name}' not found"

        try:
            import json

            args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args
            result = await tool.execute(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _completion(self, **kwargs):
        """Call LiteLLM completion (sync)"""
        if self.config.router:
            return self.config.router.completion(model=self.config.model, **kwargs)
        else:
            return litellm.completion(model=self.config.model, **kwargs)

    async def _acompletion(self, **kwargs):
        """Call LiteLLM completion (async)"""
        if self.config.router:
            return await self.config.router.acompletion(
                model=self.config.model, **kwargs
            )
        else:
            return await litellm.acompletion(model=self.config.model, **kwargs)
