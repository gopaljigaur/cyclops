"""Core agent implementation"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import asyncio
import litellm


class AgentConfig(BaseModel):
    """Configuration for an agent"""

    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None


class Message(BaseModel):
    """Message representation"""

    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """Base abstract agent class"""

    def __init__(self, config: AgentConfig, tools: Optional[List] = None):
        self.config = config
        self.messages: List[Message] = []
        self.tools = tools or []

    @abstractmethod
    def run(self, input_message: str) -> str:
        """Run the agent with input and return response"""
        pass

    async def arun(self, input_message: str) -> str:
        """Async version of run"""
        return self.run(input_message)

    def add_message(
        self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the conversation"""
        self.messages.append(
            Message(role=role, content=content, metadata=metadata or {})
        )


class Agent(BaseAgent):
    """Basic LLM agent implementation"""

    def run(self, input_message: str) -> str:
        """Run the agent with input using LiteLLM (sync)"""
        return asyncio.run(self.arun(input_message))

    async def arun(self, input_message: str) -> str:
        """Run the agent with input using LiteLLM (async)"""
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
        response = await litellm.acompletion(
            model=self.config.model,
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

            final_response = await litellm.acompletion(
                model=self.config.model,
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
