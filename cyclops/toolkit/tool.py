"""Tool definitions and base classes"""

from abc import ABC
from typing import Any, Callable
import inspect
from cyclops.toolkit.types import ToolParameter, ToolDefinition


class BaseTool(ABC):
    """Abstract base tool class"""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._definition = self._build_definition()

    async def execute(self, **kwargs) -> Any:
        """Execute the tool. Subclasses should override with their own signature."""
        raise NotImplementedError(f"Tool '{self.name}' must implement execute()")

    def _build_definition(self) -> ToolDefinition:
        """Build tool definition from method signature"""
        sig = inspect.signature(self.execute)
        parameters = {}

        for param_name, param in sig.parameters.items():
            if param_name == "kwargs":
                continue

            param_type = (
                str(param.annotation)
                if param.annotation != inspect.Parameter.empty
                else "str"
            )
            required = param.default == inspect.Parameter.empty
            default = param.default if not required else None

            parameters[param_name] = ToolParameter(
                name=param_name, type=param_type, required=required, default=default
            )

        return ToolDefinition(
            name=self.name, description=self.description, parameters=parameters
        )

    @property
    def definition(self) -> ToolDefinition:
        """Get tool definition"""
        return self._definition


class Tool(BaseTool):
    """Simple function-based tool"""

    def __init__(self, name: str, description: str, func: Callable):
        self.func = func
        super().__init__(name, description)

    async def execute(self, **kwargs) -> Any:
        """Execute the wrapped function"""
        if inspect.iscoroutinefunction(self.func):
            return await self.func(**kwargs)
        return self.func(**kwargs)
