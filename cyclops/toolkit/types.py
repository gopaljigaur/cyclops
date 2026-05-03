"""Toolkit type definitions"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """Tool parameter definition"""

    name: str
    type: str
    description: Optional[str] = None
    required: bool = True
    default: Any = None


class ToolDefinition(BaseModel):
    """Tool definition metadata"""

    name: str
    description: str
    parameters: Dict[str, ToolParameter] = Field(default_factory=dict)

    def to_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                param.name: {
                    "type": param.type,
                    "description": param.description or "",
                }
                for param in self.parameters.values()
            },
            "required": [
                param.name for param in self.parameters.values() if param.required
            ],
        }


class ToolResult(BaseModel):
    """Result from tool execution"""

    success: bool = Field(description="Whether the tool execution succeeded")
    result: Any = Field(description="The actual result from the tool")
    error: Optional[str] = Field(
        default=None, description="Error message if execution failed"
    )

    @classmethod
    def success_result(cls, result: Any) -> "ToolResult":
        """Create a successful tool result"""
        return cls(success=True, result=result)

    @classmethod
    def error_result(cls, error: str, result: Any = None) -> "ToolResult":
        """Create an error tool result"""
        return cls(success=False, result=result, error=error)
