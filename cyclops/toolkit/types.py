"""Toolkit type definitions"""

import json
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
from pydantic_core import to_jsonable_python


class ToolParameter(BaseModel):
    """Tool parameter definition"""

    name: str
    type: str
    description: Optional[str] = None
    required: bool = True
    default: Any = None


class ToolAnnotations(BaseModel):
    """Behaviour hints for a tool, mirroring MCP tool annotations"""

    read_only: Optional[bool] = None
    destructive: Optional[bool] = None
    idempotent: Optional[bool] = None
    open_world: Optional[bool] = None


class ToolDefinition(BaseModel):
    """Tool definition metadata"""

    name: str
    description: str
    parameters: Dict[str, ToolParameter] = Field(default_factory=dict)
    input_schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Full JSON Schema of the parameters"
    )
    annotations: ToolAnnotations = Field(default_factory=ToolAnnotations)

    def to_json_schema(self) -> Dict[str, Any]:
        if self.input_schema is not None:
            return self.input_schema
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


class TextContent(BaseModel):
    """Text content block"""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Base64 encoded image content block"""

    type: Literal["image"] = "image"
    data: str
    mime_type: str


ContentBlock = Annotated[Union[TextContent, ImageContent], Field(discriminator="type")]


class ToolResult(BaseModel):
    """Result from tool execution"""

    success: bool = Field(
        default=True, description="Whether the tool execution succeeded"
    )
    result: Any = Field(
        default=None, description="Structured data returned by the tool"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if execution failed"
    )
    content: List[ContentBlock] = Field(
        default_factory=list, description="Content blocks shown to the model"
    )

    @property
    def is_error(self) -> bool:
        return not self.success

    @classmethod
    def success_result(cls, result: Any) -> "ToolResult":
        """Create a successful tool result"""
        return cls(success=True, result=result)

    @classmethod
    def error_result(cls, error: str, result: Any = None) -> "ToolResult":
        """Create an error tool result"""
        return cls(success=False, result=result, error=error)

    def to_message_content(self) -> Union[str, List[Dict[str, Any]]]:
        """Render as the content of a tool message in OpenAI chat format.

        Text only results render as a string. Results with images render as
        a list of ``text`` and ``image_url`` parts; OpenAI itself accepts only
        text parts in a tool message, so images need another route there.
        ``result`` is rendered as JSON only when there are no content blocks,
        since those usually already describe it.
        """
        parts: List[Dict[str, Any]] = []
        if self.is_error:
            error = self.error or "tool execution failed"
            parts.append({"type": "text", "text": f"Error: {error}"})
        for block in self.content:
            if isinstance(block, TextContent):
                parts.append({"type": "text", "text": block.text})
            else:
                url = f"data:{block.mime_type};base64,{block.data}"
                parts.append({"type": "image_url", "image_url": {"url": url}})
        if not self.content and self.result is not None:
            text = (
                self.result
                if isinstance(self.result, str)
                else json.dumps(to_jsonable_python(self.result, fallback=str))
            )
            parts.append({"type": "text", "text": text})
        if all(part["type"] == "text" for part in parts):
            return "\n".join(part["text"] for part in parts)
        return parts

    @staticmethod
    def message_text(content: Union[str, List[Dict[str, Any]]]) -> str:
        """The text of content from to_message_content(), without its images."""
        if isinstance(content, str):
            return content
        return "\n".join(part["text"] for part in content if part["type"] == "text")
