"""Configuration management"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import os
import json


class Config(BaseModel):
    """Configuration model"""

    # LLM settings
    default_model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: Optional[int] = None

    # API keys (loaded from environment)
    openai_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    anthropic_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    # Logging
    log_level: str = "INFO"

    # Custom settings
    custom: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, file_path: str) -> "Config":
        """Load configuration from JSON file"""
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_file(self, file_path: str) -> None:
        """Save configuration to JSON file"""
        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
