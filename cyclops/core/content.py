"""User message content: text, or OpenAI chat content parts that LiteLLM maps."""

from typing import Any, Dict, List, Optional, Sequence, Union, cast

from litellm.types.llms.openai import (
    ChatCompletionAudioObject,
    ChatCompletionFileObject,
    ChatCompletionImageObject,
    ChatCompletionTextObject,
)
from pydantic import ConfigDict, Field, TypeAdapter
from typing_extensions import Annotated

from cyclops.core.providers import _INPUT_SUPPORT, _every_deployment, _takes_input
from cyclops.core.types import AgentConfig

# An OpenAI chat content part, as LiteLLM types it.
ContentPart = Annotated[
    Union[
        ChatCompletionTextObject,
        ChatCompletionImageObject,
        ChatCompletionFileObject,
        ChatCompletionAudioObject,
    ],
    Field(discriminator="type"),
]

# A user message: text, or a list of text, image_url, file and input_audio parts.
UserContent = Union[str, Sequence[ContentPart]]
# User content as a message holds it.
_Content = Union[str, List[Dict[str, Any]]]

# Errors leave the input out: it may be megabytes of base64.
_PARTS: TypeAdapter[List[Any]] = TypeAdapter(
    List[ContentPart], config=ConfigDict(hide_input_in_errors=True)
)


def _input_kind(part: Dict[str, Any]) -> Optional[str]:
    """What a part needs the model to take: "image", "pdf" or "audio", else None."""
    kind = part["type"]
    if kind == "image_url":
        return "image"
    if kind == "input_audio":
        return "audio"
    if kind != "file":
        return None
    # A file's media type is in its data URL or format; a file_id may have neither.
    file = part["file"]
    data = file.get("file_data", "")
    media_type = data[5:].split(";", 1)[0] if data.startswith("data:") else ""
    media_type = media_type or file.get("format", "")
    if media_type == "application/pdf":
        return "pdf"
    major = media_type.split("/", 1)[0]
    return major if major in ("image", "audio") else None


def _user_content(config: AgentConfig, content: UserContent) -> _Content:
    """Content for a user message, checked against what the model takes.

    Raises ValueError for parts that are not OpenAI chat content parts, and for
    images, PDFs or audio that LiteLLM's model map marks a deployment as not taking.
    """
    if isinstance(content, str):
        return content
    parts = cast(List[Dict[str, Any]], list(content))
    _PARTS.validate_python(parts)
    if not parts:
        raise ValueError("Message content has no parts")
    for kind in dict.fromkeys(filter(None, map(_input_kind, parts))):
        flag = _INPUT_SUPPORT[kind]
        # Fallback models read the same history.
        if not _every_deployment(config, _takes_input, flag, fallbacks=True):
            raise ValueError(
                f"{config.model}, one of its deployments or one of its fallback "
                f"models can't take {kind} input: LiteLLM's model map sets {flag} "
                "to false for one of them; if that is wrong, correct its entry with "
                "litellm.register_model at startup"
            )
    return parts


__all__ = ["ContentPart", "UserContent"]
