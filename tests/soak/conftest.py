import json
import logging

import litellm
import pytest

ANSWER = "Two words."


@pytest.fixture(autouse=True)
def mock_completions(monkeypatch):
    """Real litellm.acompletion with mock responses: a tool call, then text.

    With tools, the step after the user message calls the first tool with
    every argument set to 1; any other step answers. A mock_response in
    completion_kwargs wins over the answer.
    """
    acompletion = litellm.acompletion

    async def scripted(**kwargs):
        if kwargs.get("tools") and kwargs["messages"][-1]["role"] == "user":
            function = kwargs["tools"][0]["function"]
            arguments = {name: 1 for name in function["parameters"]["properties"]}
            call = {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": function["name"],
                    "arguments": json.dumps(arguments),
                },
            }
            # A ModelResponse, not mock_tool_calls, so streamed steps call tools too.
            kwargs["mock_response"] = litellm.ModelResponse(
                choices=[
                    {
                        "finish_reason": "tool_calls",
                        "message": {"role": "assistant", "tool_calls": [call]},
                    }
                ]
            )
        else:
            kwargs.setdefault("mock_response", ANSWER)
        return await acompletion(**kwargs)

    monkeypatch.setattr(litellm, "acompletion", scripted)


@pytest.fixture(autouse=True)
def no_asyncio_errors(caplog):
    """Fail on what asyncio logs: unretrieved exceptions, destroyed tasks."""
    yield
    errors = [
        r.getMessage()
        for r in caplog.records
        if r.name == "asyncio" and r.levelno >= logging.ERROR
    ]
    assert not errors
