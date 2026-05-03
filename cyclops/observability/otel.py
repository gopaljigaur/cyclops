from __future__ import annotations

import time
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span

from cyclops.core.hooks import AgentHooks


class OTelHooks(AgentHooks):
    """AgentHooks that emits OpenTelemetry spans for every agent event.

    Requires a TracerProvider to be configured before instantiation:

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
        from opentelemetry import trace

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(provider)

    Span hierarchy per agent.run():
        agent.run
        ├── llm.completion   (model, prompt_tokens, completion_tokens, latency_ms)
        ├── tool.<name>      (tool.name, tool.args_count, tool.result.length)
        └── ...

    Note: on_run_end is not called for stream()/astream(), so the root span
    is not closed for streaming runs.
    """

    def __init__(self, tracer_name: str = "cyclops") -> None:
        self._tracer = trace.get_tracer(tracer_name)
        self._root_span: Span | None = None
        self._llm_span: Span | None = None
        self._llm_start_ns: int = 0
        self._tool_span_stack: list[tuple[str, Span]] = []

    def _root_ctx(self) -> Any:
        if self._root_span is not None:
            return trace.set_span_in_context(self._root_span)
        return None

    # ── Run lifecycle ─────────────────────────────────────────────────────────

    def on_run_start(self, input_message: str) -> None:
        self._root_span = self._tracer.start_span("agent.run")
        self._root_span.set_attribute("agent.input.length", len(input_message))

    def on_run_end(self, content: str) -> None:
        if self._root_span is not None:
            self._root_span.set_attribute("agent.output.length", len(content))
            self._root_span.end()
            self._root_span = None

    # ── LLM lifecycle ─────────────────────────────────────────────────────────

    def on_llm_start(self, messages: list[dict[str, Any]]) -> None:
        self._llm_start_ns = time.time_ns()
        self._llm_span = self._tracer.start_span(
            "llm.completion", context=self._root_ctx()
        )
        self._llm_span.set_attribute("llm.message_count", len(messages))

    def on_llm_end(self, response: Any) -> None:
        if self._llm_span is not None:
            elapsed_ms = (time.time_ns() - self._llm_start_ns) // 1_000_000
            self._llm_span.set_attribute("llm.latency_ms", elapsed_ms)
            try:
                usage = response.usage
                self._llm_span.set_attribute("llm.prompt_tokens", usage.prompt_tokens)
                self._llm_span.set_attribute(
                    "llm.completion_tokens", usage.completion_tokens
                )
                self._llm_span.set_attribute("llm.total_tokens", usage.total_tokens)
            except Exception:
                pass
            try:
                self._llm_span.set_attribute("llm.model", response.model)
            except Exception:
                pass
            self._llm_span.end()
            self._llm_span = None

    def on_llm_error(self, error: Exception) -> None:
        if self._llm_span is not None:
            self._llm_span.set_attribute("error", True)
            self._llm_span.set_attribute("error.message", str(error))
            self._llm_span.set_attribute("error.type", type(error).__name__)
            self._llm_span.end()
            self._llm_span = None

    # ── Tool lifecycle ────────────────────────────────────────────────────────

    def on_tool_start(self, tool_name: str, args: dict[str, Any]) -> None:
        span = self._tracer.start_span(f"tool.{tool_name}", context=self._root_ctx())
        span.set_attribute("tool.name", tool_name)
        span.set_attribute("tool.args_count", len(args))
        self._tool_span_stack.append((tool_name, span))

    def on_tool_end(self, tool_name: str, args: dict[str, Any], result: str) -> None:
        span = self._pop_tool_span(tool_name)
        if span is not None:
            span.set_attribute("tool.result.length", len(result))
            span.end()

    def on_tool_error(
        self, tool_name: str, args: dict[str, Any], error: Exception
    ) -> None:
        span = self._pop_tool_span(tool_name)
        if span is not None:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(error))
            span.set_attribute("error.type", type(error).__name__)
            span.end()

    def _pop_tool_span(self, tool_name: str) -> Span | None:
        for i in range(len(self._tool_span_stack) - 1, -1, -1):
            if self._tool_span_stack[i][0] == tool_name:
                return self._tool_span_stack.pop(i)[1]
        return None
