from __future__ import annotations

import asyncio
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from cyclops.core.hooks import RUN_STOPPED, AgentHooks, current_tool_call_id
from cyclops.toolkit.types import ToolResult

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider


@dataclass(frozen=True)
class _Frame:
    """An open span. Frames form an immutable stack per context, so tasks never share one."""

    span: Span
    parent: _Frame | None
    token: Token[Context] | None
    run: bool


# Module level: contexts keep strong references to their variables.
_frame: ContextVar[_Frame | None] = ContextVar("cyclops_telemetry_frame", default=None)


class TelemetryHooks(AgentHooks):
    """AgentHooks that emits OpenTelemetry spans for every agent event.

    One instance can serve many concurrent runs: span state lives in a
    ContextVar, so each thread or asyncio task sees only its own run.
    Span names and ``gen_ai.*`` attributes follow the OpenTelemetry GenAI
    semantic conventions in part: the hooks do not get the request model or
    provider, so ``gen_ai.request.model`` and ``gen_ai.provider.name`` are not set.

        hooks=TelemetryHooks.console()                          # print spans to stdout
        hooks=TelemetryHooks.otlp("http://localhost:4317")      # send to Jaeger / Tempo / Datadog
        hooks=TelemetryHooks(tracer_provider=provider)          # your own provider

    Without a provider, spans go to the global one. Flushing is left to the
    application (``hooks.flush()`` for the factories).

    Span hierarchy per agent.run(), named after the GenAI semantic conventions:
        invoke_agent
        ├── chat
        ├── execute_tool <name>
        └── ...
    """

    def __init__(
        self,
        tracer_name: str = "cyclops",
        tracer_provider: trace.TracerProvider | None = None,
    ) -> None:
        self._tracer = trace.get_tracer(tracer_name, tracer_provider=tracer_provider)
        self._provider: SDKTracerProvider | None = None

    @classmethod
    def console(cls, tracer_name: str = "cyclops") -> TelemetryHooks:
        """Return a TelemetryHooks instance with its own console (stdout) exporter."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        provider = TracerProvider()
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        instance = cls(tracer_name=tracer_name, tracer_provider=provider)
        instance._provider = provider
        return instance

    @classmethod
    def otlp(
        cls,
        endpoint: str = "http://localhost:4317",
        tracer_name: str = "cyclops",
        **exporter_kwargs: Any,
    ) -> TelemetryHooks:
        """Return a TelemetryHooks instance with its own OTLP gRPC exporter.

        Requires: uv add opentelemetry-exporter-otlp-proto-grpc
        """
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore[import]
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, **exporter_kwargs))
        )
        instance = cls(tracer_name=tracer_name, tracer_provider=provider)
        instance._provider = provider
        return instance

    def flush(self) -> None:
        """Force-flush the provider built by console() or otlp()."""
        if self._provider is not None:
            self._provider.force_flush()

    # ── Span stack ────────────────────────────────────────────────────────────

    def _push(
        self,
        name: str,
        kind: SpanKind,
        attributes: dict[str, Any],
        run: bool = False,
        attach: bool = False,
    ) -> None:
        parent = _frame.get()
        # A run's parent is the current OTel span: the caller's, or a tool span for a nested run.
        ctx = trace.set_span_in_context(parent.span) if parent and not run else None
        span = self._tracer.start_span(
            name, context=ctx, kind=kind, attributes=attributes
        )
        # Only tool spans become current, so a tool and any run nested in it
        # are traced under the tool's span.
        token = otel_context.attach(trace.set_span_in_context(span)) if attach else None
        _frame.set(_Frame(span, parent, token, run))

    def _pop(self) -> _Frame | None:
        frame = _frame.get()
        if frame is None:
            return None
        if frame.token is not None:
            otel_context.detach(frame.token)
        _frame.set(frame.parent)
        return frame

    @staticmethod
    def _fail(frame: _Frame | None, error: BaseException) -> None:
        if frame is None:
            return
        if isinstance(error, asyncio.CancelledError) and error.args == (RUN_STOPPED,):
            # The caller stopped the run, e.g. by closing its stream; not a failure.
            frame.span.end()
            return
        frame.span.set_attribute("error.type", type(error).__qualname__)
        frame.span.record_exception(error)
        frame.span.set_status(Status(StatusCode.ERROR, str(error)))
        frame.span.end()

    # ── Run lifecycle ─────────────────────────────────────────────────────────

    def on_run_start(self, input_message: str | list[dict[str, Any]]) -> None:
        # Only the text's length: parts may hold base64 data.
        text = ToolResult.message_text(input_message)
        self._push(
            "invoke_agent",
            SpanKind.INTERNAL,
            {
                "gen_ai.operation.name": "invoke_agent",
                "cyclops.input.length": len(text),
            },
            run=True,
        )

    def on_run_end(self, content: str) -> None:
        frame = self._pop()
        if frame is not None:
            frame.span.set_attribute("cyclops.output.length", len(content))
            frame.span.end()

    def on_run_error(self, error: BaseException) -> None:
        # Also ends an LLM or tool span the error cut short.
        while (frame := self._pop()) is not None:
            self._fail(frame, error)
            if frame.run:
                break

    # ── LLM lifecycle ─────────────────────────────────────────────────────────

    def on_llm_start(self, messages: list[dict[str, Any]]) -> None:
        self._push(
            "chat",
            SpanKind.CLIENT,
            {
                "gen_ai.operation.name": "chat",
                "cyclops.llm.message_count": len(messages),
            },
        )

    def on_llm_end(self, response: Any) -> None:
        frame = self._pop()
        if frame is None:
            return
        span = frame.span
        model = getattr(response, "model", None)
        if isinstance(model, str):
            span.set_attribute("gen_ai.response.model", model)
        response_id = getattr(response, "id", None)
        if isinstance(response_id, str):
            span.set_attribute("gen_ai.response.id", response_id)
        reasons = [
            choice.finish_reason
            for choice in getattr(response, "choices", None) or []
            if isinstance(getattr(choice, "finish_reason", None), str)
        ]
        if reasons:
            span.set_attribute("gen_ai.response.finish_reasons", reasons)
        usage = getattr(response, "usage", None)
        prompt = getattr(usage, "prompt_tokens_details", None)
        completion = getattr(usage, "completion_tokens_details", None)
        for attr, source, field in (
            ("gen_ai.usage.input_tokens", usage, "prompt_tokens"),
            ("gen_ai.usage.output_tokens", usage, "completion_tokens"),
            ("gen_ai.usage.cache_read.input_tokens", prompt, "cached_tokens"),
            ("gen_ai.usage.cache_write.input_tokens", prompt, "cache_write_tokens"),
            ("gen_ai.usage.reasoning.output_tokens", completion, "reasoning_tokens"),
        ):
            value = getattr(source, field, None)
            if isinstance(value, int):
                span.set_attribute(attr, value)
        span.end()

    def on_llm_error(self, error: Exception) -> None:
        # The run may recover (naive tool fallback), so only the chat span ends.
        self._fail(self._pop(), error)

    # ── Tool lifecycle ────────────────────────────────────────────────────────

    def on_tool_start(self, tool_name: str, args: dict[str, Any]) -> str | None:
        # Each native tool call runs in its own task, so it pairs with its own end.
        attributes = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": tool_name,
            "cyclops.tool.args_count": len(args),
        }
        call_id = current_tool_call_id()
        if call_id is not None:
            attributes["gen_ai.tool.call.id"] = call_id
        self._push(
            f"execute_tool {tool_name}", SpanKind.INTERNAL, attributes, attach=True
        )
        return None

    def on_tool_end(self, tool_name: str, args: dict[str, Any], result: str) -> None:
        frame = self._pop()
        if frame is not None:
            frame.span.set_attribute("cyclops.tool.result.length", len(result))
            frame.span.end()

    def on_tool_error(
        self, tool_name: str, args: dict[str, Any], error: BaseException
    ) -> None:
        self._fail(self._pop(), error)
