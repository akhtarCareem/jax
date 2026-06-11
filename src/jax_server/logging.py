import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional


class DynatraceJsonFormatter(logging.Formatter):
    """Formats log records as JSON compatible with Dynatrace."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            "loglevel": record.levelname,
            "content": record.getMessage(),
        }

        if hasattr(record, "dt_trace_id"):
            log_obj["dt.trace_id"] = record.dt_trace_id
        if hasattr(record, "dt_span_id"):
            log_obj["dt.span_id"] = record.dt_span_id

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        extra_fields = {k: v for k, v in record.__dict__.items()
                       if k not in {"name", "msg", "args", "created", "filename", "funcName",
                                   "levelname", "levelno", "lineno", "module", "msecs", "message",
                                   "pathname", "process", "processName", "relativeCreated", "thread",
                                   "threadName", "exc_info", "exc_text", "stack_info", "dt_trace_id",
                                   "dt_span_id"}}
        if extra_fields:
            log_obj.update(extra_fields)

        return json.dumps(log_obj)


def setup_dynatrace_logging(level: int = logging.INFO) -> None:
    """Configure root logger with Dynatrace JSON formatting."""
    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(DynatraceJsonFormatter())
    root.addHandler(handler)


def add_trace_context(logger: logging.LoggerAdapter, trace_id: Optional[str], span_id: Optional[str]) -> None:
    """Add Dynatrace trace context to a logger adapter."""
    if trace_id:
        logger.extra["dt_trace_id"] = trace_id
    if span_id:
        logger.extra["dt_span_id"] = span_id
