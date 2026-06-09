from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.request_latency = Histogram(
            "jax_server_request_latency_seconds",
            "Inference latency by model/backend/mode.",
            labelnames=("model", "backend", "mode"),
            registry=self.registry,
        )
        self.request_total = Counter(
            "jax_server_requests_total",
            "Total inference requests.",
            labelnames=("model", "backend", "mode"),
            registry=self.registry,
        )
        self.request_errors = Counter(
            "jax_server_request_errors_total",
            "Inference errors by model.",
            labelnames=("model",),
            registry=self.registry,
        )
        self.backend_fallbacks = Counter(
            "jax_server_backend_fallbacks_total",
            "Automatic backend fallbacks.",
            labelnames=("model", "from_backend", "to_backend"),
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "jax_server_batch_size",
            "Observed request batch sizes.",
            labelnames=("model",),
            buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "jax_server_model_ready",
            "Whether a model is loaded and ready.",
            labelnames=("model",),
            registry=self.registry,
        )
        self.model_load_time = Histogram(
            "jax_server_model_load_seconds",
            "Model load duration in seconds.",
            labelnames=("model",),
            registry=self.registry,
        )

    def prometheus_response(self):
        from starlette.responses import Response
        return Response(generate_latest(self.registry), media_type=CONTENT_TYPE_LATEST)
