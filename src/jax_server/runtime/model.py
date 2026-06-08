from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from jax_server.artifacts.discovery import discover_artifacts
from jax_server.config import ModelConfig
from jax_server.env import normalize_platform
from jax_server.exceptions import ClientInputError, ExecutionError, ModelLoadError
from jax_server.hf.store import ArtifactStore
from jax_server.inference.convert import infer_batch_size, to_jax_pytree, to_numpy_pytree
from jax_server.metrics import Metrics

logger = logging.getLogger("jax_server.runtime")
from jax_server.runtime.exported import ExportedFunction
from jax_server.runtime.params import load_params


def _pin_params_to_gpu(params: Any) -> Any:
    import jax

    gpus = jax.devices("gpu") if any(d.platform == "gpu" for d in jax.devices()) else []
    if not gpus:
        return params
    return jax.device_put(params, gpus[0])


class ServedModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.params: Any | None = None
        self.exports: dict[tuple[str, str], ExportedFunction] = {}
        self.snapshot_path: Path | None = None
        self.loaded = False
        self._metric_handles: dict[tuple[str, str], Any] = {}

    def load(
        self,
        store: ArtifactStore,
        metrics: Metrics | None = None,
        shared_params_cache: dict[tuple[str, str, str], Any] | None = None,
    ) -> None:
        started_at = time.perf_counter()
        snapshot_path = store.fetch_snapshot(self.config)
        params_path = snapshot_path / self.config.params_path
        cache_key = (str(snapshot_path), self.config.params_path, self.config.params_format)
        if shared_params_cache is not None and cache_key in shared_params_cache:
            self.params = shared_params_cache[cache_key]
        else:
            self.params = _pin_params_to_gpu(load_params(params_path, self.config.params_format))
            if shared_params_cache is not None:
                shared_params_cache[cache_key] = self.params

        artifacts_root = snapshot_path / self.config.artifact_dir if self.config.artifact_dir else snapshot_path
        artifacts = discover_artifacts(artifacts_root, self.config.artifact_prefix)
        self.exports = {
            key: ExportedFunction.from_file(path) for key, path in artifacts.items()
        }
        self.snapshot_path = snapshot_path
        self.loaded = True
        if metrics is not None:
            metrics.model_ready.labels(model=self.config.name).set(1)
            metrics.model_load_time.labels(model=self.config.name).observe(
                time.perf_counter() - started_at
            )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.config.name,
            "source": self.config.source,
            "hf_repo": self.config.hf_repo,
            "revision": self.config.revision,
            "local_path": self.config.local_path,
            "artifact_name": self.config.artifact_prefix,
            "default_platform": self.config.default_platform,
            "max_batch_size": self.config.max_batch_size,
            "available_exports": [
                {"backend": backend, "mode": mode}
                for backend, mode in sorted(self.exports.keys())
            ],
            "loaded": self.loaded,
        }

    def _available_backends(self) -> set[str]:
        return {backend for backend, _ in self.exports}

    def _detect_gpu_available(self) -> bool:
        try:
            import jax
        except ImportError:
            return False
        return any(normalize_platform(device.platform) == "gpu" for device in jax.devices())

    def _resolve_mode(self, mode: str, batch_size: int) -> str:
        if mode != "auto":
            return mode

        if batch_size > 1:
            if self.config.max_batch_size is None:
                raise ClientInputError(
                    f"Model '{self.config.name}' received batch size {batch_size}, "
                    "but batching is disabled."
                )
            if batch_size > self.config.max_batch_size:
                raise ClientInputError(
                    f"Model '{self.config.name}' received batch size {batch_size}, "
                    f"which exceeds max_batch_size={self.config.max_batch_size}."
                )
            return "batch"
        return "single"

    def _resolve_backend(self, backend: str, mode: str, metrics: Metrics | None) -> str:
        available = self._available_backends()
        if backend in {"cpu", "gpu"}:
            if (backend, mode) not in self.exports:
                raise ClientInputError(
                    f"Model '{self.config.name}' has no {backend}/{mode} export."
                )
            return backend

        preferred = "gpu" if self._detect_gpu_available() else self.config.default_platform
        if (preferred, mode) in self.exports:
            return preferred
        if preferred != "cpu" and ("cpu", mode) in self.exports:
            if metrics is not None:
                metrics.backend_fallbacks.labels(
                    model=self.config.name,
                    from_backend=preferred,
                    to_backend="cpu",
                ).inc()
            return "cpu"
        if self.config.default_platform in available and (self.config.default_platform, mode) in self.exports:
            return self.config.default_platform
        raise ClientInputError(
            f"Model '{self.config.name}' has no compatible export for mode '{mode}'."
        )

    def predict(
        self,
        inputs: Any,
        backend: str = "auto",
        mode: str = "auto",
        metrics: Metrics | None = None,
    ) -> dict[str, Any]:
        if not self.loaded or self.params is None:
            raise ModelLoadError(f"Model '{self.config.name}' is not loaded.")

        convert_start = time.perf_counter()
        converted_inputs = to_jax_pytree(inputs)
        try:
            batch_size = infer_batch_size(converted_inputs)
        except ValueError as exc:
            raise ClientInputError(str(exc)) from exc

        resolved_mode = self._resolve_mode(mode, batch_size)
        resolved_backend = self._resolve_backend(backend, resolved_mode, metrics)
        exported = self.exports[(resolved_backend, resolved_mode)]
        handles = self._metric_handles_for(metrics, resolved_backend, resolved_mode)

        compute_start = time.perf_counter()
        try:
            outputs = exported.call(self.params, converted_inputs)
        except Exception as exc:
            if handles is not None:
                handles["errors"].inc()
            logger.exception(
                "inference error model=%s backend=%s mode=%s",
                self.config.name, resolved_backend, resolved_mode,
            )
            raise ExecutionError(
                f"Prediction failed for model '{self.config.name}'."
            ) from exc

        serialize_start = time.perf_counter()
        jsonable_outputs = to_numpy_pytree(outputs)
        finished_at = time.perf_counter()

        if handles is not None:
            handles["batch_size"].observe(batch_size)
            handles["latency"].observe(finished_at - compute_start)
            handles["total"].inc()

        logger.debug(
            "predict timing model=%s convert_ms=%.3f compute_ms=%.3f serialize_ms=%.3f",
            self.config.name,
            (compute_start - convert_start) * 1000,
            (serialize_start - compute_start) * 1000,
            (finished_at - serialize_start) * 1000,
        )

        return {
            "model": self.config.name,
            "backend": resolved_backend,
            "mode": resolved_mode,
            "outputs": jsonable_outputs,
        }

    def _metric_handles_for(
        self, metrics: Metrics | None, backend: str, mode: str
    ) -> dict[str, Any] | None:
        if metrics is None:
            return None
        key = (backend, mode)
        handles = self._metric_handles.get(key)
        if handles is None:
            name = self.config.name
            handles = {
                "batch_size": metrics.batch_size.labels(model=name),
                "latency": metrics.request_latency.labels(model=name, backend=backend, mode=mode),
                "total": metrics.request_total.labels(model=name, backend=backend, mode=mode),
                "errors": metrics.request_errors.labels(model=name),
            }
            self._metric_handles[key] = handles
        return handles
