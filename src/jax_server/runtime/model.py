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
from jax_server.inference.convert import infer_batch_size, to_jax_pytree
from jax_server.metrics import Metrics

logger = logging.getLogger("jax_server.runtime")
from jax_server.runtime.exported import ExportedFunction
from jax_server.runtime.params import load_params


class ServedModel:
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.params: Any | None = None
        self.exports: dict[tuple[str, str], ExportedFunction] = {}
        self.snapshot_path: Path | None = None
        self.loaded = False

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
            self.params = load_params(params_path, self.config.params_format)
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

        converted_inputs = to_jax_pytree(inputs)
        try:
            batch_size = infer_batch_size(converted_inputs)
        except ValueError as exc:
            raise ClientInputError(str(exc)) from exc

        resolved_mode = self._resolve_mode(mode, batch_size)
        resolved_backend = self._resolve_backend(backend, resolved_mode, metrics)
        exported = self.exports[(resolved_backend, resolved_mode)]

        timer = None
        if metrics is not None:
            metrics.batch_size.labels(model=self.config.name).observe(batch_size)
            timer = metrics.request_latency.labels(
                model=self.config.name,
                backend=resolved_backend,
                mode=resolved_mode,
            ).time()
            timer.__enter__()

        try:
            outputs = exported.call(self.params, converted_inputs)
        except Exception as exc:
            if metrics is not None:
                metrics.request_errors.labels(model=self.config.name).inc()
            logger.exception(
                "inference error model=%s backend=%s mode=%s",
                self.config.name, resolved_backend, resolved_mode,
            )
            raise ExecutionError(
                f"Prediction failed for model '{self.config.name}'."
            ) from exc
        finally:
            if timer is not None:
                timer.__exit__(None, None, None)

        if metrics is not None:
            metrics.request_total.labels(
                model=self.config.name,
                backend=resolved_backend,
                mode=resolved_mode,
            ).inc()

        return {
            "model": self.config.name,
            "backend": resolved_backend,
            "mode": resolved_mode,
            "outputs": outputs,
        }
