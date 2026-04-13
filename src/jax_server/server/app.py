from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import orjson
from fastapi import FastAPI, HTTPException

from jax_server.config import AppConfig, load_config
from jax_server.env import configure_jax_environment
from jax_server.exceptions import JaxServerError
from jax_server.hf.store import ArtifactStore
from jax_server.inference.convert import to_jsonable
from jax_server.metrics import Metrics
from jax_server.runtime.model import ServedModel
from jax_server.runtime.registry import ModelRegistry
from jax_server.server.schemas import PredictRequest, PredictResponse


class AppState:
    def __init__(self, config: AppConfig, registry: ModelRegistry, metrics: Metrics) -> None:
        self.config = config
        self.registry = registry
        self.metrics = metrics
        self.ready = False


def _ensure_config(config: AppConfig | str | Path) -> AppConfig:
    if isinstance(config, AppConfig):
        return config
    return load_config(config)


async def _load_models(app_state: AppState) -> None:
    configure_jax_environment(app_state.config.cache_dir)
    store = ArtifactStore(
        cache_dir=app_state.config.cache_dir,
        local_files_only=app_state.config.allow_local_files_only,
    )
    shared_params_cache: dict[tuple[str, str, str], Any] = {}

    for model_config in app_state.config.models:
        model = ServedModel(model_config)
        model.load(store=store, metrics=app_state.metrics, shared_params_cache=shared_params_cache)
        if app_state.config.warmup_enabled:
            for warmup_request in model_config.warmup_requests:
                prediction = model.predict(warmup_request, metrics=app_state.metrics)
                orjson.dumps(to_jsonable(prediction["outputs"]))
        app_state.registry.add(model)
    app_state.ready = True


async def load_app_state(app: FastAPI) -> None:
    await _load_models(app.state.jax_server)


def initialize_app_state(app: FastAPI) -> None:
    asyncio.run(load_app_state(app))


def create_app(
    config: AppConfig | str | Path,
    *,
    startup_enabled: bool = True,
    registry: ModelRegistry | None = None,
    metrics: Metrics | None = None,
) -> FastAPI:
    resolved_config = _ensure_config(config)
    resolved_registry = registry or ModelRegistry()
    resolved_metrics = metrics or Metrics()
    app_state = AppState(resolved_config, resolved_registry, resolved_metrics)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if startup_enabled:
            await _load_models(app_state)
        yield

    app = FastAPI(
        title="jax-server",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.jax_server = app_state

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        if not app.state.jax_server.ready and startup_enabled:
            raise HTTPException(status_code=503, detail="models not loaded")
        return {"status": "ready"}

    @app.get("/metrics")
    async def metrics_endpoint():
        return app.state.jax_server.metrics.prometheus_response()

    @app.get("/v1/models")
    async def list_models() -> dict[str, list[dict[str, Any]]]:
        models = [model.describe() for model in app.state.jax_server.registry.all()]
        return {"models": models}

    @app.get("/v1/models/{name}")
    async def get_model(name: str) -> dict[str, Any]:
        try:
            model = app.state.jax_server.registry.get(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown model: {name}") from exc
        return model.describe()

    @app.post("/v1/models/{name}:predict", response_model=PredictResponse)
    async def predict(name: str, request: PredictRequest) -> dict[str, Any]:
        try:
            model = app.state.jax_server.registry.get(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"unknown model: {name}") from exc

        try:
            result = model.predict(
                request.inputs,
                backend=request.backend,
                mode=request.mode,
                metrics=app.state.jax_server.metrics,
            )
        except JaxServerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return {
            "model": result["model"],
            "backend": result["backend"],
            "mode": result["mode"],
            "outputs": to_jsonable(result["outputs"]),
        }

    return app
