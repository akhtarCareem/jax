from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import orjson
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi import FastAPI, Header, HTTPException, Request

from jax_server.config import AppConfig, load_config
from jax_server.env import configure_jax_environment
from jax_server.exceptions import ClientInputError, ExecutionError, JaxServerError, ModelLoadError
from jax_server.hf.store import ArtifactStore
from jax_server.inference.convert import to_jsonable
from jax_server.metrics import Metrics
from jax_server.runtime.model import ServedModel
from jax_server.runtime.registry import ModelRegistry
from jax_server.server.admission import AdmissionGate
from jax_server.server.guards import validate_input_shape
from jax_server.server.schemas import PredictRequest, PredictResponse

logger = logging.getLogger("jax_server.server")


class AppState:
    def __init__(self, config: AppConfig, registry: ModelRegistry, metrics: Metrics) -> None:
        self.config = config
        self.registry = registry
        self.metrics = metrics
        self.ready = False
        self.auth_token = (
            os.environ.get(config.auth_token_env) if config.auth_token_env else None
        )
        self.request_gates: dict[str, AdmissionGate] = {}


def _ensure_config(config: AppConfig | str | Path) -> AppConfig:
    if isinstance(config, AppConfig):
        return config
    return load_config(config)


async def load_registry(app_state: AppState) -> None:
    configure_jax_environment(app_state.config.cache_dir)
    store = ArtifactStore(
        cache_dir=app_state.config.cache_dir,
        local_files_only=app_state.config.allow_local_files_only,
    )
    shared_params_cache: dict[tuple[str, str, str], Any] = {}

    for model_config in app_state.config.models:
        model = ServedModel(model_config)
        started_at = time.perf_counter()
        model.load(
            store=store,
            metrics=app_state.metrics,
            shared_params_cache=shared_params_cache,
        )
        if app_state.config.warmup_enabled:
            for warmup_request in model_config.warmup_requests:
                prediction = model.predict(warmup_request, metrics=None)
                orjson.dumps(to_jsonable(prediction["outputs"]))
        app_state.registry.add(model)
        app_state.request_gates[model_config.name] = AdmissionGate(
            app_state.config.max_concurrent_requests_per_model
        )
        _log_event(
            "model_loaded",
            model=model_config.name,
            source=model_config.source,
            hf_repo=model_config.hf_repo,
            artifact_name=model_config.artifact_prefix,
            load_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )
    app_state.ready = True


async def load_app_state(app: FastAPI) -> None:
    await load_registry(app.state.jax_server)


def initialize_app_state(app: FastAPI) -> None:
    asyncio.run(load_app_state(app))


def _log_event(event: str, **fields: Any) -> None:
    logger.info(json.dumps({"event": event, **fields}, sort_keys=True, default=str))


def _authorize_inference(app: FastAPI, authorization: str | None) -> None:
    expected_token = app.state.jax_server.auth_token
    if not expected_token:
        return

    scheme, _, token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not hmac.compare_digest(token, expected_token):
        raise HTTPException(
            status_code=401,
            detail="invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_app(
    config: AppConfig | str | Path,
    *,
    startup_enabled: bool = True,
    registry: ModelRegistry | None = None,
    metrics: Metrics | None = None,
) -> FastAPI:
    """Create the FastAPI app.

    By default, model state is loaded during FastAPI lifespan startup.
    Set `startup_enabled=False` only when the caller will preload state
    manually via `load_app_state(...)`, such as snapshot-oriented deploy flows.
    """
    resolved_config = _ensure_config(config)
    resolved_registry = registry or ModelRegistry()
    resolved_metrics = metrics or Metrics()
    app_state = AppState(resolved_config, resolved_registry, resolved_metrics)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if startup_enabled:
            await load_registry(app_state)
        yield

    app = FastAPI(
        title="jax-server",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.jax_server = app_state
    for existing_model in app_state.registry.all():
        app_state.request_gates.setdefault(
            existing_model.config.name,
            AdmissionGate(app_state.config.max_concurrent_requests_per_model),
        )

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > app.state.jax_server.config.max_request_bytes:
            return JSONResponse({"detail": "request body too large"}, status_code=413)
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

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
    async def predict(
        request_http: Request,
        name: str,
        request: PredictRequest,
        authorization: str | None = Header(default=None),
    ) -> dict[str, Any]:
        request_id = request_http.state.request_id
        _authorize_inference(app, authorization)
        validate_input_shape(
            request.inputs,
            max_elements=app.state.jax_server.config.max_input_elements,
            max_depth=app.state.jax_server.config.max_input_depth,
        )
        try:
            model = app.state.jax_server.registry.get(name)
        except KeyError as exc:
            _log_event("predict_missing_model", request_id=request_id, model=name)
            raise HTTPException(status_code=404, detail=f"unknown model: {name}") from exc

        started_at = time.perf_counter()
        try:
            gate = app.state.jax_server.request_gates[name]
            async with gate.acquire():
                result = await run_in_threadpool(
                    model.predict,
                    request.inputs,
                    request.backend,
                    request.mode,
                    app.state.jax_server.metrics,
                )
        except ClientInputError as exc:
            _log_event(
                "predict_rejected",
                request_id=request_id,
                model=name,
                requested_backend=request.backend,
                requested_mode=request.mode,
                error=str(exc),
            )
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ModelLoadError as exc:
            _log_event(
                "predict_model_unavailable",
                request_id=request_id,
                model=name,
                error=str(exc),
            )
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except ExecutionError as exc:
            _log_event(
                "predict_failed",
                request_id=request_id,
                model=name,
                requested_backend=request.backend,
                requested_mode=request.mode,
                error=str(exc),
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except JaxServerError as exc:
            _log_event(
                "predict_failed",
                request_id=request_id,
                model=name,
                requested_backend=request.backend,
                requested_mode=request.mode,
                error=str(exc),
            )
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        _log_event(
            "predict_succeeded",
            request_id=request_id,
            model=result["model"],
            backend=result["backend"],
            mode=result["mode"],
            latency_ms=round((time.perf_counter() - started_at) * 1000, 3),
        )

        return {
            "model": result["model"],
            "backend": result["backend"],
            "mode": result["mode"],
            "outputs": to_jsonable(result["outputs"]),
        }

    return app
