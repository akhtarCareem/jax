from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from jax_server.config import AppConfig, load_config
from jax_server.env import configure_jax_environment
from jax_server.hf.store import ArtifactStore
from jax_server.runtime.model import ServedModel

logger = logging.getLogger("jax_server.runtime")


def build_models(config: AppConfig | str | Path) -> list[ServedModel]:
    if not isinstance(config, AppConfig):
        config = load_config(config)

    configure_jax_environment(config.cache_dir)
    store = ArtifactStore(
        cache_dir=config.cache_dir,
        local_files_only=config.allow_local_files_only,
    )
    shared_params_cache: dict[tuple[str, str, str], Any] = {}
    models: list[ServedModel] = []

    for model_config in config.models:
        model = ServedModel(model_config)
        started_at = time.perf_counter()
        model.load(store=store, metrics=None, shared_params_cache=shared_params_cache)
        if config.warmup_enabled:
            for warmup_request in model_config.warmup_requests:
                model.predict(warmup_request, metrics=None)
        models.append(model)
        logger.info(
            "loaded model=%s load_ms=%.3f",
            model_config.name,
            (time.perf_counter() - started_at) * 1000,
        )

    return models
