from pathlib import Path

import pytest

from jax_server.config import ModelConfig
from jax_server.exceptions import ModelLoadError
from jax_server.hf.store import ArtifactStore


def test_local_store_returns_local_path(tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    store = ArtifactStore(cache_dir=tmp_path / "cache")
    config = ModelConfig(
        name="local-model",
        source="local",
        local_path=str(model_dir),
        params_path="params",
        params_format="pickle",
    )

    assert store.fetch_snapshot(config) == model_dir.resolve()


def test_local_store_errors_for_missing_path(tmp_path):
    store = ArtifactStore(cache_dir=tmp_path / "cache")
    config = ModelConfig(
        name="local-model",
        source="local",
        local_path=str(tmp_path / "missing"),
        params_path="params",
        params_format="pickle",
    )

    with pytest.raises(ModelLoadError):
        store.fetch_snapshot(config)
