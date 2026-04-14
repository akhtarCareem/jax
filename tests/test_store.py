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
        params_format="msgpack",
    )

    assert store.fetch_snapshot(config) == model_dir.resolve()


def test_local_store_errors_for_missing_path(tmp_path):
    store = ArtifactStore(cache_dir=tmp_path / "cache")
    config = ModelConfig(
        name="local-model",
        source="local",
        local_path=str(tmp_path / "missing"),
        params_path="params",
        params_format="msgpack",
    )

    with pytest.raises(ModelLoadError):
        store.fetch_snapshot(config)


def test_hf_store_restricts_snapshot_patterns(tmp_path, monkeypatch):
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        snapshot_dir = tmp_path / "snapshot"
        snapshot_dir.mkdir()
        return str(snapshot_dir)

    monkeypatch.setattr("jax_server.hf.store.snapshot_download", fake_snapshot_download)

    store = ArtifactStore(cache_dir=tmp_path / "cache")
    config = ModelConfig(
        name="hf-model",
        source="hf",
        hf_repo="org/repo",
        params_path="params",
        params_format="orbax_standard",
        artifact_name="tower",
    )

    snapshot_path = store.fetch_snapshot(config)

    assert snapshot_path == (tmp_path / "snapshot")
    assert calls["repo_id"] == "org/repo"
    assert sorted(calls["allow_patterns"]) == sorted(
        [
            "tower_cpu.bin",
            "tower_cpu_batch.bin",
            "tower_gpu.bin",
            "tower_gpu_batch.bin",
            "params",
            "params/*",
            "params/**",
        ]
    )
