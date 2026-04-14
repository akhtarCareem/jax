from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from jax_server.server.app import create_app


def _build_export_bundle(output_dir: Path) -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    ocp = pytest.importorskip("orbax.checkpoint")
    export = jax.export

    params = {
        "w": jnp.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, -1.0],
                [0.5, 0.5],
            ],
            dtype=jnp.float32,
        ),
        "b": jnp.asarray([0.25, -0.25], dtype=jnp.float32),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    params_dir = output_dir / "params"
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(params_dir, params)
    checkpointer.wait_until_finished()

    def serving_fn(model_params, inputs):
        return inputs["features"] @ model_params["w"] + model_params["b"]

    single_spec = {"features": jax.ShapeDtypeStruct((1, 4), jnp.float32)}
    single_exported = export.export(jax.jit(serving_fn))(params, single_spec)
    (output_dir / "test_model_cpu.bin").write_bytes(single_exported.serialize())

    batch_dim, = export.symbolic_shape("b,")
    batch_spec = {"features": jax.ShapeDtypeStruct((batch_dim, 4), jnp.float32)}
    batch_exported = export.export(jax.jit(serving_fn))(params, batch_spec)
    (output_dir / "test_model_cpu_batch.bin").write_bytes(batch_exported.serialize())


def test_local_runtime_serves_real_exported_model(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("orbax.checkpoint")

    model_dir = tmp_path / "bundle"
    _build_export_bundle(model_dir)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
models:
  - name: linear
    source: local
    local_path: {model_dir}
    params_path: params
    params_format: orbax_standard
    artifact_name: test_model
    default_platform: cpu
    max_batch_size: 8
cache_dir: {tmp_path / "cache"}
"""
    )

    with TestClient(create_app(config_path)) as client:
        single = client.post(
            "/v1/models/linear:predict",
            json={"inputs": {"features": [[1.0, 2.0, 3.0, 4.0]]}},
        )
        batch = client.post(
            "/v1/models/linear:predict",
            json={
                "inputs": {
                    "features": [
                        [1.0, 2.0, 3.0, 4.0],
                        [0.0, 1.0, 0.0, 1.0],
                    ]
                }
            },
        )

    assert single.status_code == 200
    assert single.json()["mode"] == "single"
    assert single.json()["outputs"] == [[6.25, 0.75]]

    assert batch.status_code == 200
    assert batch.json()["mode"] == "batch"
    assert batch.json()["outputs"] == [[6.25, 0.75], [0.75, 1.25]]


def test_local_runtime_rejects_inconsistent_batch_sizes(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("orbax.checkpoint")

    model_dir = tmp_path / "bundle"
    _build_export_bundle(model_dir)

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
models:
  - name: linear
    source: local
    local_path: {model_dir}
    params_path: params
    params_format: orbax_standard
    artifact_name: test_model
    default_platform: cpu
    max_batch_size: 8
cache_dir: {tmp_path / "cache"}
"""
    )

    with TestClient(create_app(config_path)) as client:
        response = client.post(
            "/v1/models/linear:predict",
            json={
                "inputs": {
                    "left": [[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]],
                    "right": [[1.0, 2.0, 3.0, 4.0]],
                }
            },
        )

    assert response.status_code == 400
    assert "Inconsistent batch sizes" in response.json()["detail"]
