from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


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
        return {"out": inputs["features"] @ model_params["w"] + model_params["b"]}

    single_spec = {"features": jax.ShapeDtypeStruct((1, 4), jnp.float32)}
    single_exported = export.export(jax.jit(serving_fn))(params, single_spec)
    (output_dir / "test_model_cpu.bin").write_bytes(single_exported.serialize())

    batch_dim, = export.symbolic_shape("b,")
    batch_spec = {"features": jax.ShapeDtypeStruct((batch_dim, 4), jnp.float32)}
    batch_exported = export.export(jax.jit(serving_fn))(params, batch_spec)
    (output_dir / "test_model_cpu_batch.bin").write_bytes(batch_exported.serialize())


def _load_model(model_dir: Path, tmp_path: Path):
    from jax_server.config import load_config
    from jax_server.runtime.bootstrap import build_models

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
    config = load_config(config_path)
    models = build_models(config)
    return models[0]


def test_local_runtime_serves_real_exported_model(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("orbax.checkpoint")

    model_dir = tmp_path / "bundle"
    _build_export_bundle(model_dir)
    model = _load_model(model_dir, tmp_path)

    single = model.predict({"features": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)})
    batch = model.predict(
        {"features": np.array([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 0.0, 1.0]], dtype=np.float32)}
    )

    assert single["mode"] == "single"
    assert np.allclose(single["outputs"]["out"], [[6.25, 0.75]], atol=1e-5)

    assert batch["mode"] == "batch"
    assert np.allclose(batch["outputs"]["out"], [[6.25, 0.75], [0.75, 1.25]], atol=1e-5)


def test_local_runtime_rejects_inconsistent_batch_sizes(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("orbax.checkpoint")

    model_dir = tmp_path / "bundle"
    _build_export_bundle(model_dir)
    model = _load_model(model_dir, tmp_path)

    from jax_server.exceptions import ClientInputError

    with pytest.raises(ClientInputError, match="Inconsistent batch sizes"):
        model.predict(
            {
                "left": np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 1.0, 1.0, 1.0]], dtype=np.float32),
                "right": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32),
            }
        )
