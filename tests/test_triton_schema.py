from __future__ import annotations

import numpy as np
import pytest


def _build_export(features_shape):
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    export = jax.export

    params = {"w": jnp.ones((4, 2), dtype=jnp.float32)}

    def fn(model_params, inputs):
        return {"score": inputs["features"].sum(axis=-1, keepdims=True)}

    spec = {"features": jax.ShapeDtypeStruct(features_shape, jnp.float32)}
    exported = export.export(jax.jit(fn))(params, spec)
    return exported


@pytest.mark.parametrize("features_shape,expected_leading", [
    ((1, 4), -1),
    ((8, 4), -1),
])
def test_derive_triton_io_recovers_names_dtypes_and_dynamic_leading_dim(features_shape, expected_leading):
    from jax_server.runtime.exported import derive_triton_io

    exported = _build_export(features_shape)
    inputs_spec, outputs_spec = derive_triton_io(exported)

    assert "features" in inputs_spec
    in_dtype, in_shape = inputs_spec["features"]
    assert in_dtype == np.dtype("float32")
    assert in_shape[0] == expected_leading
    assert in_shape[1:] == (4,)

    assert "score" in outputs_spec
    out_dtype, out_shape = outputs_spec["score"]
    assert out_dtype == np.dtype("float32")
    assert out_shape[0] == expected_leading


def test_derive_triton_io_trailing_dims_consistent_across_single_and_batch():
    from jax_server.runtime.exported import derive_triton_io

    jax = pytest.importorskip("jax")

    single_export = _build_export((1, 4))
    batch_dim, = jax.export.symbolic_shape("b,")

    jnp = jax.numpy
    params = {"w": jnp.ones((4, 2), dtype=jnp.float32)}

    def fn(model_params, inputs):
        return {"score": inputs["features"].sum(axis=-1, keepdims=True)}

    batch_spec = {"features": jax.ShapeDtypeStruct((batch_dim, 4), jnp.float32)}
    batch_export = jax.export.export(jax.jit(fn))(params, batch_spec)

    _, single_out = derive_triton_io(single_export)
    _, batch_out = derive_triton_io(batch_export)

    assert single_out["score"][0] == batch_out["score"][0]
    assert single_out["score"][1:] == batch_out["score"][1:]
