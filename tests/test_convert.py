import pytest
import numpy as np

from jax_server.inference.convert import to_jax_pytree


def test_to_jax_pytree_turns_nested_numeric_lists_into_arrays():
    pytest.importorskip("jax")
    converted = to_jax_pytree({"features": [[0.1, 0.2], [0.3, 0.4]]})

    assert np.asarray(converted["features"]).shape == (2, 2)
