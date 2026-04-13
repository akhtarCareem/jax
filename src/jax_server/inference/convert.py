from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _import_jax_numpy():
    try:
        import jax.numpy as jnp
    except ImportError as exc:
        raise RuntimeError("jax is required for inference conversion.") from exc
    return jnp


def to_jax_pytree(value: Any) -> Any:
    jnp = _import_jax_numpy()

    if isinstance(value, Mapping):
        return {key: to_jax_pytree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(to_jax_pytree(item) for item in value)
    if isinstance(value, list):
        if value and any(isinstance(item, Mapping) for item in value):
            return [to_jax_pytree(item) for item in value]
        try:
            return jnp.asarray(value)
        except Exception:
            return [to_jax_pytree(item) for item in value]
    return jnp.asarray(value)


def _array_to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def to_jsonable(value: Any) -> Any:
    try:
        import jax
    except ImportError:
        jax = None

    if jax is not None and isinstance(value, jax.Array):
        return np.asarray(value).tolist()
    if isinstance(value, Mapping):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return _array_to_jsonable(value)


def infer_batch_size(value: Any) -> int:
    try:
        import jax
        leaves = jax.tree_util.tree_leaves(value)
    except ImportError:
        leaves = [value]

    sizes: list[int] = []
    for leaf in leaves:
        shape = getattr(leaf, "shape", None)
        if shape:
            sizes.append(int(shape[0]))
    return max(sizes, default=1)
