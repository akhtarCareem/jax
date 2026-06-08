from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import numpy as np


def _canonical_numpy(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.float64:
        return arr.astype(np.float32, copy=False)
    if arr.dtype == np.int64:
        return arr.astype(np.int32, copy=False)
    return arr


def _to_numpy_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_numpy_tree(item) for item in value)
    if isinstance(value, list):
        if value and any(isinstance(item, Mapping) for item in value):
            return [_to_numpy_tree(item) for item in value]
        try:
            return _canonical_numpy(value)
        except (ValueError, TypeError):
            return [_to_numpy_tree(item) for item in value]
    return _canonical_numpy(value)


def to_jax_pytree(value: Any) -> Any:
    return jax.device_put(_to_numpy_tree(value))


def to_numpy_pytree(value: Any) -> Any:
    if isinstance(value, jax.Array):
        return np.asarray(value)
    if isinstance(value, Mapping):
        return {key: to_numpy_pytree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_numpy_pytree(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def infer_batch_size(value: Any) -> int:
    leaves = jax.tree_util.tree_leaves(value)

    sizes: set[int] = set()
    for leaf in leaves:
        shape = getattr(leaf, "shape", None)
        if shape and len(shape) > 0:
            sizes.add(int(shape[0]))
    if len(sizes) > 1:
        raise ValueError(f"Inconsistent batch sizes across pytree leaves: {sorted(sizes)}")
    return next(iter(sizes), 1)
