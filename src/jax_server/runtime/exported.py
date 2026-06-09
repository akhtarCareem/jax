from __future__ import annotations

from pathlib import Path
from typing import Any


class ExportedFunction:
    def __init__(self, path: Path, exported: Any) -> None:
        self.path = path
        self.exported = exported

    @classmethod
    def from_file(cls, path: str | Path) -> "ExportedFunction":
        try:
            from jax import export
        except ImportError as exc:
            raise RuntimeError("jax is required to deserialize exported artifacts.") from exc

        artifact_path = Path(path)
        payload = artifact_path.read_bytes()
        exported = export.deserialize(payload)
        return cls(path=artifact_path, exported=exported)

    def call(self, params: Any, inputs: Any) -> Any:
        return self.exported.call(params, inputs)


def derive_triton_io(exported: Any) -> tuple[dict, dict]:
    """Derive Triton Tensor specs from a jax.export.Exported object.

    Returns (inputs_spec, outputs_spec) mapping tensor name to (np.dtype, shape).
    Axis 0 is set to -1 (dynamic batch dim). Symbolic non-leading dims become -1.
    """
    import jax
    import numpy as np

    args, kwargs = jax.tree_util.tree_unflatten(
        exported.in_tree, list(exported.in_avals)
    )
    if kwargs:
        raise ValueError(f"Unexpected kwargs in export signature: {list(kwargs)}")
    if len(args) != 2:
        raise ValueError(f"Expected (params, inputs); got {len(args)} positional args")
    _params_struct, inputs_struct = args
    if not isinstance(inputs_struct, dict):
        raise ValueError(f"inputs arg is not a dict: {type(inputs_struct)}")

    out_struct = jax.tree_util.tree_unflatten(
        exported.out_tree, list(exported.out_avals)
    )
    if not isinstance(out_struct, dict):
        raise ValueError(f"outputs are not a dict: {type(out_struct)}")

    def _spec(aval):
        dims = [-1 if (i == 0 or not isinstance(d, int)) else d for i, d in enumerate(aval.shape)]
        return (np.dtype(aval.dtype), tuple(dims))

    return (
        {name: _spec(av) for name, av in inputs_struct.items()},
        {name: _spec(av) for name, av in out_struct.items()},
    )
