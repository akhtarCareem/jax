from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal


ParamsFormat = Literal["orbax_standard", "pickle", "msgpack"]


def load_params(path: str | Path, params_format: ParamsFormat) -> Any:
    params_path = Path(path)
    if params_format == "pickle":
        with params_path.open("rb") as handle:
            return pickle.load(handle)
    if params_format == "msgpack":
        try:
            import msgpack
        except ImportError as exc:
            raise RuntimeError("msgpack dependency is required for msgpack params.") from exc
        with params_path.open("rb") as handle:
            return msgpack.unpackb(handle.read(), raw=False)
    if params_format == "orbax_standard":
        try:
            import orbax.checkpoint as ocp
        except ImportError as exc:
            raise RuntimeError(
                "orbax-checkpoint dependency is required for orbax_standard params."
            ) from exc
        checkpointer = ocp.StandardCheckpointer()
        return checkpointer.restore(params_path)
    raise ValueError(f"Unsupported params format: {params_format}")
