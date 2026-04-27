from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

ParamsFormat = Literal["orbax_standard", "msgpack"]


def load_params(path: str | Path, params_format: ParamsFormat) -> Any:
    params_path = Path(path)
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
        try:
            import jax
        except ImportError as exc:
            raise RuntimeError("jax dependency is required for orbax_standard params.") from exc
        devices = jax.local_devices()
        fallback_sharding = jax.sharding.SingleDeviceSharding(devices[0]) if devices else None
        checkpointer = ocp.Checkpointer(ocp.StandardCheckpointHandler())
        return checkpointer.restore(
            params_path,
            args=ocp.args.StandardRestore(fallback_sharding=fallback_sharding),
        )
    raise ValueError(f"Unsupported params format: {params_format}")
