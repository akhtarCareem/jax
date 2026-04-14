from __future__ import annotations

import os
from pathlib import Path


def configure_jax_environment(cache_dir: str | Path | None) -> None:
    if cache_dir is None:
        return
    cache_path = Path(cache_dir).expanduser().resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache_path / "compilation"))


def normalize_platform(platform: str) -> str:
    lowered = platform.lower()
    if lowered in {"cuda", "rocm"}:
        return "gpu"
    return lowered
