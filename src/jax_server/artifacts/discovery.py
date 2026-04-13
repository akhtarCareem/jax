from __future__ import annotations

from pathlib import Path

from jax_server.exceptions import ArtifactNotFoundError

Platform = str
Mode = str


def artifact_filename(prefix: str, platform: Platform, mode: Mode) -> str:
    suffix = f"_{platform}"
    if mode == "batch":
        suffix += "_batch"
    return f"{prefix}{suffix}.bin"


def discover_artifacts(root: Path, artifact_prefix: str) -> dict[tuple[Platform, Mode], Path]:
    found: dict[tuple[Platform, Mode], Path] = {}
    for platform in ("cpu", "gpu"):
        for mode in ("single", "batch"):
            path = root / artifact_filename(artifact_prefix, platform, mode)
            if path.exists():
                found[(platform, mode)] = path
    if not found:
        raise ArtifactNotFoundError(
            f"No artifacts found for prefix '{artifact_prefix}' in '{root}'."
        )
    return found
