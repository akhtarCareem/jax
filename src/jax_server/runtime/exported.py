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
