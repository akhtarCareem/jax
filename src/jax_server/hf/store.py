from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from jax_server.config import ModelConfig
from jax_server.exceptions import ModelLoadError


def _artifact_patterns(model_config: ModelConfig) -> list[str]:
    artifact_prefix = model_config.artifact_prefix
    params_path = model_config.params_path.rstrip("/")
    patterns = [
        f"{artifact_prefix}_cpu.bin",
        f"{artifact_prefix}_cpu_batch.bin",
        f"{artifact_prefix}_gpu.bin",
        f"{artifact_prefix}_gpu_batch.bin",
        model_config.params_path,
    ]
    if params_path:
        patterns.extend([f"{params_path}/*", f"{params_path}/**"])
    return patterns


class ArtifactStore:
    def __init__(self, cache_dir: str | Path | None, local_files_only: bool = False) -> None:
        self.cache_dir = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir).expanduser().resolve()
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.local_files_only = local_files_only

    def fetch_snapshot(self, model_config: ModelConfig) -> Path:
        if model_config.source == "local":
            if model_config.local_path is None:
                raise ModelLoadError(
                    f"Model '{model_config.name}' is configured for local loading without a local_path."
                )
            local_path = Path(model_config.local_path).expanduser().resolve()
            if not local_path.exists():
                raise ModelLoadError(
                    f"Local model path does not exist for model '{model_config.name}': {local_path}"
                )
            return local_path

        snapshot_kwargs = dict(
            repo_id=model_config.hf_repo,
            revision=model_config.revision,
            local_files_only=self.local_files_only,
            allow_patterns=_artifact_patterns(model_config),
        )
        if self.cache_dir is not None:
            snapshot_kwargs["cache_dir"] = str(self.cache_dir / "hf")
        snapshot_path = snapshot_download(**snapshot_kwargs)
        return Path(snapshot_path)
