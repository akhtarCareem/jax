from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

from jax_server.config import ModelConfig
from jax_server.exceptions import ModelLoadError


class ArtifactStore:
    def __init__(self, cache_dir: str | Path, local_files_only: bool = False) -> None:
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

        snapshot_path = snapshot_download(
            repo_id=model_config.hf_repo,
            revision=model_config.revision,
            cache_dir=str(self.cache_dir / "hf"),
            local_files_only=self.local_files_only,
        )
        return Path(snapshot_path)


HFArtifactStore = ArtifactStore
