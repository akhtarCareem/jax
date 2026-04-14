from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic import model_validator


class ModelConfig(BaseModel):
    name: str
    source: Literal["hf", "local"] = "hf"
    hf_repo: str | None = None
    revision: str | None = None
    local_path: str | None = None
    params_path: str
    params_format: Literal["orbax_standard", "msgpack"]
    artifact_name: str | None = None
    default_platform: Literal["cpu", "gpu"] = "cpu"
    max_batch_size: int | None = None
    warmup_requests: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_source_fields(self) -> "ModelConfig":
        if self.source == "hf" and not self.hf_repo:
            raise ValueError("hf_repo is required when source='hf'.")
        if self.source == "local" and not self.local_path:
            raise ValueError("local_path is required when source='local'.")
        if self.max_batch_size is not None and self.max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1 when provided.")
        return self

    @property
    def artifact_prefix(self) -> str:
        return self.artifact_name or "model"


class AppConfig(BaseModel):
    models: list[ModelConfig]
    cache_dir: str | None = None
    warmup_enabled: bool = True
    allow_local_files_only: bool = False
    auth_token_env: str | None = "JAX_SERVER_AUTH_TOKEN"
    max_request_bytes: int = 1_048_576
    max_input_elements: int = 100_000
    max_input_depth: int = 8
    max_concurrent_requests_per_model: int = 8


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) or {}
    base_dir = config_path.parent.resolve()

    if "cache_dir" in data and data["cache_dir"] is not None:
        data["cache_dir"] = str((base_dir / data["cache_dir"]).resolve())

    for model in data.get("models", []):
        local_path = model.get("local_path")
        if local_path:
            model["local_path"] = str((base_dir / local_path).resolve())

    return AppConfig.model_validate(data)
