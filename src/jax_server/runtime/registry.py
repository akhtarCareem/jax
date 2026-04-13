from __future__ import annotations

from typing import Iterable

from jax_server.runtime.model import ServedModel


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, ServedModel] = {}

    def add(self, model: ServedModel) -> None:
        self._models[model.config.name] = model

    def get(self, name: str) -> ServedModel:
        return self._models[name]

    def all(self) -> Iterable[ServedModel]:
        return self._models.values()

    def names(self) -> list[str]:
        return sorted(self._models.keys())
