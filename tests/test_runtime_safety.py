from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from jax_server.config import AppConfig, ModelConfig
from jax_server.exceptions import ClientInputError
from jax_server.server.admission import AdmissionGate
from jax_server.server.app import create_app
from jax_server.server.guards import validate_input_shape


def test_validate_input_shape_rejects_too_many_elements():
    with pytest.raises(ClientInputError) as exc_info:
        validate_input_shape({"x": list(range(6))}, max_elements=5, max_depth=10)

    assert "max_input_elements" in str(exc_info.value)


def test_validate_input_shape_rejects_too_deep():
    with pytest.raises(ClientInputError) as exc_info:
        validate_input_shape({"x": [[[[1]]]]}, max_elements=10, max_depth=3)

    assert "max_input_depth" in str(exc_info.value)


def test_admission_gate_rejects_when_full():
    gate = AdmissionGate(1)

    async def exercise():
        async with gate.acquire():
            with pytest.raises(ClientInputError) as exc_info:
                async with gate.acquire():
                    pass
            assert "concurrency capacity" in str(exc_info.value)

    asyncio.run(exercise())


def test_request_too_large_is_rejected():
    model_config = ModelConfig(
        name="sample",
        hf_repo="org/repo",
        params_path="params/",
        params_format="msgpack",
    )

    class FakeModel:
        def __init__(self, config):
            self.config = config

        def describe(self):
            return {"name": self.config.name, "loaded": True, "available_exports": []}

        def predict(self, inputs, backend="auto", mode="auto", metrics=None):
            return {
                "model": self.config.name,
                "backend": "cpu",
                "mode": "single",
                "outputs": {"ok": True},
            }

    from jax_server.runtime.registry import ModelRegistry

    registry = ModelRegistry()
    registry.add(FakeModel(model_config))
    app = create_app(
        AppConfig(models=[model_config], max_request_bytes=10),
        startup_enabled=False,
        registry=registry,
    )
    client = TestClient(app)

    response = client.post("/v1/models/sample:predict", json={"inputs": {"x": [1, 2, 3]}})

    assert response.status_code == 413
