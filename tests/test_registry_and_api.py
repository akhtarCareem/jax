from fastapi.testclient import TestClient

from jax_server.config import AppConfig, ModelConfig
from jax_server.runtime.registry import ModelRegistry
from jax_server.server.app import create_app


class FakeModel:
    def __init__(self, config):
        self.config = config

    def describe(self):
        return {
            "name": self.config.name,
            "loaded": True,
            "available_exports": [{"backend": "cpu", "mode": "single"}],
        }

    def predict(self, inputs, backend="auto", mode="auto", metrics=None):
        return {
            "model": self.config.name,
            "backend": "cpu",
            "mode": "single",
            "outputs": {"echo": inputs},
        }


def test_api_endpoints():
    model_config = ModelConfig(
        name="sample",
        hf_repo="org/repo",
        params_path="params/",
        params_format="pickle",
    )
    registry = ModelRegistry()
    registry.add(FakeModel(model_config))
    app = create_app(
        AppConfig(models=[model_config]),
        startup_enabled=False,
        registry=registry,
    )

    client = TestClient(app)

    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json() == {"status": "ready"}
    assert client.get("/v1/models").json()["models"][0]["name"] == "sample"

    response = client.post("/v1/models/sample:predict", json={"inputs": {"x": [1, 2, 3]}})
    assert response.status_code == 200
    assert response.json()["outputs"] == {"echo": {"x": [1, 2, 3]}}
