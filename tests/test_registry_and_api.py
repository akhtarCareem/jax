from fastapi.testclient import TestClient

from jax_server.config import AppConfig, ModelConfig
from jax_server.exceptions import ClientInputError, ExecutionError, ModelLoadError
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
        params_format="msgpack",
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
    assert response.headers["x-request-id"]


def test_request_id_header_is_preserved():
    model_config = ModelConfig(
        name="sample",
        hf_repo="org/repo",
        params_path="params/",
        params_format="msgpack",
    )
    registry = ModelRegistry()
    registry.add(FakeModel(model_config))
    client = TestClient(
        create_app(AppConfig(models=[model_config]), startup_enabled=False, registry=registry)
    )

    response = client.post(
        "/v1/models/sample:predict",
        json={"inputs": {"x": [1]}},
        headers={"X-Request-ID": "req-123"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"


def test_api_maps_input_errors_to_400():
    class BadRequestModel(FakeModel):
        def predict(self, inputs, backend="auto", mode="auto", metrics=None):
            raise ClientInputError("bad request")

    model_config = ModelConfig(
        name="sample",
        hf_repo="org/repo",
        params_path="params/",
        params_format="msgpack",
    )
    registry = ModelRegistry()
    registry.add(BadRequestModel(model_config))
    client = TestClient(create_app(AppConfig(models=[model_config]), startup_enabled=False, registry=registry))

    response = client.post("/v1/models/sample:predict", json={"inputs": {"x": [1]}})

    assert response.status_code == 400


def test_api_maps_model_load_errors_to_503():
    class UnloadedModel(FakeModel):
        def predict(self, inputs, backend="auto", mode="auto", metrics=None):
            raise ModelLoadError("not ready")

    model_config = ModelConfig(
        name="sample",
        hf_repo="org/repo",
        params_path="params/",
        params_format="msgpack",
    )
    registry = ModelRegistry()
    registry.add(UnloadedModel(model_config))
    client = TestClient(create_app(AppConfig(models=[model_config]), startup_enabled=False, registry=registry))

    response = client.post("/v1/models/sample:predict", json={"inputs": {"x": [1]}})

    assert response.status_code == 503


def test_api_maps_execution_errors_to_500():
    class BrokenModel(FakeModel):
        def predict(self, inputs, backend="auto", mode="auto", metrics=None):
            raise ExecutionError("runtime failure")

    model_config = ModelConfig(
        name="sample",
        hf_repo="org/repo",
        params_path="params/",
        params_format="msgpack",
    )
    registry = ModelRegistry()
    registry.add(BrokenModel(model_config))
    client = TestClient(create_app(AppConfig(models=[model_config]), startup_enabled=False, registry=registry))

    response = client.post("/v1/models/sample:predict", json={"inputs": {"x": [1]}})

    assert response.status_code == 500


def test_predict_does_not_require_auth_when_token_not_configured():
    model_config = ModelConfig(
        name="sample",
        hf_repo="org/repo",
        params_path="params/",
        params_format="msgpack",
    )
    registry = ModelRegistry()
    registry.add(FakeModel(model_config))
    client = TestClient(create_app(AppConfig(models=[model_config]), startup_enabled=False, registry=registry))

    response = client.post("/v1/models/sample:predict", json={"inputs": {"x": [1]}})

    assert response.status_code == 200


def test_predict_requires_bearer_token_when_configured(monkeypatch):
    monkeypatch.setenv("JAX_SERVER_AUTH_TOKEN", "secret-token")

    model_config = ModelConfig(
        name="sample",
        hf_repo="org/repo",
        params_path="params/",
        params_format="msgpack",
    )
    registry = ModelRegistry()
    registry.add(FakeModel(model_config))
    client = TestClient(create_app(AppConfig(models=[model_config]), startup_enabled=False, registry=registry))

    missing = client.post("/v1/models/sample:predict", json={"inputs": {"x": [1]}})
    wrong = client.post(
        "/v1/models/sample:predict",
        json={"inputs": {"x": [1]}},
        headers={"Authorization": "Bearer wrong-token"},
    )
    ok = client.post(
        "/v1/models/sample:predict",
        json={"inputs": {"x": [1]}},
        headers={"Authorization": "Bearer secret-token"},
    )

    assert missing.status_code == 401
    assert wrong.status_code == 401
    assert ok.status_code == 200
