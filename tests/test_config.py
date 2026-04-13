from jax_server.config import load_config


def test_load_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
models:
  - name: sample
    source: hf
    hf_repo: org/repo
    params_path: params/
    params_format: pickle
"""
    )

    config = load_config(config_path)

    assert config.cache_dir == ".cache/jax-server"
    assert config.models[0].artifact_prefix == "model"


def test_load_config_resolves_local_paths(tmp_path):
    model_dir = tmp_path / "artifacts"
    model_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
models:
  - name: local-sample
    source: local
    local_path: ./artifacts
    params_path: params/
    params_format: pickle
cache_dir: .cache/test-cache
"""
    )

    config = load_config(config_path)

    assert config.models[0].local_path == str(model_dir.resolve())
    assert config.cache_dir == str((tmp_path / ".cache/test-cache").resolve())
