from jax_server.artifacts.discovery import artifact_filename, discover_artifacts


def test_artifact_filename():
    assert artifact_filename("tower", "cpu", "single") == "tower_cpu.bin"
    assert artifact_filename("tower", "gpu", "batch") == "tower_gpu_batch.bin"


def test_discover_artifacts(tmp_path):
    (tmp_path / "tower_cpu.bin").write_bytes(b"cpu")
    (tmp_path / "tower_gpu_batch.bin").write_bytes(b"gpu-batch")

    found = discover_artifacts(tmp_path, "tower")

    assert ("cpu", "single") in found
    assert ("gpu", "batch") in found
