from __future__ import annotations

import modal

CONFIG_FILE = "configs/hf-example.yaml"
SECOND = 1
MINUTE = 60 * SECOND

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .uv_pip_install(
        "jax-server[jax] @ git+https://github.com/dmitryBe/jax-server.git@main"
    )
    .add_local_file(CONFIG_FILE, f"/root/{CONFIG_FILE}", copy=True)
    .env({"JAX_SERVER_CONFIG": f"/root/{CONFIG_FILE}"})
)

app = modal.App("jax-server-example")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    enable_memory_snapshot=True,
    max_containers=3,
    scaledown_window=1 * MINUTE,
    timeout=100 * SECOND,
)
@modal.asgi_app(label="jax-server-example")
def fastapi_app():
    from jax_server.server.app import create_app

    return create_app("/root/configs/hf-example.yaml")
