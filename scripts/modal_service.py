from __future__ import annotations

import modal


image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .uv_pip_install(
        "jax-server[jax] @ git+https://github.com/dmitryBe/jax-server.git@main"
    )
    .add_local_file(
        "configs/hf-example.yaml", "/root/configs/hf-example.yaml", copy=True
    )
    .env({"JAX_SERVER_CONFIG": "/root/configs/hf-example.yaml"})
)

app = modal.App("jax-server-example")


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
)
@modal.asgi_app(label="jax-server-example")
def fastapi_app():
    from jax_server.server.app import create_app

    return create_app("/root/configs/hf-example.yaml")
