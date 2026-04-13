from __future__ import annotations

import asyncio

import modal

CONFIG_FILE = "configs/hf-example.yaml"
SECOND = 1
MINUTE = 60 * SECOND
USE_MEMORY_SNAPSHOT = True

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


@app.cls(
    image=image,
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    enable_memory_snapshot=USE_MEMORY_SNAPSHOT,
    max_containers=3,
    scaledown_window=5 * MINUTE,
    timeout=10 * MINUTE,
    startup_timeout=10 * MINUTE,
)
class JaxServer:
    @modal.enter(snap=USE_MEMORY_SNAPSHOT)
    def setup(self):
        from jax_server.server.app import _load_models, create_app

        self.web_app = create_app("/root/configs/hf-example.yaml", startup_enabled=False)
        asyncio.run(_load_models(self.web_app.state.jax_server))

    @modal.asgi_app(label="jax-server-example")
    def fastapi_app(self):
        return self.web_app
