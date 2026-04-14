from __future__ import annotations

import asyncio

import modal

CONFIG_FILE = "configs/hf-example.yaml"
SECOND = 1
MINUTE = 60 * SECOND
USE_MEMORY_SNAPSHOT = True
JAX_SERVER_GIT_REF = "a348ac3f4fde0f51f349f779d7aa9fded0c8d718"
PERSIST_ROOT = "/persist_vol"
HF_HOME = f"{PERSIST_ROOT}/.hf"
JAX_COMPILATION_CACHE_DIR = f"{PERSIST_ROOT}/.jax_compilation_cache"


image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .uv_pip_install(
        f"jax-server[jax] @ git+https://github.com/dmitryBe/jax-server.git@{JAX_SERVER_GIT_REF}"
    )
    .add_local_file(CONFIG_FILE, f"/root/{CONFIG_FILE}", copy=True)
    .env(
        {
            "JAX_SERVER_CONFIG": f"/root/{CONFIG_FILE}",
            "HF_HOME": HF_HOME,
            "JAX_COMPILATION_CACHE_DIR": JAX_COMPILATION_CACHE_DIR,
        }
    )
)

volume = modal.Volume.from_name("jax-server", create_if_missing=True)

app = modal.App(
    "jax-server-example",
    image=image,
    secrets=[modal.Secret.from_name("huggingface-token", required_keys=["HF_TOKEN"])],
    volumes={PERSIST_ROOT: volume},
)

@app.cls(
    enable_memory_snapshot=USE_MEMORY_SNAPSHOT,
    max_containers=3,
    scaledown_window=3 * MINUTE,
    timeout=10 * MINUTE,
    startup_timeout=10 * MINUTE,
)
class JaxServer:
    @modal.enter(snap=USE_MEMORY_SNAPSHOT)
    def setup(self):
        from jax_server.server.app import create_app, load_app_state

        self.web_app = create_app(f"/root/{CONFIG_FILE}", startup_enabled=False)
        asyncio.run(load_app_state(self.web_app))
        volume.commit()

    @modal.asgi_app(label="jax-server-example")
    def fastapi_app(self):
        return self.web_app
