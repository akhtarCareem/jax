from __future__ import annotations

import os

from jax_server.server.app import create_app


def fastapi_app():
    return create_app(os.environ.get("JAX_SERVER_CONFIG", "configs/example.yaml"))
