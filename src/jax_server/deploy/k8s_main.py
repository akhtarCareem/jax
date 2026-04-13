from __future__ import annotations

import os

from jax_server.server.app import create_app


app = create_app(os.environ.get("JAX_SERVER_CONFIG", "configs/example.yaml"))
