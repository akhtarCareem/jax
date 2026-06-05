from __future__ import annotations

import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from jax_server.server.app import create_app


app = create_app(os.environ.get("JAX_SERVER_CONFIG", "configs/example.yaml"))
