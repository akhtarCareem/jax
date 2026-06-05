ARG GIT_REF=main

FROM ghcr.io/astral-sh/uv:0.7 AS uv

FROM python:3.12-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG GIT_REF

RUN git clone https://github.com/akhtarCareem/jax.git /src \
    && git -C /src checkout ${GIT_REF}

RUN uv venv /app/.venv \
    && uv pip install --python /app/.venv "/src[cuda]"


FROM python:3.12-slim AS runtime

RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /src/configs/ /app/configs/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    JAX_SERVER_CONFIG=/app/configs/gpu-example.yaml \
    HF_HOME=/cache/hf \
    JAX_COMPILATION_CACHE_DIR=/cache/jax

RUN mkdir -p /cache && chown app:app /cache

EXPOSE 8080

USER app

CMD ["uvicorn", "jax_server.deploy.k8s_main:app", "--host", "0.0.0.0", "--port", "8080"]
