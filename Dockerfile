ARG GIT_REF=2a7cdce10cb5173dc4c443de18a0d97d809a3be9

FROM ghcr.io/astral-sh/uv:0.7 AS uv

FROM python:3.12-slim AS builder

COPY --from=uv /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

ARG GIT_REF

RUN git clone https://github.com/akhtarCareem/jax.git /src \
    && git -C /src checkout ${GIT_REF}

RUN uv venv /app/.venv \
    && uv pip install --python /app/.venv "/src[cuda]" nvidia-pytriton


FROM python:3.12-slim AS runtime

# nvidia-pytriton bundles tritonserver, which dynamically links libs absent from -slim.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libnuma1 libgomp1 libssl3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /src/configs/ /app/configs/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    JAX_SERVER_CONFIG=/app/configs/gpu-example.yaml \
    HF_HOME=/cache/hf \
    JAX_COMPILATION_CACHE_DIR=/cache/jax \
    XLA_PYTHON_CLIENT_PREALLOCATE=false

RUN mkdir -p /cache && chown app:app /cache

# Triton uses shared memory between its HTTP frontend and the Python callback.
# For docker run: --shm-size=2g
# For k8s: mount a medium:Memory emptyDir at /dev/shm (~2Gi).
EXPOSE 8000 8001 8002

USER app

# Single process: one GPU per pod — concurrent JAX on the same device would contend
# for VRAM. The infer_fn uses a threading.Lock to serialize GPU access internally.
CMD ["python", "-m", "jax_server.deploy.triton_main"]
