# jax-server

`jax-server` is a lightweight FastAPI runtime for serving JAX models exported via `jax.export`.

It loads serialized export artifacts from Hugging Face snapshots or local folders, restores parameters separately, and exposes a generic inference API for low-latency single and batched requests across CPU and GPU deployments.

For end-to-end examples, use [configs/local-example.yaml](/Users/dmitry/git/github.com/dmitryBe/jax-server/configs/local-example.yaml) for local artifacts or [configs/hf-example.yaml](/Users/dmitry/git/github.com/dmitryBe/jax-server/configs/hf-example.yaml) for the published Hugging Face sample.

When exporting a model for `jax-server`, keep the exported boundary runtime-generic: the exported function should accept plain JAX pytrees made of dicts, lists, tuples, and arrays, and any model-specific wrapping such as `NamedTuple` reconstruction should happen inside the export wrapper. That lets the server accept ordinary JSON without importing custom application types. In practice, export one fixed-shape artifact for single-request latency and a separate batch artifact with a symbolic batch dimension, for example `b, = export.symbolic_shape("b,")`, so the same batch export can serve multiple batch sizes.

## Local example

Export the example model:

```bash
uv sync --extra dev --extra jax
uv run --extra dev --extra jax python scripts/export_example_model.py
```

Export and publish the generated bundle to a dedicated Hugging Face model repo:

```bash
HF_TOKEN=hf_xxx uv run --extra dev --extra jax python scripts/export_example_model.py \
  --hf-repo your-org/test-mlp
```

Start the server with the local example config:

```bash
JAX_SERVER_CONFIG=configs/local-example.yaml uv run uvicorn jax_server.deploy.k8s_main:app --host 0.0.0.0 --port 8000
```

Start the server with the Hugging Face example config:

```bash
JAX_SERVER_CONFIG=configs/hf-example.yaml uv run uvicorn jax_server.deploy.k8s_main:app --host 0.0.0.0 --port 8000
```

Send a request using plain JSON inputs:

```bash
curl -X POST http://127.0.0.1:8000/v1/models/feature_encoder:predict \
  -H 'content-type: application/json' \
  -d '{"inputs":{"features":[[0.1,0.2,0.3,0.4]]},"backend":"auto","mode":"auto"}'
```

Send a batch request:

```bash
curl -X POST http://127.0.0.1:8000/v1/models/feature_encoder:predict \
  -H 'content-type: application/json' \
  -d '{"inputs":{"features":[[0.1,0.2,0.3,0.4],[0.5,0.6,0.7,0.8],[0.9,1.0,1.1,1.2]]},"backend":"auto","mode":"auto"}'
```
