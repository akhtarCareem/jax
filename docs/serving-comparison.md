# Serving the JAX Ranker: PyTriton vs FastAPI

Two ways to serve the `jax.export` ranker over HTTP. Both wrap the **same
inference core** (`predict`, dtype conversion, export schema) — they differ only
in the HTTP layer. Choosing one is a choice about the serving stack, not the
model.

- **FastAPI** — custom Python app on uvicorn (uvloop/httptools).
- **PyTriton** — NVIDIA's embedded Triton C++ server, KServe v2 protocol.

## Bottom line

**Latency is a tie: ~13ms p99 both.** GPU compute is ~4ms; the rest is JSON
parse/encode of the ~14KB history payload. The payload dominates, not the
server — so neither stack wins on speed, and the only real latency lever (for
both) is shrinking history tensors or moving to binary/gRPC tensors.

The decision is therefore **where cross-cutting concerns live** — auth,
rate-limiting, request-size limits, observability — in our app, or in the
platform.

## Comparison

| Dimension | FastAPI | PyTriton |
|---|---|---|
| p99 latency | ~13ms | ~13ms |
| HTTP frontend | Python ASGI | Triton C++ |
| Protocol | custom JSON `{inputs,backend,mode}` | KServe v2 typed-tensor |
| gRPC | no | yes (free) |
| Auth / rate-limit / size-cap | in-app | **must be upstream** (mesh/gateway) |
| Concurrency control | per-model admission gate | single GPU lock |
| Metrics / health | app routes (Prometheus) | Triton-native :8002 |
| Code we own | the whole HTTP app | just the inference callback |
| Client dtypes | lenient (silently downcasts) | strict INT32/FP32 |
| Image | slim Python | + tritonserver binary, `/dev/shm` ≥2Gi |
| Dependency risk | low | jax[cuda] + pytriton CUDA coexistence |

## What you give up / gain

**PyTriton** trades app-level control for less code. We supply only the
inference callback; Triton provides the C++ frontend, gRPC, native metrics, and
health. But it has **no custom routes** — auth, rate-limiting, request-size
limits, and any custom request contract must be owned upstream. It also adds a
heavier image (bundled `tritonserver`, shared-memory IPC) and a real dependency
risk: `jax[cuda]` and `nvidia-pytriton` both ship CUDA libs and must be verified
together on GPU.

**FastAPI** keeps everything in-app: auth, admission control, request validation,
the existing contract. No new dependency risk. The cost is that all of it is our
code to maintain, and the request path is Python.

## Recommendation

Latency no longer favors either — decide by ownership of cross-cutting concerns:

- **Platform (mesh/gateway) already owns auth + rate-limiting → PyTriton.**
  We delete the most code, get gRPC and native metrics for free, and the
  features it drops are exactly what the platform should own. This is the better
  long-term shape.

- **Cross-cutting concerns must stay in-app, or CUDA coexistence is unverified
  → FastAPI.** It already hits the latency target with zero new dependency risk.

## Open items before adopting PyTriton

1. Verify `jax[cuda]` + `nvidia-pytriton` run together on Linux+GPU (highest risk).
2. Confirm auth + rate-limiting are owned by the mesh/gateway.
3. Set `/dev/shm` ≥ 2Gi in the deployment.
4. Migrate clients to strict INT32/FP32 tensors.
