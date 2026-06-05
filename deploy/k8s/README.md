# Deploying jax-server on Yoda (GPU)

jax-server is deployed as a `serving.yoda/v1` Serving CR. The r2-d2 operator
reconciles it into a Kubernetes Deployment behind a Service on port 8080.

## Prerequisites

- Access to the `careem/boba` repo to trigger the image build.
- A `serving.yoda/v1` CRD installed (managed by r2-d2).
- GPU nodes in the cluster with the correct label/taint (confirm with platform team).
- A Yoda project namespace already created via Kamino/the Yoda UI.

## Image build

Images are built and pushed to ECR via `careem/boba`. Trigger the
`deploy-image.yml` `workflow_dispatch` in that repo with:

```
resource:        jax-server
context_path:    jax-server/
dockerfile_path: Dockerfile
extra_tags:      <version>
```

This produces:
```
848569320300.dkr.ecr.eu-west-1.amazonaws.com/boba/jax-server:<sha>,<version>,latest
```

For a quick local smoke test (no GPU needed for basic wiring):

```bash
docker buildx build --platform linux/amd64 -t jax-server:local .
docker run --rm \
  -e HF_TOKEN=hf_xxx \
  -p 8080:8080 \
  jax-server:local
curl http://localhost:8080/healthz
```

Wait for `/readyz` to return 200, then send a predict request:

```bash
curl -X POST http://localhost:8080/v1/models/feature_encoder:predict \
  -H 'content-type: application/json' \
  -d '{"inputs":{"features":[[0.1,0.2,0.3,0.4]]},"backend":"auto","mode":"auto"}'
```

## Apply order

1. **Edit `serving.yaml`** — substitute your namespace, image tag, and GPU
   `nodeSelector` label (ask the platform team for the correct key).
2. **Create the Secret** (do not commit real tokens):
   ```bash
   kubectl create secret generic jax-server-secrets \
     --from-literal=HF_TOKEN=hf_xxx \
     -n <your-yoda-project-namespace>
   ```
3. **Dry-run validate** (never skip on prod):
   ```bash
   kubectl apply --dry-run=client -f deploy/k8s/serving.yaml
   ```
4. **Apply on non-prod**:
   ```bash
   kubectl apply -f deploy/k8s/serving.yaml
   ```
5. **Watch rollout**:
   ```bash
   kubectl get serving,deploy,pods -n <namespace> -w
   ```

## GPU node scheduling

Add the correct GPU node label under `spec.nodeSelector` and the corresponding
taint toleration under `spec.tolerations` in `serving.yaml`. The r2-d2 operator
passes both through verbatim to the pod. The GPU resource limit
(`nvidia.com/gpu: "1"`) also goes in `spec.container.resources.limits`; the
operator does not inject it.

## HF model cache

The serving.yaml mounts `/cache` as an emptyDir equivalent — artifacts are
re-downloaded on every pod restart. For large models or faster cold starts,
replace the volume with a PVC that survives pod restarts and add it via the
Deployment or ask the platform team for a shared model cache PVC.

## GPU artifacts

The `gpu-example.yaml` config sets `default_platform: gpu`. This requires
`_gpu.bin` and `_gpu_batch.bin` artifacts to be present in the HF repo. If
the target repo only has CPU artifacts, export GPU variants first with
`scripts/export_example_model.py` and push them, then reference the GPU
artifact dir from the config.
