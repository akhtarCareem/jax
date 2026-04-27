from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import NamedTuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from huggingface_hub import HfApi
from jax import export


class ExampleBatch(NamedTuple):
    features: jax.Array


# JAX export needs a stable serialized name for NamedTuple pytrees that appear
# inside the exported function; the serving runtime still accepts plain dict JSON.
export.register_namedtuple_serialization(
    ExampleBatch,
    serialized_name="jax_server.examples.ExampleBatch",
)


def build_forward_fn(hidden_size: int, output_size: int):
    def forward(batch: ExampleBatch) -> jax.Array:
        mlp = hk.nets.MLP(
            [hidden_size, output_size],
            activate_final=False,
            name="feature_encoder",
        )
        return mlp(batch.features)

    return forward


def export_variant(
    transformed: hk.Transformed,
    params,
    output_dir: Path,
    artifact_name: str,
    *,
    mode: str,
) -> None:
    if mode == "single":
        input_struct = {
            "features": jax.ShapeDtypeStruct((1, 4), jnp.float32),
        }
        suffix = "cpu.bin"
    elif mode == "batch":
        batch_dim, = export.symbolic_shape("b,")
        input_struct = {
            "features": jax.ShapeDtypeStruct((batch_dim, 4), jnp.float32),
        }
        suffix = "cpu_batch.bin"
    else:
        raise ValueError(f"Unsupported export mode: {mode}")

    def serving_fn(model_params, inputs):
        batch = ExampleBatch(features=inputs["features"])
        return transformed.apply(model_params, batch)

    exported = export.export(jax.jit(serving_fn))(params, input_struct)
    (output_dir / f"{artifact_name}_{suffix}").write_bytes(exported.serialize())


def save_params(params, output_dir: Path) -> None:
    params_dir = output_dir / "params"
    if params_dir.exists():
        shutil.rmtree(params_dir)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(params_dir, params)
    checkpointer.wait_until_finished()


def upload_to_hub(
    output_dir: Path,
    repo_id: str,
    *,
    token: str,
    private: bool = False,
    commit_message: str = "Upload example JAX export bundle",
) -> None:
    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(output_dir),
        path_in_repo=".",
        repo_type="model",
        commit_message=commit_message,
    )


def export_example_model(
    output_dir: Path,
    artifact_name: str,
    hidden_size: int,
    output_size: int,
    *,
    hf_repo: str | None = None,
    hf_private: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(0)
    sample_batch = ExampleBatch(features=jnp.ones((1, 4), dtype=jnp.float32))
    transformed = hk.without_apply_rng(hk.transform(build_forward_fn(hidden_size, output_size)))
    params = transformed.init(rng, sample_batch)

    compiled_dir = output_dir / "compiled"
    compiled_dir.mkdir(parents=True, exist_ok=True)
    save_params(params, output_dir)
    export_variant(transformed, params, compiled_dir, artifact_name, mode="single")
    export_variant(transformed, params, compiled_dir, artifact_name, mode="batch")

    sample_inputs = {"features": np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32).tolist()}
    print(f"Exported example model to {output_dir}")
    print("Sample request body:")
    print({"inputs": sample_inputs, "backend": "auto", "mode": "auto"})
    if hf_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError(
                "--hf-repo was provided but HF_TOKEN is not set."
            )
        upload_to_hub(
            output_dir,
            hf_repo,
            token=token,
            private=hf_private,
            commit_message="Upload example JAX export bundle",
        )
        print(f"Uploaded example model bundle to https://huggingface.co/{hf_repo}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a local Haiku MLP example model.")
    parser.add_argument(
        "--output-dir",
        default="examples/test_mlp",
        help="Directory where params and artifacts will be written.",
    )
    parser.add_argument("--artifact-name", default="feature_encoder")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--output-size", type=int, default=2)
    parser.add_argument(
        "--hf-repo",
        help="Optional Hugging Face model repo to create/update with the exported bundle.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create the Hugging Face repo as private when used with --hf-repo.",
    )
    args = parser.parse_args()

    export_example_model(
        output_dir=Path(args.output_dir).resolve(),
        artifact_name=args.artifact_name,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        hf_repo=args.hf_repo,
        hf_private=args.hf_private,
    )


if __name__ == "__main__":
    main()
