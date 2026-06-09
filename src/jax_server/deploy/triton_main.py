from __future__ import annotations

import logging
import os
import threading
from typing import Any

import numpy as np

from jax_server.config import load_config
from jax_server.exceptions import ClientInputError, ExecutionError, ModelLoadError
from jax_server.runtime.bootstrap import build_models
from jax_server.runtime.exported import derive_triton_io
from jax_server.runtime.model import ServedModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jax_server.deploy")


def _unified_schema(model: ServedModel) -> tuple[dict, dict]:
    specs = {
        key: derive_triton_io(ef.exported)
        for key, ef in model.exports.items()
    }
    first_key = next(iter(specs))
    first_inputs, first_outputs = specs[first_key]

    for key, (inputs_spec, outputs_spec) in specs.items():
        for name, (dtype, shape) in inputs_spec.items():
            ref_dtype, ref_shape = first_inputs[name]
            if dtype != ref_dtype or shape[1:] != ref_shape[1:]:
                raise ValueError(
                    f"Model '{model.config.name}' export {key} input '{name}' "
                    f"has incompatible dtype/trailing-shape {dtype}/{shape[1:]} "
                    f"vs reference {ref_dtype}/{ref_shape[1:]}"
                )
        for name, (dtype, shape) in outputs_spec.items():
            ref_dtype, ref_shape = first_outputs[name]
            if dtype != ref_dtype or shape[1:] != ref_shape[1:]:
                raise ValueError(
                    f"Model '{model.config.name}' export {key} output '{name}' "
                    f"has incompatible dtype/trailing-shape {dtype}/{shape[1:]} "
                    f"vs reference {ref_dtype}/{ref_shape[1:]}"
                )

    return first_inputs, first_outputs


def _decode_optional(request: Any, name: str) -> str:
    val = request.get(name)
    if val is None:
        return "auto"
    raw = np.asarray(val).flat[0]
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _make_infer(
    model: ServedModel,
    gpu_lock: threading.Lock,
    input_names: list[str],
    output_names: list[str],
):
    def infer_fn(requests):
        responses = []
        for request in requests:
            inputs = {name: np.asarray(request[name]) for name in input_names}
            backend = _decode_optional(request, "backend")
            mode = _decode_optional(request, "mode")
            try:
                with gpu_lock:
                    result = model.predict(inputs, backend, mode, metrics=None)
            except (ClientInputError, ExecutionError, ModelLoadError) as exc:
                logger.error("predict error model=%s: %s", model.config.name, exc)
                raise
            responses.append(
                {name: np.asarray(result["outputs"][name]) for name in output_names}
            )
        return responses

    return infer_fn


def main() -> None:
    from pytriton.model_config import ModelConfig, Tensor
    from pytriton.triton import Triton

    config_path = os.environ.get("JAX_SERVER_CONFIG", "configs/example.yaml")
    config = load_config(config_path)
    models = build_models(config)

    gpu_lock = threading.Lock()

    with Triton() as triton:
        for model in models:
            inputs_spec, outputs_spec = _unified_schema(model)
            input_names = list(inputs_spec)
            output_names = list(outputs_spec)

            triton.bind(
                model_name=model.config.name,
                infer_func=_make_infer(model, gpu_lock, input_names, output_names),
                inputs=[
                    Tensor(name=n, dtype=dt, shape=sh)
                    for n, (dt, sh) in inputs_spec.items()
                ] + [
                    Tensor(name="backend", dtype=np.bytes_, shape=(1,), optional=True),
                    Tensor(name="mode", dtype=np.bytes_, shape=(1,), optional=True),
                ],
                outputs=[
                    Tensor(name=n, dtype=dt, shape=sh)
                    for n, (dt, sh) in outputs_spec.items()
                ],
                config=ModelConfig(batching=False),
            )
            logger.info(
                "bound model=%s inputs=%s outputs=%s",
                model.config.name, input_names, output_names,
            )

        logger.info("Triton serving — HTTP :8000  gRPC :8001  metrics :8002")
        triton.serve()


if __name__ == "__main__":
    main()
