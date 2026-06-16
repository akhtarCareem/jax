from __future__ import annotations

import logging
import os
import threading
import time

from jax_server.config import load_config
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
        for kind, spec, ref in (
            ("input", inputs_spec, first_inputs),
            ("output", outputs_spec, first_outputs),
        ):
            for name, value in spec.items():
                if value != ref[name]:
                    raise ValueError(
                        f"Model '{model.config.name}' export {key} {kind} '{name}' "
                        f"spec {value} != reference {ref[name]}"
                    )

    return first_inputs, first_outputs


def _make_infer(model: ServedModel, gpu_lock: threading.Lock, input_names: list[str]):
    def infer_fn(requests):
        t0 = time.perf_counter()
        responses = []
        for request in requests:
            inputs = {name: request[name] for name in input_names}
            with gpu_lock:
                result = model.predict(inputs, backend="auto", mode="auto", metrics=None)
            responses.append(result["outputs"])
        total_ms = (time.perf_counter() - t0) * 1000
        logger.info("infer total_ms=%.3f", total_ms)
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

            triton.bind(
                model_name=model.config.name,
                infer_func=_make_infer(model, gpu_lock, list(inputs_spec.keys())),
                inputs=[
                    Tensor(name=n, dtype=dt, shape=sh)
                    for n, (dt, sh) in inputs_spec.items()
                ],
                outputs=[
                    Tensor(name=n, dtype=dt, shape=sh)
                    for n, (dt, sh) in outputs_spec.items()
                ],
                config=ModelConfig(batching=False),
            )
            logger.info(
                "bound model=%s batching=False inputs=%s outputs=%s",
                model.config.name, list(inputs_spec), list(outputs_spec),
            )

        logger.info("Triton serving — HTTP :8000  gRPC :8001  metrics :8002")
        triton.serve()


if __name__ == "__main__":
    main()
