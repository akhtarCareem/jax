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

MAX_QUEUE_DELAY_US = int(os.environ.get("TRITON_MAX_QUEUE_DELAY_US", "2000"))


def _unified_schema(model: ServedModel) -> tuple[dict, dict]:
    """Per-sample tensor specs (leading batch dim stripped) for Triton @batch.

    Validates every (backend, mode) export shares the same per-sample signature.
    """
    specs = {
        key: derive_triton_io(ef.exported, strip_leading=True)
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
                        f"per-sample spec {value} != reference {ref[name]}"
                    )

    return first_inputs, first_outputs


def _make_infer(model: ServedModel, gpu_lock: threading.Lock, batch_decorator, scalar_inputs):
    def infer_fn(**inputs):
        t0 = time.perf_counter()
        for name in scalar_inputs:
            arr = inputs[name]
            inputs[name] = arr.reshape(arr.shape[0])
        with gpu_lock:
            result = model.predict(inputs, backend="auto", mode="batch", metrics=None)
        total_ms = (time.perf_counter() - t0) * 1000
        logger.info("infer total_ms=%.3f", total_ms)
        return result["outputs"]

    return batch_decorator(infer_fn)


def main() -> None:
    from pytriton.decorators import batch
    from pytriton.model_config import DynamicBatcher, ModelConfig, Tensor
    from pytriton.triton import Triton

    config_path = os.environ.get("JAX_SERVER_CONFIG", "configs/example.yaml")
    config = load_config(config_path)
    models = build_models(config)

    gpu_lock = threading.Lock()

    with Triton() as triton:
        for model in models:
            inputs_spec, outputs_spec = _unified_schema(model)
            max_batch_size = model.config.max_batch_size or 8
            scalar_inputs = {n for n, (_, sh) in inputs_spec.items() if sh == ()}

            triton.bind(
                model_name=model.config.name,
                infer_func=_make_infer(model, gpu_lock, batch, scalar_inputs),
                inputs=[
                    Tensor(name=n, dtype=dt, shape=(sh if sh != () else (1,)))
                    for n, (dt, sh) in inputs_spec.items()
                ],
                outputs=[
                    Tensor(name=n, dtype=dt, shape=sh)
                    for n, (dt, sh) in outputs_spec.items()
                ],
                config=ModelConfig(
                    max_batch_size=max_batch_size,
                    batcher=DynamicBatcher(max_queue_delay_microseconds=MAX_QUEUE_DELAY_US),
                ),
            )
            logger.info(
                "bound model=%s max_batch_size=%d queue_delay_us=%d inputs=%s outputs=%s",
                model.config.name, max_batch_size, MAX_QUEUE_DELAY_US,
                list(inputs_spec), list(outputs_spec),
            )

        logger.info("Triton serving — HTTP :8000  gRPC :8001  metrics :8002")
        triton.serve()


if __name__ == "__main__":
    main()
