from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class PredictRequest(BaseModel):
    inputs: Any
    backend: Literal["auto", "cpu", "gpu"] = "auto"
    mode: Literal["auto", "single", "batch"] = "auto"


class PredictResponse(BaseModel):
    model: str
    backend: Literal["cpu", "gpu"]
    mode: Literal["single", "batch"]
    outputs: Any
