from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from jax_server.exceptions import ClientInputError


class AdmissionGate:
    def __init__(self, capacity: int) -> None:
        self.capacity = max(capacity, 1)
        self._tokens: asyncio.Queue[object] = asyncio.Queue(maxsize=self.capacity)
        for _ in range(self.capacity):
            self._tokens.put_nowait(object())

    @asynccontextmanager
    async def acquire(self):
        try:
            token = self._tokens.get_nowait()
        except asyncio.QueueEmpty as exc:
            raise ClientInputError("Model is at concurrency capacity, retry later.") from exc
        try:
            yield
        finally:
            self._tokens.put_nowait(token)
