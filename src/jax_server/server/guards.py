from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from jax_server.exceptions import ClientInputError


def validate_input_shape(
    value: Any,
    *,
    max_elements: int,
    max_depth: int,
) -> None:
    element_count = 0

    def walk(node: Any, depth: int) -> None:
        nonlocal element_count
        if depth > max_depth:
            raise ClientInputError(
                f"Input nesting exceeds max_input_depth={max_depth}."
            )
        if isinstance(node, Mapping):
            for item in node.values():
                walk(item, depth + 1)
            return
        if isinstance(node, tuple):
            for item in node:
                walk(item, depth + 1)
            return
        if isinstance(node, list):
            for item in node:
                walk(item, depth + 1)
            return
        element_count += 1
        if element_count > max_elements:
            raise ClientInputError(
                f"Input element count exceeds max_input_elements={max_elements}."
            )

    walk(value, 1)
