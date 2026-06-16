"""Read/assemble values at nested process-bigraph state paths.

A *path* is a ``list[str]`` naming a nested store, e.g. ``["cell", "mass"]``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def port_key(path: Sequence[str]) -> str:
    """Stable, unique port name derived from a state path."""
    return "__".join(path)


def get_path(state: dict, path: Sequence[str]) -> float:
    """Return the (scalar) value at ``path`` in ``state``."""
    node = state
    walked = []
    for key in path:
        walked.append(key)
        if not isinstance(node, dict) or key not in node:
            raise KeyError("/".join(walked))
        node = node[key]
    return node


def assemble_vector(state: dict, paths: Sequence[Sequence[str]]) -> np.ndarray:
    """Read each path in order into a float64 vector."""
    return np.array([float(get_path(state, p)) for p in paths], dtype=np.float64)
