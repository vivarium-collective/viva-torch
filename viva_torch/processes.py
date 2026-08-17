"""NeuralProcess: a drop-in surrogate Process backed by a trained SurrogateNet."""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
from process_bigraph import Process

from viva_torch.model import SurrogateNet
from viva_torch.pathing import port_key
from viva_torch.spec import SurrogateSpec


class NeuralProcess(Process):
    """Approximates a target subsystem's per-step dynamics.

    Reads feature ports (targets + drivers), predicts next absolute target
    values, and writes them with ``overwrite`` semantics.
    """

    config_schema = {
        "checkpoint": {"_type": "string", "_default": ""},
    }

    def __init__(self, config: Optional[dict] = None, core: Any = None) -> None:
        super().__init__(config, core)
        self.net = SurrogateNet.load(self.config["checkpoint"])
        self.spec: SurrogateSpec = self.net.spec
        self._feature_keys = [port_key(p) for p in self.spec.feature_paths]
        self._target_keys = [port_key(p) for p in self.spec.target_paths_t]

    def inputs(self):
        return {k: "float" for k in self._feature_keys}

    def outputs(self):
        return {k: "overwrite[float]" for k in self._target_keys}

    def update(self, state, interval):
        if abs(interval - self.spec.timestep) > 1e-9:
            raise ValueError(
                f"NeuralProcess was trained at dt={self.spec.timestep} but the "
                f"engine requested interval={interval}. Sub-stepping is not yet "
                f"supported; wire the node at the trained timestep."
            )
        x = np.array([[float(state[k]) for k in self._feature_keys]], dtype=np.float64)
        nxt = self.net.predict_next(x)[0]
        return {k: float(nxt[i]) for i, k in enumerate(self._target_keys)}


def register_neural_process(core: Any, name: str = "NeuralProcess") -> bool:
    """Re-register NeuralProcess into ``core``; returns True if newly added.

    ``NeuralProcess`` is normally auto-discovered into every ``core`` via
    bigraph-schema package discovery, so on a fresh ``allocate_core()`` it is
    already present and this returns False. This helper exists for cores that
    don't auto-discover. Genuine registration errors are NOT swallowed.
    """
    if name in (getattr(core, "link_registry", {}) or {}):
        return False
    core.register_link(name, NeuralProcess)
    return True


def neural_process_node(*, checkpoint: str, spec: SurrogateSpec, interval: float = None,
                        address: str = "local:NeuralProcess") -> dict:
    """Build a composite node for a NeuralProcess, wiring ports to their paths.

    ``interval`` defaults to the spec's trained timestep to avoid desync.
    """
    if interval is None:
        interval = spec.timestep
    inputs = {port_key(p): list(p) for p in spec.feature_paths}
    outputs = {port_key(p): list(p) for p in spec.target_paths_t}
    return {
        "_type": "process",
        "address": address,
        "config": {"checkpoint": checkpoint},
        "interval": interval,
        "inputs": inputs,
        "outputs": outputs,
    }
