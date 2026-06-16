"""A cheap analytic target Process for end-to-end pbg-torch tests.

Logistic growth dx/dt = r*x*(1 - x/K), with ``r`` supplied as a *driver*
(constant per trajectory) and ``x`` the predicted *target*.
"""
from __future__ import annotations

from bigraph_schema import allocate_core
from process_bigraph import Process

from pbg_torch.spec import SurrogateSpec


class LogisticGrowth(Process):
    config_schema = {"K": {"_type": "float", "_default": 10.0}}

    def inputs(self):
        return {"x": "float", "r": "float"}

    def outputs(self):
        return {"x": "overwrite[float]"}

    def update(self, state, interval):
        x = state["x"]
        r = state["r"]
        K = self.config["K"]
        dx = r * x * (1.0 - x / K) * interval
        return {"x": x + dx}


def make_core():
    core = allocate_core()
    core.register_link("LogisticGrowth", LogisticGrowth)
    return core


def target_document(x0: float, r: float, K: float = 10.0) -> dict:
    return {
        "state": {
            "x": float(x0),
            "r": float(r),
            "cell": {
                "_type": "process",
                "address": "local:LogisticGrowth",
                "config": {"K": K},
                "interval": 1.0,
                "inputs": {"x": ["x"], "r": ["r"]},
                "outputs": {"x": ["x"]},
            },
        }
    }


def logistic_spec() -> SurrogateSpec:
    return SurrogateSpec(target_paths=[["x"]], driver_paths=[["r"]],
                         timestep=1.0, parameterization="delta")
