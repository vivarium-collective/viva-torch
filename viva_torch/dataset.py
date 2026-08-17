"""Standardized transition dataset: (features_t, targets_{t+1}, dt) + spec."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from viva_torch.spec import SurrogateSpec


@dataclass
class TransitionDataset:
    X: np.ndarray        # (n, n_features) raw feature values at t
    Y: np.ndarray        # (n, n_targets) raw target values at t+1
    DT: np.ndarray       # (n,) timestep of each transition
    traj_id: np.ndarray  # (n,) trajectory index each transition came from
    spec: SurrogateSpec

    @property
    def n_transitions(self) -> int:
        return self.X.shape[0]

    def save(self, path) -> None:
        np.savez(
            path,
            X=self.X, Y=self.Y, DT=self.DT, traj_id=self.traj_id,
            spec_json=np.array(json.dumps(self.spec.to_dict())),
        )

    @classmethod
    def load(cls, path) -> "TransitionDataset":
        with np.load(path, allow_pickle=False) as data:
            spec = SurrogateSpec.from_dict(json.loads(str(data["spec_json"])))
            return cls(X=data["X"], Y=data["Y"], DT=data["DT"],
                       traj_id=data["traj_id"], spec=spec)
