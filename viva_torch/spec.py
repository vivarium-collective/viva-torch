"""Declarative description of what a surrogate approximates, plus normalization."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

_EPS = 1e-8


@dataclass
class SurrogateSpec:
    """Which state paths a surrogate predicts (targets, fed back) vs. consumes
    as exogenous drivers (supplied, not predicted)."""

    target_paths: List[List[str]]
    driver_paths: List[List[str]] = field(default_factory=list)
    timestep: float = 1.0
    parameterization: str = "delta"  # "delta" | "absolute"

    @property
    def target_paths_t(self) -> List[Tuple[str, ...]]:
        return [tuple(p) for p in self.target_paths]

    @property
    def feature_paths(self) -> List[Tuple[str, ...]]:
        return [tuple(p) for p in self.target_paths] + [tuple(p) for p in self.driver_paths]

    @property
    def n_targets(self) -> int:
        return len(self.target_paths)

    @property
    def n_features(self) -> int:
        return len(self.target_paths) + len(self.driver_paths)

    def to_dict(self) -> dict:
        return {
            "target_paths": [list(p) for p in self.target_paths],
            "driver_paths": [list(p) for p in self.driver_paths],
            "timestep": self.timestep,
            "parameterization": self.parameterization,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SurrogateSpec":
        return cls(
            target_paths=[list(p) for p in d["target_paths"]],
            driver_paths=[list(p) for p in d.get("driver_paths", [])],
            timestep=d.get("timestep", 1.0),
            parameterization=d.get("parameterization", "delta"),
        )


@dataclass
class Normalizer:
    """Per-column standardization (mean/std), invertible, JSON-serializable."""

    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_data(cls, data: np.ndarray) -> "Normalizer":
        data = np.asarray(data, dtype=np.float64)
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std = np.where(std < _EPS, 1.0, std)  # constant columns -> no scaling
        return cls(mean=mean, std=std)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def denormalize(self, z: np.ndarray) -> np.ndarray:
        return np.asarray(z, dtype=np.float64) * self.std + self.mean

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(mean=np.array(d["mean"], dtype=np.float64),
                   std=np.array(d["std"], dtype=np.float64))
