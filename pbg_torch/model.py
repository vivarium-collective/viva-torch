"""Residual-MLP surrogate network with baked-in normalization."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from pbg_torch.spec import SurrogateSpec, Normalizer


class SurrogateNet(nn.Module):
    """Predicts the per-step target signal (delta or absolute) from features.

    Normalization is stored as buffers so a saved checkpoint is self-contained.
    """

    def __init__(self, spec: SurrogateSpec, feature_norm: Normalizer,
                 target_norm: Normalizer, hidden: Sequence[int] = (64, 64)):
        super().__init__()
        self.spec = spec
        self.hidden = tuple(hidden)

        layers = []
        dim = spec.n_features
        for h in self.hidden:
            layers += [nn.Linear(dim, h), nn.ReLU()]
            dim = h
        layers += [nn.Linear(dim, spec.n_targets)]
        self.net = nn.Sequential(*layers)

        self.register_buffer("feat_mean", torch.tensor(feature_norm.mean, dtype=torch.float32))
        self.register_buffer("feat_std", torch.tensor(feature_norm.std, dtype=torch.float32))
        self.register_buffer("targ_mean", torch.tensor(target_norm.mean, dtype=torch.float32))
        self.register_buffer("targ_std", torch.tensor(target_norm.std, dtype=torch.float32))

    def forward_normalized(self, x_raw: torch.Tensor) -> torch.Tensor:
        z = (x_raw - self.feat_mean) / self.feat_std
        return self.net(z)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """Denormalized prediction of the training signal (raw delta/absolute)."""
        return self.forward_normalized(x_raw) * self.targ_std + self.targ_mean

    def predict_next(self, x_raw: np.ndarray) -> np.ndarray:
        """Next absolute target values for a batch of raw feature rows."""
        self.eval()
        with torch.no_grad():
            xt = torch.tensor(np.asarray(x_raw, dtype=np.float32))
            raw = self.forward(xt).numpy()
        if self.spec.parameterization == "delta":
            current_targets = np.asarray(x_raw, dtype=np.float64)[:, : self.spec.n_targets]
            return current_targets + raw
        return raw

    def save(self, path) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "spec": self.spec.to_dict(),
            "hidden": list(self.hidden),
        }, path)

    @classmethod
    def load(cls, path) -> "SurrogateNet":
        blob = torch.load(path, weights_only=False)
        spec = SurrogateSpec.from_dict(blob["spec"])
        dummy = Normalizer(mean=np.zeros(spec.n_features), std=np.ones(spec.n_features))
        dummy_t = Normalizer(mean=np.zeros(spec.n_targets), std=np.ones(spec.n_targets))
        net = cls(spec, dummy, dummy_t, hidden=blob["hidden"])
        net.load_state_dict(blob["state_dict"])
        return net
