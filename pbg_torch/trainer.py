"""Train a SurrogateNet on a TransitionDataset (one-step prediction)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from torch import nn

from pbg_torch.dataset import TransitionDataset
from pbg_torch.model import SurrogateNet
from pbg_torch.spec import Normalizer


def _training_signal(ds: TransitionDataset) -> np.ndarray:
    """The quantity the net regresses on, per parameterization."""
    if ds.spec.parameterization == "delta":
        current_targets = ds.X[:, : ds.spec.n_targets]
        return ds.Y - current_targets
    return ds.Y


def train_surrogate(ds: TransitionDataset, hidden: Sequence[int] = (64, 64),
                    epochs: int = 200, lr: float = 1e-3, batch_size: int = 256,
                    val_fraction: float = 0.2, seed: int = 0):
    """Returns (trained SurrogateNet, history dict). Splits by trajectory."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    traj_ids = np.unique(ds.traj_id)
    rng.shuffle(traj_ids)
    n_val = max(1, int(round(len(traj_ids) * val_fraction)))
    val_ids = set(traj_ids[:n_val].tolist())
    is_val = np.array([t in val_ids for t in ds.traj_id])

    signal = _training_signal(ds)
    X_tr, sig_tr = ds.X[~is_val], signal[~is_val]
    X_val, sig_val = ds.X[is_val], signal[is_val]

    feat_norm = Normalizer.from_data(X_tr)
    targ_norm = Normalizer.from_data(sig_tr)
    net = SurrogateNet(ds.spec, feat_norm, targ_norm, hidden=hidden)

    def to_t(a):
        return torch.tensor(np.asarray(a, dtype=np.float32))

    Xtr_t, sigtr_t = to_t(X_tr), to_t(sig_tr)
    Xval_t = to_t(X_val)
    sigval_norm = to_t(targ_norm.normalize(sig_val))

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = Xtr_t.shape[0]
    history = {"train_loss": [], "val_loss": []}

    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, sigb = Xtr_t[idx], sigtr_t[idx]
            target_norm_b = (sigb - net.targ_mean) / net.targ_std
            opt.zero_grad()
            pred = net.forward_normalized(xb)
            loss = loss_fn(pred, target_norm_b)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.shape[0]
        history["train_loss"].append(epoch_loss / n)

        net.eval()
        with torch.no_grad():
            val_pred = net.forward_normalized(Xval_t)
            history["val_loss"].append(loss_fn(val_pred, sigval_norm).item())

    return net, history
