import numpy as np

from pbg_torch.sampler import TrajectorySampler, SamplingPlan
from pbg_torch.trainer import train_surrogate
from tests.analytic import make_core, target_document, logistic_spec


def _dataset():
    spec = logistic_spec()
    inits = [{"x": float(x0), "r": float(r)}
             for x0 in np.linspace(0.05, 2.0, 8)
             for r in (0.3, 0.6)]
    sampler = TrajectorySampler(target_document(0.1, 0.5), spec, make_core())
    return sampler.sample(SamplingPlan(n_steps=25, timestep=1.0, initial_states=inits))


def test_training_drives_val_loss_down():
    ds = _dataset()
    net, history = train_surrogate(ds, hidden=(32, 32), epochs=300, lr=1e-2, seed=0)
    assert history["val_loss"][-1] < history["val_loss"][0]
    assert history["val_loss"][-1] < 1e-2


def test_trainer_returns_self_contained_net():
    ds = _dataset()
    net, _ = train_surrogate(ds, hidden=(16, 16), epochs=10, seed=0)
    x = ds.X[:3]
    nxt = net.predict_next(x)
    assert nxt.shape == (3, ds.spec.n_targets)
    assert np.isfinite(nxt).all()
