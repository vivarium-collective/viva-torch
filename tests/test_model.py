import numpy as np
import torch

from viva_torch.spec import SurrogateSpec, Normalizer
from viva_torch.model import SurrogateNet


def _net(parameterization="delta"):
    spec = SurrogateSpec(target_paths=[["x"]], driver_paths=[["r"]],
                         parameterization=parameterization)
    feat_norm = Normalizer.from_data(np.array([[0.1, 0.5], [1.0, 0.7]]))
    targ_norm = Normalizer.from_data(np.array([[0.05], [0.2]]))
    return SurrogateNet(spec, feat_norm, targ_norm, hidden=(8, 8))


def test_forward_shapes():
    net = _net()
    x = torch.tensor([[0.1, 0.5], [0.2, 0.7]], dtype=torch.float32)
    out = net.forward_normalized(x)
    assert out.shape == (2, 1)


def test_predict_next_delta_adds_to_current_targets():
    net = _net("delta")
    x = np.array([[0.3, 0.5]])  # current x = 0.3
    raw_delta = net.forward(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    nxt = net.predict_next(x)
    np.testing.assert_allclose(nxt, x[:, :1] + raw_delta, rtol=1e-5)


def test_save_load_round_trip(tmp_path):
    net = _net()
    x = np.array([[0.42, 0.6]])
    before = net.predict_next(x)
    p = tmp_path / "ckpt.pt"
    net.save(p)
    reloaded = SurrogateNet.load(p)
    after = reloaded.predict_next(x)
    np.testing.assert_allclose(before, after, rtol=1e-6)
    assert reloaded.spec == net.spec


def test_predict_next_absolute_returns_raw_prediction():
    # In "absolute" mode the net predicts next values directly (no residual add).
    net = _net("absolute")
    x = np.array([[0.3, 0.5]])
    raw = net.forward(torch.tensor(x, dtype=torch.float32)).detach().numpy()
    nxt = net.predict_next(x)
    np.testing.assert_allclose(nxt, raw, rtol=1e-5)
