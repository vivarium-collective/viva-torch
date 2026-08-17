import numpy as np
from viva_torch.spec import SurrogateSpec
from viva_torch.dataset import TransitionDataset


def _toy():
    spec = SurrogateSpec(target_paths=[["x"]], driver_paths=[["r"]])
    X = np.array([[0.1, 0.5], [0.2, 0.5], [0.3, 0.7]])   # [x, r]
    Y = np.array([[0.2], [0.3], [0.45]])                 # next x
    DT = np.array([1.0, 1.0, 1.0])
    traj_id = np.array([0, 0, 1])
    return TransitionDataset(X=X, Y=Y, DT=DT, traj_id=traj_id, spec=spec)


def test_dataset_dimensions():
    ds = _toy()
    assert ds.n_transitions == 3
    assert ds.X.shape == (3, 2)
    assert ds.Y.shape == (3, 1)


def test_dataset_save_load_round_trip(tmp_path):
    ds = _toy()
    p = tmp_path / "data.npz"
    ds.save(p)
    loaded = TransitionDataset.load(p)
    np.testing.assert_allclose(loaded.X, ds.X)
    np.testing.assert_allclose(loaded.Y, ds.Y)
    np.testing.assert_allclose(loaded.DT, ds.DT)
    np.testing.assert_array_equal(loaded.traj_id, ds.traj_id)
    assert loaded.spec == ds.spec
