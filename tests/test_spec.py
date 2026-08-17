import numpy as np
from viva_torch.spec import SurrogateSpec, Normalizer


def test_feature_ordering_is_targets_then_drivers():
    spec = SurrogateSpec(target_paths=[["x"], ["y"]], driver_paths=[["r"]])
    assert spec.feature_paths == [("x",), ("y",), ("r",)]
    assert spec.target_paths_t == [("x",), ("y",)]
    assert spec.n_features == 3
    assert spec.n_targets == 2


def test_spec_round_trips_through_dict():
    spec = SurrogateSpec(
        target_paths=[["cell", "mass"]], driver_paths=[["media", "glc"]],
        timestep=2.0, parameterization="absolute",
    )
    again = SurrogateSpec.from_dict(spec.to_dict())
    assert again == spec


def test_normalizer_round_trip_is_identity():
    rng = np.random.default_rng(0)
    data = rng.normal(5.0, 3.0, size=(100, 2))
    norm = Normalizer.from_data(data)
    z = norm.normalize(data)
    np.testing.assert_allclose(norm.denormalize(z), data, atol=1e-9)


def test_normalizer_handles_zero_variance_column():
    data = np.column_stack([np.ones(10), np.arange(10.0)])
    norm = Normalizer.from_data(data)
    z = norm.normalize(data)
    assert np.isfinite(z).all()


def test_normalizer_dict_round_trip():
    norm = Normalizer.from_data(np.arange(20.0).reshape(10, 2))
    again = Normalizer.from_dict(norm.to_dict())
    np.testing.assert_allclose(again.mean, norm.mean)
    np.testing.assert_allclose(again.std, norm.std)
