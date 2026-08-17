import numpy as np
from viva_torch.pathing import get_path, assemble_vector, port_key


def test_get_path_nested():
    state = {"x": 0.1, "cell": {"mass": 3.0}}
    assert get_path(state, ["x"]) == 0.1
    assert get_path(state, ["cell", "mass"]) == 3.0


def test_get_path_missing_raises_with_path_name():
    with __import__("pytest").raises(KeyError, match="cell/missing"):
        get_path({"cell": {}}, ["cell", "missing"])


def test_assemble_vector_order_matches_paths():
    state = {"x": 1.0, "r": 0.5, "cell": {"mass": 2.0}}
    vec = assemble_vector(state, [["x"], ["cell", "mass"], ["r"]])
    assert isinstance(vec, np.ndarray)
    np.testing.assert_array_equal(vec, np.array([1.0, 2.0, 0.5]))


def test_port_key_is_stable_join():
    assert port_key(["cell", "mass"]) == "cell__mass"
    assert port_key(["x"]) == "x"
