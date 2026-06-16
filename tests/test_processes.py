import numpy as np
from bigraph_schema import allocate_core
from process_bigraph import Composite

from pbg_torch.sampler import TrajectorySampler, SamplingPlan
from pbg_torch.trainer import train_surrogate
from pbg_torch.processes import (
    NeuralProcess, register_neural_process, neural_process_node,
)
from tests.analytic import make_core, target_document, logistic_spec


def test_register_is_idempotent():
    core = allocate_core()
    register_neural_process(core)
    assert register_neural_process(core) is False
    assert "NeuralProcess" in (getattr(core, "link_registry", {}) or {})


def test_ports_derive_from_spec():
    from pbg_torch.pathing import port_key
    assert port_key(["x"]) == "x"
    assert port_key(["r"]) == "r"


def test_surrogate_runs_in_a_composite(tmp_path):
    spec = logistic_spec()
    inits = [{"x": float(x0), "r": 0.5} for x0 in np.linspace(0.05, 2.0, 12)]
    ds = TrajectorySampler(target_document(0.1, 0.5), spec, make_core()).sample(
        SamplingPlan(n_steps=25, timestep=1.0, initial_states=inits))
    net, _ = train_surrogate(ds, hidden=(32, 32), epochs=300, lr=1e-2, seed=0)
    ckpt = tmp_path / "ckpt.pt"
    net.save(ckpt)

    core = make_core()
    register_neural_process(core)
    node = neural_process_node(checkpoint=str(ckpt), spec=spec, interval=1.0)
    comp = Composite({"state": {"x": 0.2, "r": 0.5, "surrogate": node}}, core=core)
    xs = []
    for _ in range(10):
        comp.run(1.0)
        xs.append(comp.state["x"])
    xs = np.array(xs)
    assert np.isfinite(xs).all()
    assert (xs > 0).all()
    assert xs[-1] > xs[0]


def test_update_rejects_interval_mismatch(tmp_path):
    # A NeuralProcess trained at dt=1.0 must refuse a different interval rather
    # than silently advancing the wrong amount of dynamics.
    spec = logistic_spec()  # timestep == 1.0
    inits = [{"x": float(x0), "r": 0.5} for x0 in np.linspace(0.05, 2.0, 6)]
    ds = TrajectorySampler(target_document(0.1, 0.5), spec, make_core()).sample(
        SamplingPlan(n_steps=10, timestep=1.0, initial_states=inits))
    net, _ = train_surrogate(ds, hidden=(8, 8), epochs=5, seed=0)
    ckpt = tmp_path / "ckpt.pt"
    net.save(ckpt)

    proc = NeuralProcess({"checkpoint": str(ckpt)}, make_core())
    proc.update({"x": 0.2, "r": 0.5}, 1.0)  # matched dt: fine
    with __import__("pytest").raises(ValueError, match="trained at dt"):
        proc.update({"x": 0.2, "r": 0.5}, 2.0)


def test_node_interval_defaults_to_trained_timestep():
    spec = logistic_spec()  # timestep == 1.0
    node = neural_process_node(checkpoint="x.pt", spec=spec)
    assert node["interval"] == spec.timestep
