import numpy as np
from process_bigraph import Composite

from viva_torch.sampler import TrajectorySampler, SamplingPlan
from tests.analytic import make_core, target_document, logistic_spec


def test_sampler_produces_well_formed_dataset():
    spec = logistic_spec()
    core = make_core()
    doc = target_document(x0=0.1, r=0.5)
    sampler = TrajectorySampler(target_document=doc, spec=spec, core=core)
    plan = SamplingPlan(
        n_steps=5, timestep=1.0,
        initial_states=[{"x": 0.1, "r": 0.5}, {"x": 0.2, "r": 0.7}],
    )
    ds = sampler.sample(plan)
    assert ds.n_transitions == 10
    assert ds.X.shape == (10, 2)   # [x, r]
    assert ds.Y.shape == (10, 1)   # next x
    assert set(np.unique(ds.traj_id)) == {0, 1}


def test_sampler_records_true_transitions():
    spec = logistic_spec()
    core = make_core()
    sampler = TrajectorySampler(target_document=target_document(0.1, 0.5),
                                spec=spec, core=core)
    ds = sampler.sample(SamplingPlan(n_steps=1, timestep=1.0,
                                     initial_states=[{"x": 0.1, "r": 0.5}]))
    comp = Composite(target_document(0.1, 0.5), core=make_core())
    comp.run(1.0)
    np.testing.assert_allclose(ds.Y[0, 0], comp.state["x"], rtol=1e-9)
    np.testing.assert_allclose(ds.X[0], [0.1, 0.5], rtol=1e-9)
