import numpy as np

from pbg_torch.sampler import TrajectorySampler, SamplingPlan
from pbg_torch.trainer import train_surrogate
from pbg_torch.evaluate import evaluate_surrogate
from tests.analytic import make_core, target_document, logistic_spec


def test_surrogate_matches_target_within_tolerance(tmp_path):
    # INTERPOLATION gate: eval driver values (r=0.45, 0.55) are in the training
    # set and eval x0 lies within the sampled range. Proves the pipeline works
    # end-to-end, not driver-axis extrapolation.
    spec = logistic_spec()
    train_inits = [{"x": float(x0), "r": float(r)}
                   for x0 in np.linspace(0.05, 2.0, 10)
                   for r in (0.3, 0.45, 0.55, 0.6)]
    ds = TrajectorySampler(target_document(0.1, 0.5), spec, make_core()).sample(
        SamplingPlan(n_steps=30, timestep=1.0, initial_states=train_inits))
    net, _ = train_surrogate(ds, hidden=(32, 32), epochs=500, lr=1e-2, seed=0)
    ckpt = tmp_path / "ckpt.pt"
    net.save(ckpt)

    eval_inits = [{"x": 0.15, "r": 0.45}, {"x": 1.3, "r": 0.55}]
    metrics = evaluate_surrogate(
        target_document=target_document(0.1, 0.5),
        spec=spec, checkpoint=str(ckpt), core_factory=make_core,
        initial_states=eval_inits, n_steps=30, timestep=1.0,
        report_path=str(tmp_path / "report.html"),
    )
    assert metrics["per_target"]["x"]["nrmse"] < 0.1
    assert (tmp_path / "report.html").exists()
