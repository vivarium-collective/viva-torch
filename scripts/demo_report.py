"""Demo: approximate logistic growth and write an evaluation report."""
import numpy as np

from viva_torch.sampler import TrajectorySampler, SamplingPlan
from viva_torch.trainer import train_surrogate
from viva_torch.evaluate import evaluate_surrogate
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from analytic import make_core, target_document, logistic_spec  # noqa: E402


def main():
    spec = logistic_spec()
    inits = [{"x": float(x0), "r": float(r)}
             for x0 in np.linspace(0.05, 2.0, 10) for r in (0.3, 0.45, 0.55, 0.6)]
    ds = TrajectorySampler(target_document(0.1, 0.5), spec, make_core()).sample(
        SamplingPlan(n_steps=30, timestep=1.0, initial_states=inits))
    net, _ = train_surrogate(ds, hidden=(32, 32), epochs=500, lr=1e-2, seed=0)
    net.save("demo_ckpt.pt")
    metrics = evaluate_surrogate(
        target_document=target_document(0.1, 0.5), spec=spec,
        checkpoint="demo_ckpt.pt", core_factory=make_core,
        initial_states=[{"x": 0.15, "r": 0.45}, {"x": 1.3, "r": 0.55}],
        n_steps=30, timestep=1.0, report_path="demo_report.html")
    print("overall nRMSE:", metrics["overall_nrmse"])


if __name__ == "__main__":
    main()
