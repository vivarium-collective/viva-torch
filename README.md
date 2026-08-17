# viva-torch

Neural-network surrogate Processes for [process-bigraph](https://github.com/vivarium-collective/process-bigraph).

`viva-torch` trains a PyTorch network to approximate the per-step dynamics of
any target `Process`/`Composite` over a curated panel of observables, and emits
a drop-in `NeuralProcess` that rolls out autoregressively as a coarse-grained
surrogate.

## Pipeline

```python
from viva_torch import (
    SurrogateSpec, TrajectorySampler, SamplingPlan,
    train_surrogate, neural_process_node, evaluate_surrogate,
)

spec = SurrogateSpec(target_paths=[["x"]], driver_paths=[["r"]], timestep=1.0)
ds = TrajectorySampler(target_document, spec, core).sample(
    SamplingPlan(n_steps=30, timestep=1.0, initial_states=[...]))
net, history = train_surrogate(ds)
net.save("surrogate.pt")
# drop into any composite:
node = neural_process_node(checkpoint="surrogate.pt", spec=spec, interval=1.0)
```

- **SurrogateSpec** — declares predicted targets vs. exogenous drivers.
- **TrajectorySampler** — rolls out the target to build a `TransitionDataset`.
- **train_surrogate** — fits a residual-MLP `SurrogateNet` (normalized per-step delta).
- **NeuralProcess** — the drop-in surrogate Process.
- **evaluate_surrogate** — compares surrogate vs. target rollouts.

Surrogates interpolate within their training distribution: evaluation initial
conditions (including driver values) should lie inside the sampled range.

See `docs/superpowers/specs/` for the full design and `scripts/demo_report.py`
for a runnable logistic-growth example.

## Install

```bash
uv venv && uv pip install -e ".[dev]"
```
