# viva-torch

Neural-network surrogate Processes for
[process-bigraph](https://github.com/vivarium-collective/process-bigraph).

`viva-torch` trains a [PyTorch](https://pytorch.org/) network to approximate the
per-step dynamics of a target `Process`/`Composite` over a declared panel of
observables, then emits a drop-in `NeuralProcess` that rolls out
autoregressively as a coarse-grained surrogate. The package auto-registers into
any workspace `core` via bigraph-schema package discovery.

## Install

```bash
uv venv
uv pip install -e ".[dev]"
```

Requires Python >= 3.11. Core dependencies: `process-bigraph`, `bigraph-schema`,
`numpy`, `torch` (see `pyproject.toml`).

## How it works

A `SurrogateSpec` declares which state paths the network **predicts** (targets,
which are fed back on each step) versus which it **consumes** as exogenous
**drivers** (supplied, not predicted). The workflow is:

1. **Sample** — `TrajectorySampler` runs the real target through the
   process-bigraph engine and records, for each step, the feature vector at `t`
   and the target vector at `t+1`, producing a `TransitionDataset`.
2. **Train** — `train_surrogate` fits a `SurrogateNet` (an MLP over normalized
   features; `"delta"` parameterization learns the per-step change by default).
   It splits train/validation by trajectory and returns `(net, history)`.
3. **Deploy** — `neural_process_node` builds a composite node wiring the
   `NeuralProcess` ports to their state paths; it drops into any composite.
4. **Evaluate** — `evaluate_surrogate` rolls out both the target and the
   surrogate from each initial state and reports RMSE / nRMSE / R2 per target.

Surrogates interpolate within their training distribution: evaluation initial
conditions (including driver values) should lie inside the sampled range.

## Quick start

A complete, runnable example (logistic growth) lives in
`scripts/demo_report.py`:

```python
from viva_torch import (
    SurrogateSpec, TrajectorySampler, SamplingPlan,
    train_surrogate, evaluate_surrogate,
)

# target_document / spec / make_core come from your model;
# see tests/analytic.py for the logistic-growth fixtures used below.
spec = SurrogateSpec(target_paths=[["x"]], driver_paths=[["r"]], timestep=1.0)

# 1. Sample transitions from the real target across a panel of initial states.
inits = [{"x": x0, "r": r} for x0 in (0.05, 0.5, 1.0, 2.0) for r in (0.3, 0.5, 0.6)]
ds = TrajectorySampler(target_document, spec, make_core()).sample(
    SamplingPlan(n_steps=30, timestep=1.0, initial_states=inits))

# 2. Train and save a checkpoint.
net, history = train_surrogate(ds, hidden=(32, 32), epochs=500, lr=1e-2, seed=0)
net.save("surrogate.pt")

# 3. Evaluate against the real target.
metrics = evaluate_surrogate(
    target_document=target_document, spec=spec, checkpoint="surrogate.pt",
    core_factory=make_core, initial_states=[{"x": 0.15, "r": 0.45}],
    n_steps=30, timestep=1.0, report_path="report.html")
print(metrics["overall_nrmse"])
```

Drop the trained surrogate into a composite:

```python
from viva_torch import neural_process_node

node = neural_process_node(checkpoint="surrogate.pt", spec=spec)  # interval defaults to spec.timestep
```

Run the demo directly:

```bash
python scripts/demo_report.py
```

## API overview

Everything below is exported from the top-level `viva_torch` package:

- **`SurrogateSpec`** — declares predicted `target_paths` vs. exogenous
  `driver_paths`, the `timestep`, and the `parameterization` (`"delta"` or
  `"absolute"`).
- **`Normalizer`** — invertible per-column standardization (mean/std).
- **`TrajectorySampler`** / **`SamplingPlan`** — roll out the target to build a
  `TransitionDataset`.
- **`TransitionDataset`** — the sampled `(X, Y, dt, traj_id)` transitions.
- **`SurrogateNet`** — the MLP surrogate; serializable via `.save()`.
- **`train_surrogate`** — fit a `SurrogateNet`; returns `(net, history)`.
- **`NeuralProcess`** — the drop-in surrogate `Process`.
- **`register_neural_process`** / **`neural_process_node`** — register the
  process type into a `core` and build a wired composite node.
- **`evaluate_surrogate`** — compare surrogate vs. target rollouts; optional HTML
  report.
- **`build_core`** — construct/augment a process-bigraph `core` with the
  `NeuralProcess` type registered.

A back-compatibility shim distribution (`pbg-torch`, under `shim/`) re-exports
`viva_torch` for the package's former name.

## Links

- Repository: https://github.com/vivarium-collective/viva-torch
- process-bigraph: https://github.com/vivarium-collective/process-bigraph
- PyTorch: https://pytorch.org/
