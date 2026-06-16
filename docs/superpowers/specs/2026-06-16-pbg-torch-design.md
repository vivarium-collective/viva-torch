# pbg-torch — neural-network surrogate processes for process-bigraph

**Date:** 2026-06-16
**Status:** Design approved, pending spec review

## Purpose

A reusable `pbg-<name>` library that wraps a PyTorch neural network as a
process-bigraph `Process` which **approximates another `Process`/`Composite`**.
Given a target and a declared panel of observables, `pbg-torch` samples
rollouts of the target, trains a net to predict the per-step delta of those
observables, and emits a self-contained **drop-in surrogate Process** that can
be swapped into a bigraph and rolled out autoregressively.

Its first real consumer is a v2ecoli investigation that trains surrogates of
baseline behavior across multiple target categories (growth, exchange, fluxes,
protein abundance, transcript abundance, chromosome state, cell-cycle state).
That investigation is a **separate** spec/plan/build cycle; this document
covers the generic library only.

## Settled design decisions

1. **Surrogate mode:** per-step update — a true drop-in `Process` implementing
   `update(state, interval) -> delta`, composable and rolled out
   autoregressively. (Not a trajectory emulator or static Step.)
2. **Variable space:** curated observables, autoregressive — the same reduced
   vector is both input and predicted output, so the surrogate closes on
   itself and rolls out standalone. Inputs may additionally include exogenous
   drivers (e.g. media) that are supplied but not predicted.
3. **Data source:** the library rolls out the target itself — it holds the
   target Composite spec + observable paths + a sampling plan and runs it
   through the process-bigraph engine to produce a standardized transition
   dataset.
4. **Framework:** PyTorch. Repo name `pbg-torch`.

## Components

Each component is independently testable with a single clear responsibility.

### `SurrogateSpec` (`spec.py`)
Declarative, serializable description of what is being approximated:
- `target_paths`: bigraph state paths that are **predicted and fed back**.
- `driver_paths`: exogenous input paths **supplied each step, not predicted**
  (e.g. media). Features = `target_paths ∪ driver_paths`.
- `timestep`: the fixed interval the surrogate is trained at.
- `output_parameterization`: `delta` (default) | `absolute` | `log_delta`.
- `normalization`: per-variable mean/std or `log1p` flags.

Round-trips to/from JSON; carries no learned state itself.

### `TrajectorySampler` (`sampler.py`)
Holds the target Composite document, the spec, and a `SamplingPlan`
(initial-condition distribution or explicit config list, `n_trajectories`,
`n_steps`, `seeds`). Runs the target through the process-bigraph engine,
collects observable vectors per step, and writes a standardized
**`TransitionDataset`**: arrays of `(x_features_t, y_targets_{t+1}, dt)` plus a
JSON manifest carrying the `SurrogateSpec` and computed normalization stats.
Stored as `.npz`/parquet.

### `SurrogateNet` (`model.py`)
A `torch.nn.Module`. Default: residual MLP predicting the **normalized delta**
(`ŷ = x_targets + f(x_features)`), with normalization baked in as buffers so a
checkpoint is fully self-contained. Configurable depth/width/activation. The
interface is structured so an RNN/neural-ODE backend can slot in later, but no
such backend is built now (YAGNI).

### `train_surrogate` (`trainer.py`)
Loads a `TransitionDataset`, **splits by trajectory** (not by step, to avoid
leakage), trains a one-step prediction loss with per-target-group weighting,
plus an optional **multi-step rollout loss** (curriculum on k-step unrolled
predictions to combat autoregressive drift). Logs per-group metrics. Saves one
artifact: weights + spec + normalization stats.

### `NeuralProcess(Process)` (`processes.py`)
The drop-in surrogate, registered via the pbg discovery convention.
- `config` points at the checkpoint artifact.
- `initialize` loads the torch model + spec + norm stats and sets eval mode.
- `inputs()` exposes the feature ports; `outputs()` exposes the target ports as
  deltas.
- `update(state, interval)` assembles the feature vector from the state ports,
  runs the net, and returns a delta update so it composes like any process.
  Handles interval mismatch (trained at fixed dt → sub-step or raise).

### `evaluate_surrogate` (`evaluate.py`)
Rolls out the surrogate **and** the target from matched held-out initial
conditions and compares per target group (RMSE / MAPE / R², trajectory
overlays), producing an HTML report. These metrics double as the acceptance
gates for the downstream v2ecoli studies.

## Data flow

```
target Composite spec ─Sampler→ TransitionDataset ─Trainer→ checkpoint(weights+spec+norm)
                                                              │
                                          NeuralProcess (drop-in) ─Evaluator→ report
```

Autoregressive closure: predicted targets become next-step features; drivers
come from the surrounding bigraph (or are held constant in a standalone
rollout).

## Error handling

- **Interval mismatch:** trained at a fixed `dt`; a different requested
  interval is sub-stepped or raises a clear error.
- **Missing feature path** in the incoming state → explicit error naming the
  path.
- **NaN / instability** during autoregressive rollout → clamp and report
  divergence rather than propagating NaNs silently.
- **Out-of-training-distribution features** → optional range-check warning
  derived from the dataset manifest.

## Testing strategy (drives the build via TDD)

Prove the **entire pipeline on a cheap analytic target first** — e.g. logistic
growth or an ODE oscillator from `core-processes`: sample → train → drop-in →
evaluate within tolerance. This validates `SurrogateSpec` round-trip,
vector↔state-path assembly, normalization invertibility, and autoregressive
rollout **without paying v2ecoli cost**, and gives a fast regression gate.

- **Unit tests:** spec JSON round-trip; feature-vector assembly from and update
  emission to bigraph state paths; normalization forward/inverse identity;
  dataset manifest integrity.
- **Integration gate:** the analytic end-to-end recovers the known dynamics
  within tolerance.
- **Real consumer:** v2ecoli, only after the analytic gate is green.

## Packaging

Standard heavy `pbg-<name>` repo layout: `pbg_torch/` package
(`spec.py`, `sampler.py`, `model.py`, `trainer.py`, `processes.py`,
`evaluate.py`), tests, README, an HTML report, and registration of
`NeuralProcess` through the pbg discovery mechanism (`__pb_kind__` / `build_core`).
PyTorch (CPU by default) as a dependency.

## Out of scope (this spec)

- The v2ecoli investigation itself (observable panels per category, sampling
  plan over multi-generation seeds, multi-head vs per-group nets, wiring the
  surrogate as a coarse-grained WCM replacement) — its own spec/plan later.
- Non-PyTorch backends; trajectory-emulator and static-Step surrogate modes;
  GPU/distributed training (can be added once the core is proven).
