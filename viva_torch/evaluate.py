"""Compare surrogate vs. target rollouts from matched initial conditions."""
from __future__ import annotations

import copy
from typing import Callable, List

import numpy as np
from process_bigraph import Composite

from viva_torch.pathing import assemble_vector, port_key
from viva_torch.processes import register_neural_process, neural_process_node
from viva_torch.spec import SurrogateSpec


def _rollout_target(target_document, spec, core, overrides, n_steps, timestep):
    doc = copy.deepcopy(target_document)
    doc["state"].update(copy.deepcopy(overrides))
    comp = Composite(doc, core=core)
    traj = []
    for _ in range(n_steps):
        comp.run(timestep)
        traj.append(assemble_vector(comp.state, spec.target_paths_t))
    return np.array(traj)  # (n_steps, n_targets)


def _rollout_surrogate(spec, checkpoint, core, overrides, n_steps, timestep):
    register_neural_process(core)
    node = neural_process_node(checkpoint=checkpoint, spec=spec, interval=timestep)
    state = {}
    for p in spec.feature_paths:           # initialize feature stores
        node_ptr = state
        for key in p[:-1]:
            node_ptr = node_ptr.setdefault(key, {})
        node_ptr[p[-1]] = 0.0
    state.update(copy.deepcopy(overrides))
    state["surrogate"] = node
    comp = Composite({"state": state}, core=core)
    traj = []
    for _ in range(n_steps):
        comp.run(timestep)
        traj.append(assemble_vector(comp.state, spec.target_paths_t))
    return np.array(traj)


def evaluate_surrogate(target_document: dict, spec: SurrogateSpec, checkpoint: str,
                       core_factory: Callable, initial_states: List[dict],
                       n_steps: int, timestep: float, report_path: str = None) -> dict:
    """Roll out target and surrogate from each initial state; return metrics."""
    tgt_all, sur_all = [], []
    for overrides in initial_states:
        tgt = _rollout_target(target_document, spec, core_factory(), overrides, n_steps, timestep)
        sur = _rollout_surrogate(spec, checkpoint, core_factory(), overrides, n_steps, timestep)
        tgt_all.append(tgt)
        sur_all.append(sur)
    tgt_all = np.concatenate(tgt_all, axis=0)   # (n_states*n_steps, n_targets)
    sur_all = np.concatenate(sur_all, axis=0)

    per_target = {}
    for i, path in enumerate(spec.target_paths_t):
        t = tgt_all[:, i]
        s = sur_all[:, i]
        rmse = float(np.sqrt(np.mean((t - s) ** 2)))
        scale = float(np.std(t)) or 1.0
        per_target[port_key(path)] = {
            "rmse": rmse,
            "nrmse": rmse / scale,
            "r2": float(1.0 - np.sum((t - s) ** 2) / (np.sum((t - t.mean()) ** 2) or 1.0)),
        }

    metrics = {"per_target": per_target,
               "overall_nrmse": float(np.mean([m["nrmse"] for m in per_target.values()]))}

    if report_path:
        _write_report(report_path, metrics, spec)
    return metrics


def _write_report(path: str, metrics: dict, spec: SurrogateSpec) -> None:
    rows = "".join(
        f"<tr><td>{k}</td><td>{m['rmse']:.4g}</td>"
        f"<td>{m['nrmse']:.4g}</td><td>{m['r2']:.4g}</td></tr>"
        for k, m in metrics["per_target"].items()
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>viva-torch surrogate evaluation</title></head><body>
<h1>Surrogate evaluation</h1>
<p>Overall normalized RMSE: {metrics['overall_nrmse']:.4g}</p>
<table border="1" cellpadding="4"><tr><th>target</th><th>RMSE</th>
<th>nRMSE</th><th>R&sup2;</th></tr>{rows}</table>
</body></html>"""
    with open(path, "w") as fh:
        fh.write(html)
