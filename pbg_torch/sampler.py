"""Roll out a target Composite to build a TransitionDataset."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List

import numpy as np
from process_bigraph import Composite

from pbg_torch.dataset import TransitionDataset
from pbg_torch.pathing import assemble_vector
from pbg_torch.spec import SurrogateSpec


@dataclass
class SamplingPlan:
    n_steps: int
    timestep: float
    initial_states: List[dict] = field(default_factory=list)  # state overrides per trajectory


class TrajectorySampler:
    """Runs the target through the process-bigraph engine and records, for each
    step, the feature vector at ``t`` and the target vector at ``t+1``."""

    def __init__(self, target_document: dict, spec: SurrogateSpec, core):
        self.target_document = target_document
        self.spec = spec
        self.core = core

    def _build(self, overrides: dict) -> Composite:
        doc = copy.deepcopy(self.target_document)
        doc["state"].update(copy.deepcopy(overrides))
        return Composite(doc, core=self.core)

    def sample(self, plan: SamplingPlan) -> TransitionDataset:
        feat_paths = self.spec.feature_paths
        targ_paths = self.spec.target_paths_t
        Xs, Ys, DTs, ids = [], [], [], []
        for traj_index, overrides in enumerate(plan.initial_states):
            comp = self._build(overrides)
            for _ in range(plan.n_steps):
                x_t = assemble_vector(comp.state, feat_paths)
                comp.run(plan.timestep)
                y_next = assemble_vector(comp.state, targ_paths)
                Xs.append(x_t)
                Ys.append(y_next)
                DTs.append(plan.timestep)
                ids.append(traj_index)
        return TransitionDataset(
            X=np.array(Xs), Y=np.array(Ys),
            DT=np.array(DTs), traj_id=np.array(ids), spec=self.spec,
        )
