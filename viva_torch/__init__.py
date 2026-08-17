"""viva-torch: neural-network surrogate Processes for process-bigraph.

A ``NeuralProcess`` learns to approximate the per-step dynamics of a target
Process/Composite over a declared panel of observables, then rolls out
autoregressively as a coarse-grained drop-in surrogate. Auto-registers into any
workspace ``core`` via bigraph-schema package discovery.
"""

__version__ = "0.1.0"

from viva_torch.spec import SurrogateSpec, Normalizer
from viva_torch.dataset import TransitionDataset
from viva_torch.sampler import TrajectorySampler, SamplingPlan
from viva_torch.model import SurrogateNet
from viva_torch.trainer import train_surrogate
from viva_torch.processes import (
    NeuralProcess, register_neural_process, neural_process_node,
)
from viva_torch.evaluate import evaluate_surrogate

__all__ = [
    "SurrogateSpec", "Normalizer", "TransitionDataset",
    "TrajectorySampler", "SamplingPlan", "SurrogateNet", "train_surrogate",
    "NeuralProcess", "register_neural_process", "neural_process_node",
    "evaluate_surrogate",
]
