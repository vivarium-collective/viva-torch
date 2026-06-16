"""pbg-torch: neural-network surrogate Processes for process-bigraph.

A ``NeuralProcess`` learns to approximate the per-step dynamics of a target
Process/Composite over a declared panel of observables, then rolls out
autoregressively as a coarse-grained drop-in surrogate. Auto-registers into any
workspace ``core`` via bigraph-schema package discovery.
"""

__version__ = "0.1.0"
