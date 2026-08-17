from bigraph_schema import allocate_core
from viva_torch.processes import register_neural_process


def test_neural_process_registers_into_core():
    core = allocate_core()
    # NeuralProcess may already be auto-registered via entry points; call is
    # idempotent.  The important invariant is that the key is present afterward.
    register_neural_process(core)
    assert "NeuralProcess" in (getattr(core, "link_registry", {}) or {})
