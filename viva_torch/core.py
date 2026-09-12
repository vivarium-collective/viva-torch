"""build_core() — the uniform viva-/pbg- core entrypoint for viva-torch.

Follows the shared convention: ``build_core(core=None)`` allocates a fresh
process-bigraph core when ``core is None`` (else composes onto the passed one),
registers THIS repo's own Process/Step classes by name so they are first-class,
browsable dashboard Registry entries, and returns ``core``.

viva-torch's process is :class:`viva_torch.processes.NeuralProcess`. It is
normally auto-discovered into a fresh ``allocate_core()`` via bigraph-schema
package discovery, but composites that instantiate it directly (rather than
referencing it by a registered address) leave it invisible in the dashboard
Registry — so we register it explicitly here as well.
"""
from __future__ import annotations

from typing import Any

from process_bigraph import allocate_core

from viva_torch.processes import NeuralProcess


def build_core(core: Any = None):
    """Return a process-bigraph core with viva-torch's processes registered.

    Composites that address ``local:NeuralProcess`` (and the test suite) should
    build their ``Composite`` against a core returned from here.
    """
    if core is None:
        core = allocate_core()

    # Register every process-bigraph-native Process/Step in this package's
    # ``processes`` module as a first-class Registry entry. Best-effort +
    # idempotent; a missing helper must never break core construction.
    try:
        from viva_superpowers.core_compose import register_package_processes
        register_package_processes(core, "viva_torch.processes")
    except Exception:
        pass

    # NeuralProcess is also top-level (``from viva_torch import NeuralProcess``),
    # so register it explicitly to guarantee it lands regardless of the helper.
    try:
        core.register_link("NeuralProcess", NeuralProcess)
    except Exception:
        pass

    return core
