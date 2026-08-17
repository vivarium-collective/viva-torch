"""The pbg_torch import name stays working (deprecated) after the rename."""
import sys
import warnings


def test_pbg_torch_still_imports_and_warns():
    # pbg_torch ships inside the same distribution as viva_torch, so an
    # earlier test's allocate_core() (bigraph-schema package discovery) may
    # have already imported it -- Python won't re-execute a cached module, so
    # force a fresh import here to reliably observe the one-time warning.
    for name in [k for k in sys.modules if k == "pbg_torch" or k.startswith("pbg_torch.")]:
        del sys.modules[name]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        import pbg_torch  # noqa: F401
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_pbg_torch_submodule_redirects_to_viva_torch():
    import viva_torch.spec as real
    import pbg_torch.spec as shimmed
    assert shimmed is real            # meta-path finder aliases to the real module
