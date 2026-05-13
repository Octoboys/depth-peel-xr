"""Phase 0 smoke tests — verify the dev venv and pytest wiring.

These tests do *not* import `bpy`: that's deliberate. `fake-bpy-module`
installs stub files for Pylance autocomplete but nothing runtime-importable
under that name. `depth_peel_xr/core/` is also bpy-free by contract
(enforced by tools/run_core_tests.sh), so future numpy-only tests can live
alongside these without pulling Blender into the picture.
"""


def test_numpy_importable():
    import numpy

    assert numpy.__version__


def test_usd_core_importable():
    from pxr import Usd

    assert Usd is not None
