from . import properties
from .operators import portal_rig as op_portal_rig
from .operators import inspect as op_inspect
from .operators import layer_placement as op_layer_placement
from .operators import capture as op_capture
from .ui import n_panel
from .ui import overlays


_MODULES = (
    properties,
    op_portal_rig,
    op_inspect,
    op_layer_placement,
    op_capture,
    n_panel,
    overlays,
)


def register():
    for mod in _MODULES:
        mod.register()


def unregister():
    for mod in reversed(_MODULES):
        mod.unregister()
