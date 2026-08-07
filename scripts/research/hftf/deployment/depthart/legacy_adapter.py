"""Compatibility adapter for the staged DepthART deployment migration."""

from importlib import import_module

_MODULES = (
    "rewrite_depthart_qairt_onnx",
    "rewrite_depthart_qairt_static_shape",
    "rewrite_depthart_qairt_hygiene",
    "depthart_admission_r1",
)

for _name in _MODULES:
    globals()[_name] = import_module(f"{__package__}.{_name}")
