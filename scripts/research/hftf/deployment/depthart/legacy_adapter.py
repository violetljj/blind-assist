"""Compatibility adapter for the staged DepthART deployment migration.

The implementation remains at the historical HFTF paths until P0-A regression
is complete. New callers may import through this module without changing the
old protocol and receipt paths.
"""

from importlib import import_module

_MODULES = (
    "rewrite_depthart_qairt_onnx",
    "rewrite_depthart_qairt_static_shape",
    "rewrite_depthart_qairt_hygiene",
    "depthart_admission_r1",
)

for _name in _MODULES:
    globals()[_name] = import_module(f"scripts.research.hftf.{_name}")

