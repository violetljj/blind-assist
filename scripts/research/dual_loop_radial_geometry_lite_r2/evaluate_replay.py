#!/usr/bin/env python3
"""R2 evaluator entrypoint reusing the hash-bound R1 pre-truth and gates."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import radial_geometry as _R2_GEOMETRY


_R1_PATH = (
    Path(__file__).resolve().parents[1]
    / "dual_loop_radial_geometry_lite_r1"
    / "evaluate_replay.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "dual_loop_radial_geometry_lite_r1_evaluator_for_r2",
    _R1_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load immutable R1 evaluator")
_previous = sys.modules.get("radial_geometry")
sys.modules["radial_geometry"] = _R2_GEOMETRY
try:
    _R1 = importlib.util.module_from_spec(_SPEC)
    sys.modules[_SPEC.name] = _R1
    _SPEC.loader.exec_module(_R1)
finally:
    if _previous is None:
        sys.modules.pop("radial_geometry", None)
    else:
        sys.modules["radial_geometry"] = _previous

evaluate_files = _R1.evaluate_files
evaluate_records = _R1.evaluate_records
validate_output_ledger = _R1.validate_output_ledger
validate_shape_audit_binding = _R1.validate_shape_audit_binding
SCIENTIFIC_GATE_CONTRACT = _R1.SCIENTIFIC_GATE_CONTRACT
SCIENTIFIC_GATE_CONTRACT_SHA256 = _R1.SCIENTIFIC_GATE_CONTRACT_SHA256
main = _R1.main


if __name__ == "__main__":
    raise SystemExit(main())
