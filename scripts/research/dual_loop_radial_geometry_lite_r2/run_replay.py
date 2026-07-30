#!/usr/bin/env python3
"""R2 producer entrypoint reusing the hash-bound R1 producer mechanics."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import radial_geometry as _R2_GEOMETRY


_R1_PATH = (
    Path(__file__).resolve().parents[1]
    / "dual_loop_radial_geometry_lite_r1"
    / "run_replay.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "dual_loop_radial_geometry_lite_r1_producer_for_r2",
    _R1_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load immutable R1 producer")
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

_ORIGINAL_R1_RUN = _R1.run


def _assert_not_r1_execution_evidence(path: Path) -> None:
    normalized = path.resolve(strict=False).as_posix().lower()
    forbidden_namespace = (
        "target-track-causal-radial-geometry-lite-r1/run-r1"
    )
    if forbidden_namespace in normalized:
        raise ValueError("R1 producer execution evidence is forbidden input")


def run(
    replay_input: Path,
    image_root: Path,
    output_path: Path,
    expected_replay_input_sha256: str,
    **kwargs: object,
) -> dict[str, object]:
    _assert_not_r1_execution_evidence(replay_input)
    _assert_not_r1_execution_evidence(image_root)
    return _ORIGINAL_R1_RUN(
        replay_input,
        image_root,
        output_path,
        expected_replay_input_sha256,
        **kwargs,
    )


_R1.run = run
main = _R1.main
_sha256 = _R1._sha256
FORMAL_INPUT_ROWS = _R1.FORMAL_INPUT_ROWS
FORMAL_OUTPUT_ROWS = _R1.FORMAL_OUTPUT_ROWS
FORMAL_SHAPE_CHANGE_OPPORTUNITIES = _R1.FORMAL_SHAPE_CHANGE_OPPORTUNITIES
FORMAL_SHAPE_CHANGE_ARM_ROWS = _R1.FORMAL_SHAPE_CHANGE_ARM_ROWS


if __name__ == "__main__":
    raise SystemExit(main())
