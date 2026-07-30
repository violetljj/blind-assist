"""R2 identity wrapper over the frozen R1 scientific implementation."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any


_R1_PATH = (
    Path(__file__).resolve().parents[1]
    / "dual_loop_radial_geometry_lite_r1"
    / "radial_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "dual_loop_radial_geometry_lite_r1_immutable_core_for_r2",
    _R1_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("cannot load immutable R1 geometry")
_R1 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _R1
_SPEC.loader.exec_module(_R1)

PROTOCOL_ID = "DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R2"
IMPLEMENTATION_ID = "DUAL_LOOP_RADIAL_GEOMETRY_LITE_R2_IMPL_R0"
PARAMETERS = _R1.PARAMETERS
PARAMETER_SHA256 = _R1.PARAMETER_SHA256
TTL_NS = _R1.TTL_NS
ARM_BBOX = _R1.ARM_BBOX
ARM_FLOW = _R1.ARM_FLOW
ARMS = _R1.ARMS
FrameObservation = _R1.FrameObservation


def _rewrite(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["protocol_id"] = PROTOCOL_ID
    result["implementation_id"] = IMPLEMENTATION_ID
    result["parameter_sha256"] = PARAMETER_SHA256
    return result


def bbox_log_area_growth(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> dict[str, Any]:
    return _rewrite(_R1.bbox_log_area_growth(previous, current))


def roi_sparse_radial_flow(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> dict[str, Any]:
    return _rewrite(_R1.roi_sparse_radial_flow(previous, current))


def evaluate_pair(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> list[dict[str, Any]]:
    return [_rewrite(row) for row in _R1.evaluate_pair(previous, current)]


def apply_consumer_time(
    row: dict[str, Any],
    consumer_timestamp_ns: int,
) -> dict[str, Any]:
    return _R1.apply_consumer_time(row, consumer_timestamp_ns)
