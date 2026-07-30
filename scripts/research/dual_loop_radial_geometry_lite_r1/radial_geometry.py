"""R1 common native-shape guard over the immutable R0 two-arm core."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


_R0_PATH = (
    Path(__file__).resolve().parents[1]
    / "dual_loop_radial_geometry_lite_r0"
    / "radial_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "dual_loop_radial_geometry_lite_r0_immutable_core",
    _R0_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load immutable R0 core: {_R0_PATH}")
_R0 = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _R0
_SPEC.loader.exec_module(_R0)


PROTOCOL_ID = "DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R1"
IMPLEMENTATION_ID = "DUAL_LOOP_RADIAL_GEOMETRY_LITE_R1_IMPL_R0"
TTL_NS = _R0.TTL_NS
ARM_BBOX = _R0.ARM_BBOX
ARM_FLOW = _R0.ARM_FLOW
ARMS = _R0.ARMS
FrameObservation = _R0.FrameObservation
PARAMETERS: dict[str, Any] = deepcopy(_R0.PARAMETERS)
PARAMETERS["common_shape_guard"] = {
    "enabled": True,
    "comparison": "exact_native_decoded_grayscale_hw",
    "abstention_reason": "FRAME_SHAPE_CHANGE",
    "history_after_trigger": "current_observation_replaces_previous",
    "resize_pad_crop_letterbox": False,
}
PARAMETER_SHA256 = hashlib.sha256(
    json.dumps(PARAMETERS, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _rewrite_identity(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    result["protocol_id"] = PROTOCOL_ID
    result["implementation_id"] = IMPLEMENTATION_ID
    result["parameter_sha256"] = PARAMETER_SHA256
    return result


def _valid_history_shape_mismatch(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> bool:
    if (
        previous is None
        or current.history_reset
        or previous.track_epoch != current.track_epoch
    ):
        return False
    delta_ns = int(current.captured_at_ns) - int(previous.captured_at_ns)
    if delta_ns <= 0 or delta_ns > TTL_NS:
        return False
    return tuple(previous.gray.shape) != tuple(current.gray.shape)


def _shape_abstention(
    previous: FrameObservation,
    current: FrameObservation,
    arm_id: str,
) -> dict[str, Any]:
    row = _R0._abstain(
        current,
        arm_id,
        "FRAME_SHAPE_CHANGE",
        previous_frame_shape_hw=[
            int(previous.gray.shape[0]),
            int(previous.gray.shape[1]),
        ],
        current_frame_shape_hw=[
            int(current.gray.shape[0]),
            int(current.gray.shape[1]),
        ],
    )
    return _rewrite_identity(row)


def bbox_log_area_growth(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> dict[str, Any]:
    if _valid_history_shape_mismatch(previous, current):
        assert previous is not None
        return _shape_abstention(previous, current, ARM_BBOX)
    return _rewrite_identity(_R0.bbox_log_area_growth(previous, current))


def roi_sparse_radial_flow(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> dict[str, Any]:
    if _valid_history_shape_mismatch(previous, current):
        assert previous is not None
        return _shape_abstention(previous, current, ARM_FLOW)
    return _rewrite_identity(_R0.roi_sparse_radial_flow(previous, current))


def apply_consumer_time(
    row: dict[str, Any],
    consumer_timestamp_ns: int,
) -> dict[str, Any]:
    return _R0.apply_consumer_time(row, consumer_timestamp_ns)


def evaluate_pair(
    previous: FrameObservation | None,
    current: FrameObservation,
) -> list[dict[str, Any]]:
    """Return both frozen arms after the shared pre-geometry shape guard."""
    return [
        bbox_log_area_growth(previous, current),
        roi_sparse_radial_flow(previous, current),
    ]
