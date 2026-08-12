from __future__ import annotations

import numpy as np
import pytest

from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.contract import (
    ContractError,
)
from scripts.research.assistive_geometry.ag_r2_cross_sensor_confirmation.source_geometry import (
    derive_session_context,
    materialize_score_truth,
    source_parent_summary,
)


def _floor_frame(frame_id: str, height_m: float = 1.5) -> dict:
    shape = (64, 64)
    fx = fy = 40.0
    cx = cy = 31.5
    rows, _ = np.indices(shape, dtype=np.float64)
    known = rows >= 32
    depth = np.full(shape, np.nan, dtype=np.float64)
    depth[known] = height_m * fy / (rows[known] - cy)
    pose = np.eye(4, dtype=np.float64)
    pose[1, 3] = -height_m
    return {
        "parent_id": "plant_scene_2",
        "frame_id": frame_id,
        "depth_m": depth,
        "depth_known": known,
        "intrinsics": np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64),
        "camera_to_world": pose,
        "gravity_up_camera_xyz": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    }


def test_calibration_context_uses_lowest_persistent_support_and_score_truth() -> None:
    calibration = [_floor_frame(f"cal-{index:02d}") for index in range(12)]
    context, identity = derive_session_context("plant_scene_2", calibration)
    assert context["camera_height_m"] == pytest.approx(1.5, abs=0.03)
    assert context["camera_height_mad_m"] == pytest.approx(0.0, abs=1e-9)
    assert identity["selected_lowest_mode"]["persistent_frame_count"] >= 8
    score = [_floor_frame(f"score-{index:02d}") for index in range(12)]
    truths = [materialize_score_truth(frame, identity, (32, 32)) for frame in score]
    assert all(np.mean(row["truth"]["depth_known"]) > 0.4 for row in truths)
    assert all(np.mean(row["truth"]["support_known"]) > 0.1 for row in truths)
    summary = source_parent_summary("plant_scene_2", 24, context, truths)
    assert summary["source_depth_known_coverage"] >= 0.5
    assert summary["source_support_known_coverage"] >= 0.1
    assert summary["source_boundary_known_coverage"] >= 0.05


def test_gravity_inconsistency_and_unknown_depth_fail_closed() -> None:
    calibration = [_floor_frame(f"cal-{index:02d}") for index in range(12)]
    calibration[-1]["gravity_up_camera_xyz"] = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    with pytest.raises(ContractError, match="F2_CALIBRATION_GRAVITY_INCONSISTENT"):
        derive_session_context("plant_scene_2", calibration)
    invalid = _floor_frame("invalid")
    invalid["depth_known"][0, 0] = False
    invalid["depth_m"][0, 0] = 1.0
    frames = [_floor_frame(f"cal-{index:02d}") for index in range(11)] + [invalid]
    with pytest.raises(ContractError, match="F2_SOURCE_DEPTH_UNKNOWN_NOT_NAN"):
        derive_session_context("plant_scene_2", frames)


def test_score_height_outside_range_is_unknown_not_negative() -> None:
    calibration = [_floor_frame(f"cal-{index:02d}") for index in range(12)]
    _, identity = derive_session_context("plant_scene_2", calibration)
    score = _floor_frame("score")
    score["camera_to_world"][1, 3] = -0.2
    truth = materialize_score_truth(score, identity, (32, 32))["truth"]
    assert not truth["support_known"].any()
    assert not truth["evidence_known"].any()
    assert np.isnan(truth["support_probability"]).all()
    assert np.isnan(truth["obstacle_probability"]).all()
