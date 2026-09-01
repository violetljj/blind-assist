"""X76 rejects self-motion from a zero-shift reconstructed parent hull."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x73_credentialed_parent_hull_reconstruction as x73  # noqa: E402
import dtr_carla_x75_collision_credentialed_object_permanence as x75  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X76_ZERO_SHIFT_PARENT_HULL_MOTION_REJECTION"
ARM_X76 = "X76_ISSUED_PLAN_ZERO_SHIFT_PARENT_HULL_MOTION_REJECTION"
ZERO_SHIFT = "ZERO_SHIFT_SURFACE_SUPPORT_CARRY"


def fixed_constants() -> dict[str, Any]:
    return {
        **x75.fixed_constants(),
        "representation": "X75_WITH_ZERO_SHIFT_PARENT_HULL_MOTION_REJECTION",
        "retained_core": "X75",
        "release_rule": (
            "EVERY_CONFIRMED_CARRIER_IS_A_CREDENTIALLED_PARENT_CURRENT_FRAGMENT_"
            "HULL_WITH_ZERO_SHIFT_ZERO_CONTRADICTION_TRANSPORT_BUT_NONZERO_"
            "RECONSTRUCTED_VELOCITY"
        ),
        "zero_test": "INHERITED_NUMERIC_EPSILON",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _is_zero_shift_self_motion(row: Mapping[str, Any]) -> bool:
    velocity_nonzero = (
        abs(float(row.get("velocity_forward_mps", 0.0))) > x24.EPSILON
        or abs(float(row.get("velocity_right_mps", 0.0))) > x24.EPSILON
    )
    return (
        row.get("support_footprint_mode")
        == x73.SUPPORT_MODE
        and row.get("transport_state") == ZERO_SHIFT
        and int(row.get("transport_contradictions", 0)) == 0
        and velocity_nonzero
    )


def apply_zero_shift_parent_hull_motion_rejection_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x75.ARM_X75]
        arm["x76_zero_shift_parent_hull_motion_release_used"] = False
        arm["x76_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue
        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x76_carrier_reference")
        if not all(_is_zero_shift_self_motion(rows[track_id]) for track_id in confirmed_ids):
            continue

        candidate_ids = {
            str(track_id) for track_id in arm.get("candidate_risk_track_ids", [])
        } - confirmed_ids
        arm.update(
            {
                "route_risk": False,
                "minimum_entry_s": None,
                "candidate_risk_track_ids": sorted(candidate_ids),
                "confirmed_risk_track_ids": [],
                "candidate_risk_parent_track_ids": sorted(
                    {
                        str(rows[track_id]["parent_track_id"])
                        for track_id in candidate_ids
                        if track_id in rows and rows[track_id].get("parent_track_id")
                    }
                ),
                "confirmed_risk_parent_track_ids": [],
                "x76_zero_shift_parent_hull_motion_release_used": True,
                "x76_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X76] = value["arms"].pop(x75.ARM_X75)
    value["diagnostics"]["x76_route_mode_counts"] = value["diagnostics"].pop(
        "x75_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x76_zero_shift_parent_hull_motion_release_frames": released_frames,
            "x76_zero_shift_parent_hull_motion_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X76] = frame["arms"].pop(x75.ARM_X75)
    return value


def self_check() -> dict[str, Any]:
    row = {
        "support_footprint_mode": x73.SUPPORT_MODE,
        "transport_state": ZERO_SHIFT,
        "transport_contradictions": 0,
        "velocity_forward_mps": 0.0,
        "velocity_right_mps": -1.625,
    }
    x24.require(
        _is_zero_shift_self_motion(row)
        and not _is_zero_shift_self_motion({**row, "velocity_right_mps": 0.0})
        and not _is_zero_shift_self_motion({**row, "transport_contradictions": 1})
        and not _is_zero_shift_self_motion(
            {**row, "transport_state": "DIRECTION_CONSISTENT_BRANCH_CONTINUATION"}
        ),
        "x76_zero_shift_motion_partition",
    )
    return {
        "status": "X76_ZERO_SHIFT_PARENT_HULL_MOTION_FALSIFIER_MET",
        "release_only": True,
        "contradicted_history_retained": True,
        "consistent_or_stationary_hull_retained": True,
        "zero_test_inherited": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
