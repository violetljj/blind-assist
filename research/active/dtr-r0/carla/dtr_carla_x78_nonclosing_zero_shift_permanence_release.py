"""X78 separates non-closing zero-shift permanence from route-risk authority.

Object identity may persist after a measurement disappears, but a pure
object-permanence belief whose surface transport stayed at zero shift has no
current closing-motion evidence.  X78 therefore clears route risk only when
every confirmed carrier is such a zero-contradiction belief and its inherited
forward velocity is stationary or receding.  The track rows remain intact as
existence memory.  Approaching beliefs and every measured or metric carrier
remain conservative.  No detector, route, association, or numeric threshold is
changed; consumed cohorts provide Development evidence only.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x46_evidence_terminated_object_permanence as x46  # noqa: E402
import dtr_carla_x76_zero_shift_parent_hull_motion_rejection as x76  # noqa: E402
import dtr_carla_x77_receding_metric_temporal_handoff_rejection as x77  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X78_NONCLOSING_ZERO_SHIFT_PERMANENCE_RELEASE"
ARM_X78 = "X78_ISSUED_PLAN_NONCLOSING_ZERO_SHIFT_PERMANENCE_RELEASE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x77.fixed_constants(),
        "representation": "X77_WITH_NONCLOSING_ZERO_SHIFT_PERMANENCE_RELEASE",
        "retained_core": "X77",
        "release_rule": (
            "EVERY_CONFIRMED_CARRIER_IS_A_ZERO_CONTRADICTION_OBJECT_"
            "PERMANENCE_BELIEF_WITH_ZERO_SHIFT_TRANSPORT_AND_NONNEGATIVE_"
            "FORWARD_VELOCITY"
        ),
        "identity_memory_retained": True,
        "velocity_sign_test": "INHERITED_NUMERIC_EPSILON",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _is_nonclosing_zero_shift_permanence(row: Mapping[str, Any]) -> bool:
    return (
        row.get("disposition") == x46.BELIEF_DISPOSITION
        and row.get("transport_state") == x76.ZERO_SHIFT
        and int(row.get("transport_contradictions", 0)) == 0
        and float(row.get("velocity_forward_mps", 0.0)) >= -x24.EPSILON
    )


def apply_nonclosing_zero_shift_permanence_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x77.ARM_X77]
        arm["x78_nonclosing_zero_shift_permanence_release_used"] = False
        arm["x78_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue
        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x78_carrier_reference")
        if not all(
            _is_nonclosing_zero_shift_permanence(rows[track_id])
            for track_id in confirmed_ids
        ):
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
                "x78_nonclosing_zero_shift_permanence_release_used": True,
                "x78_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X78] = value["arms"].pop(x77.ARM_X77)
    value["diagnostics"]["x78_route_mode_counts"] = value["diagnostics"].pop(
        "x77_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x78_nonclosing_zero_shift_permanence_release_frames": released_frames,
            "x78_nonclosing_zero_shift_permanence_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X78] = frame["arms"].pop(x77.ARM_X77)
    return value


def self_check() -> dict[str, Any]:
    row = {
        "disposition": x46.BELIEF_DISPOSITION,
        "transport_state": x76.ZERO_SHIFT,
        "transport_contradictions": 0,
        "velocity_forward_mps": 0.0,
    }
    x24.require(
        _is_nonclosing_zero_shift_permanence(row)
        and _is_nonclosing_zero_shift_permanence(
            {**row, "velocity_forward_mps": x24.EPSILON}
        )
        and not _is_nonclosing_zero_shift_permanence(
            {**row, "velocity_forward_mps": -0.8}
        )
        and not _is_nonclosing_zero_shift_permanence(
            {**row, "disposition": "MEASURED"}
        )
        and not _is_nonclosing_zero_shift_permanence(
            {**row, "transport_state": "DIRECTION_CONSISTENT_BRANCH_CONTINUATION"}
        )
        and not _is_nonclosing_zero_shift_permanence(
            {**row, "transport_contradictions": 1}
        ),
        "x78_nonclosing_zero_shift_permanence_partition",
    )
    return {
        "status": "X78_NONCLOSING_ZERO_SHIFT_PERMANENCE_FALSIFIER_MET",
        "release_only": True,
        "identity_memory_retained": True,
        "approaching_permanence_retained": True,
        "measured_and_metric_carriers_retained": True,
        "velocity_sign_test_inherited": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
