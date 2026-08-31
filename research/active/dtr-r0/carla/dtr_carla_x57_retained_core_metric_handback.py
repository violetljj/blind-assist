"""X57 reliability-routed metric handback on the retained X54 core.

C26 showed that X55 parent-sibling promotion contributed eleven false-positive
frames while X56's zero-eligible metric handback contributed three true-positive
frames.  X57 therefore keeps X54 as the route-risk core and composes only the
observable X56 handback rule.  It adds no detector, score, duration, route,
association, or numeric threshold.

C26 is consumed synthetic Development and cannot confirm X57.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x54_metric_bootstrap_dropout_continuation as x54  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X57_RETAINED_CORE_METRIC_HANDBACK"
ARM_X57 = "X57_ISSUED_PLAN_RETAINED_CORE_METRIC_HANDBACK"
HANDBACK_DISPOSITION = "X24_CONFIRMED_METRIC_HANDBACK"


def fixed_constants() -> dict[str, Any]:
    return {
        **x54.fixed_constants(),
        "representation": "X54_RETAINED_CORE_WITH_ZERO_ELIGIBLE_METRIC_HANDBACK",
        "retained_core": "X54",
        "excluded_promotion_path": "X55_PARENT_SIBLING_STATE_CYCLE_CONSENSUS",
        "handback_precondition": (
            "X54_RISK_ELIGIBLE_TRACK_COUNT_ZERO_AND_CURRENT_X24_ISSUED_PLAN_"
            "RISK_CONFIRMED"
        ),
        "handback_conflict_rule": "NO_X44_OR_X45_CAUSAL_CONFLICT",
        "handback_disposition": HANDBACK_DISPOSITION,
        "handback_is_current_fused_measurement": False,
        "metric_confirmation_threshold": "INHERITED_X24",
        "new_numeric_threshold_added": False,
    }


def handback_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    source_track_id = str(row["track_id"])
    value.update(
        {
            "track_id": f"x57-handback::{source_track_id}",
            "parent_track_id": f"x57-metric-parent::{source_track_id}",
            "disposition": HANDBACK_DISPOSITION,
            "motion_authority": x27.RIGID_DYNAMIC,
            "risk_eligible": True,
            "support_footprint_mode": "X24_CONFIRMED_METRIC_TRACK",
            "x57_metric_source_track_id": source_track_id,
            "x57_zero_eligible_metric_handback": True,
        }
    )
    return value


def apply_retained_core_metric_handback_episode(
    value: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(value)
    handback_frames = 0
    handed_back_tracks = 0

    for core_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(core_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(core_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x57_frame_alignment",
        )
        arm = core_frame["arms"][x54.ARM_X54]
        arm["x57_zero_eligible_metric_handback_used"] = False
        arm["x57_zero_eligible_metric_handback_track_ids"] = []
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        if (
            bool(arm.get("route_risk"))
            or int(core_frame.get("risk_eligible_tracks", 0)) != 0
            or bool(arm.get("x44_velocity_cycle_suppressed", False))
            or bool(arm.get("x45_state_cycle_suppressed", False))
            or not bool(metric_arm.get("route_risk"))
            or metric_arm.get("authority") != "VALID"
            or metric_arm.get("route_mode") != "ISSUED_PLAN"
        ):
            continue

        metric_tracks = {
            str(row["track_id"]): row for row in metric_frame["tracks"]
        }
        sources = [
            metric_tracks[str(track_id)]
            for track_id in metric_arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in metric_tracks
        ]
        if not sources:
            continue
        carriers = [handback_row(row) for row in sources]
        existing_ids = {str(row["track_id"]) for row in core_frame["tracks"]}
        for row in carriers:
            x24.require(
                str(row["track_id"]) not in existing_ids,
                "x57_handback_track_id_collision",
            )
            core_frame["tracks"].append(row)
            core_frame["risk_eligible_tracks"] = int(
                core_frame["risk_eligible_tracks"]
            ) + 1
        track_ids = sorted(str(row["track_id"]) for row in carriers)
        parent_ids = sorted(str(row["parent_track_id"]) for row in carriers)
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": metric_arm.get("minimum_entry_s"),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x57_zero_eligible_metric_handback_used": True,
                "x57_zero_eligible_metric_handback_track_ids": track_ids,
            }
        )
        handback_frames += 1
        handed_back_tracks += len(carriers)

    value["arms"][ARM_X57] = value["arms"].pop(x54.ARM_X54)
    value["diagnostics"]["x57_route_mode_counts"] = value["diagnostics"].pop(
        "x54_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x57_zero_eligible_metric_handback_frames": handback_frames,
            "x57_zero_eligible_metric_handback_tracks": handed_back_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X57] = frame["arms"].pop(x54.ARM_X54)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_retained_core_metric_handback_episode(
        x54.predict_episode(episode, candidate_values, calibration),
        x24.predict_episode(episode, candidate_values, calibration),
    )


def self_check() -> dict[str, Any]:
    row = handback_row(
        {
            "track_id": "metric-1",
            "position_forward_m": 4.0,
            "position_right_m": 0.0,
            "velocity_forward_mps": -1.0,
            "velocity_right_mps": 0.0,
        }
    )
    x24.require(
        row["motion_authority"] == x27.RIGID_DYNAMIC
        and row["risk_eligible"] is True
        and row["disposition"] == HANDBACK_DISPOSITION,
        "x57_authorized_handback_row",
    )
    return {
        "status": "X57_RETAINED_CORE_METRIC_HANDBACK_FALSIFIER_MET",
        "retained_core": "X54",
        "x55_route_promotion_inherited": False,
        "zero_core_eligible_tracks_required": True,
        "current_x24_issued_plan_confirmation_required": True,
        "causal_conflict_release_preserved": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
