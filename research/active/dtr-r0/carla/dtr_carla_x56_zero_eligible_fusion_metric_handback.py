"""X56 confirmed-metric handback for a zero-eligible fused representation.

X56 preserves X55.  If the fused branch has no risk-eligible track at all,
X55 is silent, and current X24 issued-plan risk is already confirmed, X56
hands the confirmed metric state back as an explicit rigid-dynamic belief.
The exception is unavailable when either X44 or X45 reports a causal conflict.

This repairs representation collapse; it does not lower detector, score,
duration, route, or association thresholds.  C24 is consumed synthetic
Development and cannot confirm X56.
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
import dtr_carla_x55_parent_sibling_state_cycle_consensus as x55  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X56_ZERO_ELIGIBLE_FUSION_METRIC_HANDBACK"
ARM_X56 = "X56_ISSUED_PLAN_ZERO_ELIGIBLE_FUSION_METRIC_HANDBACK"
HANDBACK_DISPOSITION = "X24_CONFIRMED_METRIC_HANDBACK"


def fixed_constants() -> dict[str, Any]:
    return {
        **x55.fixed_constants(),
        "representation": "FUSED_WITH_ZERO_ELIGIBLE_METRIC_HANDBACK",
        "handback_precondition": (
            "FUSED_RISK_ELIGIBLE_TRACK_COUNT_ZERO_AND_CURRENT_X24_ISSUED_PLAN_"
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
            "track_id": f"x56-handback::{source_track_id}",
            "parent_track_id": f"x56-metric-parent::{source_track_id}",
            "disposition": HANDBACK_DISPOSITION,
            "motion_authority": x27.RIGID_DYNAMIC,
            "risk_eligible": True,
            "support_footprint_mode": "X24_CONFIRMED_METRIC_TRACK",
            "x56_metric_source_track_id": source_track_id,
            "x56_zero_eligible_fusion_metric_handback": True,
        }
    )
    return value


def apply_zero_eligible_metric_handback_episode(
    value: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(value)
    handback_frames = 0
    handed_back_tracks = 0

    for fused_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(fused_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(fused_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x56_frame_alignment",
        )
        arm = fused_frame["arms"][x55.ARM_X55]
        arm["x56_zero_eligible_metric_handback_used"] = False
        arm["x56_zero_eligible_metric_handback_track_ids"] = []
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        if (
            bool(arm.get("route_risk"))
            or int(fused_frame.get("risk_eligible_tracks", 0)) != 0
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
        existing_ids = {str(row["track_id"]) for row in fused_frame["tracks"]}
        for row in carriers:
            x24.require(
                str(row["track_id"]) not in existing_ids,
                "x56_handback_track_id_collision",
            )
            fused_frame["tracks"].append(row)
            fused_frame["risk_eligible_tracks"] = int(
                fused_frame["risk_eligible_tracks"]
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
                "x56_zero_eligible_metric_handback_used": True,
                "x56_zero_eligible_metric_handback_track_ids": track_ids,
            }
        )
        handback_frames += 1
        handed_back_tracks += len(carriers)

    value["arms"][ARM_X56] = value["arms"].pop(x55.ARM_X55)
    value["diagnostics"]["x56_route_mode_counts"] = value["diagnostics"].pop(
        "x55_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x56_zero_eligible_metric_handback_frames": handback_frames,
            "x56_zero_eligible_metric_handback_tracks": handed_back_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X56] = frame["arms"].pop(x55.ARM_X55)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_zero_eligible_metric_handback_episode(
        x55.predict_episode(episode, candidate_values, calibration),
        x24.predict_episode(episode, candidate_values, calibration),
    )


def self_check() -> dict[str, Any]:
    inherited = x55.self_check()
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
        "x56_authorized_handback_row",
    )
    return {
        "status": "X56_ZERO_ELIGIBLE_FUSION_METRIC_HANDBACK_FALSIFIER_MET",
        "x55_structural_status": inherited["status"],
        "zero_fused_eligible_tracks_required": True,
        "current_x24_issued_plan_confirmation_required": True,
        "causal_conflict_release_preserved": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
