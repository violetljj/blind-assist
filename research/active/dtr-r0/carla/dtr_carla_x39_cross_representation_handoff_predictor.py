"""X39 cross-representation temporal handoff for brief surface silence.

X39 keeps X38 as the primary risk source.  When X38 has just issued route
risk, it may preserve that risk through a later X38-negative frame only when
the inherited X24 metric tracker still confirms the same currently held track
and its last measured motion is longitudinally dominant.  This is a causal,
one-frame-at-a-time handoff: metric HOLD cannot originate a new alert, but a
successful handoff may support the next frame.  No score, speed, duration, or
detector threshold is introduced.

C16 is consumed same-source synthetic Development and cannot confirm X39.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x38_metric_closing_bootstrap_predictor as x38  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X39_CROSS_REPRESENTATION_TEMPORAL_HANDOFF"
ARM_X39 = "X39_ISSUED_PLAN_CROSS_REPRESENTATION_TEMPORAL_HANDOFF"


def fixed_constants() -> dict[str, Any]:
    return {
        **x38.fixed_constants(),
        "representation": "DUAL_REPRESENTATION_CAUSAL_TEMPORAL_HANDOFF",
        "handoff_origin": "PREVIOUS_FUSED_ROUTE_RISK_REQUIRED",
        "handoff_source": "INHERITED_X24_CONFIRMED_METRIC_HOLD",
        "handoff_motion_rule": "ABS_LONGITUDINAL_STRICTLY_DOMINATES_ABS_LATERAL",
        "handoff_class_rule": "CLASS_INDEPENDENT",
        "handoff_numeric_speed_threshold": None,
        "handoff_duration_threshold": None,
        "x24_generic_union": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def eligible_metric_hold(row: Mapping[str, Any]) -> bool:
    return (
        row.get("disposition") == "HOLD"
        and abs(float(row["velocity_forward_mps"]))
        > abs(float(row["velocity_right_mps"]))
    )


def handoff_track(row: Mapping[str, Any]) -> dict[str, Any]:
    track_id = f"x24-handoff::{row['track_id']}"
    return {
        **row,
        "track_id": track_id,
        "parent_track_id": track_id,
        "disposition": "METRIC_TEMPORAL_HANDOFF",
        "motion_authority": x27.RIGID_DYNAMIC,
        "risk_eligible": True,
        "surface_transport_branch_count": 0,
        "authorized_surface_transport_branches": 0,
        "transport_lineage_pairs": 0,
        "transport_anchor_pairs": 0,
        "support_footprint_mode": "X24_METRIC_TEMPORAL_HANDOFF",
        "metric_temporal_handoff": True,
    }


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    value = x38.predict_episode(episode, candidate_values, calibration)
    metric = x24.predict_episode(episode, candidate_values, calibration)
    previous_fused_risk = False
    handoff_frames = 0
    handoff_track_ids: set[str] = set()

    for fused_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(fused_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(fused_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x39_frame_alignment",
        )
        arm = fused_frame["arms"][x38.ARM_X38]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        selected: list[dict[str, Any]] = []
        if not bool(arm["route_risk"]) and previous_fused_risk:
            confirmed = set(metric_arm.get("confirmed_risk_track_ids", []))
            selected = [
                handoff_track(row)
                for row in metric_frame["tracks"]
                if str(row["track_id"]) in confirmed and eligible_metric_hold(row)
            ]

        if selected:
            handoff_frames += 1
            fused_frame["tracks"].extend(selected)
            fused_frame["risk_eligible_tracks"] = int(
                fused_frame["risk_eligible_tracks"]
            ) + len(selected)
            track_ids = sorted(str(row["track_id"]) for row in selected)
            handoff_track_ids.update(track_ids)
            arm.update(
                {
                    "route_risk": True,
                    "minimum_entry_s": metric_arm.get("minimum_entry_s"),
                    "candidate_risk_track_ids": sorted(
                        set(arm.get("candidate_risk_track_ids", [])) | set(track_ids)
                    ),
                    "confirmed_risk_track_ids": track_ids,
                    "candidate_risk_parent_track_ids": sorted(
                        set(arm.get("candidate_risk_parent_track_ids", []))
                        | set(track_ids)
                    ),
                    "confirmed_risk_parent_track_ids": track_ids,
                    "metric_temporal_handoff_used": True,
                    "metric_temporal_handoff_track_ids": track_ids,
                }
            )
        else:
            arm["metric_temporal_handoff_used"] = False
            arm["metric_temporal_handoff_track_ids"] = []

        previous_fused_risk = bool(arm["route_risk"])

    value["arms"][ARM_X39] = value["arms"].pop(x38.ARM_X38)
    value["diagnostics"]["x39_route_mode_counts"] = value["diagnostics"].pop(
        "x38_route_mode_counts"
    )
    value["diagnostics"]["metric_temporal_handoff_frames"] = handoff_frames
    value["diagnostics"]["metric_temporal_handoff_unique_tracks"] = len(
        handoff_track_ids
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X39] = frame["arms"].pop(x38.ARM_X38)
    return value


def self_check() -> dict[str, Any]:
    inherited = x38.self_check()
    x24.require(
        eligible_metric_hold(
            {
                "disposition": "HOLD",
                "velocity_forward_mps": -2.0,
                "velocity_right_mps": 0.2,
            }
        ),
        "x39_handoff_rule_positive",
    )
    x24.require(
        not eligible_metric_hold(
            {
                "disposition": "MEASURED",
                "velocity_forward_mps": -2.0,
                "velocity_right_mps": 0.2,
            }
        ),
        "x39_measured_rejected_from_handoff",
    )
    return {
        "status": "X39_CROSS_REPRESENTATION_HANDOFF_STRUCTURAL_FALSIFIER_MET",
        "x38_structural_status": inherited["status"],
        "causal_previous_risk_required": True,
        "class_independent": True,
        "numeric_speed_threshold_added": False,
        "generic_x24_union": False,
    }
