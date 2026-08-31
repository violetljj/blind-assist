"""X42 dual-representation instant closing-motion consensus.

X42 keeps X41 and may originate route risk before X24's second route-confirming
frame only when two current representations agree on motion: X24 supplies a
currently measured, depth-supported route candidate with longitudinal closing
motion, and the surface arm supplies a currently measured rigid-dynamic branch
with the same dominant axis and a positive velocity dot product.  The metric
candidate becomes the confirmed risk reference; the surface branch remains an
explicit witness rather than being fused as the same object.

The rule is causal and class-independent.  It adds no numeric speed, score,
duration, detector, association, or route threshold.

C16 is consumed same-source synthetic Development and cannot confirm X42.
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
import dtr_carla_x41_metric_credentialed_parent_continuation as x41  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X42_INSTANT_CLOSING_MOTION_CONSENSUS"
ARM_X42 = "X42_ISSUED_PLAN_INSTANT_CLOSING_MOTION_CONSENSUS"


def fixed_constants() -> dict[str, Any]:
    return {
        **x41.fixed_constants(),
        "representation": "DUAL_REPRESENTATION_INSTANT_CLOSING_CONSENSUS",
        "metric_evidence": "CURRENT_MEASURED_DEPTH_SUPPORTED_X24_ROUTE_CANDIDATE",
        "surface_evidence": "CURRENT_MEASURED_RIGID_DYNAMIC_BRANCH",
        "consensus_rule": "SAME_DOMINANT_AXIS_AND_POSITIVE_VELOCITY_DOT_PRODUCT",
        "consensus_identity_merge": False,
        "consensus_class_rule": "CLASS_INDEPENDENT",
        "consensus_numeric_speed_threshold": None,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def metric_closing_candidate(row: Mapping[str, Any]) -> bool:
    return (
        row.get("disposition") == "MEASURED"
        and row.get("depth_grid_support") is not None
        and float(row["velocity_forward_mps"])
        < -abs(float(row["velocity_right_mps"]))
    )


def longitudinal_dominant(row: Mapping[str, Any]) -> bool:
    return abs(float(row["velocity_forward_mps"])) > abs(
        float(row["velocity_right_mps"])
    )


def positive_velocity_dot(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return (
        float(left["velocity_forward_mps"]) * float(right["velocity_forward_mps"])
        + float(left["velocity_right_mps"]) * float(right["velocity_right_mps"])
        > 0.0
    )


def surface_witness(
    metric_row: Mapping[str, Any], surface_row: Mapping[str, Any]
) -> bool:
    return (
        surface_row.get("disposition") == "MEASURED"
        and surface_row.get("motion_authority") == x27.RIGID_DYNAMIC
        and longitudinal_dominant(metric_row) == longitudinal_dominant(surface_row)
        and positive_velocity_dot(metric_row, surface_row)
    )


def consensus_track(row: Mapping[str, Any]) -> dict[str, Any]:
    track_id = f"x24-consensus::{row['track_id']}"
    return {
        **row,
        "track_id": track_id,
        "parent_track_id": track_id,
        "disposition": "METRIC_SURFACE_CLOSING_CONSENSUS",
        "motion_authority": x27.RIGID_DYNAMIC,
        "risk_eligible": True,
        "surface_transport_branch_count": 0,
        "authorized_surface_transport_branches": 0,
        "transport_lineage_pairs": 0,
        "transport_anchor_pairs": 0,
        "support_footprint_mode": "X24_METRIC_WITH_SURFACE_MOTION_WITNESS",
        "instant_closing_motion_consensus": True,
    }


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    value = x41.predict_episode(episode, candidate_values, calibration)
    metric = x24.predict_episode(episode, candidate_values, calibration)
    consensus_frames = 0

    for fused_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(fused_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(fused_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x42_frame_alignment",
        )
        arm = fused_frame["arms"][x41.ARM_X41]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        candidate_ids = set(metric_arm.get("candidate_risk_track_ids", []))
        metric_candidates = [
            row
            for row in metric_frame["tracks"]
            if str(row["track_id"]) in candidate_ids and metric_closing_candidate(row)
        ]
        surface_rows = [
            row
            for row in fused_frame["tracks"]
            if row.get("disposition") == "MEASURED"
            and row.get("motion_authority") == x27.RIGID_DYNAMIC
        ]
        selected: list[dict[str, Any]] = []
        witness_parents: set[str] = set()
        if not bool(arm["route_risk"]):
            for metric_row in metric_candidates:
                witnesses = [
                    row for row in surface_rows if surface_witness(metric_row, row)
                ]
                if not witnesses:
                    continue
                selected.append(consensus_track(metric_row))
                witness_parents.update(str(row["parent_track_id"]) for row in witnesses)

        if selected:
            consensus_frames += 1
            fused_frame["tracks"].extend(selected)
            fused_frame["risk_eligible_tracks"] = int(
                fused_frame["risk_eligible_tracks"]
            ) + len(selected)
            track_ids = sorted(str(row["track_id"]) for row in selected)
            arm.update(
                {
                    "route_risk": True,
                    "minimum_entry_s": metric_arm.get("minimum_entry_s"),
                    "candidate_risk_track_ids": track_ids,
                    "confirmed_risk_track_ids": track_ids,
                    "candidate_risk_parent_track_ids": track_ids,
                    "confirmed_risk_parent_track_ids": track_ids,
                    "instant_closing_motion_consensus_used": True,
                    "instant_closing_motion_consensus_track_ids": track_ids,
                    "instant_closing_motion_surface_witness_parent_ids": sorted(
                        witness_parents
                    ),
                }
            )
        else:
            arm["instant_closing_motion_consensus_used"] = False
            arm["instant_closing_motion_consensus_track_ids"] = []
            arm["instant_closing_motion_surface_witness_parent_ids"] = []

    value["arms"][ARM_X42] = value["arms"].pop(x41.ARM_X41)
    value["diagnostics"]["x42_route_mode_counts"] = value["diagnostics"].pop(
        "x41_route_mode_counts"
    )
    value["diagnostics"]["instant_closing_motion_consensus_frames"] = (
        consensus_frames
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X42] = frame["arms"].pop(x41.ARM_X41)
    return value


def self_check() -> dict[str, Any]:
    inherited = x41.self_check()
    metric = {
        "disposition": "MEASURED",
        "depth_grid_support": 1,
        "velocity_forward_mps": -2.0,
        "velocity_right_mps": 0.2,
    }
    surface = {
        "disposition": "MEASURED",
        "motion_authority": x27.RIGID_DYNAMIC,
        "velocity_forward_mps": -1.0,
        "velocity_right_mps": 0.0,
    }
    x24.require(metric_closing_candidate(metric), "x42_metric_closing_positive")
    x24.require(surface_witness(metric, surface), "x42_surface_witness_positive")
    return {
        "status": "X42_INSTANT_CLOSING_CONSENSUS_STRUCTURAL_FALSIFIER_MET",
        "x41_structural_status": inherited["status"],
        "identity_merge": False,
        "class_independent": True,
        "numeric_speed_threshold_added": False,
    }
