"""X38 metric closing-motion bootstrap for the surface-flow arm.

X38 runs the inherited X24 metric tracker beside X37 on the same detector and
depth observations.  It may fill an X37-negative frame only when X24 has
already confirmed route risk from a currently measured, depth-supported track
whose longitudinal closing motion strictly dominates its lateral motion.  The
rule is class-independent and sign/direction based; no score or speed threshold
is introduced.  All other X24-only risks are ignored.

C16 is consumed same-source synthetic Development and cannot confirm X38.
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
import dtr_carla_x37_motion_evidence_credit_predictor as x37  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X38_METRIC_CLOSING_MOTION_BOOTSTRAP"
ARM_X38 = "X38_ISSUED_PLAN_METRIC_CLOSING_MOTION_BOOTSTRAP"


def fixed_constants() -> dict[str, Any]:
    return {
        **x37.fixed_constants(),
        "representation": "DUAL_REPRESENTATION_METRIC_CLOSING_BOOTSTRAP",
        "bootstrap_source": "INHERITED_X24_CONFIRMED_METRIC_TRACK",
        "bootstrap_eligibility": (
            "CURRENTLY_MEASURED_AND_DEPTH_SUPPORTED_AND_LONGITUDINAL_CLOSING_DOMINATES_LATERAL"
        ),
        "bootstrap_class_rule": "CLASS_INDEPENDENT",
        "bootstrap_numeric_speed_threshold": None,
        "x24_generic_union": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def dominant_longitudinal_closing(row: Mapping[str, Any]) -> bool:
    forward = float(row["velocity_forward_mps"])
    lateral = float(row["velocity_right_mps"])
    return (
        row.get("disposition") == "MEASURED"
        and row.get("depth_grid_support") is not None
        and forward < -abs(lateral)
    )


def bootstrap_track(row: Mapping[str, Any]) -> dict[str, Any]:
    track_id = f"x24-closing::{row['track_id']}"
    return {
        **row,
        "track_id": track_id,
        "parent_track_id": track_id,
        "motion_authority": x27.RIGID_DYNAMIC,
        "risk_eligible": True,
        "surface_transport_branch_count": 0,
        "authorized_surface_transport_branches": 0,
        "transport_lineage_pairs": 0,
        "transport_anchor_pairs": 0,
        "support_footprint_mode": "X24_METRIC_POINT_BOOTSTRAP",
        "metric_closing_bootstrap": True,
    }


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    value = x37.predict_episode(episode, candidate_values, calibration)
    metric = x24.predict_episode(episode, candidate_values, calibration)
    bootstrap_frames = 0
    for surface_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(surface_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(surface_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x38_frame_alignment",
        )
        arm = surface_frame["arms"][x37.ARM_X37]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        selected: list[dict[str, Any]] = []
        if not bool(arm["route_risk"]) and bool(metric_arm["route_risk"]):
            confirmed = set(metric_arm.get("confirmed_risk_track_ids", []))
            selected = [
                bootstrap_track(row)
                for row in metric_frame["tracks"]
                if str(row["track_id"]) in confirmed
                and dominant_longitudinal_closing(row)
            ]
        if selected:
            bootstrap_frames += 1
            surface_frame["tracks"].extend(selected)
            surface_frame["risk_eligible_tracks"] = int(
                surface_frame["risk_eligible_tracks"]
            ) + len(selected)
            track_ids = sorted(str(row["track_id"]) for row in selected)
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
                    "metric_closing_bootstrap_used": True,
                    "metric_closing_bootstrap_track_ids": track_ids,
                }
            )
        else:
            arm["metric_closing_bootstrap_used"] = False
            arm["metric_closing_bootstrap_track_ids"] = []

    value["arms"][ARM_X38] = value["arms"].pop(x37.ARM_X37)
    value["diagnostics"]["x38_route_mode_counts"] = value["diagnostics"].pop(
        "x37_route_mode_counts"
    )
    value["diagnostics"]["metric_closing_bootstrap_frames"] = bootstrap_frames
    for frame in value["frames"]:
        frame["arms"][ARM_X38] = frame["arms"].pop(x37.ARM_X37)
    return value


def self_check() -> dict[str, Any]:
    inherited = x37.self_check()
    x24.require(
        dominant_longitudinal_closing(
            {
                "disposition": "MEASURED",
                "depth_grid_support": 10,
                "velocity_forward_mps": -2.0,
                "velocity_right_mps": 0.2,
            }
        ),
        "x38_closing_rule_positive",
    )
    x24.require(
        not dominant_longitudinal_closing(
            {
                "disposition": "HOLD",
                "depth_grid_support": None,
                "velocity_forward_mps": -2.0,
                "velocity_right_mps": 0.2,
            }
        ),
        "x38_hold_rejected",
    )
    return {
        "status": "X38_METRIC_CLOSING_BOOTSTRAP_STRUCTURAL_FALSIFIER_MET",
        "x37_structural_status": inherited["status"],
        "class_independent": True,
        "numeric_speed_threshold_added": False,
        "generic_x24_union": False,
    }
