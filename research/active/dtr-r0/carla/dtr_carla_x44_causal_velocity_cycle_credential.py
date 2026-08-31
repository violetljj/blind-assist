"""X44 causal velocity-cycle credential for dual-representation route risk.

X44 preserves X43 except when an available cross-representation velocity edge
is directionally inconsistent.  A fused risk with a current X24 route
candidate must have a positive velocity dot product with at least one current
metric candidate.  A metric temporal handoff must additionally close the
cycle back to an authorized fused risk in the immediately previous frame.

This is a class-independent causal consistency rule.  It adds no numeric
speed, score, duration, detector, association, or route threshold.

C21 is consumed posthoc synthetic Development and cannot confirm X44.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x43_authority_preserving_credential_belief as x43  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X44_CAUSAL_VELOCITY_CYCLE_CREDENTIAL"
ARM_X44 = "X44_ISSUED_PLAN_CAUSAL_VELOCITY_CYCLE_CREDENTIAL"


def fixed_constants() -> dict[str, Any]:
    return {
        **x43.fixed_constants(),
        "representation": "DUAL_REPRESENTATION_CAUSAL_VELOCITY_CYCLE",
        "current_metric_edge_rule": (
            "IF_CURRENT_X24_ROUTE_CANDIDATE_EXISTS_THEN_POSITIVE_VELOCITY_DOT_"
            "WITH_CONFIRMED_FUSED_RISK_REQUIRED"
        ),
        "handoff_cycle_rule": (
            "METRIC_TEMPORAL_HANDOFF_MUST_HAVE_POSITIVE_VELOCITY_DOT_WITH_"
            "IMMEDIATELY_PREVIOUS_AUTHORIZED_FUSED_RISK"
        ),
        "velocity_cycle_class_rule": "CLASS_INDEPENDENT",
        "velocity_cycle_numeric_speed_threshold": None,
        "velocity_cycle_duration_threshold": None,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def positive_velocity_dot(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return (
        float(left["velocity_forward_mps"])
        * float(right["velocity_forward_mps"])
        + float(left["velocity_right_mps"])
        * float(right["velocity_right_mps"])
        > 0.0
    )


def any_positive_velocity_edge(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> bool:
    return any(
        positive_velocity_dot(left_row, right_row)
        for left_row in left
        for right_row in right
    )


def selected_rows(
    arm: Mapping[str, Any], tracks: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    return [
        dict(tracks[str(track_id)])
        for track_id in arm.get("confirmed_risk_track_ids", [])
        if str(track_id) in tracks
    ]


def suppress_arm(
    arm: dict[str, Any], *, reason: str, track_ids: Sequence[str]
) -> None:
    arm.update(
        {
            "route_risk": False,
            "minimum_entry_s": None,
            "candidate_risk_track_ids": [],
            "confirmed_risk_track_ids": [],
            "candidate_risk_parent_track_ids": [],
            "confirmed_risk_parent_track_ids": [],
            "x44_velocity_cycle_suppressed": True,
            "x44_velocity_cycle_suppression_reason": reason,
            "x44_velocity_cycle_suppressed_track_ids": sorted(
                str(value) for value in track_ids
            ),
        }
    )


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    value = x43.predict_episode(episode, candidate_values, calibration)
    metric = x24.predict_episode(episode, candidate_values, calibration)
    previous_authorized_rows: list[dict[str, Any]] = []
    previous_authorized_risk = False
    current_edge_suppressions = 0
    handoff_cycle_suppressions = 0

    for fused_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(fused_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(fused_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x44_frame_alignment",
        )
        arm = fused_frame["arms"][x43.ARM_X43]
        fused_tracks = {
            str(row["track_id"]): row for row in fused_frame["tracks"]
        }
        metric_tracks = {
            str(row["track_id"]): row for row in metric_frame["tracks"]
        }
        confirmed = selected_rows(arm, fused_tracks)
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        metric_candidates = [
            metric_tracks[str(track_id)]
            for track_id in metric_arm.get("candidate_risk_track_ids", [])
            if str(track_id) in metric_tracks
        ]
        confirmed_ids = [str(row["track_id"]) for row in confirmed]
        reason: str | None = None

        if (
            bool(arm.get("route_risk"))
            and metric_candidates
            and not any_positive_velocity_edge(confirmed, metric_candidates)
        ):
            reason = "CURRENT_METRIC_EDGE_DIRECTION_CONFLICT"
            current_edge_suppressions += 1
        elif bool(arm.get("metric_temporal_handoff_used", False)) and not (
            previous_authorized_risk
            and any_positive_velocity_edge(previous_authorized_rows, confirmed)
        ):
            reason = "HANDOFF_VELOCITY_CYCLE_NOT_CLOSED"
            handoff_cycle_suppressions += 1

        if reason is not None:
            suppress_arm(arm, reason=reason, track_ids=confirmed_ids)
        else:
            arm["x44_velocity_cycle_suppressed"] = False
            arm["x44_velocity_cycle_suppression_reason"] = None
            arm["x44_velocity_cycle_suppressed_track_ids"] = []

        previous_authorized_risk = bool(arm.get("route_risk"))
        previous_authorized_rows = confirmed if previous_authorized_risk else []

    value["arms"][ARM_X44] = value["arms"].pop(x43.ARM_X43)
    value["diagnostics"]["x44_route_mode_counts"] = value["diagnostics"].pop(
        "x43_route_mode_counts"
    )
    value["diagnostics"]["x44_current_metric_edge_suppressions"] = (
        current_edge_suppressions
    )
    value["diagnostics"]["x44_handoff_cycle_suppressions"] = (
        handoff_cycle_suppressions
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X44] = frame["arms"].pop(x43.ARM_X43)
    return value


def self_check() -> dict[str, Any]:
    inherited = x43.self_check()
    closing = {"velocity_forward_mps": -2.0, "velocity_right_mps": 0.2}
    agreeing = {"velocity_forward_mps": -1.0, "velocity_right_mps": 0.1}
    conflicting = {"velocity_forward_mps": 1.0, "velocity_right_mps": 0.0}
    x24.require(
        positive_velocity_dot(closing, agreeing),
        "x44_positive_velocity_edge",
    )
    x24.require(
        not positive_velocity_dot(closing, conflicting),
        "x44_conflicting_velocity_edge",
    )
    return {
        "status": "X44_CAUSAL_VELOCITY_CYCLE_STRUCTURAL_FALSIFIER_MET",
        "x43_structural_status": inherited["status"],
        "current_metric_edge_required_when_observable": True,
        "handoff_cycle_closure_required": True,
        "class_independent": True,
        "numeric_speed_threshold_added": False,
    }
