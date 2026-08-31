"""X40 cross-representation adjudication for lateral-only surface risk.

X40 keeps X39 unless every currently confirmed surface-flow branch is lateral
dominant and the inherited X24 metric arm has no current route-risk candidate.
This treats metric and surface tracking as complementary witnesses: a purely
lateral collision claim needs current path-level corroboration from the metric
representation, while any longitudinal-dominant branch remains untouched.

The adjudication is causal, class-independent, and adds no numeric speed,
duration, score, detector, or route threshold.  A suppressed frame cannot seed
X39's temporal handoff in the following frame.

C16 is consumed same-source synthetic Development and cannot confirm X40.
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
import dtr_carla_x39_cross_representation_handoff_predictor as x39  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X40_CROSS_REPRESENTATION_LATERAL_ADJUDICATION"
ARM_X40 = "X40_ISSUED_PLAN_CROSS_REPRESENTATION_LATERAL_ADJUDICATION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x39.fixed_constants(),
        "representation": "DUAL_REPRESENTATION_LATERAL_RISK_ADJUDICATION",
        "adjudication_scope": "ALL_CONFIRMED_BRANCHES_LATERAL_DOMINANT",
        "adjudication_corroboration": "CURRENT_X24_ROUTE_RISK_CANDIDATE",
        "adjudication_class_rule": "CLASS_INDEPENDENT",
        "adjudication_numeric_speed_threshold": None,
        "adjudication_duration_threshold": None,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def lateral_only(rows: Sequence[Mapping[str, Any]]) -> bool:
    return bool(rows) and all(
        abs(float(row["velocity_forward_mps"]))
        <= abs(float(row["velocity_right_mps"]))
        for row in rows
    )


def suppress_arm(
    arm: dict[str, Any],
    reason: str,
    confirmed_ids: Sequence[str],
) -> None:
    suppressed_minimum_entry_s = arm.get("minimum_entry_s")
    arm.update(
        {
            "route_risk": False,
            "minimum_entry_s": None,
            "candidate_risk_track_ids": [],
            "confirmed_risk_track_ids": [],
            "candidate_risk_parent_track_ids": [],
            "confirmed_risk_parent_track_ids": [],
            "cross_representation_adjudication_suppressed": True,
            "cross_representation_adjudication_reason": reason,
            "cross_representation_suppressed_track_ids": sorted(confirmed_ids),
            "cross_representation_suppressed_minimum_entry_s": (
                suppressed_minimum_entry_s
            ),
        }
    )


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    value = x39.predict_episode(episode, candidate_values, calibration)
    metric = x24.predict_episode(episode, candidate_values, calibration)
    previous_x40_risk = False
    suppressed_frames = 0
    invalidated_handoffs = 0

    for fused_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(fused_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(fused_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x40_frame_alignment",
        )
        arm = fused_frame["arms"][x39.ARM_X39]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        confirmed_ids = [str(value) for value in arm.get("confirmed_risk_track_ids", [])]
        confirmed_set = set(confirmed_ids)
        confirmed_rows = [
            row for row in fused_frame["tracks"] if str(row["track_id"]) in confirmed_set
        ]
        x24_candidate = bool(metric_arm.get("candidate_risk_track_ids", []))
        invalid_handoff = bool(arm.get("metric_temporal_handoff_used", False)) and not (
            previous_x40_risk
        )
        uncorroborated_lateral = (
            bool(arm["route_risk"])
            and lateral_only(confirmed_rows)
            and not x24_candidate
        )

        if invalid_handoff:
            invalidated_handoffs += 1
            suppressed_frames += 1
            suppress_arm(arm, "HANDOFF_SOURCE_SUPPRESSED", confirmed_ids)
        elif uncorroborated_lateral:
            suppressed_frames += 1
            suppress_arm(
                arm,
                "LATERAL_ONLY_WITHOUT_CURRENT_METRIC_ROUTE_CANDIDATE",
                confirmed_ids,
            )
        else:
            arm["cross_representation_adjudication_suppressed"] = False
            arm["cross_representation_adjudication_reason"] = None
            arm["cross_representation_suppressed_track_ids"] = []
            arm["cross_representation_suppressed_minimum_entry_s"] = None

        previous_x40_risk = bool(arm["route_risk"])

    value["arms"][ARM_X40] = value["arms"].pop(x39.ARM_X39)
    value["diagnostics"]["x40_route_mode_counts"] = value["diagnostics"].pop(
        "x39_route_mode_counts"
    )
    value["diagnostics"]["lateral_adjudication_suppressed_frames"] = (
        suppressed_frames
    )
    value["diagnostics"]["invalidated_temporal_handoff_frames"] = invalidated_handoffs
    for frame in value["frames"]:
        frame["arms"][ARM_X40] = frame["arms"].pop(x39.ARM_X39)
    return value


def self_check() -> dict[str, Any]:
    inherited = x39.self_check()
    x24.require(
        lateral_only(
            [
                {"velocity_forward_mps": 0.0, "velocity_right_mps": -1.0},
                {"velocity_forward_mps": 1.0, "velocity_right_mps": 1.0},
            ]
        ),
        "x40_lateral_only_positive",
    )
    x24.require(
        not lateral_only(
            [{"velocity_forward_mps": -2.0, "velocity_right_mps": 0.5}]
        ),
        "x40_longitudinal_branch_preserved",
    )
    return {
        "status": "X40_LATERAL_ADJUDICATION_STRUCTURAL_FALSIFIER_MET",
        "x39_structural_status": inherited["status"],
        "causal_handoff_revalidated": True,
        "class_independent": True,
        "numeric_speed_threshold_added": False,
    }
