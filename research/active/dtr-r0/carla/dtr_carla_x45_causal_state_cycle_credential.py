"""X45 causal state-cycle credential for dual-representation route risk.

X45 preserves X44 except when a current X24 route candidate exists but no
single fused/metric pair closes both the velocity-direction edge and the
position-association edge.  The position edge reuses X24's frozen association
distance; X45 introduces no numeric threshold.

C22 is consumed posthoc synthetic Development and cannot confirm X45.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x44_causal_velocity_cycle_credential as x44  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X45_CAUSAL_STATE_CYCLE_CREDENTIAL"
ARM_X45 = "X45_ISSUED_PLAN_CAUSAL_STATE_CYCLE_CREDENTIAL"


def fixed_constants() -> dict[str, Any]:
    return {
        **x44.fixed_constants(),
        "representation": "DUAL_REPRESENTATION_CAUSAL_STATE_CYCLE",
        "current_state_cycle_rule": (
            "IF_CURRENT_X24_ROUTE_CANDIDATE_EXISTS_THEN_ONE_FUSED_METRIC_PAIR_"
            "MUST_CLOSE_BOTH_POSITION_ASSOCIATION_AND_POSITIVE_VELOCITY_DOT"
        ),
        "position_association_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "position_threshold_source": "INHERITED_X24_ASSOCIATION_DISTANCE",
        "state_cycle_class_rule": "CLASS_INDEPENDENT",
        "new_numeric_threshold_added": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "score_threshold_change": False,
    }


def position_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    return math.hypot(
        float(left["position_forward_m"]) - float(right["position_forward_m"]),
        float(left["position_right_m"]) - float(right["position_right_m"]),
    )


def closes_state_cycle(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        position_distance(left, right) <= x24.ASSOCIATION_DISTANCE_M + x31.EPSILON
        and x44.positive_velocity_dot(left, right)
    )


def any_closed_state_cycle(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> bool:
    return any(
        closes_state_cycle(left_row, right_row)
        for left_row in left
        for right_row in right
    )


def suppress_arm(arm: dict[str, Any], track_ids: Sequence[str]) -> None:
    arm.update(
        {
            "route_risk": False,
            "minimum_entry_s": None,
            "candidate_risk_track_ids": [],
            "confirmed_risk_track_ids": [],
            "candidate_risk_parent_track_ids": [],
            "confirmed_risk_parent_track_ids": [],
            "x45_state_cycle_suppressed": True,
            "x45_state_cycle_suppression_reason": (
                "CURRENT_FUSED_METRIC_STATE_CYCLE_NOT_CLOSED"
            ),
            "x45_state_cycle_suppressed_track_ids": sorted(
                str(value) for value in track_ids
            ),
        }
    )


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    value = x44.predict_episode(episode, candidate_values, calibration)
    metric = x24.predict_episode(episode, candidate_values, calibration)
    state_cycle_suppressions = 0

    for fused_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(fused_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(fused_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x45_frame_alignment",
        )
        arm = fused_frame["arms"][x44.ARM_X44]
        fused_tracks = {
            str(row["track_id"]): row for row in fused_frame["tracks"]
        }
        metric_tracks = {
            str(row["track_id"]): row for row in metric_frame["tracks"]
        }
        confirmed = x44.selected_rows(arm, fused_tracks)
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        metric_candidates = [
            metric_tracks[str(track_id)]
            for track_id in metric_arm.get("candidate_risk_track_ids", [])
            if str(track_id) in metric_tracks
        ]
        confirmed_ids = [str(row["track_id"]) for row in confirmed]

        if (
            bool(arm.get("route_risk"))
            and metric_candidates
            and not any_closed_state_cycle(confirmed, metric_candidates)
        ):
            suppress_arm(arm, confirmed_ids)
            state_cycle_suppressions += 1
        else:
            arm["x45_state_cycle_suppressed"] = False
            arm["x45_state_cycle_suppression_reason"] = None
            arm["x45_state_cycle_suppressed_track_ids"] = []

    value["arms"][ARM_X45] = value["arms"].pop(x44.ARM_X44)
    value["diagnostics"]["x45_route_mode_counts"] = value["diagnostics"].pop(
        "x44_route_mode_counts"
    )
    value["diagnostics"]["x45_current_state_cycle_suppressions"] = (
        state_cycle_suppressions
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X45] = frame["arms"].pop(x44.ARM_X44)
    return value


def self_check() -> dict[str, Any]:
    inherited = x44.self_check()
    fused = {
        "position_forward_m": 4.0,
        "position_right_m": 0.2,
        "velocity_forward_mps": -2.0,
        "velocity_right_mps": 0.1,
    }
    associated = {
        "position_forward_m": 4.4,
        "position_right_m": 0.3,
        "velocity_forward_mps": -1.0,
        "velocity_right_mps": 0.1,
    }
    distant = {**associated, "position_forward_m": 7.0}
    x24.require(closes_state_cycle(fused, associated), "x45_closed_state_cycle")
    x24.require(
        not closes_state_cycle(fused, distant), "x45_position_cycle_conflict"
    )
    return {
        "status": "X45_CAUSAL_STATE_CYCLE_STRUCTURAL_FALSIFIER_MET",
        "x44_structural_status": inherited["status"],
        "same_pair_position_and_velocity_closure_required": True,
        "position_threshold_inherited_from_x24": True,
        "class_independent": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
