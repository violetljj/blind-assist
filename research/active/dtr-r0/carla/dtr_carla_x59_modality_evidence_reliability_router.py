"""X59 modality-evidence reliability routing around X54 plus X57 fallback.

X59 retains X54 and the X57 zero-eligible fallback.  A second positive route
admits only currently MEASURED X24 confirmations whose forward velocity is
negative (closing), while a negative route releases a same-track receding
object-permanence belief only when the current frame still contains direct
metric footprint measurements.  Thus both decisions are conditioned on active
modality evidence rather than a weather or lighting label.

No detector, score, duration, route, association, or numeric threshold is
added.  C26 and C27 are consumed synthetic Development and cannot confirm X59.
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
import dtr_carla_x57_retained_core_metric_handback as x57  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X59_MODALITY_EVIDENCE_RELIABILITY_ROUTER"
ARM_X59 = "X59_ISSUED_PLAN_MODALITY_EVIDENCE_RELIABILITY_ROUTER"
HANDBACK_DISPOSITION = "X24_CURRENT_MEASURED_CLOSING_METRIC_HANDBACK"
PERMANENCE_DISPOSITION = "OBJECT_PERMANENCE_BELIEF_HOLD"


def fixed_constants() -> dict[str, Any]:
    return {
        **x57.fixed_constants(),
        "representation": "X54_X57_WITH_MODALITY_EVIDENCE_RELIABILITY_ROUTER",
        "retained_core": "X54",
        "retained_component": "X57_ZERO_ELIGIBLE_METRIC_HANDBACK",
        "active_positive_route": (
            "CURRENT_MEASURED_X24_CONFIRMATION_WITH_NEGATIVE_FORWARD_VELOCITY"
        ),
        "hold_only_active_handback_allowed": False,
        "active_positive_route_conflict_rule": "NO_X44_VELOCITY_CYCLE_CONFLICT",
        "active_negative_route": (
            "SAME_RECEDING_OBJECT_PERMANENCE_BELIEF_ON_SECOND_CONSECUTIVE_FRAME_"
            "WITH_CURRENT_DIRECT_METRIC_MEASUREMENT_AND_X24_SILENT"
        ),
        "receding_rule": "VELOCITY_FORWARD_MPS_POSITIVE",
        "direct_metric_evidence_rule": "METRIC_FOOTPRINT_MEASUREMENTS_NONZERO",
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def handback_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    source_track_id = str(row["track_id"])
    value.update(
        {
            "track_id": f"x59-handback::{source_track_id}",
            "parent_track_id": f"x59-metric-parent::{source_track_id}",
            "disposition": HANDBACK_DISPOSITION,
            "motion_authority": x27.RIGID_DYNAMIC,
            "risk_eligible": True,
            "support_footprint_mode": "X24_CURRENT_MEASURED_CLOSING_METRIC_TRACK",
            "x59_metric_source_track_id": source_track_id,
            "x59_current_measured_closing_metric_handback": True,
        }
    )
    return value


def apply_modality_evidence_reliability_router_episode(
    core: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    value = x57.apply_retained_core_metric_handback_episode(core, metric)
    value = copy.deepcopy(value)
    closing_handback_frames = 0
    handed_back_tracks = 0
    evidence_supported_release_frames = 0
    released_tracks = 0
    previous_receding_track_ids: set[str] = set()

    for routed_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(routed_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(routed_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x59_frame_alignment",
        )
        arm = routed_frame["arms"][x57.ARM_X57]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        arm["x59_current_measured_closing_handback_used"] = False
        arm["x59_current_measured_closing_handback_track_ids"] = []
        arm["x59_evidence_supported_receding_release_used"] = False
        arm["x59_evidence_supported_receding_release_track_ids"] = []

        routed_tracks = {
            str(row["track_id"]): row for row in routed_frame["tracks"]
        }
        confirmed_rows = [
            routed_tracks[str(track_id)]
            for track_id in arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in routed_tracks
        ]
        current_receding_track_ids = {
            str(row["track_id"])
            for row in confirmed_rows
            if row.get("disposition") == PERMANENCE_DISPOSITION
            and float(row.get("velocity_forward_mps", 0.0)) > 0.0
        }
        release_ids = current_receding_track_ids & previous_receding_track_ids
        release = (
            bool(arm.get("route_risk"))
            and bool(confirmed_rows)
            and len(current_receding_track_ids) == len(confirmed_rows)
            and bool(release_ids)
            and int(routed_frame.get("metric_footprint_measurements", 0)) > 0
            and not bool(metric_arm.get("route_risk"))
        )
        if release:
            released = sorted(str(row["track_id"]) for row in confirmed_rows)
            arm.update(
                {
                    "route_risk": False,
                    "minimum_entry_s": None,
                    "candidate_risk_track_ids": [],
                    "confirmed_risk_track_ids": [],
                    "candidate_risk_parent_track_ids": [],
                    "confirmed_risk_parent_track_ids": [],
                    "x59_evidence_supported_receding_release_used": True,
                    "x59_evidence_supported_receding_release_track_ids": released,
                }
            )
            evidence_supported_release_frames += 1
            released_tracks += len(released)
        previous_receding_track_ids = current_receding_track_ids

        if (
            bool(arm.get("route_risk"))
            or bool(arm.get("x44_velocity_cycle_suppressed", False))
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
            and metric_tracks[str(track_id)].get("disposition") == "MEASURED"
            and float(
                metric_tracks[str(track_id)].get("velocity_forward_mps", 0.0)
            )
            < 0.0
        ]
        if not sources:
            continue
        carriers = [handback_row(row) for row in sources]
        existing_ids = {str(row["track_id"]) for row in routed_frame["tracks"]}
        for row in carriers:
            x24.require(
                str(row["track_id"]) not in existing_ids,
                "x59_handback_track_id_collision",
            )
            routed_frame["tracks"].append(row)
            routed_frame["risk_eligible_tracks"] = int(
                routed_frame["risk_eligible_tracks"]
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
                "x59_current_measured_closing_handback_used": True,
                "x59_current_measured_closing_handback_track_ids": track_ids,
            }
        )
        closing_handback_frames += 1
        handed_back_tracks += len(carriers)

    value["arms"][ARM_X59] = value["arms"].pop(x57.ARM_X57)
    value["diagnostics"]["x59_route_mode_counts"] = value["diagnostics"].pop(
        "x57_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x59_current_measured_closing_handback_frames": closing_handback_frames,
            "x59_current_measured_closing_handback_tracks": handed_back_tracks,
            "x59_evidence_supported_receding_release_frames": (
                evidence_supported_release_frames
            ),
            "x59_evidence_supported_receding_release_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X59] = frame["arms"].pop(x57.ARM_X57)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_modality_evidence_reliability_router_episode(
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
        "x59_authorized_handback_row",
    )
    return {
        "status": "X59_MODALITY_EVIDENCE_RELIABILITY_ROUTER_FALSIFIER_MET",
        "retained_core": "X54",
        "retained_x57_zero_eligible_component": True,
        "current_measured_closing_only_handback": True,
        "receding_release_requires_current_direct_metric_evidence": True,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
