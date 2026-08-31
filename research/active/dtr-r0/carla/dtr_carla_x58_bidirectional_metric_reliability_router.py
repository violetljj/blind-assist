"""X58 bidirectional reliability routing around the retained X54 core.

The positive route rescues only currently MEASURED X24 confirmed metric tracks
when X54 is silent and X44 reports no velocity conflict.  The negative route
releases an X54 object-permanence risk after the same belief is observed moving
forward away from the wearer for a second consecutive frame while X24 is also
silent.  HOLD-only metric rescues are never admitted.

The router adds no detector, score, duration, route, association, or numeric
threshold.  C26 and C27 are consumed synthetic Development and cannot confirm
X58.
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


EXPERIMENT_ID = "DTR_CARLA_X58_BIDIRECTIONAL_METRIC_RELIABILITY_ROUTER"
ARM_X58 = "X58_ISSUED_PLAN_BIDIRECTIONAL_METRIC_RELIABILITY_ROUTER"
HANDBACK_DISPOSITION = "X24_CURRENT_MEASURED_METRIC_HANDBACK"
PERMANENCE_DISPOSITION = "OBJECT_PERMANENCE_BELIEF_HOLD"


def fixed_constants() -> dict[str, Any]:
    return {
        **x54.fixed_constants(),
        "representation": "X54_RETAINED_CORE_WITH_BIDIRECTIONAL_RELIABILITY_ROUTER",
        "retained_core": "X54",
        "positive_route": "CURRENT_MEASURED_X24_CONFIRMATION_ONLY",
        "hold_only_handback_allowed": False,
        "positive_route_conflict_rule": "NO_X44_VELOCITY_CYCLE_CONFLICT",
        "negative_route": (
            "SAME_RECEDING_OBJECT_PERMANENCE_BELIEF_ON_SECOND_CONSECUTIVE_FRAME_"
            "AND_CURRENT_X24_SILENT"
        ),
        "receding_rule": "VELOCITY_FORWARD_MPS_POSITIVE",
        "metric_confirmation_threshold": "INHERITED_X24",
        "new_numeric_threshold_added": False,
    }


def handback_row(row: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    source_track_id = str(row["track_id"])
    value.update(
        {
            "track_id": f"x58-handback::{source_track_id}",
            "parent_track_id": f"x58-metric-parent::{source_track_id}",
            "disposition": HANDBACK_DISPOSITION,
            "motion_authority": x27.RIGID_DYNAMIC,
            "risk_eligible": True,
            "support_footprint_mode": "X24_CURRENT_MEASURED_METRIC_TRACK",
            "x58_metric_source_track_id": source_track_id,
            "x58_current_measured_metric_handback": True,
        }
    )
    return value


def apply_bidirectional_metric_reliability_router_episode(
    value: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(value)
    measured_handback_frames = 0
    handed_back_tracks = 0
    receding_release_frames = 0
    released_tracks = 0
    previous_receding_track_ids: set[str] = set()

    for core_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(core_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(core_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x58_frame_alignment",
        )
        arm = core_frame["arms"][x54.ARM_X54]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        arm["x58_current_measured_metric_handback_used"] = False
        arm["x58_current_measured_metric_handback_track_ids"] = []
        arm["x58_receding_permanence_release_used"] = False
        arm["x58_receding_permanence_release_track_ids"] = []

        core_tracks = {
            str(row["track_id"]): row for row in core_frame["tracks"]
        }
        confirmed_core_rows = [
            core_tracks[str(track_id)]
            for track_id in arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in core_tracks
        ]
        current_receding_track_ids = {
            str(row["track_id"])
            for row in confirmed_core_rows
            if row.get("disposition") == PERMANENCE_DISPOSITION
            and float(row.get("velocity_forward_mps", 0.0)) > 0.0
        }
        release_ids = current_receding_track_ids & previous_receding_track_ids
        release = (
            bool(arm.get("route_risk"))
            and bool(confirmed_core_rows)
            and len(current_receding_track_ids) == len(confirmed_core_rows)
            and bool(release_ids)
            and not bool(metric_arm.get("route_risk"))
        )
        if release:
            released = sorted(str(row["track_id"]) for row in confirmed_core_rows)
            arm.update(
                {
                    "route_risk": False,
                    "minimum_entry_s": None,
                    "candidate_risk_track_ids": [],
                    "confirmed_risk_track_ids": [],
                    "candidate_risk_parent_track_ids": [],
                    "confirmed_risk_parent_track_ids": [],
                    "x58_receding_permanence_release_used": True,
                    "x58_receding_permanence_release_track_ids": released,
                }
            )
            receding_release_frames += 1
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
        ]
        if not sources:
            continue
        carriers = [handback_row(row) for row in sources]
        existing_ids = {str(row["track_id"]) for row in core_frame["tracks"]}
        for row in carriers:
            x24.require(
                str(row["track_id"]) not in existing_ids,
                "x58_handback_track_id_collision",
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
                "x58_current_measured_metric_handback_used": True,
                "x58_current_measured_metric_handback_track_ids": track_ids,
            }
        )
        measured_handback_frames += 1
        handed_back_tracks += len(carriers)

    value["arms"][ARM_X58] = value["arms"].pop(x54.ARM_X54)
    value["diagnostics"]["x58_route_mode_counts"] = value["diagnostics"].pop(
        "x54_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x58_current_measured_metric_handback_frames": measured_handback_frames,
            "x58_current_measured_metric_handback_tracks": handed_back_tracks,
            "x58_receding_permanence_release_frames": receding_release_frames,
            "x58_receding_permanence_release_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X58] = frame["arms"].pop(x54.ARM_X54)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_bidirectional_metric_reliability_router_episode(
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
        "x58_authorized_handback_row",
    )
    return {
        "status": "X58_BIDIRECTIONAL_RELIABILITY_ROUTER_FALSIFIER_MET",
        "retained_core": "X54",
        "current_measured_only_handback": True,
        "hold_only_handback_allowed": False,
        "same_receding_belief_second_frame_release": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
