"""X61 conflict-localized route-entry handback around frozen X59.

X44's velocity-cycle conflict is evidence against the suppressed surface
track, not against every independently measured representation in the frame.
X61 keeps X59 unchanged except when that exact conflict suppresses the frame
while X24 still has a valid issued-plan metric track entering the route.  Only
that metric track is handed back, and HOLD continuation requires the same
credentialed metric identity and the same active X44 conflict.

No detector, association, duration, route, weather label, or numeric threshold
is added.  C26-C28 are consumed synthetic Development only and cannot confirm
X61.
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
import dtr_carla_x31_ambiguity_preserving_transport_predictor as x31  # noqa: E402
import dtr_carla_x59_modality_evidence_reliability_router as x59  # noqa: E402
import dtr_carla_x60_route_entry_credential_memory as x60  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X61_CONFLICT_LOCALIZED_ROUTE_ENTRY"
ARM_X61 = "X61_ISSUED_PLAN_CONFLICT_LOCALIZED_ROUTE_ENTRY"
X44_EDGE_CONFLICT = "CURRENT_METRIC_EDGE_DIRECTION_CONFLICT"


def fixed_constants() -> dict[str, Any]:
    return {
        **x59.fixed_constants(),
        "representation": "X59_WITH_CONFLICT_LOCALIZED_ROUTE_ENTRY_HANDBACK",
        "retained_core": "X59",
        "handback_scope": (
            "X44_SUPPRESSED_SURFACE_TRACK_ONLY_WHILE_INDEPENDENT_X24_METRIC_"
            "TRACK_REMAINS_ISSUED_PLAN_ROUTE_RISK"
        ),
        "credential_birth_rule": "CURRENT_X24_MEASURED_PLAN_ROUTE_ENTRY",
        "credential_hold_rule": (
            "SAME_X24_TRACK_ON_HOLD_WITH_SAME_ACTIVE_X44_EDGE_CONFLICT"
        ),
        "credential_release_rule": (
            "X44_CONFLICT_OR_TRACK_IDENTITY_OR_X24_ROUTE_RISK_DISAPPEARS"
        ),
        "x59_direct_negative_evidence_has_priority": True,
        "global_x44_frame_veto_used": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _carrier(row: Mapping[str, Any], *, held: bool) -> dict[str, Any]:
    value = x60.handback_row(row, held=held)
    value.update(
        {
            "track_id": f"x61-handback::{row['track_id']}",
            "parent_track_id": f"x61-metric-parent::{row['track_id']}",
            "support_footprint_mode": (
                "X61_HELD_CONFLICT_LOCALIZED_ROUTE_ENTRY"
                if held
                else "X61_MEASURED_CONFLICT_LOCALIZED_ROUTE_ENTRY"
            ),
            "x61_conflict_localized_route_entry_handback": True,
        }
    )
    return value


def apply_conflict_localized_route_entry_episode(
    core: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    value = x59.apply_modality_evidence_reliability_router_episode(core, metric)
    value = copy.deepcopy(value)
    credentialed_track_ids: set[str] = set()
    measured_frames = 0
    held_frames = 0
    handed_back_tracks = 0

    for routed_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(routed_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(routed_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x61_frame_alignment",
        )
        arm = routed_frame["arms"][x59.ARM_X59]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        arm["x61_measured_conflict_localized_handback_used"] = False
        arm["x61_held_conflict_localized_handback_used"] = False
        arm["x61_conflict_localized_track_ids"] = []

        active_conflict = (
            bool(arm.get("x44_velocity_cycle_suppressed", False))
            and arm.get("x44_velocity_cycle_suppression_reason")
            == X44_EDGE_CONFLICT
            and bool(arm.get("x44_velocity_cycle_suppressed_track_ids", []))
        )
        valid_route_evidence = (
            bool(metric_arm.get("route_risk"))
            and metric_arm.get("authority") == "VALID"
            and metric_arm.get("route_mode") == "ISSUED_PLAN"
        )
        metric_tracks = {
            str(row["track_id"]): row for row in metric_frame["tracks"]
        }
        confirmed_ids = {
            str(track_id)
            for track_id in metric_arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in metric_tracks
        }
        previous_credentials = set(credentialed_track_ids)
        credentialed_track_ids = (
            previous_credentials & confirmed_ids
            if active_conflict and valid_route_evidence
            else set()
        )
        measured_sources: list[Mapping[str, Any]] = []
        held_sources: list[Mapping[str, Any]] = []
        if active_conflict and valid_route_evidence:
            for track_id in sorted(confirmed_ids):
                row = metric_tracks[track_id]
                if row.get("disposition") == "MEASURED":
                    credentialed_track_ids.add(track_id)
                    measured_sources.append(row)
                elif (
                    row.get("disposition") == "HOLD"
                    and track_id in previous_credentials
                ):
                    held_sources.append(row)

        if (
            bool(arm.get("route_risk"))
            or bool(arm.get("x59_evidence_supported_receding_release_used", False))
        ):
            continue
        sources = measured_sources + held_sources
        if not sources:
            continue

        carriers = [
            _carrier(row, held=row.get("disposition") == "HOLD")
            for row in sources
        ]
        existing_ids = {str(row["track_id"]) for row in routed_frame["tracks"]}
        for row in carriers:
            x24.require(
                str(row["track_id"]) not in existing_ids,
                "x61_handback_track_id_collision",
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
                "x61_measured_conflict_localized_handback_used": bool(
                    measured_sources
                ),
                "x61_held_conflict_localized_handback_used": bool(held_sources),
                "x61_conflict_localized_track_ids": track_ids,
            }
        )
        measured_frames += int(bool(measured_sources))
        held_frames += int(bool(held_sources))
        handed_back_tracks += len(carriers)

    value["arms"][ARM_X61] = value["arms"].pop(x59.ARM_X59)
    value["diagnostics"]["x61_route_mode_counts"] = value["diagnostics"].pop(
        "x59_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x61_measured_conflict_localized_handback_frames": measured_frames,
            "x61_held_conflict_localized_handback_frames": held_frames,
            "x61_conflict_localized_handback_tracks": handed_back_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X61] = frame["arms"].pop(x59.ARM_X59)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_conflict_localized_route_entry_episode(
        x59.x54.predict_episode(episode, candidate_values, calibration),
        x24.predict_episode(episode, candidate_values, calibration),
    )


def self_check() -> dict[str, Any]:
    measured = _carrier(
        {
            "track_id": "metric-lateral",
            "position_forward_m": 1.8,
            "position_right_m": 4.2,
            "velocity_forward_mps": 0.051,
            "velocity_right_mps": -1.591,
        },
        held=False,
    )
    x24.require(
        measured["risk_eligible"] is True
        and measured["x61_conflict_localized_route_entry_handback"] is True,
        "x61_authorized_conflict_localized_handback",
    )
    return {
        "status": "X61_CONFLICT_LOCALIZED_ROUTE_ENTRY_FALSIFIER_MET",
        "retained_core": "X59",
        "x44_conflict_required": True,
        "same_metric_track_hold_only_memory": True,
        "global_x44_frame_veto_used": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
