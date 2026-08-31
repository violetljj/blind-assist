"""X60 route-entry evidence credential memory around frozen X59.

X59 correctly gates a positive metric handback on direct evidence, but its
forward-velocity sign test rejects lateral cross-traffic whose plan-adherent
trajectory still enters the wearer route.  X60 credentials the stronger fact:
an X24 track is currently MEASURED and already confirmed by the frozen issued-
plan route geometry.  The credential may continue only while the same X24
track is on HOLD and remains a confirmed route risk.  It ends immediately when
that identity or route-risk evidence disappears.

No detector, score, duration, route, association, weather label, or numeric
threshold is added.  C26-C28 are consumed synthetic Development only and
cannot confirm X60.
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
import dtr_carla_x59_modality_evidence_reliability_router as x59  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X60_ROUTE_ENTRY_CREDENTIAL_MEMORY"
ARM_X60 = "X60_ISSUED_PLAN_ROUTE_ENTRY_CREDENTIAL_MEMORY"
MEASURED_HANDBACK = "X24_MEASURED_PLAN_ROUTE_ENTRY_HANDBACK"
HELD_HANDBACK = "X24_HELD_PLAN_ROUTE_ENTRY_CREDENTIAL_HANDBACK"


def fixed_constants() -> dict[str, Any]:
    return {
        **x59.fixed_constants(),
        "representation": "X59_WITH_ROUTE_ENTRY_EVIDENCE_CREDENTIAL_MEMORY",
        "retained_core": "X59",
        "credential_birth_rule": (
            "CURRENT_X24_MEASURED_TRACK_CONFIRMED_BY_ISSUED_PLAN_ROUTE_GEOMETRY"
        ),
        "credential_hold_rule": (
            "SAME_X24_TRACK_ON_HOLD_AND_STILL_CONFIRMED_BY_ISSUED_PLAN_ROUTE_GEOMETRY"
        ),
        "credential_release_rule": (
            "TRACK_IDENTITY_OR_X24_ROUTE_RISK_OR_VALID_ISSUED_PLAN_EVIDENCE_DISAPPEARS"
        ),
        "x59_direct_negative_evidence_has_priority": True,
        "forward_velocity_sign_gate_used": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def handback_row(row: Mapping[str, Any], *, held: bool) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    source_track_id = str(row["track_id"])
    value.update(
        {
            "track_id": f"x60-handback::{source_track_id}",
            "parent_track_id": f"x60-metric-parent::{source_track_id}",
            "disposition": HELD_HANDBACK if held else MEASURED_HANDBACK,
            "motion_authority": x27.RIGID_DYNAMIC,
            "risk_eligible": True,
            "support_footprint_mode": (
                "X24_HELD_PLAN_ROUTE_ENTRY_CREDENTIAL"
                if held
                else "X24_CURRENT_MEASURED_PLAN_ROUTE_ENTRY"
            ),
            "x60_metric_source_track_id": source_track_id,
            "x60_route_entry_credential_held": held,
            "x60_route_entry_credential_handback": True,
        }
    )
    return value


def apply_route_entry_credential_memory_episode(
    core: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    value = x59.apply_modality_evidence_reliability_router_episode(core, metric)
    value = copy.deepcopy(value)
    credentialed_track_ids: set[str] = set()
    measured_handback_frames = 0
    held_handback_frames = 0
    handed_back_tracks = 0

    for routed_frame, metric_frame in zip(
        value["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(routed_frame["sample_index"]) == int(metric_frame["sample_index"])
            and abs(float(routed_frame["time_s"]) - float(metric_frame["time_s"]))
            <= x31.EPSILON,
            "x60_frame_alignment",
        )
        arm = routed_frame["arms"][x59.ARM_X59]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        arm["x60_measured_route_entry_handback_used"] = False
        arm["x60_held_route_entry_credential_handback_used"] = False
        arm["x60_route_entry_credential_track_ids"] = []

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
            previous_credentials & confirmed_ids if valid_route_evidence else set()
        )
        measured_sources: list[Mapping[str, Any]] = []
        held_sources: list[Mapping[str, Any]] = []
        if valid_route_evidence:
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
            or bool(arm.get("x44_velocity_cycle_suppressed", False))
            or bool(arm.get("x59_evidence_supported_receding_release_used", False))
        ):
            continue
        sources = measured_sources + held_sources
        if not sources:
            continue

        carriers = [
            handback_row(row, held=row.get("disposition") == "HOLD")
            for row in sources
        ]
        existing_ids = {str(row["track_id"]) for row in routed_frame["tracks"]}
        for row in carriers:
            x24.require(
                str(row["track_id"]) not in existing_ids,
                "x60_handback_track_id_collision",
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
                "x60_measured_route_entry_handback_used": bool(measured_sources),
                "x60_held_route_entry_credential_handback_used": bool(held_sources),
                "x60_route_entry_credential_track_ids": track_ids,
            }
        )
        measured_handback_frames += int(bool(measured_sources))
        held_handback_frames += int(bool(held_sources))
        handed_back_tracks += len(carriers)

    value["arms"][ARM_X60] = value["arms"].pop(x59.ARM_X59)
    value["diagnostics"]["x60_route_mode_counts"] = value["diagnostics"].pop(
        "x59_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x60_measured_route_entry_handback_frames": measured_handback_frames,
            "x60_held_route_entry_credential_handback_frames": held_handback_frames,
            "x60_route_entry_handback_tracks": handed_back_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X60] = frame["arms"].pop(x59.ARM_X59)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_route_entry_credential_memory_episode(
        x59.x54.predict_episode(episode, candidate_values, calibration),
        x24.predict_episode(episode, candidate_values, calibration),
    )


def self_check() -> dict[str, Any]:
    measured = handback_row(
        {
            "track_id": "metric-lateral",
            "position_forward_m": 1.8,
            "position_right_m": 4.2,
            "velocity_forward_mps": 0.051,
            "velocity_right_mps": -1.591,
        },
        held=False,
    )
    held = handback_row(measured, held=True)
    x24.require(
        measured["motion_authority"] == x27.RIGID_DYNAMIC
        and measured["risk_eligible"] is True
        and measured["disposition"] == MEASURED_HANDBACK
        and held["disposition"] == HELD_HANDBACK,
        "x60_authorized_route_entry_credential_rows",
    )
    return {
        "status": "X60_ROUTE_ENTRY_CREDENTIAL_MEMORY_FALSIFIER_MET",
        "retained_core": "X59",
        "measured_plan_route_entry_birth": True,
        "same_track_hold_only_memory": True,
        "lateral_route_entry_supported": True,
        "x59_direct_negative_evidence_has_priority": True,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
