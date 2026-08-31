"""X70 triple-credential X25 handback through surface dropout around X69.

An X25 rigid-footprint track may continue a collision credential after current
surface support disappears, but it may not create that credential alone.  X70
records an X25 identity only when X69 surface risk, X25 rigid-footprint risk,
and X24 metric-point risk currently agree on the same object.  The same
continuously confirmed X25 identity may then hand risk back only while no
current measured risk-eligible surface track spatially supports it.

Current surface evidence always wins over handback, and an explicit X69 rigid
contradiction release clears credentials before any continuation.  Matching and
track lifetime reuse X24 constants.  Consumed cohorts are Development evidence
only.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x69_mature_cross_route_rigid_contradiction as x69  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X70_TRIPLE_CREDENTIAL_SURFACE_DROPOUT_HANDBACK"
ARM_X70 = "X70_ISSUED_PLAN_TRIPLE_CREDENTIAL_SURFACE_DROPOUT_HANDBACK"
SUPPORT_MODE = "X25_TRIPLE_CREDENTIAL_SURFACE_DROPOUT_HANDBACK"


def fixed_constants() -> dict[str, Any]:
    return {
        **x69.fixed_constants(),
        "representation": "X69_WITH_TRIPLE_CREDENTIALED_X25_DROPOUT_HANDBACK",
        "retained_core": "X69",
        "credential_birth_rule": (
            "CURRENT_X69_SURFACE_X25_RIGID_FOOTPRINT_AND_X24_METRIC_POINT_"
            "ROUTE_RISK_SPATIALLY_AGREE"
        ),
        "credential_continuation_rule": "SAME_CONTINUOUSLY_CONFIRMED_X25_IDENTITY",
        "handback_rule": "ZERO_CURRENT_MEASURED_MATCHING_SURFACE_SUPPORT",
        "release_precedence": "X69_EXPLICIT_RIGID_CONTRADICTION_RELEASE_FIRST",
        "association_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "new_numeric_threshold_added": False,
        "weather_or_lighting_label_used": False,
    }


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    forward = float(left["position_forward_m"]) - float(
        right["position_forward_m"]
    )
    route_right = float(left["position_right_m"]) - float(
        right["position_right_m"]
    )
    return (forward * forward + route_right * route_right) ** 0.5


def _matches_any(row: Mapping[str, Any], others: list[Mapping[str, Any]]) -> bool:
    return any(
        _distance(row, other) <= x24.ASSOCIATION_DISTANCE_M + x24.EPSILON
        for other in others
    )


def _carrier(row: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    source_id = str(row["track_id"])
    value.update(
        {
            "track_id": f"x70-handback::{source_id}",
            "parent_track_id": source_id,
            "risk_eligible": True,
            "motion_authority": x27.RIGID_DYNAMIC,
            "support_footprint_mode": SUPPORT_MODE,
            "x70_triple_credential_surface_dropout_handback": True,
            "x70_x25_source_track_id": source_id,
        }
    )
    return value


def apply_triple_credential_surface_dropout_handback_episode(
    core: dict[str, Any],
    rigid: dict[str, Any],
    metric: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_x25_ids: set[str] = set()
    handback_frames = 0
    measured_handbacks = 0
    held_handbacks = 0
    triple_credential_births = 0
    surface_supported_rejections = 0

    x24.require(
        len(value["frames"]) == len(rigid["frames"]) == len(metric["frames"]),
        "x70_frame_count",
    )
    for frame, rigid_frame, metric_frame in zip(
        value["frames"], rigid["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(frame["sample_index"])
            == int(rigid_frame["sample_index"])
            == int(metric_frame["sample_index"]),
            "x70_frame_alignment",
        )
        arm = frame["arms"][x69.ARM_X69]
        arm["x70_triple_credential_surface_dropout_handback_used"] = False
        arm["x70_triple_credential_x25_source_track_ids"] = []
        arm["x70_triple_credential_handback_track_ids"] = []

        rigid_arm = rigid_frame["arms"][x25.ARM_X25]
        rigid_rows = {str(row["track_id"]): row for row in rigid_frame["tracks"]}
        rigid_confirmed = {
            str(track_id)
            for track_id in rigid_arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in rigid_rows
        }
        rigid_risk = bool(rigid_arm.get("route_risk"))
        credentialed_x25_ids = (
            credentialed_x25_ids & rigid_confirmed if rigid_risk else set()
        )

        if bool(
            arm.get("x69_mature_cross_route_rigid_contradiction_release_used")
        ):
            credentialed_x25_ids.clear()
            continue

        surface_rows = {str(row["track_id"]): row for row in frame["tracks"]}
        if bool(arm.get("route_risk")) and rigid_risk:
            surface_carriers = [
                surface_rows[str(track_id)]
                for track_id in arm.get("confirmed_risk_track_ids", [])
                if str(track_id) in surface_rows
            ]
            metric_arm = metric_frame["arms"][x24.ARM_X24]
            valid_metric_risk = bool(
                metric_arm.get("route_risk")
                and metric_arm.get("authority") == "VALID"
                and metric_arm.get("route_mode") == "ISSUED_PLAN"
            )
            metric_rows = {str(row["track_id"]): row for row in metric_frame["tracks"]}
            metric_carriers = [
                metric_rows[str(track_id)]
                for track_id in metric_arm.get("confirmed_risk_track_ids", [])
                if valid_metric_risk and str(track_id) in metric_rows
            ]
            births = {
                track_id
                for track_id in rigid_confirmed
                if _matches_any(rigid_rows[track_id], surface_carriers)
                and _matches_any(rigid_rows[track_id], metric_carriers)
            }
            before = len(credentialed_x25_ids)
            credentialed_x25_ids.update(births)
            triple_credential_births += max(0, len(credentialed_x25_ids) - before)
            continue

        if bool(arm.get("route_risk")) or not credentialed_x25_ids:
            continue
        current_surface = [
            row
            for row in frame["tracks"]
            if row.get("disposition") == "MEASURED"
            and bool(row.get("risk_eligible", False))
        ]
        sources = [
            rigid_rows[track_id]
            for track_id in sorted(credentialed_x25_ids & rigid_confirmed)
            if not _matches_any(rigid_rows[track_id], current_surface)
        ]
        surface_supported_rejections += len(credentialed_x25_ids) - len(sources)
        if not sources:
            continue

        carriers = [_carrier(row) for row in sources]
        existing_ids = {str(row["track_id"]) for row in frame["tracks"]}
        for carrier in carriers:
            x24.require(
                str(carrier["track_id"]) not in existing_ids,
                "x70_handback_track_id_collision",
            )
            frame["tracks"].append(carrier)
            frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
        track_ids = sorted(str(row["track_id"]) for row in carriers)
        parent_ids = sorted(str(row["parent_track_id"]) for row in carriers)
        source_ids = sorted(str(row["track_id"]) for row in sources)
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": rigid_arm.get("minimum_entry_s"),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x70_triple_credential_surface_dropout_handback_used": True,
                "x70_triple_credential_x25_source_track_ids": source_ids,
                "x70_triple_credential_handback_track_ids": track_ids,
            }
        )
        handback_frames += 1
        measured_handbacks += int(
            any(row.get("disposition") == "MEASURED" for row in sources)
        )
        held_handbacks += int(any(row.get("disposition") == "HOLD" for row in sources))

    value["arms"][ARM_X70] = value["arms"].pop(x69.ARM_X69)
    value["diagnostics"]["x70_route_mode_counts"] = value["diagnostics"].pop(
        "x69_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x70_triple_credential_births": triple_credential_births,
            "x70_surface_dropout_handback_frames": handback_frames,
            "x70_surface_dropout_measured_handback_frames": measured_handbacks,
            "x70_surface_dropout_held_handback_frames": held_handbacks,
            "x70_current_surface_supported_rejections": surface_supported_rejections,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X70] = frame["arms"].pop(x69.ARM_X69)
    return value


def self_check() -> dict[str, Any]:
    row = {"position_forward_m": 4.0, "position_right_m": 0.0}
    x24.require(
        _matches_any(row, [{"position_forward_m": 4.5, "position_right_m": 0.0}])
        and not _matches_any(
            row, [{"position_forward_m": 6.0, "position_right_m": 0.0}]
        ),
        "x70_spatial_identity_partition",
    )
    carrier = _carrier(
        {
            **row,
            "track_id": "footprint-1",
            "disposition": "HOLD",
            "footprint_xy": [[0, 0], [1, 0], [1, 1]],
            "velocity_forward_mps": -1.0,
            "velocity_right_mps": 0.0,
        }
    )
    x24.require(
        carrier["risk_eligible"]
        and carrier["motion_authority"] == x27.RIGID_DYNAMIC
        and carrier["parent_track_id"] == "footprint-1",
        "x70_carrier_authority",
    )
    return {
        "status": "X70_TRIPLE_CREDENTIAL_SURFACE_DROPOUT_HANDBACK_FALSIFIER_MET",
        "triple_credential_birth_required": True,
        "current_surface_support_blocks_handback": True,
        "x69_release_precedence": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
