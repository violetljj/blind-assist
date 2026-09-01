"""X75 collision-credentialed object permanence around X74.

An occupancy-peak anchor supports object existence through dropout, but a
conflict-free surface extrapolation that never obtained an independent
collision credential should not indefinitely retain route-risk authority.
X75 records a surface-parent collision credential only when current surface,
X25 rigid-footprint, and X24 metric-point confirmed route risk spatially agree.

If every current confirmed carrier is an occupancy-peak-anchored object-
permanence belief, has zero observed transport contradictions, and its parent
never obtained that triple credential, X75 clears route risk while preserving
the belief row as existence memory.  Contradicted histories and all other
carriers remain conservative.  X75 cannot create or prolong an alert and adds
no numeric threshold.  Consumed cohorts are Development evidence only.
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
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x46_evidence_terminated_object_permanence as x46  # noqa: E402
import dtr_carla_x74_metric_handback_class_contradiction as x74  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X75_COLLISION_CREDENTIALED_OBJECT_PERMANENCE"
ARM_X75 = "X75_ISSUED_PLAN_COLLISION_CREDENTIALED_OBJECT_PERMANENCE"
OCCUPANCY_PEAK = "OCCUPANCY_PEAK_ANCHORED_BRANCH"


def fixed_constants() -> dict[str, Any]:
    return {
        **x74.fixed_constants(),
        "representation": "X74_WITH_COLLISION_CREDENTIALED_OBJECT_PERMANENCE",
        "retained_core": "X74",
        "credential_birth_rule": (
            "CURRENT_SURFACE_X25_RIGID_FOOTPRINT_AND_X24_METRIC_POINT_"
            "CONFIRMED_ROUTE_RISK_SPATIALLY_AGREE"
        ),
        "existence_only_rule": (
            "UNCREDENTIALED_OCCUPANCY_PEAK_OBJECT_PERMANENCE_WITH_ZERO_"
            "TRANSPORT_CONTRADICTIONS_CANNOT_INDEPENDENTLY_CARRY_ROUTE_RISK"
        ),
        "association_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "existence_memory_retained": True,
        "transport_contradiction_protection": "ANY_NONZERO_HISTORY_RETAINS_RISK",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    forward = float(left["position_forward_m"]) - float(right["position_forward_m"])
    route_right = float(left["position_right_m"]) - float(right["position_right_m"])
    return (forward * forward + route_right * route_right) ** 0.5


def _matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _distance(left, right) <= x24.ASSOCIATION_DISTANCE_M + x24.EPSILON


def _is_existence_only(
    row: Mapping[str, Any], credentialed_parent_ids: set[str]
) -> bool:
    parent_id = str(row.get("parent_track_id") or row["track_id"])
    return (
        row.get("disposition") == x46.BELIEF_DISPOSITION
        and row.get("transport_state") == OCCUPANCY_PEAK
        and int(row.get("transport_contradictions", 0)) == 0
        and parent_id not in credentialed_parent_ids
    )


def apply_collision_credentialed_object_permanence_episode(
    core: dict[str, Any], rigid: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_parent_ids: set[str] = set()
    credential_births = 0
    released_frames = 0
    released_tracks = 0

    x24.require(
        len(value["frames"]) == len(rigid["frames"]) == len(metric["frames"]),
        "x75_frame_count",
    )
    for frame, rigid_frame, metric_frame in zip(
        value["frames"], rigid["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(frame["sample_index"])
            == int(rigid_frame["sample_index"])
            == int(metric_frame["sample_index"]),
            "x75_frame_alignment",
        )
        arm = frame["arms"][x74.ARM_X74]
        arm["x75_collision_credential_birth_parent_ids"] = []
        arm["x75_existence_only_permanence_release_used"] = False
        arm["x75_existence_only_released_track_ids"] = []

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids.issubset(rows), "x75_confirmed_track_reference")
        surface_carriers = [
            rows[track_id]
            for track_id in sorted(confirmed_ids)
            if rows[track_id].get("class_name") == "WORLD_OCCUPANCY_COMPONENT"
        ]
        rigid_arm = rigid_frame["arms"][x25.ARM_X25]
        rigid_rows = {str(row["track_id"]): row for row in rigid_frame["tracks"]}
        rigid_carriers = [
            rigid_rows[str(track_id)]
            for track_id in rigid_arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in rigid_rows
        ]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        metric_rows = {str(row["track_id"]): row for row in metric_frame["tracks"]}
        valid_metric_risk = bool(
            metric_arm.get("route_risk")
            and metric_arm.get("authority") == "VALID"
            and metric_arm.get("route_mode") == "ISSUED_PLAN"
        )
        metric_carriers = [
            metric_rows[str(track_id)]
            for track_id in metric_arm.get("confirmed_risk_track_ids", [])
            if valid_metric_risk and str(track_id) in metric_rows
        ]
        births = {
            str(surface.get("parent_track_id") or surface["track_id"])
            for surface in surface_carriers
            if any(
                _matches(surface, rigid_row)
                and any(_matches(rigid_row, metric_row) for metric_row in metric_carriers)
                for rigid_row in rigid_carriers
            )
        }
        new_births = births - credentialed_parent_ids
        credentialed_parent_ids.update(births)
        credential_births += len(new_births)
        arm["x75_collision_credential_birth_parent_ids"] = sorted(new_births)

        if not bool(arm.get("route_risk")) or not confirmed_ids:
            continue
        carriers = [rows[track_id] for track_id in sorted(confirmed_ids)]
        if not all(_is_existence_only(row, credentialed_parent_ids) for row in carriers):
            continue

        candidate_ids = {
            str(track_id) for track_id in arm.get("candidate_risk_track_ids", [])
        } - confirmed_ids
        arm.update(
            {
                "route_risk": False,
                "minimum_entry_s": None,
                "candidate_risk_track_ids": sorted(candidate_ids),
                "confirmed_risk_track_ids": [],
                "candidate_risk_parent_track_ids": sorted(
                    {
                        str(rows[track_id]["parent_track_id"])
                        for track_id in candidate_ids
                        if track_id in rows and rows[track_id].get("parent_track_id")
                    }
                ),
                "confirmed_risk_parent_track_ids": [],
                "x75_existence_only_permanence_release_used": True,
                "x75_existence_only_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X75] = value["arms"].pop(x74.ARM_X74)
    value["diagnostics"]["x75_route_mode_counts"] = value["diagnostics"].pop(
        "x74_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x75_collision_credential_births": credential_births,
            "x75_existence_only_permanence_release_frames": released_frames,
            "x75_existence_only_permanence_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X75] = frame["arms"].pop(x74.ARM_X74)
    return value


def self_check() -> dict[str, Any]:
    row = {
        "track_id": "surface-1::cone-1",
        "parent_track_id": "surface-1",
        "disposition": x46.BELIEF_DISPOSITION,
        "transport_state": OCCUPANCY_PEAK,
        "transport_contradictions": 0,
    }
    x24.require(
        _is_existence_only(row, set())
        and not _is_existence_only(row, {"surface-1"})
        and not _is_existence_only({**row, "transport_contradictions": 1}, set()),
        "x75_collision_credentialed_permanence_partition",
    )
    return {
        "status": "X75_COLLISION_CREDENTIALED_OBJECT_PERMANENCE_FALSIFIER_MET",
        "release_only": True,
        "existence_memory_retained": True,
        "triple_collision_credential_protects_permanence": True,
        "transport_contradiction_protects_permanence": True,
        "association_distance_inherited": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
