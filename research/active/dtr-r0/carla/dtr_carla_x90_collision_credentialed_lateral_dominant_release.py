"""X90 requires collision credentials for lateral-dominant future entry.

X79 separates pure lateral motion evidence from collision-timing authority.
A small route-forward component does not remove that epistemic distinction
when lateral speed still dominates: the carrier establishes cross-route motion,
but not synchronized collision with the issued wearer plan. X90 therefore
extends the existing credential rule ordinally rather than with a fitted ratio.

X90 clears only a positive-time future entry whose every confirmed carrier is
an uncredentialed, zero-contradiction surface branch with absolute lateral
velocity greater than absolute route-forward velocity. Current overlap,
credentialed parents, contradicted histories, non-surface carriers, and
forward-dominant or static motion remain conservative. No numeric threshold is
added.
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
import dtr_carla_x79_collision_credentialed_lateral_only_release as x79  # noqa: E402
import dtr_carla_x89_branch_overloaded_receding_release as x89  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X90_COLLISION_CREDENTIALED_LATERAL_DOMINANT_RELEASE"
ARM_X90 = "X90_ISSUED_PLAN_COLLISION_CREDENTIALED_LATERAL_DOMINANT_RELEASE"
SURFACE_CLASS = x79.SURFACE_CLASS


def fixed_constants() -> dict[str, Any]:
    return {
        **x89.fixed_constants(),
        "representation": "X89_WITH_COLLISION_CREDENTIALED_LATERAL_DOMINANT_RELEASE",
        "retained_core": "X89",
        "release_rule": (
            "POSITIVE_TIME_ENTRY_AND_EVERY_CONFIRMED_CARRIER_IS_AN_"
            "UNCREDENTIALED_ZERO_CONTRADICTION_SURFACE_BRANCH_WITH_ABSOLUTE_"
            "LATERAL_VELOCITY_GREATER_THAN_ABSOLUTE_ROUTE_FORWARD_VELOCITY"
        ),
        "authority_rule": (
            "LATERAL_DOMINANT_MOTION_ESTABLISHES_CROSS_ROUTE_KINEMATICS_BUT_"
            "REQUIRES_X75_CROSS_REPRESENTATION_AGREEMENT_FOR_COLLISION_TIMING"
        ),
        "collision_credential_source": "X75_SURFACE_X25_X24_ROUTE_RISK_AGREEMENT",
        "ordinal_dominance_test": "ABS_LATERAL_GREATER_THAN_ABS_ROUTE_FORWARD",
        "current_overlap_retained": True,
        "credentialed_parent_retained": True,
        "contradicted_history_retained": True,
        "non_lateral_dominant_motion_retained": True,
        "track_geometry_motion_and_lineage_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _is_uncredentialed_lateral_dominant_surface(
    row: Mapping[str, Any], credentialed_parent_ids: set[str]
) -> bool:
    parent_id = str(row.get("parent_track_id") or row["track_id"])
    forward = abs(float(row.get("velocity_forward_mps", 0.0)))
    lateral = abs(float(row.get("velocity_right_mps", 0.0)))
    return (
        row.get("class_name") == SURFACE_CLASS
        and lateral > forward + x24.EPSILON
        and int(row.get("transport_contradictions", 0)) == 0
        and parent_id not in credentialed_parent_ids
    )


def _release_partition(
    arm: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    confirmed_ids: set[str],
    credentialed_parent_ids: set[str],
) -> bool:
    entry = arm.get("minimum_entry_s")
    return (
        bool(confirmed_ids)
        and entry is not None
        and float(entry) > x24.EPSILON
        and all(
            _is_uncredentialed_lateral_dominant_surface(
                rows[track_id], credentialed_parent_ids
            )
            for track_id in confirmed_ids
        )
    )


def apply_collision_credentialed_lateral_dominant_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_parent_ids: set[str] = set()
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x89.ARM_X89]
        credentialed_parent_ids.update(
            str(parent_id)
            for parent_id in arm.get("x75_collision_credential_birth_parent_ids", [])
        )
        arm["x90_collision_credentialed_lateral_dominant_release_used"] = False
        arm["x90_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x90_carrier_reference")
        if not _release_partition(
            arm, rows, confirmed_ids, credentialed_parent_ids
        ):
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
                        str(rows[track_id].get("parent_track_id") or track_id)
                        for track_id in candidate_ids
                        if track_id in rows
                    }
                ),
                "confirmed_risk_parent_track_ids": [],
                "x90_collision_credentialed_lateral_dominant_release_used": True,
                "x90_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X90] = value["arms"].pop(x89.ARM_X89)
    value["diagnostics"]["x90_route_mode_counts"] = value["diagnostics"].pop(
        "x89_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x90_collision_credentialed_lateral_dominant_release_frames": released_frames,
            "x90_collision_credentialed_lateral_dominant_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X90] = frame["arms"].pop(x89.ARM_X89)
    return value


def self_check() -> dict[str, Any]:
    track_id = "surface-1::cone-1"
    row = {
        "track_id": track_id,
        "parent_track_id": "surface-1",
        "class_name": SURFACE_CLASS,
        "velocity_forward_mps": -1.0,
        "velocity_right_mps": 2.0,
        "transport_contradictions": 0,
    }
    arm = {"minimum_entry_s": 0.1}
    rows = {track_id: row}
    x24.require(
        _release_partition(arm, rows, set(rows), set())
        and not _release_partition(arm, rows, set(rows), {"surface-1"})
        and not _release_partition(
            {"minimum_entry_s": 0.0}, rows, set(rows), set()
        )
        and not _release_partition(
            arm,
            {track_id: {**row, "velocity_forward_mps": -2.0}},
            set(rows),
            set(),
        )
        and not _release_partition(
            arm,
            {track_id: {**row, "transport_contradictions": 1}},
            set(rows),
            set(),
        ),
        "x90_collision_credentialed_lateral_dominant_partition",
    )
    return {
        "status": "X90_COLLISION_CREDENTIALED_LATERAL_DOMINANT_FALSIFIER_MET",
        "release_only": True,
        "ordinal_lateral_dominance_required": True,
        "positive_time_entry_required": True,
        "credentialed_parent_retained": True,
        "current_overlap_retained": True,
        "contradicted_history_retained": True,
        "track_geometry_motion_and_lineage_retained": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
