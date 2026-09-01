"""X93 rejects uncredentialed conflicted non-closing future timing.

X89 shows that receding branch consensus is insufficient when correspondence
is underdetermined. X90 shows that lateral-dominant motion establishes
cross-route kinematics but not collision timing without an X75 credential.
X93 unifies those authority limits for transport histories that are themselves
contradicted. A contradiction is diagnostic evidence against a stable
association; it cannot substitute for a cross-representation collision
credential and authorize a positive-time future entry.

X93 clears only when every confirmed carrier is a contradicted,
uncredentialed surface branch and the carrier set is either entirely receding
or entirely lateral-dominant. Current overlap, any closing/non-dominant mixed
motion, credentialed parents, non-surface carriers, and zero-contradiction
histories remain unchanged. No new numeric threshold is added.
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
import dtr_carla_x92_held_risk_birth_horizon_latch as x92  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X93_CONFLICTED_NONCLOSING_FUTURE_RELEASE"
ARM_X93 = "X93_ISSUED_PLAN_CONFLICTED_NONCLOSING_FUTURE_RELEASE"
SURFACE_CLASS = x79.SURFACE_CLASS


def fixed_constants() -> dict[str, Any]:
    return {
        **x92.fixed_constants(),
        "representation": "X92_WITH_CONFLICTED_NONCLOSING_FUTURE_RELEASE",
        "retained_core": "X92",
        "release_rule": (
            "POSITIVE_TIME_ENTRY_AND_EVERY_CONFIRMED_CARRIER_IS_A_"
            "CONTRADICTED_UNCREDENTIALED_SURFACE_BRANCH_AND_CARRIER_SET_IS_"
            "EITHER_ALL_RECEDING_OR_ALL_LATERAL_DOMINANT"
        ),
        "authority_rule": (
            "TRANSPORT_CONTRADICTION_CANNOT_SUBSTITUTE_FOR_COLLISION_"
            "CREDENTIAL_WHEN_MOTION_DOES_NOT_ESTABLISH_ROUTE_FORWARD_CLOSING"
        ),
        "collision_credential_source": "X75_SURFACE_X25_X24_ROUTE_RISK_AGREEMENT",
        "current_overlap_retained": True,
        "closing_or_mixed_motion_retained": True,
        "credentialed_parent_retained": True,
        "zero_contradiction_history_retained": True,
        "non_surface_carriers_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _is_conflicted_uncredentialed_surface(
    row: Mapping[str, Any], credentialed_parent_ids: set[str]
) -> bool:
    parent_id = str(row.get("parent_track_id") or row["track_id"])
    return (
        row.get("class_name") == SURFACE_CLASS
        and int(row.get("transport_contradictions", 0)) > 0
        and parent_id not in credentialed_parent_ids
    )


def _release_partition(
    arm: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    confirmed_ids: set[str],
    credentialed_parent_ids: set[str],
) -> bool:
    entry = arm.get("minimum_entry_s")
    carriers = [rows[track_id] for track_id in confirmed_ids]
    all_receding = all(
        float(row.get("velocity_forward_mps", 0.0)) > x24.EPSILON
        for row in carriers
    )
    all_lateral_dominant = all(
        abs(float(row.get("velocity_right_mps", 0.0)))
        > abs(float(row.get("velocity_forward_mps", 0.0))) + x24.EPSILON
        for row in carriers
    )
    return (
        bool(confirmed_ids)
        and entry is not None
        and float(entry) > x24.EPSILON
        and (all_receding or all_lateral_dominant)
        and all(
            _is_conflicted_uncredentialed_surface(
                rows[track_id], credentialed_parent_ids
            )
            for track_id in confirmed_ids
        )
    )


def apply_conflicted_nonclosing_future_release_episode(core: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_parent_ids: set[str] = set()
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x92.ARM_X92]
        credentialed_parent_ids.update(
            str(parent_id)
            for parent_id in arm.get("x75_collision_credential_birth_parent_ids", [])
        )
        arm["x93_conflicted_nonclosing_future_release_used"] = False
        arm["x93_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(
            confirmed_ids and confirmed_ids.issubset(rows),
            "x93_carrier_reference",
        )
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
                "x93_conflicted_nonclosing_future_release_used": True,
                "x93_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X93] = value["arms"].pop(x92.ARM_X92)
    value["diagnostics"]["x93_route_mode_counts"] = value["diagnostics"].pop(
        "x92_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x93_conflicted_nonclosing_future_release_frames": released_frames,
            "x93_conflicted_nonclosing_future_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X93] = frame["arms"].pop(x92.ARM_X92)
    return value


def self_check() -> dict[str, Any]:
    track_id = "surface-1::branch-1"
    carrier = {
        "track_id": track_id,
        "parent_track_id": "surface-1",
        "class_name": SURFACE_CLASS,
        "transport_contradictions": 1,
        "velocity_forward_mps": 1.0,
        "velocity_right_mps": 0.5,
    }
    rows = {track_id: carrier}
    future = {"minimum_entry_s": 0.1}
    x24.require(
        _release_partition(future, rows, set(rows), set())
        and not _release_partition(
            {"minimum_entry_s": 0.0}, rows, set(rows), set()
        )
        and not _release_partition(future, rows, set(rows), {"surface-1"})
        and not _release_partition(
            future,
            {track_id: {**carrier, "transport_contradictions": 0}},
            set(rows),
            set(),
        )
        and not _release_partition(
            future,
            {
                track_id: {
                    **carrier,
                    "velocity_forward_mps": -1.0,
                    "velocity_right_mps": 0.5,
                }
            },
            set(rows),
            set(),
        ),
        "x93_conflicted_nonclosing_future_partition",
    )
    return {
        "status": "X93_CONFLICTED_NONCLOSING_FUTURE_FALSIFIER_MET",
        "release_only": True,
        "positive_time_entry_required": True,
        "all_receding_or_all_lateral_dominant_required": True,
        "every_carrier_contradicted_and_uncredentialed": True,
        "current_overlap_retained": True,
        "closing_or_mixed_motion_retained": True,
        "credentialed_parent_retained": True,
        "zero_contradiction_history_retained": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
