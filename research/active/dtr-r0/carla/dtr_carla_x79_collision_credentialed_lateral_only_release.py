"""X79 separates lateral motion evidence from collision-timing authority.

A surface track can strongly support that an object exists and moves laterally
without independently establishing that its crossing is synchronized with the
issued route.  X75 already records a parent collision credential only when the
surface, X25 rigid footprint, and X24 metric point all agree on route risk.
X79 keeps every track and motion row, but clears route risk when every current
carrier is an uncredentialed, conflict-free surface branch with zero
longitudinal and nonzero lateral velocity.  A credentialed branch, a static
obstacle, longitudinal motion, or any transport contradiction remains
conservative.  No detector, route, association, or numeric threshold changes.
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
import dtr_carla_x78_nonclosing_zero_shift_permanence_release as x78  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X79_COLLISION_CREDENTIALED_LATERAL_ONLY_RELEASE"
ARM_X79 = "X79_ISSUED_PLAN_COLLISION_CREDENTIALED_LATERAL_ONLY_RELEASE"
SURFACE_CLASS = "WORLD_OCCUPANCY_COMPONENT"


def fixed_constants() -> dict[str, Any]:
    return {
        **x78.fixed_constants(),
        "representation": "X78_WITH_COLLISION_CREDENTIALED_LATERAL_ONLY_RELEASE",
        "retained_core": "X78",
        "release_rule": (
            "EVERY_CONFIRMED_CARRIER_IS_A_ZERO_CONTRADICTION_SURFACE_BRANCH_"
            "WITH_ZERO_LONGITUDINAL_AND_NONZERO_LATERAL_VELOCITY_WHOSE_PARENT_"
            "NEVER_OBTAINED_THE_X75_TRIPLE_COLLISION_CREDENTIAL"
        ),
        "identity_and_motion_memory_retained": True,
        "collision_credential_source": "X75_SURFACE_X25_X24_ROUTE_RISK_AGREEMENT",
        "zero_and_nonzero_tests": "INHERITED_NUMERIC_EPSILON",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _is_uncredentialed_lateral_only_surface(
    row: Mapping[str, Any], credentialed_parent_ids: set[str]
) -> bool:
    parent_id = str(row.get("parent_track_id") or row["track_id"])
    return (
        row.get("class_name") == SURFACE_CLASS
        and abs(float(row.get("velocity_forward_mps", 0.0))) <= x24.EPSILON
        and abs(float(row.get("velocity_right_mps", 0.0))) > x24.EPSILON
        and int(row.get("transport_contradictions", 0)) == 0
        and parent_id not in credentialed_parent_ids
    )


def apply_collision_credentialed_lateral_only_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_parent_ids: set[str] = set()
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x78.ARM_X78]
        credentialed_parent_ids.update(
            str(parent_id)
            for parent_id in arm.get("x75_collision_credential_birth_parent_ids", [])
        )
        arm["x79_uncredentialed_lateral_only_release_used"] = False
        arm["x79_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x79_carrier_reference")
        if not all(
            _is_uncredentialed_lateral_only_surface(
                rows[track_id], credentialed_parent_ids
            )
            for track_id in confirmed_ids
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
                        str(rows[track_id]["parent_track_id"])
                        for track_id in candidate_ids
                        if track_id in rows and rows[track_id].get("parent_track_id")
                    }
                ),
                "confirmed_risk_parent_track_ids": [],
                "x79_uncredentialed_lateral_only_release_used": True,
                "x79_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X79] = value["arms"].pop(x78.ARM_X78)
    value["diagnostics"]["x79_route_mode_counts"] = value["diagnostics"].pop(
        "x78_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x79_uncredentialed_lateral_only_release_frames": released_frames,
            "x79_uncredentialed_lateral_only_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X79] = frame["arms"].pop(x78.ARM_X78)
    return value


def self_check() -> dict[str, Any]:
    row = {
        "track_id": "surface-1::cone-1",
        "parent_track_id": "surface-1",
        "class_name": SURFACE_CLASS,
        "velocity_forward_mps": 0.0,
        "velocity_right_mps": -1.0,
        "transport_contradictions": 0,
    }
    x24.require(
        _is_uncredentialed_lateral_only_surface(row, set())
        and not _is_uncredentialed_lateral_only_surface(row, {"surface-1"})
        and not _is_uncredentialed_lateral_only_surface(
            {**row, "velocity_forward_mps": -1.0}, set()
        )
        and not _is_uncredentialed_lateral_only_surface(
            {**row, "velocity_right_mps": 0.0}, set()
        )
        and not _is_uncredentialed_lateral_only_surface(
            {**row, "transport_contradictions": 1}, set()
        )
        and not _is_uncredentialed_lateral_only_surface(
            {**row, "class_name": "car"}, set()
        ),
        "x79_collision_credentialed_lateral_only_partition",
    )
    return {
        "status": "X79_COLLISION_CREDENTIALED_LATERAL_ONLY_FALSIFIER_MET",
        "release_only": True,
        "identity_and_motion_memory_retained": True,
        "credentialed_lateral_risk_retained": True,
        "static_and_longitudinal_risk_retained": True,
        "contradicted_history_retained": True,
        "zero_and_nonzero_tests_inherited": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
