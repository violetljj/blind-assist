"""X84 releases branch-overloaded held closing continuations.

X41 may bridge a metric dropout through a credentialed surface parent.  That
bridge is useful when current occupancy peaks or enough direct transport
anchors support the active branch hypotheses.  A held, forward-closing row
that survives only as direction-consistent branch continuation must not own
route risk when its authorized branch hypotheses outnumber its direct anchor
pairs.  In that partition, direction is being carried more strongly than the
correspondence that would identify which branch actually closes the route.

X84 clears only that inherited route-risk ownership while retaining all track,
geometry, motion, lineage, and suppressed-cross-representation evidence.  The
rule is class-independent and uses a relational evidence condition rather than
a fitted metric threshold.
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
import dtr_carla_x83_rigid_risk_reference_projection as x83  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X84_BRANCH_OVERLOADED_CLOSING_CONTINUATION_RELEASE"
ARM_X84 = "X84_ISSUED_PLAN_BRANCH_OVERLOADED_CLOSING_CONTINUATION_RELEASE"
TRANSPORT_STATE = "DIRECTION_CONSISTENT_BRANCH_CONTINUATION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x83.fixed_constants(),
        "representation": "X83_WITH_BRANCH_OVERLOADED_CLOSING_CONTINUATION_RELEASE",
        "retained_core": "X83",
        "release_rule": (
            "METRIC_CREDENTIALED_PARENT_CONTINUATION_AND_EVERY_CONFIRMED_"
            "CARRIER_IS_HOLD_DIRECTION_CONSISTENT_FORWARD_CLOSING_AND_"
            "AUTHORIZED_BRANCHES_OUTNUMBER_DIRECT_ANCHOR_PAIRS"
        ),
        "authority_rule": (
            "CARRIED_DIRECTION_CANNOT_OWN_CLOSING_RISK_WHEN_BRANCH_"
            "HYPOTHESES_OUTNUMBER_DIRECT_CORRESPONDENCE_ANCHORS"
        ),
        "occupancy_peak_anchored_carriers_retained": True,
        "non_closing_carriers_retained": True,
        "anchor_covered_branch_continuations_retained": True,
        "track_geometry_motion_and_lineage_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _is_branch_overloaded_closing_continuation(row: Mapping[str, Any]) -> bool:
    return (
        row.get("disposition") == "HOLD"
        and row.get("transport_state") == TRANSPORT_STATE
        and float(row.get("velocity_forward_mps", 0.0)) < 0.0
        and int(row.get("authorized_surface_transport_branches", 0))
        > int(row.get("transport_anchor_pairs", 0))
    )


def _release_partition(
    rows: Mapping[str, Mapping[str, Any]], confirmed_ids: set[str]
) -> bool:
    return bool(confirmed_ids) and all(
        _is_branch_overloaded_closing_continuation(rows[track_id])
        for track_id in confirmed_ids
    )


def apply_branch_overloaded_closing_continuation_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x83.ARM_X83]
        arm["x84_branch_overloaded_closing_continuation_release_used"] = False
        arm["x84_released_track_ids"] = []
        if not bool(arm.get("route_risk")) or not bool(
            arm.get("metric_credentialed_parent_continuation_used", False)
        ):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x84_carrier_reference")
        if not _release_partition(rows, confirmed_ids):
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
                "x84_branch_overloaded_closing_continuation_release_used": True,
                "x84_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X84] = value["arms"].pop(x83.ARM_X83)
    value["diagnostics"]["x84_route_mode_counts"] = value["diagnostics"].pop(
        "x83_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x84_branch_overloaded_closing_continuation_release_frames": released_frames,
            "x84_branch_overloaded_closing_continuation_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X84] = frame["arms"].pop(x83.ARM_X83)
    return value


def self_check() -> dict[str, Any]:
    overloaded = {
        "track_id": "held-closing-1",
        "disposition": "HOLD",
        "transport_state": TRANSPORT_STATE,
        "velocity_forward_mps": -1.0,
        "authorized_surface_transport_branches": 5,
        "transport_anchor_pairs": 4,
    }
    rows = {overloaded["track_id"]: overloaded}
    x24.require(
        _release_partition(rows, set(rows))
        and not _release_partition(
            {overloaded["track_id"]: {**overloaded, "disposition": "MEASURED"}},
            set(rows),
        )
        and not _release_partition(
            {
                overloaded["track_id"]: {
                    **overloaded,
                    "transport_state": "OCCUPANCY_PEAK_ANCHORED_BRANCH",
                }
            },
            set(rows),
        )
        and not _release_partition(
            {overloaded["track_id"]: {**overloaded, "velocity_forward_mps": 0.0}},
            set(rows),
        )
        and not _release_partition(
            {
                overloaded["track_id"]: {
                    **overloaded,
                    "authorized_surface_transport_branches": 4,
                }
            },
            set(rows),
        ),
        "x84_relational_release_partition",
    )
    return {
        "status": "X84_BRANCH_OVERLOADED_CLOSING_CONTINUATION_FALSIFIER_MET",
        "release_only": True,
        "occupancy_peak_carriers_retained": True,
        "non_closing_carriers_retained": True,
        "anchor_covered_continuations_retained": True,
        "track_geometry_motion_and_lineage_retained": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
