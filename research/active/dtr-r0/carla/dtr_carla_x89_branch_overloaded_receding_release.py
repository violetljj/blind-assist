"""X89 rejects anchor-deficient receding branch consensus.

Surface transport may retain several authorized branch hypotheses after direct
correspondence becomes sparse. A receding velocity alone is not sufficient to
clear risk because lateral motion can still cross the route. But when every
confirmed carrier is receding and every carrier has more authorized branches
than direct anchor pairs, the frame has neither a closing direction nor enough
correspondence to identify which branch owns the forecast collision.

X89 clears only that all-carrier partition. Any closing or static carrier, any
anchor-covered carrier, any non-surface carrier, or any contradiction history
keeps the inherited decision. Track, geometry, motion, lineage, and diagnostic
history remain available. The rule is relational and adds no numeric threshold.
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
import dtr_carla_x88_motion_epoch_contradiction_release as x88  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X89_BRANCH_OVERLOADED_RECEDING_RELEASE"
ARM_X89 = "X89_ISSUED_PLAN_BRANCH_OVERLOADED_RECEDING_RELEASE"
SURFACE_CLASS = x79.SURFACE_CLASS


def fixed_constants() -> dict[str, Any]:
    return {
        **x88.fixed_constants(),
        "representation": "X88_WITH_BRANCH_OVERLOADED_RECEDING_RELEASE",
        "retained_core": "X88",
        "release_rule": (
            "EVERY_CONFIRMED_CARRIER_IS_A_ZERO_CONTRADICTION_SURFACE_BRANCH_"
            "WITH_FORWARD_RECEDING_MOTION_AND_AUTHORIZED_BRANCHES_OUTNUMBERING_"
            "DIRECT_TRANSPORT_ANCHOR_PAIRS"
        ),
        "authority_rule": (
            "RECEDING_BRANCH_CONSENSUS_CANNOT_OWN_COLLISION_TIMING_WHEN_EVERY_"
            "BRANCH_SET_OUTNUMBERS_ITS_DIRECT_CORRESPONDENCE_ANCHORS"
        ),
        "closing_or_static_carriers_retained": True,
        "anchor_covered_carriers_retained": True,
        "contradicted_history_retained": True,
        "mixed_or_non_surface_carriers_retained": True,
        "track_geometry_motion_and_lineage_retained": True,
        "zero_and_positive_tests": "INHERITED_NUMERIC_EPSILON",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _is_branch_overloaded_receding_surface(row: Mapping[str, Any]) -> bool:
    return (
        row.get("class_name") == SURFACE_CLASS
        and float(row.get("velocity_forward_mps", 0.0)) > x24.EPSILON
        and int(row.get("authorized_surface_transport_branches", 0))
        > int(row.get("transport_anchor_pairs", 0))
        and int(row.get("transport_contradictions", 0)) == 0
    )


def _release_partition(
    rows: Mapping[str, Mapping[str, Any]], confirmed_ids: set[str]
) -> bool:
    return bool(confirmed_ids) and all(
        _is_branch_overloaded_receding_surface(rows[track_id])
        for track_id in confirmed_ids
    )


def apply_branch_overloaded_receding_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x88.ARM_X88]
        arm["x89_branch_overloaded_receding_release_used"] = False
        arm["x89_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x89_carrier_reference")
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
                "x89_branch_overloaded_receding_release_used": True,
                "x89_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X89] = value["arms"].pop(x88.ARM_X88)
    value["diagnostics"]["x89_route_mode_counts"] = value["diagnostics"].pop(
        "x88_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x89_branch_overloaded_receding_release_frames": released_frames,
            "x89_branch_overloaded_receding_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X89] = frame["arms"].pop(x88.ARM_X88)
    return value


def self_check() -> dict[str, Any]:
    track_id = "surface-1::cone-1"
    overloaded = {
        "track_id": track_id,
        "class_name": SURFACE_CLASS,
        "velocity_forward_mps": 1.0,
        "authorized_surface_transport_branches": 5,
        "transport_anchor_pairs": 4,
        "transport_contradictions": 0,
    }
    rows = {track_id: overloaded}
    x24.require(
        _release_partition(rows, set(rows))
        and not _release_partition(
            {track_id: {**overloaded, "velocity_forward_mps": 0.0}}, set(rows)
        )
        and not _release_partition(
            {track_id: {**overloaded, "authorized_surface_transport_branches": 4}},
            set(rows),
        )
        and not _release_partition(
            {track_id: {**overloaded, "transport_contradictions": 1}}, set(rows)
        )
        and not _release_partition(
            {track_id: {**overloaded, "class_name": "car"}}, set(rows)
        ),
        "x89_branch_overloaded_receding_partition",
    )
    return {
        "status": "X89_BRANCH_OVERLOADED_RECEDING_FALSIFIER_MET",
        "release_only": True,
        "all_carriers_receding_required": True,
        "all_carriers_branch_overloaded_required": True,
        "anchor_covered_carriers_retained": True,
        "closing_or_static_carriers_retained": True,
        "contradicted_history_retained": True,
        "track_geometry_motion_and_lineage_retained": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
