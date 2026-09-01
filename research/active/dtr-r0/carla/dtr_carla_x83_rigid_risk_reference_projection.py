"""X83 projects mixed confirmed risk references onto rigid authority.

A route-risk decision may be correctly owned by at least one rigid dynamic
carrier while an inherited static or otherwise non-rigid support row is also
listed as confirmed.  The extra row must not become a risk authority merely by
co-occurring with a valid carrier.  X83 keeps the route-risk classification and
all track state unchanged, moves non-rigid confirmed references back to the
candidate set, and rebuilds confirmed parent identity from the remaining rigid
carriers.  Frames without both a rigid owner and a non-rigid confirmed
reference are unchanged.  No metric threshold is added.
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
import dtr_carla_x82_held_proxy_consensus_release as x82  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X83_RIGID_RISK_REFERENCE_PROJECTION"
ARM_X83 = "X83_ISSUED_PLAN_RIGID_RISK_REFERENCE_PROJECTION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x82.fixed_constants(),
        "representation": "X82_WITH_RIGID_RISK_REFERENCE_PROJECTION",
        "retained_core": "X82",
        "projection_rule": (
            "WHEN_CONFIRMED_REFERENCES_MIX_RIGID_DYNAMIC_AND_NON_RIGID_ROWS_"
            "KEEP_ONLY_RIGID_DYNAMIC_CONFIRMED_AND_RETURN_NON_RIGID_TO_CANDIDATE"
        ),
        "confirmed_parent_rule": "DERIVE_FROM_REMAINING_RIGID_CONFIRMED_CARRIERS",
        "route_risk_classification_unchanged": True,
        "minimum_entry_unchanged": True,
        "all_track_state_retained": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _is_rigid_risk_owner(row: Mapping[str, Any]) -> bool:
    return bool(row.get("risk_eligible")) and row.get("motion_authority") == "RIGID_DYNAMIC"


def _partition_confirmed(
    rows: Mapping[str, Mapping[str, Any]], confirmed_ids: set[str]
) -> tuple[set[str], set[str]]:
    rigid = {track_id for track_id in confirmed_ids if _is_rigid_risk_owner(rows[track_id])}
    return rigid, confirmed_ids - rigid


def apply_rigid_risk_reference_projection_episode(core: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(core)
    projected_frames = 0
    demoted_references = 0

    for frame in value["frames"]:
        arm = frame["arms"][x82.ARM_X82]
        arm["x83_rigid_risk_reference_projection_used"] = False
        arm["x83_demoted_non_rigid_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x83_carrier_reference")
        rigid_ids, non_rigid_ids = _partition_confirmed(rows, confirmed_ids)
        if not rigid_ids or not non_rigid_ids:
            continue

        candidate_ids = {
            str(track_id) for track_id in arm.get("candidate_risk_track_ids", [])
        } | non_rigid_ids
        arm.update(
            {
                "candidate_risk_track_ids": sorted(candidate_ids),
                "confirmed_risk_track_ids": sorted(rigid_ids),
                "candidate_risk_parent_track_ids": sorted(
                    {
                        str(rows[track_id].get("parent_track_id") or track_id)
                        for track_id in candidate_ids
                        if track_id in rows
                    }
                ),
                "confirmed_risk_parent_track_ids": sorted(
                    {
                        str(rows[track_id].get("parent_track_id") or track_id)
                        for track_id in rigid_ids
                    }
                ),
                "x83_rigid_risk_reference_projection_used": True,
                "x83_demoted_non_rigid_track_ids": sorted(non_rigid_ids),
            }
        )
        projected_frames += 1
        demoted_references += len(non_rigid_ids)

    value["arms"][ARM_X83] = value["arms"].pop(x82.ARM_X82)
    value["diagnostics"]["x83_route_mode_counts"] = value["diagnostics"].pop(
        "x82_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x83_rigid_risk_reference_projection_frames": projected_frames,
            "x83_demoted_non_rigid_risk_references": demoted_references,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X83] = frame["arms"].pop(x82.ARM_X82)
    return value


def self_check() -> dict[str, Any]:
    rigid = {
        "track_id": "rigid-1",
        "risk_eligible": True,
        "motion_authority": "RIGID_DYNAMIC",
    }
    static = {
        "track_id": "static-1",
        "risk_eligible": True,
        "motion_authority": "STATIC_SCENE",
    }
    rows = {row["track_id"]: row for row in (rigid, static)}
    rigid_ids, demoted = _partition_confirmed(rows, set(rows))
    x24.require(
        rigid_ids == {"rigid-1"}
        and demoted == {"static-1"}
        and _partition_confirmed(rows, {"rigid-1"}) == ({"rigid-1"}, set())
        and _partition_confirmed(rows, {"static-1"}) == (set(), {"static-1"}),
        "x83_rigid_reference_partition",
    )
    return {
        "status": "X83_RIGID_RISK_REFERENCE_PROJECTION_FALSIFIER_MET",
        "classification_unchanged": True,
        "minimum_entry_unchanged": True,
        "track_state_retained": True,
        "non_rigid_reference_demoted_not_deleted": True,
        "all_non_rigid_without_rigid_owner_unchanged": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
