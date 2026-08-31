"""X74 object-local class contradiction release around X73.

X57 can hand a currently confirmed X24 metric track back to the retained
occupancy core when that core has zero eligible tracks.  C35 exposed a failure
mode where the old metric identity remained a route-risk ``truck`` while the
nearest current X25 rigid footprint was a non-route ``person``.  X74 treats
that current, object-local class disagreement as contradiction evidence.

The release is deliberately narrow: every confirmed carrier must be an X57
metric handback; its nearest current measured X25 footprint inside the
inherited X24 association radius must have a different detector class and must
not itself be an X25 route candidate.  X74 cannot create or prolong an alert,
and adds no numeric threshold.  Consumed cohorts are Development evidence only.
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
import dtr_carla_x57_retained_core_metric_handback as x57  # noqa: E402
import dtr_carla_x73_credentialed_parent_hull_reconstruction as x73  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X74_METRIC_HANDBACK_CLASS_CONTRADICTION"
ARM_X74 = "X74_ISSUED_PLAN_METRIC_HANDBACK_CLASS_CONTRADICTION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x73.fixed_constants(),
        "representation": "X73_WITH_OBJECT_LOCAL_METRIC_HANDBACK_CLASS_CONTRADICTION",
        "retained_core": "X73",
        "release_scope": "ALL_CONFIRMED_CARRIERS_X57_METRIC_HANDBACK",
        "contradiction_source": "NEAREST_CURRENT_MEASURED_X25_RIGID_FOOTPRINT",
        "contradiction_rule": "DETECTOR_CLASS_MISMATCH_AND_MATCHED_RIGID_NOT_ROUTE_CANDIDATE",
        "association_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "detector_class_source": "CURRENT_DETECTOR_CLASS_ID",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    forward = float(left["position_forward_m"]) - float(right["position_forward_m"])
    route_right = float(left["position_right_m"]) - float(right["position_right_m"])
    return (forward * forward + route_right * route_right) ** 0.5


def _is_metric_handback(row: Mapping[str, Any]) -> bool:
    return (
        row.get("support_footprint_mode") == "X24_CONFIRMED_METRIC_TRACK"
        and row.get("disposition") == x57.HANDBACK_DISPOSITION
        and bool(row.get("x57_zero_eligible_metric_handback"))
    )


def _nearest_rigid_contradiction(
    carrier: Mapping[str, Any],
    rigid_rows: Sequence[Mapping[str, Any]],
    rigid_candidate_ids: set[str],
) -> Mapping[str, Any] | None:
    nearby = [
        row
        for row in rigid_rows
        if _distance(carrier, row) <= x24.ASSOCIATION_DISTANCE_M + x24.EPSILON
    ]
    if not nearby:
        return None
    nearest = min(nearby, key=lambda row: _distance(carrier, row))
    if (
        str(nearest["track_id"]) in rigid_candidate_ids
        or int(nearest.get("class_id", -1)) == int(carrier.get("class_id", -1))
    ):
        return None
    return nearest


def apply_metric_handback_class_contradiction_episode(
    core: dict[str, Any], rigid: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    contradicted_tracks = 0

    x24.require(len(value["frames"]) == len(rigid["frames"]), "x74_frame_count")
    for frame, rigid_frame in zip(value["frames"], rigid["frames"], strict=True):
        x24.require(
            int(frame["sample_index"]) == int(rigid_frame["sample_index"]),
            "x74_frame_alignment",
        )
        arm = frame["arms"][x73.ARM_X73]
        arm["x74_metric_handback_class_contradiction_release_used"] = False
        arm["x74_metric_handback_class_contradicted_track_ids"] = []
        arm["x74_rigid_contradiction_track_ids"] = []
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        if not bool(arm.get("route_risk")) or not confirmed_ids:
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        x24.require(confirmed_ids.issubset(rows), "x74_confirmed_track_reference")
        carriers = [rows[track_id] for track_id in sorted(confirmed_ids)]
        if not all(_is_metric_handback(row) for row in carriers):
            continue

        rigid_arm = rigid_frame["arms"][x25.ARM_X25]
        rigid_candidate_ids = {
            str(track_id)
            for track_id in rigid_arm.get("candidate_risk_track_ids", [])
        }
        rigid_rows = [
            row
            for row in rigid_frame["tracks"]
            if row.get("disposition") == "MEASURED"
        ]
        matches = [
            _nearest_rigid_contradiction(
                carrier, rigid_rows, rigid_candidate_ids
            )
            for carrier in carriers
        ]
        if any(match is None for match in matches):
            continue

        remaining_candidates = {
            str(track_id) for track_id in arm.get("candidate_risk_track_ids", [])
        } - confirmed_ids
        arm.update(
            {
                "route_risk": False,
                "minimum_entry_s": None,
                "candidate_risk_track_ids": sorted(remaining_candidates),
                "confirmed_risk_track_ids": [],
                "candidate_risk_parent_track_ids": sorted(
                    {
                        str(rows[track_id]["parent_track_id"])
                        for track_id in remaining_candidates
                        if track_id in rows and rows[track_id].get("parent_track_id")
                    }
                ),
                "confirmed_risk_parent_track_ids": [],
                "x74_metric_handback_class_contradiction_release_used": True,
                "x74_metric_handback_class_contradicted_track_ids": sorted(
                    confirmed_ids
                ),
                "x74_rigid_contradiction_track_ids": sorted(
                    str(match["track_id"]) for match in matches if match is not None
                ),
            }
        )
        released_frames += 1
        contradicted_tracks += len(confirmed_ids)

    value["arms"][ARM_X74] = value["arms"].pop(x73.ARM_X73)
    value["diagnostics"]["x74_route_mode_counts"] = value["diagnostics"].pop(
        "x73_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x74_metric_handback_class_contradiction_release_frames": released_frames,
            "x74_metric_handback_class_contradicted_tracks": contradicted_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X74] = frame["arms"].pop(x73.ARM_X73)
    return value


def self_check() -> dict[str, Any]:
    carrier = {
        "track_id": "x57-handback::metric-1",
        "class_id": 7,
        "disposition": x57.HANDBACK_DISPOSITION,
        "support_footprint_mode": "X24_CONFIRMED_METRIC_TRACK",
        "x57_zero_eligible_metric_handback": True,
        "position_forward_m": 4.0,
        "position_right_m": 1.0,
    }
    person = {
        "track_id": "footprint-1",
        "class_id": 0,
        "position_forward_m": 4.1,
        "position_right_m": 1.0,
    }
    truck = {**person, "class_id": 7}
    x24.require(
        _nearest_rigid_contradiction(carrier, [person], set()) is person
        and _nearest_rigid_contradiction(carrier, [person], {"footprint-1"}) is None
        and _nearest_rigid_contradiction(carrier, [truck], set()) is None,
        "x74_object_local_class_contradiction_partition",
    )
    return {
        "status": "X74_METRIC_HANDBACK_CLASS_CONTRADICTION_FALSIFIER_MET",
        "release_only": True,
        "current_measured_rigid_footprint_required": True,
        "matched_rigid_route_candidate_forbids_release": True,
        "class_mismatch_required": True,
        "association_distance_inherited": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
