"""X64 unanchored crossing release around frozen X62.

Direction-only object permanence remains useful for motion along the route
corridor, but a cross-route trajectory changes its route relationship while
occluded.  Without a current occupancy-peak anchor it must not independently
retain route-risk authority.  X64 releases only object-permanence tracks whose
discrete surface-transport velocity has a non-zero route-right component.

The rule is axis-topological rather than thresholded: longitudinal corridor
memory is retained, cross-route memory is existence-only.  No detector,
duration, distance, weather label, or numeric cutoff is added.  C26-C28 remain
consumed synthetic Development only.
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
import dtr_carla_x62_synchronized_conflict_handback as x62  # noqa: E402
import dtr_carla_x63_existence_only_object_permanence as x63  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X64_UNANCHORED_CROSSING_RELEASE"
ARM_X64 = "X64_ISSUED_PLAN_UNANCHORED_CROSSING_RELEASE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x62.fixed_constants(),
        "representation": "X62_WITH_UNANCHORED_CROSS_ROUTE_BELIEF_RELEASE",
        "retained_core": "X62",
        "release_rule": (
            "DIRECTION_ONLY_OBJECT_PERMANENCE_WITH_NONZERO_DISCRETE_ROUTE_"
            "RIGHT_TRANSPORT_CANNOT_INDEPENDENTLY_CARRY_ROUTE_RISK"
        ),
        "longitudinal_corridor_object_permanence_retained": True,
        "occupancy_peak_anchored_crossing_retained": True,
        "axis_topology_rule": True,
        "duration_or_distance_threshold_used": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _is_unanchored_crossing(row: Mapping[str, Any]) -> bool:
    velocity_key = row.get("surface_transport_velocity_key_cells_per_s")
    return (
        x63._is_existence_only(row)
        and isinstance(velocity_key, Sequence)
        and not isinstance(velocity_key, (str, bytes))
        and len(velocity_key) == 2
        and int(velocity_key[1]) != 0
    )


def apply_unanchored_crossing_release_episode(
    core: dict[str, Any], metric: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(
        x62.apply_synchronized_conflict_handback_episode(core, metric)
    )
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x62.ARM_X62]
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        rows = {
            str(row["track_id"]): row
            for row in frame["tracks"]
            if str(row.get("track_id")) in confirmed_ids
        }
        released_ids = {
            track_id
            for track_id, row in rows.items()
            if _is_unanchored_crossing(row)
        }
        arm["x64_unanchored_crossing_release_used"] = False
        arm["x64_unanchored_crossing_released_track_ids"] = []
        if not released_ids:
            continue

        retained_ids = confirmed_ids - released_ids
        candidate_ids = {
            str(track_id) for track_id in arm.get("candidate_risk_track_ids", [])
        } - released_ids
        retained_parent_ids = sorted(
            {
                str(rows[track_id]["parent_track_id"])
                for track_id in retained_ids
                if track_id in rows and rows[track_id].get("parent_track_id")
            }
        )
        candidate_parent_ids = sorted(
            {
                str(rows[track_id]["parent_track_id"])
                for track_id in candidate_ids
                if track_id in rows and rows[track_id].get("parent_track_id")
            }
        )
        arm.update(
            {
                "route_risk": bool(retained_ids),
                "minimum_entry_s": (
                    arm.get("minimum_entry_s") if retained_ids else None
                ),
                "candidate_risk_track_ids": sorted(candidate_ids),
                "confirmed_risk_track_ids": sorted(retained_ids),
                "candidate_risk_parent_track_ids": candidate_parent_ids,
                "confirmed_risk_parent_track_ids": retained_parent_ids,
                "x64_unanchored_crossing_release_used": True,
                "x64_unanchored_crossing_released_track_ids": sorted(released_ids),
            }
        )
        released_frames += 1
        released_tracks += len(released_ids)

    value["arms"][ARM_X64] = value["arms"].pop(x62.ARM_X62)
    value["diagnostics"]["x64_route_mode_counts"] = value["diagnostics"].pop(
        "x62_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x64_unanchored_crossing_release_frames": released_frames,
            "x64_unanchored_crossing_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X64] = frame["arms"].pop(x62.ARM_X62)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_unanchored_crossing_release_episode(
        x62.x59.x54.predict_episode(episode, candidate_values, calibration),
        x24.predict_episode(episode, candidate_values, calibration),
    )


def self_check() -> dict[str, Any]:
    base = {
        "disposition": x63.OBJECT_PERMANENCE,
        "transport_state": x63.DIRECTION_ONLY,
    }
    x24.require(
        _is_unanchored_crossing(
            {**base, "surface_transport_velocity_key_cells_per_s": [-20, 10]}
        )
        and not _is_unanchored_crossing(
            {**base, "surface_transport_velocity_key_cells_per_s": [10, 0]}
        ),
        "x64_axis_topology_partition",
    )
    return {
        "status": "X64_UNANCHORED_CROSSING_RELEASE_FALSIFIER_MET",
        "retained_core": "X62",
        "cross_route_direction_only_belief_authority": False,
        "longitudinal_corridor_object_permanence_retained": True,
        "axis_topology_rule": True,
        "duration_or_distance_threshold_used": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
