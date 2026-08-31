"""X63 existence-only object permanence around frozen X62.

Object permanence can preserve that a dynamic object may still exist, but a
pure direction-consistent extrapolation without a current occupancy-peak
anchor must not independently retain route-risk authority.  X63 therefore
removes only confirmed route-risk references carried by
``OBJECT_PERMANENCE_BELIEF_HOLD`` plus
``DIRECTION_CONSISTENT_BRANCH_CONTINUATION``.  Occupancy-peak-anchored belief,
current measurements, and every X62 handback remain unchanged.

No detector, duration, distance, weather label, or numeric threshold is added.
C26-C28 remain consumed synthetic Development only.
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


EXPERIMENT_ID = "DTR_CARLA_X63_EXISTENCE_ONLY_OBJECT_PERMANENCE"
ARM_X63 = "X63_ISSUED_PLAN_EXISTENCE_ONLY_OBJECT_PERMANENCE"
OBJECT_PERMANENCE = "OBJECT_PERMANENCE_BELIEF_HOLD"
DIRECTION_ONLY = "DIRECTION_CONSISTENT_BRANCH_CONTINUATION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x62.fixed_constants(),
        "representation": "X62_WITH_EXISTENCE_ONLY_DIRECTIONAL_OBJECT_PERMANENCE",
        "retained_core": "X62",
        "release_rule": (
            "OBJECT_PERMANENCE_BELIEF_HOLD_WITH_DIRECTION_CONSISTENT_BRANCH_"
            "CONTINUATION_CANNOT_INDEPENDENTLY_CARRY_ROUTE_RISK"
        ),
        "occupancy_peak_anchored_belief_retained": True,
        "existence_memory_retained": True,
        "duration_or_distance_threshold_used": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _is_existence_only(row: Mapping[str, Any]) -> bool:
    return (
        row.get("disposition") == OBJECT_PERMANENCE
        and row.get("transport_state") == DIRECTION_ONLY
    )


def apply_existence_only_object_permanence_episode(
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
            track_id for track_id, row in rows.items() if _is_existence_only(row)
        }
        arm["x63_existence_only_object_permanence_release_used"] = False
        arm["x63_existence_only_released_track_ids"] = []
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
                "x63_existence_only_object_permanence_release_used": True,
                "x63_existence_only_released_track_ids": sorted(released_ids),
            }
        )
        released_frames += 1
        released_tracks += len(released_ids)

    value["arms"][ARM_X63] = value["arms"].pop(x62.ARM_X62)
    value["diagnostics"]["x63_route_mode_counts"] = value["diagnostics"].pop(
        "x62_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x63_existence_only_release_frames": released_frames,
            "x63_existence_only_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X63] = frame["arms"].pop(x62.ARM_X62)
    return value


def predict_episode(
    episode: Any,
    candidate_values: Sequence[Mapping[str, Any]],
    calibration: Any,
) -> dict[str, Any]:
    return apply_existence_only_object_permanence_episode(
        x62.x59.x54.predict_episode(episode, candidate_values, calibration),
        x24.predict_episode(episode, candidate_values, calibration),
    )


def self_check() -> dict[str, Any]:
    x24.require(
        _is_existence_only(
            {
                "disposition": OBJECT_PERMANENCE,
                "transport_state": DIRECTION_ONLY,
            }
        )
        and not _is_existence_only(
            {
                "disposition": OBJECT_PERMANENCE,
                "transport_state": "OCCUPANCY_PEAK_ANCHORED_BRANCH",
            }
        ),
        "x63_object_permanence_authority_partition",
    )
    return {
        "status": "X63_EXISTENCE_ONLY_OBJECT_PERMANENCE_FALSIFIER_MET",
        "retained_core": "X62",
        "direction_only_belief_route_authority": False,
        "occupancy_peak_anchored_belief_retained": True,
        "existence_memory_retained": True,
        "duration_or_distance_threshold_used": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
