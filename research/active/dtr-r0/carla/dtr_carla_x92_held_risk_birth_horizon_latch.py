"""X92 carries X91 suppression until evidence or horizon state changes.

X91 prevents a stale held-only, post-horizon route forecast from birthing a new
alert. Clearing only that first frame is insufficient: the inherited alert
segment can otherwise reappear one frame later from the same held parent with
no new measurement. X92 therefore latches the X91 release across the same
parent lineage while every carrier remains a held envelope and route entry
remains beyond X24's evidence horizon.

The latch ends immediately when current or mixed support appears, parent
identity changes, route entry reaches the inherited horizon, or the inherited
route decision is clear for a reason other than X91. No new numeric threshold
is added.
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
import dtr_carla_x91_held_risk_birth_horizon_release as x91  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X92_HELD_RISK_BIRTH_HORIZON_LATCH"
ARM_X92 = "X92_ISSUED_PLAN_HELD_RISK_BIRTH_HORIZON_LATCH"


def fixed_constants() -> dict[str, Any]:
    return {
        **x91.fixed_constants(),
        "representation": "X91_WITH_HELD_RISK_BIRTH_HORIZON_LATCH",
        "retained_core": "X91",
        "release_rule": (
            "CONTINUE_X91_RELEASE_FOR_SAME_PARENT_LINEAGE_WHILE_EVERY_"
            "CONFIRMED_CARRIER_REMAINS_HELD_AND_MINIMUM_ROUTE_ENTRY_"
            "EXCEEDS_INHERITED_X24_HOLD_WINDOW"
        ),
        "authority_rule": (
            "SUPPRESSED_ALERT_CANNOT_REBIRTH_WITHOUT_NEW_MEASUREMENT_PARENT_"
            "CHANGE_OR_ENTRY_INTO_THE_AUTHORIZED_HORIZON"
        ),
        "evidence_authority_horizon_seconds": x24.HOLD_WINDOW_S,
        "new_or_mixed_support_terminates_latch": True,
        "parent_change_terminates_latch": True,
        "within_horizon_entry_terminates_latch": True,
        "ordinary_clear_terminates_latch": True,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _parent_ids(
    rows: Mapping[str, Mapping[str, Any]], track_ids: set[str]
) -> set[str]:
    return {
        str(rows[track_id].get("parent_track_id") or track_id)
        for track_id in track_ids
    }


def _continued_release_partition(
    arm: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    confirmed_ids: set[str],
    latched_parent_ids: set[str],
) -> bool:
    entry = arm.get("minimum_entry_s")
    return (
        bool(confirmed_ids)
        and bool(latched_parent_ids)
        and _parent_ids(rows, confirmed_ids) == latched_parent_ids
        and entry is not None
        and float(entry) > x24.HOLD_WINDOW_S + x24.EPSILON
        and all(x91._is_held_lineage_carrier(rows[track_id]) for track_id in confirmed_ids)
    )


def apply_held_risk_birth_horizon_latch_episode(core: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(core)
    latched_parent_ids: set[str] = set()
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x91.ARM_X91]
        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        arm["x92_held_risk_birth_horizon_latch_used"] = False
        arm["x92_released_track_ids"] = []

        if bool(arm.get("x91_held_risk_birth_horizon_release_used")):
            released_ids = {
                str(track_id) for track_id in arm.get("x91_released_track_ids", [])
            }
            x24.require(
                released_ids and released_ids.issubset(rows),
                "x92_x91_release_reference",
            )
            latched_parent_ids = _parent_ids(rows, released_ids)
            continue

        if not bool(arm.get("route_risk")):
            latched_parent_ids.clear()
            continue

        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(
            confirmed_ids and confirmed_ids.issubset(rows),
            "x92_carrier_reference",
        )
        if not _continued_release_partition(
            arm, rows, confirmed_ids, latched_parent_ids
        ):
            latched_parent_ids.clear()
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
                "x92_held_risk_birth_horizon_latch_used": True,
                "x92_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X92] = value["arms"].pop(x91.ARM_X91)
    value["diagnostics"]["x92_route_mode_counts"] = value["diagnostics"].pop(
        "x91_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x92_held_risk_birth_horizon_latch_frames": released_frames,
            "x92_held_risk_birth_horizon_latched_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X92] = frame["arms"].pop(x91.ARM_X91)
    return value


def self_check() -> dict[str, Any]:
    track_id = "surface-1::branch-1"
    held = {
        "track_id": track_id,
        "parent_track_id": "surface-1",
        "support_footprint_mode": x91.HELD_LINEAGE_SUPPORT,
    }
    rows = {track_id: held}
    late = {"minimum_entry_s": x24.HOLD_WINDOW_S + 0.1}
    x24.require(
        _continued_release_partition(late, rows, set(rows), {"surface-1"})
        and not _continued_release_partition(late, rows, set(rows), {"surface-2"})
        and not _continued_release_partition(
            {"minimum_entry_s": x24.HOLD_WINDOW_S},
            rows,
            set(rows),
            {"surface-1"},
        )
        and not _continued_release_partition(
            late,
            {
                track_id: {
                    **held,
                    "support_footprint_mode": "MEASURED_CONVEX_CELL_HULL",
                }
            },
            set(rows),
            {"surface-1"},
        ),
        "x92_held_risk_birth_horizon_latch_partition",
    )
    return {
        "status": "X92_HELD_RISK_BIRTH_HORIZON_LATCH_FALSIFIER_MET",
        "release_only": True,
        "inherited_x24_hold_window_reused": True,
        "same_parent_lineage_required": True,
        "new_or_mixed_support_terminates_latch": True,
        "parent_change_terminates_latch": True,
        "within_horizon_entry_terminates_latch": True,
        "ordinary_clear_terminates_latch": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
