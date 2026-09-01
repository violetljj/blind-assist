"""X77 rejects metric temporal handoffs whose obstacle is already receding."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x76_zero_shift_parent_hull_motion_rejection as x76  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X77_RECEDING_METRIC_TEMPORAL_HANDOFF_REJECTION"
ARM_X77 = "X77_ISSUED_PLAN_RECEDING_METRIC_TEMPORAL_HANDOFF_REJECTION"
SUPPORT_MODE = "X24_METRIC_TEMPORAL_HANDOFF"
TRANSPORT_STATE = "METRIC_TEMPORAL_HANDOFF"


def fixed_constants() -> dict[str, Any]:
    return {
        **x76.fixed_constants(),
        "representation": "X76_WITH_RECEDING_METRIC_TEMPORAL_HANDOFF_REJECTION",
        "retained_core": "X76",
        "release_rule": (
            "EVERY_CONFIRMED_CARRIER_IS_A_METRIC_TEMPORAL_HANDOFF_WITH_"
            "POSITIVE_FORWARD_VELOCITY"
        ),
        "velocity_sign_test": "INHERITED_NUMERIC_EPSILON",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _is_receding_metric_handoff(row: Mapping[str, Any]) -> bool:
    return (
        row.get("support_footprint_mode") == SUPPORT_MODE
        and row.get("disposition") == TRANSPORT_STATE
        and float(row.get("velocity_forward_mps", 0.0)) > x24.EPSILON
    )


def apply_receding_metric_temporal_handoff_rejection_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x76.ARM_X76]
        arm["x77_receding_metric_temporal_handoff_release_used"] = False
        arm["x77_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue
        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x77_carrier_reference")
        if not all(_is_receding_metric_handoff(rows[track_id]) for track_id in confirmed_ids):
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
                "x77_receding_metric_temporal_handoff_release_used": True,
                "x77_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X77] = value["arms"].pop(x76.ARM_X76)
    value["diagnostics"]["x77_route_mode_counts"] = value["diagnostics"].pop(
        "x76_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x77_receding_metric_temporal_handoff_release_frames": released_frames,
            "x77_receding_metric_temporal_handoff_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X77] = frame["arms"].pop(x76.ARM_X76)
    return value


def self_check() -> dict[str, Any]:
    row = {
        "support_footprint_mode": SUPPORT_MODE,
        "disposition": TRANSPORT_STATE,
        "velocity_forward_mps": 0.8,
    }
    x24.require(
        _is_receding_metric_handoff(row)
        and not _is_receding_metric_handoff({**row, "velocity_forward_mps": -0.8})
        and not _is_receding_metric_handoff({**row, "velocity_forward_mps": 0.0})
        and not _is_receding_metric_handoff({**row, "disposition": "HOLD"}),
        "x77_receding_handoff_partition",
    )
    return {
        "status": "X77_RECEDING_METRIC_TEMPORAL_HANDOFF_FALSIFIER_MET",
        "release_only": True,
        "approaching_or_stationary_handoff_retained": True,
        "velocity_sign_test_inherited": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
