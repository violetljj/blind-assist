"""X67 measurement-horizon receding release around frozen X65.

An unmeasured object may still exist after the metric track's standard hold
window expires, but a trajectory that was already reactivated from dormancy and
then disappears again has completed a full observed state cycle. Its last
receding direction-only trajectory is no longer a current collision credential.
X67 keeps the track and its existence memory, while clearing route-risk only
when every confirmed carrier has that reactivation receipt, has exhausted the
inherited X24 hold window, is not currently measured, moves longitudinally
away, and has direction-only surface transport.

No detector, route, duration, distance, speed, weather, or numeric threshold is
added. The duration boundary is the existing X24 measurement hold horizon.
Consumed C26-C28/C32/C34 are Development evidence only.
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
import dtr_carla_x63_existence_only_object_permanence as x63  # noqa: E402
import dtr_carla_x65_ancestry_synchronized_conflict_handback as x65  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X67_MEASUREMENT_HORIZON_RECEDING_RELEASE"
ARM_X67 = "X67_ISSUED_PLAN_MEASUREMENT_HORIZON_RECEDING_RELEASE"


def fixed_constants() -> dict[str, Any]:
    return {
        **x65.fixed_constants(),
        "representation": "X65_WITH_MEASUREMENT_HORIZON_RECEDING_RELEASE",
        "retained_core": "X65",
        "release_rule": (
            "ALL_CONFIRMED_RISK_CARRIERS_PREVIOUSLY_REACTIVATED_FROM_DORMANCY_"
            "THEN_UNMEASURED_AT_OR_BEYOND_INHERITED_X24_HOLD_HORIZON_WITH_"
            "POSITIVE_FORWARD_VELOCITY_AND_DIRECTION_ONLY_SURFACE_TRANSPORT"
        ),
        "existence_memory_retained": True,
        "reactivation_receipt_required": True,
        "measurement_hold_horizon_seconds": x24.HOLD_WINDOW_S,
        "measurement_hold_horizon_inherited": True,
        "receding_rule": "VELOCITY_FORWARD_MPS_POSITIVE",
        "direction_only_rule": x63.DIRECTION_ONLY,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _is_horizon_exhausted_receding(row: Mapping[str, Any]) -> bool:
    return (
        row.get("disposition") != "MEASURED"
        and int(row.get("dormant_transport_reactivations", 0)) > 0
        and float(row.get("evidence_age_s", 0.0)) + x24.EPSILON
        >= x24.HOLD_WINDOW_S
        and float(row.get("velocity_forward_mps", 0.0)) > 0.0
        and row.get("transport_state") == x63.DIRECTION_ONLY
    )


def apply_measurement_horizon_receding_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x65.ARM_X65]
        arm["x67_measurement_horizon_receding_release_used"] = False
        arm["x67_measurement_horizon_receding_released_track_ids"] = []
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        if not bool(arm.get("route_risk")) or not confirmed_ids:
            continue
        rows = {
            str(row["track_id"]): row
            for row in frame["tracks"]
            if str(row.get("track_id")) in confirmed_ids
        }
        x24.require(
            set(rows) == confirmed_ids,
            "x67_confirmed_track_reference",
        )
        if not all(_is_horizon_exhausted_receding(row) for row in rows.values()):
            continue

        released = sorted(confirmed_ids)
        arm.update(
            {
                "route_risk": False,
                "minimum_entry_s": None,
                "candidate_risk_track_ids": [],
                "confirmed_risk_track_ids": [],
                "candidate_risk_parent_track_ids": [],
                "confirmed_risk_parent_track_ids": [],
                "x67_measurement_horizon_receding_release_used": True,
                "x67_measurement_horizon_receding_released_track_ids": released,
            }
        )
        released_frames += 1
        released_tracks += len(released)

    value["arms"][ARM_X67] = value["arms"].pop(x65.ARM_X65)
    value["diagnostics"]["x67_route_mode_counts"] = value["diagnostics"].pop(
        "x65_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x67_measurement_horizon_receding_release_frames": released_frames,
            "x67_measurement_horizon_receding_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X67] = frame["arms"].pop(x65.ARM_X65)
    return value


def self_check() -> dict[str, Any]:
    base = {
        "disposition": "HOLD",
        "dormant_transport_reactivations": 1,
        "evidence_age_s": x24.HOLD_WINDOW_S,
        "velocity_forward_mps": 1.0,
        "transport_state": x63.DIRECTION_ONLY,
    }
    x24.require(
        _is_horizon_exhausted_receding(base)
        and not _is_horizon_exhausted_receding(
            {**base, "velocity_forward_mps": -1.0}
        )
        and not _is_horizon_exhausted_receding(
            {**base, "disposition": "MEASURED"}
        )
        and not _is_horizon_exhausted_receding(
            {**base, "evidence_age_s": x24.HOLD_WINDOW_S * 0.5}
        )
        and not _is_horizon_exhausted_receding(
            {**base, "dormant_transport_reactivations": 0}
        ),
        "x67_measurement_horizon_partition",
    )
    return {
        "status": "X67_MEASUREMENT_HORIZON_RECEDING_RELEASE_FALSIFIER_MET",
        "retained_core": "X65",
        "existence_memory_retained": True,
        "reactivation_receipt_required": True,
        "receding_direction_only_route_authority_after_hold_horizon": False,
        "measurement_hold_horizon_inherited": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
