"""X88 bounds transport contradiction authority to its observation epoch.

X79 conservatively retains a lateral-only surface carrier when its transport
history contains a contradiction. X68 can later replace that lattice history
with current object-local metric velocity. When this current measurement
establishes a new pure-lateral motion epoch, an old surface-transport conflict
remains diagnostic history but cannot independently authorize collision timing
for the newly measured trajectory.

X88 clears only frames whose sole active mechanism is X68 dequantization, whose
confirmed measured surface carriers were all dequantized to zero longitudinal
and nonzero lateral motion, and whose only reason for surviving X79 is a prior
transport contradiction. Independent mechanisms, non-lateral motion, held
carriers, and histories without a current metric update remain unchanged. The
contradiction record itself is retained. No new numeric threshold is added.
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
import dtr_carla_x87_solo_completion_horizon_release as x87  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X88_MOTION_EPOCH_CONTRADICTION_RELEASE"
ARM_X88 = "X88_ISSUED_PLAN_MOTION_EPOCH_CONTRADICTION_RELEASE"
X68_FLAG = "x68_object_local_lateral_dequantization_used"
SURFACE_CLASS = x79.SURFACE_CLASS


def fixed_constants() -> dict[str, Any]:
    return {
        **x87.fixed_constants(),
        "representation": "X87_WITH_MOTION_EPOCH_CONTRADICTION_RELEASE",
        "retained_core": "X87",
        "release_rule": (
            "SOLE_ACTIVE_MECHANISM_IS_X68_DEQUANTIZATION_AND_EVERY_CONFIRMED_"
            "CARRIER_IS_A_CURRENT_MEASURED_SURFACE_DEQUANTIZED_TO_PURE_LATERAL_"
            "MOTION_WITH_A_PRIOR_SURFACE_TRANSPORT_CONTRADICTION"
        ),
        "authority_rule": (
            "A_SURFACE_TRANSPORT_CONTRADICTION_REMAINS_DIAGNOSTIC_BUT_CANNOT_"
            "AUTHORIZE_COLLISION_TIMING_AFTER_CURRENT_METRIC_MOTION_STARTS_A_"
            "NEW_OBSERVATION_EPOCH"
        ),
        "current_epoch_source": "X68_OBJECT_LOCAL_METRIC_VELOCITY",
        "independent_mechanism_frames_retained": True,
        "held_or_non_lateral_carriers_retained": True,
        "contradiction_record_retained": True,
        "zero_and_nonzero_tests": "INHERITED_NUMERIC_EPSILON",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
        "new_metric_threshold_added": False,
    }


def _active_mechanism_flags(arm: Mapping[str, Any]) -> set[str]:
    return {
        str(key)
        for key, value in arm.items()
        if str(key).endswith("_used") and value is True
    }


def _is_new_epoch_lateral_carrier(
    row: Mapping[str, Any],
    track_id: str,
    dequantized_ids: set[str],
) -> bool:
    return (
        track_id in dequantized_ids
        and row.get("class_name") == SURFACE_CLASS
        and row.get("disposition") == "MEASURED"
        and abs(float(row.get("velocity_forward_mps", 0.0))) <= x24.EPSILON
        and abs(float(row.get("velocity_right_mps", 0.0))) > x24.EPSILON
        and int(row.get("transport_contradictions", 0)) > 0
    )


def _release_partition(
    arm: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
    confirmed_ids: set[str],
) -> bool:
    dequantized_ids = {
        str(track_id)
        for track_id in arm.get("x68_object_local_lateral_dequantized_track_ids", [])
    }
    metric_source_ids = {
        str(track_id)
        for track_id in arm.get("x68_object_local_metric_source_track_ids", [])
    }
    return (
        _active_mechanism_flags(arm) == {X68_FLAG}
        and bool(confirmed_ids)
        and bool(metric_source_ids)
        and all(
            _is_new_epoch_lateral_carrier(
                rows[track_id],
                track_id,
                dequantized_ids,
            )
            for track_id in confirmed_ids
        )
    )


def apply_motion_epoch_contradiction_release_episode(
    core: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    released_frames = 0
    released_tracks = 0

    for frame in value["frames"]:
        arm = frame["arms"][x87.ARM_X87]
        arm["x88_motion_epoch_contradiction_release_used"] = False
        arm["x88_released_track_ids"] = []
        if not bool(arm.get("route_risk")):
            continue

        rows = {str(row["track_id"]): row for row in frame["tracks"]}
        confirmed_ids = {
            str(track_id) for track_id in arm.get("confirmed_risk_track_ids", [])
        }
        x24.require(confirmed_ids and confirmed_ids.issubset(rows), "x88_carrier_reference")
        if not _release_partition(arm, rows, confirmed_ids):
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
                "x88_motion_epoch_contradiction_release_used": True,
                "x88_released_track_ids": sorted(confirmed_ids),
            }
        )
        released_frames += 1
        released_tracks += len(confirmed_ids)

    value["arms"][ARM_X88] = value["arms"].pop(x87.ARM_X87)
    value["diagnostics"]["x88_route_mode_counts"] = value["diagnostics"].pop(
        "x87_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x88_motion_epoch_contradiction_release_frames": released_frames,
            "x88_motion_epoch_contradiction_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X88] = frame["arms"].pop(x87.ARM_X87)
    return value


def self_check() -> dict[str, Any]:
    track_id = "surface-1::cone-1"
    row = {
        "track_id": track_id,
        "parent_track_id": "surface-1",
        "class_name": SURFACE_CLASS,
        "disposition": "MEASURED",
        "velocity_forward_mps": 0.0,
        "velocity_right_mps": -1.0,
        "transport_contradictions": 1,
    }
    arm = {
        X68_FLAG: True,
        "x68_object_local_lateral_dequantized_track_ids": [track_id],
        "x68_object_local_metric_source_track_ids": ["metric-1"],
    }
    rows = {track_id: row}
    x24.require(
        _release_partition(arm, rows, set(rows))
        and not _release_partition(
            arm,
            {track_id: {**row, "transport_contradictions": 0}},
            set(rows),
        )
        and not _release_partition(
            {**arm, "motion_evidence_credit_used": True},
            rows,
            set(rows),
        )
        and not _release_partition(
            arm,
            {track_id: {**row, "velocity_forward_mps": -1.0}},
            set(rows),
        ),
        "x88_motion_epoch_contradiction_partition",
    )
    return {
        "status": "X88_MOTION_EPOCH_CONTRADICTION_FALSIFIER_MET",
        "release_only": True,
        "sole_x68_mechanism_required": True,
        "prior_transport_contradiction_required": True,
        "current_object_local_metric_motion_required": True,
        "independent_mechanisms_retained": True,
        "held_or_non_lateral_carriers_retained": True,
        "contradiction_record_retained": True,
        "new_metric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
