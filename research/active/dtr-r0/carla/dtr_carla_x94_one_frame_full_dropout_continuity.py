"""X94 preserves one confirmed surface collision through one full dropout frame.

X93 can lose an alert when both detector candidates and metric footprints vanish
between adjacent observations even though the same surface parent remains inside
the inherited X24 hold window. X94 transports the immediately previous rigid,
confirmed surface carrier for exactly one observation when the issued-plan
receipt is unchanged and a current held row proves parent identity continuity.

Absence alone never creates risk: a previous confirmed carrier, current held
parent witness, unchanged route receipt, and zero active release are all
required. An X94 carrier cannot reseed itself. No new numeric threshold is
added; the inherited X24 hold window bounds continuity. Consumed cohorts are
Development evidence only.
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
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x73_credentialed_parent_hull_reconstruction as x73  # noqa: E402
import dtr_carla_x93_conflicted_nonclosing_future_release as x93  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X94_ONE_FRAME_FULL_DROPOUT_CONTINUITY"
ARM_X94 = "X94_ISSUED_PLAN_ONE_FRAME_FULL_DROPOUT_CONTINUITY"
SUPPORT_MODE = "PREVIOUS_CONFIRMED_SURFACE_FULL_DROPOUT_CONTINUITY"


def fixed_constants() -> dict[str, Any]:
    return {
        **x93.fixed_constants(),
        "representation": "X93_WITH_ONE_FRAME_FULL_DROPOUT_CONTINUITY",
        "retained_core": "X93",
        "continuity_birth_rule": (
            "IMMEDIATELY_PREVIOUS_CONFIRMED_RIGID_SURFACE_CARRIER_AND_"
            "CURRENT_HELD_SAME_PARENT_WITNESS"
        ),
        "full_dropout_rule": "ZERO_DETECTOR_CANDIDATES_AND_ZERO_METRIC_FOOTPRINTS",
        "route_continuity_rule": "UNCHANGED_VALID_ISSUED_PLAN_RECEIPT",
        "release_precedence": "ZERO_CURRENT_ACTIVE_RELEASE_FLAGS",
        "continuity_span": "ONE_NEXT_OBSERVATION_ONLY",
        "continuity_window_s": x24.HOLD_WINDOW_S,
        "continuity_can_reseed": False,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "new_numeric_threshold_added": False,
        "weather_or_lighting_label_used": False,
    }


def _parent_id(row: Mapping[str, Any]) -> str:
    return str(row.get("parent_track_id") or row["track_id"])


def _active_release(arm: Mapping[str, Any]) -> bool:
    return any(
        key.endswith("_release_used") and bool(value)
        for key, value in arm.items()
    )


def _transport_carrier(
    row: Mapping[str, Any], now_s: float, previous_time_s: float
) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    dt = float(now_s) - float(previous_time_s)
    parent_id = _parent_id(row)
    value.update(
        {
            "track_id": f"x94-full-dropout::{parent_id}",
            "parent_track_id": parent_id,
            "position_forward_m": float(row["position_forward_m"])
            + float(row["velocity_forward_mps"]) * dt,
            "position_right_m": float(row["position_right_m"])
            + float(row["velocity_right_mps"]) * dt,
            "disposition": "HOLD",
            "evidence_age_s": dt,
            "risk_eligible": True,
            "motion_authority": x27.RIGID_DYNAMIC,
            "support_footprint_mode": SUPPORT_MODE,
            "x94_one_frame_full_dropout_continuity": True,
            "x94_source_track_id": str(row["track_id"]),
        }
    )
    return value


def apply_one_frame_full_dropout_continuity_episode(
    core: dict[str, Any]
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    continuity_frames = 0
    continuity_tracks = 0
    parent_witness_rejections = 0

    for ordinal, frame in enumerate(value["frames"]):
        arm = frame["arms"][x93.ARM_X93]
        arm["x94_one_frame_full_dropout_continuity_used"] = False
        arm["x94_continuity_parent_ids"] = []
        arm["x94_continuity_track_ids"] = []
        arm["x94_source_track_ids"] = []
        if ordinal == 0 or bool(arm.get("route_risk")):
            continue
        if int(frame.get("raw_candidates", 0)) != 0 or int(
            frame.get("metric_footprint_measurements", 0)
        ) != 0:
            continue
        if (
            arm.get("authority") != "VALID"
            or arm.get("route_mode") != "ISSUED_PLAN"
            or bool(arm.get("route_mode_changed"))
            or _active_release(arm)
        ):
            continue

        previous = value["frames"][ordinal - 1]
        previous_arm = previous["arms"][x93.ARM_X93]
        dt = float(frame["time_s"]) - float(previous["time_s"])
        if (
            not bool(previous_arm.get("route_risk"))
            or bool(previous_arm.get("x94_one_frame_full_dropout_continuity_used"))
            or previous_arm.get("authority") != "VALID"
            or previous_arm.get("route_mode") != "ISSUED_PLAN"
            or previous_arm.get("plan_receipt_sha256")
            != arm.get("plan_receipt_sha256")
            or dt <= x24.EPSILON
            or dt > x24.HOLD_WINDOW_S + x24.EPSILON
        ):
            continue

        previous_rows = {str(row["track_id"]): row for row in previous["tracks"]}
        sources = [
            previous_rows[track_id]
            for track_id in map(
                str, previous_arm.get("confirmed_risk_track_ids", [])
            )
            if track_id in previous_rows
            and previous_rows[track_id].get("class_name") == x73.SURFACE_CLASS
            and previous_rows[track_id].get("motion_authority") == x27.RIGID_DYNAMIC
            and bool(previous_rows[track_id].get("risk_eligible"))
            and previous_rows[track_id].get("footprint_xy")
        ]
        if not sources:
            continue
        held_parent_ids = {
            _parent_id(row)
            for row in frame["tracks"]
            if row.get("disposition") == "HOLD"
            and float(row.get("evidence_age_s", float("inf")))
            <= x24.HOLD_WINDOW_S + x24.EPSILON
        }
        sources = [row for row in sources if _parent_id(row) in held_parent_ids]
        if not sources:
            parent_witness_rejections += 1
            continue

        carriers = [
            _transport_carrier(row, float(frame["time_s"]), float(previous["time_s"]))
            for row in sources
        ]
        existing_ids = {str(row["track_id"]) for row in frame["tracks"]}
        for carrier in carriers:
            x24.require(
                str(carrier["track_id"]) not in existing_ids,
                "x94_continuity_track_id_collision",
            )
            frame["tracks"].append(carrier)
            frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
        track_ids = sorted(str(row["track_id"]) for row in carriers)
        parent_ids = sorted({_parent_id(row) for row in carriers})
        source_ids = sorted(str(row["track_id"]) for row in sources)
        entry = previous_arm.get("minimum_entry_s")
        x24.require(entry is not None, "x94_previous_entry")
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": max(0.0, float(entry) - dt),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x94_one_frame_full_dropout_continuity_used": True,
                "x94_continuity_parent_ids": parent_ids,
                "x94_continuity_track_ids": track_ids,
                "x94_source_track_ids": source_ids,
            }
        )
        continuity_frames += 1
        continuity_tracks += len(carriers)

    value["arms"][ARM_X94] = value["arms"].pop(x93.ARM_X93)
    value["diagnostics"]["x94_route_mode_counts"] = value["diagnostics"].pop(
        "x93_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x94_one_frame_full_dropout_continuity_frames": continuity_frames,
            "x94_one_frame_full_dropout_continuity_tracks": continuity_tracks,
            "x94_parent_witness_rejections": parent_witness_rejections,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X94] = frame["arms"].pop(x93.ARM_X93)
    return value


def self_check() -> dict[str, Any]:
    source = {
        "track_id": "surface-parent-hull",
        "parent_track_id": "surface-parent",
        "position_forward_m": 2.0,
        "position_right_m": 0.5,
        "velocity_forward_mps": -1.0,
        "velocity_right_mps": 0.0,
        "footprint_xy": [[1.5, 0.0], [2.5, 0.0], [2.5, 1.0], [1.5, 1.0]],
    }
    carrier = _transport_carrier(source, 1.1, 1.0)
    x24.require(
        carrier["parent_track_id"] == "surface-parent"
        and carrier["motion_authority"] == x27.RIGID_DYNAMIC
        and carrier["risk_eligible"]
        and abs(float(carrier["position_forward_m"]) - 1.9) <= x24.EPSILON,
        "x94_carrier_transport",
    )
    x24.require(
        _active_release({"x93_conflicted_nonclosing_future_release_used": True})
        and not _active_release({"x73_credentialed_parent_hull_reconstruction_used": True}),
        "x94_release_precedence_partition",
    )
    return {
        "status": "X94_ONE_FRAME_FULL_DROPOUT_CONTINUITY_FALSIFIER_MET",
        "previous_confirmed_surface_carrier_required": True,
        "current_held_parent_witness_required": True,
        "unchanged_issued_plan_receipt_required": True,
        "one_next_observation_only": True,
        "continuity_can_reseed": False,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
