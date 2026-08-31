"""X71 births occupancy from X24/X25 entry-time co-transport agreement.

X71 is allowed to birth route risk without a current X69/X70 surface carrier
only when two already-confirmed route-risk representations describe the same
object.  The X24 metric point must lie inside the current X25 rigid footprint,
the semantic class and route-forward motion direction must agree, and the two
representations must remain within the inherited association distance at the
later of their predicted route-entry times.

An explicit X69 rigid contradiction release retains precedence.  The rule
reuses X24/X25 geometry, route-entry times, and association distance and adds no
numeric threshold.  Consumed cohorts are Development evidence only.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x70_triple_credential_surface_dropout_handback as x70  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X71_ENTRY_COTRANSPORT_OCCUPANCY_BIRTH"
ARM_X71 = "X71_ISSUED_PLAN_ENTRY_COTRANSPORT_OCCUPANCY_BIRTH"
SUPPORT_MODE = "X24_X25_ENTRY_COTRANSPORT_OCCUPANCY_BIRTH"


def fixed_constants() -> dict[str, Any]:
    return {
        **x70.fixed_constants(),
        "representation": "X70_WITH_X24_X25_ENTRY_COTRANSPORT_OCCUPANCY_BIRTH",
        "retained_core": "X70",
        "birth_rule": (
            "CURRENT_X24_METRIC_POINT_INSIDE_CURRENT_X25_RIGID_FOOTPRINT_"
            "WITH_CLASS_FORWARD_DIRECTION_AND_ENTRY_TIME_COTRANSPORT_AGREEMENT"
        ),
        "entry_time_rule": "LATER_OF_CURRENT_X24_AND_X25_MINIMUM_ENTRY_SECONDS",
        "future_association_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "release_precedence": "X69_EXPLICIT_RIGID_CONTRADICTION_RELEASE_FIRST",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "new_numeric_threshold_added": False,
        "weather_or_lighting_label_used": False,
        "class_specific_prior_used": False,
    }


def _position(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [row["position_forward_m"], row["position_right_m"]], dtype=np.float64
    )


def _velocity(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [row["velocity_forward_mps"], row["velocity_right_mps"]],
        dtype=np.float64,
    )


def entry_cotransport_agrees(
    rigid_row: Mapping[str, Any],
    metric_row: Mapping[str, Any],
    *,
    rigid_entry_s: float,
    metric_entry_s: float,
) -> bool:
    if int(rigid_row["class_id"]) != int(metric_row["class_id"]):
        return False
    if (
        float(rigid_row["velocity_forward_mps"])
        * float(metric_row["velocity_forward_mps"])
        < -x24.EPSILON
    ):
        return False

    footprint = np.asarray(rigid_row["footprint_xy"], dtype=np.float64).reshape(
        -1, 2
    )
    metric_point = _position(metric_row)
    if not x25.point_in_convex_polygon(metric_point, footprint):
        return False

    horizon_s = max(float(rigid_entry_s), float(metric_entry_s))
    future_footprint = footprint + _velocity(rigid_row)[None, :] * horizon_s
    future_point = metric_point + _velocity(metric_row) * horizon_s
    future_center = np.mean(future_footprint, axis=0)
    return bool(
        np.linalg.norm(future_point - future_center)
        <= x24.ASSOCIATION_DISTANCE_M + x24.EPSILON
    )


def _carrier(row: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    source_id = str(row["track_id"])
    value.update(
        {
            "track_id": f"x71-birth::{source_id}",
            "parent_track_id": source_id,
            "risk_eligible": True,
            "motion_authority": x27.RIGID_DYNAMIC,
            "support_footprint_mode": SUPPORT_MODE,
            "x71_entry_cotransport_occupancy_birth": True,
            "x71_x25_source_track_id": source_id,
        }
    )
    return value


def apply_entry_cotransport_occupancy_birth_episode(
    core: dict[str, Any],
    rigid: dict[str, Any],
    metric: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    birth_frames = 0
    birth_tracks = 0
    release_rejections = 0
    class_rejections = 0
    forward_direction_rejections = 0
    current_containment_rejections = 0
    future_cotransport_rejections = 0

    x24.require(
        len(value["frames"]) == len(rigid["frames"]) == len(metric["frames"]),
        "x71_frame_count",
    )
    for frame, rigid_frame, metric_frame in zip(
        value["frames"], rigid["frames"], metric["frames"], strict=True
    ):
        x24.require(
            int(frame["sample_index"])
            == int(rigid_frame["sample_index"])
            == int(metric_frame["sample_index"]),
            "x71_frame_alignment",
        )
        arm = frame["arms"][x70.ARM_X70]
        arm["x71_entry_cotransport_occupancy_birth_used"] = False
        arm["x71_x25_source_track_ids"] = []
        arm["x71_x24_source_track_ids"] = []
        arm["x71_entry_cotransport_birth_track_ids"] = []
        if bool(arm.get("route_risk")):
            continue
        if bool(arm.get("x69_mature_cross_route_rigid_contradiction_release_used")):
            release_rejections += 1
            continue

        rigid_arm = rigid_frame["arms"][x25.ARM_X25]
        metric_arm = metric_frame["arms"][x24.ARM_X24]
        valid_metric_risk = bool(
            metric_arm.get("route_risk")
            and metric_arm.get("authority") == "VALID"
            and metric_arm.get("route_mode") == "ISSUED_PLAN"
        )
        if not bool(rigid_arm.get("route_risk")) or not valid_metric_risk:
            continue
        rigid_entry = rigid_arm.get("minimum_entry_s")
        metric_entry = metric_arm.get("minimum_entry_s")
        if rigid_entry is None or metric_entry is None:
            continue

        rigid_rows = {str(row["track_id"]): row for row in rigid_frame["tracks"]}
        metric_rows = {str(row["track_id"]): row for row in metric_frame["tracks"]}
        rigid_confirmed = [
            rigid_rows[str(track_id)]
            for track_id in rigid_arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in rigid_rows
        ]
        metric_confirmed = [
            metric_rows[str(track_id)]
            for track_id in metric_arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in metric_rows
        ]
        matching_pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for rigid_row in rigid_confirmed:
            for metric_row in metric_confirmed:
                if int(rigid_row["class_id"]) != int(metric_row["class_id"]):
                    class_rejections += 1
                    continue
                if (
                    float(rigid_row["velocity_forward_mps"])
                    * float(metric_row["velocity_forward_mps"])
                    < -x24.EPSILON
                ):
                    forward_direction_rejections += 1
                    continue
                footprint = np.asarray(
                    rigid_row["footprint_xy"], dtype=np.float64
                ).reshape(-1, 2)
                if not x25.point_in_convex_polygon(_position(metric_row), footprint):
                    current_containment_rejections += 1
                    continue
                if not entry_cotransport_agrees(
                    rigid_row,
                    metric_row,
                    rigid_entry_s=float(rigid_entry),
                    metric_entry_s=float(metric_entry),
                ):
                    future_cotransport_rejections += 1
                    continue
                matching_pairs.append((rigid_row, metric_row))
        if not matching_pairs:
            continue

        unique_rigid = {
            str(rigid_row["track_id"]): rigid_row
            for rigid_row, _metric_row in matching_pairs
        }
        carriers = [_carrier(unique_rigid[key]) for key in sorted(unique_rigid)]
        existing_ids = {str(row["track_id"]) for row in frame["tracks"]}
        for carrier in carriers:
            x24.require(
                str(carrier["track_id"]) not in existing_ids,
                "x71_birth_track_id_collision",
            )
            frame["tracks"].append(carrier)
            frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
        track_ids = sorted(str(row["track_id"]) for row in carriers)
        parent_ids = sorted(str(row["parent_track_id"]) for row in carriers)
        rigid_ids = sorted(unique_rigid)
        metric_ids = sorted(
            {str(metric_row["track_id"]) for _rigid_row, metric_row in matching_pairs}
        )
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": min(float(rigid_entry), float(metric_entry)),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x71_entry_cotransport_occupancy_birth_used": True,
                "x71_x25_source_track_ids": rigid_ids,
                "x71_x24_source_track_ids": metric_ids,
                "x71_entry_cotransport_birth_track_ids": track_ids,
            }
        )
        birth_frames += 1
        birth_tracks += len(carriers)

    value["arms"][ARM_X71] = value["arms"].pop(x70.ARM_X70)
    value["diagnostics"]["x71_route_mode_counts"] = value["diagnostics"].pop(
        "x70_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x71_entry_cotransport_birth_frames": birth_frames,
            "x71_entry_cotransport_birth_tracks": birth_tracks,
            "x71_release_rejections": release_rejections,
            "x71_class_rejections": class_rejections,
            "x71_forward_direction_rejections": forward_direction_rejections,
            "x71_current_containment_rejections": current_containment_rejections,
            "x71_future_cotransport_rejections": future_cotransport_rejections,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X71] = frame["arms"].pop(x70.ARM_X70)
    return value


def self_check() -> dict[str, Any]:
    rigid = {
        "track_id": "footprint-1",
        "class_id": 0,
        "class_name": "person",
        "position_forward_m": 2.0,
        "position_right_m": 0.0,
        "velocity_forward_mps": -0.1,
        "velocity_right_mps": 0.0,
        "footprint_xy": [[1.5, -0.5], [2.5, -0.5], [2.5, 0.5], [1.5, 0.5]],
    }
    metric = {
        "track_id": "metric-1",
        "class_id": 0,
        "class_name": "person",
        "position_forward_m": 2.0,
        "position_right_m": 0.0,
        "velocity_forward_mps": -0.1,
        "velocity_right_mps": 0.0,
    }
    x24.require(
        entry_cotransport_agrees(
            rigid, metric, rigid_entry_s=1.0, metric_entry_s=1.2
        ),
        "x71_positive_agreement",
    )
    opposite = {**metric, "velocity_forward_mps": 0.1}
    x24.require(
        not entry_cotransport_agrees(
            rigid, opposite, rigid_entry_s=1.0, metric_entry_s=1.2
        ),
        "x71_opposite_forward_motion_rejected",
    )
    divergent = {**metric, "velocity_right_mps": 2.0}
    x24.require(
        not entry_cotransport_agrees(
            rigid, divergent, rigid_entry_s=1.0, metric_entry_s=1.2
        ),
        "x71_entry_time_divergence_rejected",
    )
    return {
        "status": "X71_ENTRY_COTRANSPORT_OCCUPANCY_BIRTH_FALSIFIER_MET",
        "current_containment_required": True,
        "entry_time_cotransport_required": True,
        "forward_direction_agreement_required": True,
        "x69_release_precedence": True,
        "new_numeric_threshold_added": False,
        "class_specific_prior_used": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
