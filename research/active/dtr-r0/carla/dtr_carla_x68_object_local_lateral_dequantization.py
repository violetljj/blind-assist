"""X68 object-local metric lateral dequantization around frozen X67.

The surface representation quantizes transport velocity to lattice steps. A
spatially matched, currently measured metric track may refine that velocity for
collision geometry only when it points in the same overall direction and does
not introduce more route-right motion than the surface estimate. This lets the
metric representation remove spurious lateral lattice motion without replacing
the surface footprint or inventing a new crossing direction.

No detector, route, duration, distance, speed, weather, or score threshold is
added. Spatial matching reuses the X24 association distance; collision geometry
and route thresholds are inherited unchanged. Consumed C26-C28/C32/C34 are
Development evidence only.
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
import dtr_carla_x30_adaptive_surface_interval_predictor as x30  # noqa: E402
import dtr_carla_x67_measurement_horizon_receding_release as x67  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X68_OBJECT_LOCAL_LATERAL_DEQUANTIZATION"
ARM_X68 = "X68_ISSUED_PLAN_OBJECT_LOCAL_LATERAL_DEQUANTIZATION"


def fixed_constants() -> dict[str, Any]:
    return {
        **x67.fixed_constants(),
        "representation": "X67_WITH_OBJECT_LOCAL_METRIC_LATERAL_DEQUANTIZATION",
        "retained_core": "X67",
        "spatial_identity_rule": "INHERITED_X24_ASSOCIATION_DISTANCE",
        "metric_velocity_rule": (
            "CURRENT_MEASURED_OBJECT_LOCAL_MATCH_WITH_POSITIVE_VELOCITY_DOT_"
            "AND_METRIC_ROUTE_RIGHT_MAGNITUDE_NOT_GREATER_THAN_SURFACE"
        ),
        "geometry_rule": (
            "RETAIN_SURFACE_FOOTPRINT_AND_REEVALUATE_INHERITED_ROUTE_CONTACT_"
            "WITH_OBJECT_LOCAL_METRIC_VELOCITY"
        ),
        "surface_footprint_retained": True,
        "metric_velocity_cannot_introduce_lateral_motion": True,
        "association_distance_m": x24.ASSOCIATION_DISTANCE_M,
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "weather_or_lighting_label_used": False,
        "new_numeric_threshold_added": False,
    }


def _metric_refines_lateral(
    surface: Mapping[str, Any], metric: Mapping[str, Any]
) -> bool:
    sf = float(surface.get("velocity_forward_mps", 0.0))
    sr = float(surface.get("velocity_right_mps", 0.0))
    mf = float(metric.get("velocity_forward_mps", 0.0))
    mr = float(metric.get("velocity_right_mps", 0.0))
    return (
        sf * mf + sr * mr > x24.EPSILON
        and abs(mr) <= abs(sr) + x24.EPSILON
    )


def _distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    forward = float(left["position_forward_m"]) - float(
        right["position_forward_m"]
    )
    route_right = float(left["position_right_m"]) - float(
        right["position_right_m"]
    )
    return (forward * forward + route_right * route_right) ** 0.5


def apply_object_local_lateral_dequantization_episode(
    core: dict[str, Any],
    metric: dict[str, Any],
    episode: Any,
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    refined_frames = 0
    released_frames = 0
    released_tracks = 0
    receipt_cache: dict[Path, Mapping[str, Any]] = {}
    previous_mode: str | None = None

    x24.require(
        len(value["frames"]) == len(metric["frames"]) == len(episode.observations),
        "x68_frame_count",
    )
    for frame, metric_frame, observation in zip(
        value["frames"], metric["frames"], episode.observations, strict=True
    ):
        x24.require(
            int(frame["sample_index"]) == int(metric_frame["sample_index"])
            == int(observation.sample_index),
            "x68_frame_alignment",
        )
        wearer_position, wearer_velocity = x24.wearer_anchor_state(
            observation, episode.route_frame
        )
        receipt = x24.load_receipt(observation, receipt_cache)
        selection = x24.route.select_route(
            receipt,
            session_id=observation.navigation_session_id,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
            previous_mode=previous_mode,
        )
        previous_mode = selection.mode
        segments = x24.route.build_route_segments(
            selection,
            receipt=receipt,
            now_s=observation.time_s,
            wearer_position_xy=wearer_position,
            wearer_velocity_xy=wearer_velocity,
        )

        arm = frame["arms"][x67.ARM_X67]
        arm["x68_object_local_lateral_dequantization_used"] = False
        arm["x68_object_local_lateral_dequantized_track_ids"] = []
        arm["x68_object_local_metric_source_track_ids"] = []
        arm["x68_object_local_lateral_released_track_ids"] = []
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
        x24.require(set(rows) == confirmed_ids, "x68_confirmed_track_reference")
        # A metric handback carrier has no surface footprint. Its inherited
        # route decision is outside X68's geometry-refinement authority, so a
        # mixed or metric-only frame remains exactly unchanged.
        if any("footprint_xy" not in row for row in rows.values()):
            continue
        metric_rows = [
            row for row in metric_frame["tracks"] if row.get("disposition") == "MEASURED"
        ]

        retained_entries: dict[str, float] = {}
        refined_ids: list[str] = []
        metric_source_ids: list[str] = []
        released_ids: list[str] = []
        for track_id, row in rows.items():
            original_entry = None
            if "footprint_xy" in row:
                original_entry = x30.first_contact_interval_entry_s(
                    row["footprint_xy"],
                    (
                        float(row["velocity_forward_mps"]),
                        float(row["velocity_right_mps"]),
                    ),
                    segments,
                )
            x24.require(original_entry is not None, "x68_confirmed_surface_entry")
            if row.get("disposition") != "MEASURED":
                retained_entries[track_id] = float(original_entry)
                continue
            matches = [
                metric_row
                for metric_row in metric_rows
                if _distance(row, metric_row)
                <= x24.ASSOCIATION_DISTANCE_M + x24.EPSILON
            ]
            if not matches:
                retained_entries[track_id] = float(original_entry)
                continue
            source = min(matches, key=lambda metric_row: _distance(row, metric_row))
            if not _metric_refines_lateral(row, source):
                retained_entries[track_id] = float(original_entry)
                continue
            refined_entry = x30.first_contact_interval_entry_s(
                row["footprint_xy"],
                (
                    float(source["velocity_forward_mps"]),
                    float(source["velocity_right_mps"]),
                ),
                segments,
            )
            refined_ids.append(track_id)
            metric_source_ids.append(str(source["track_id"]))
            if refined_entry is None:
                released_ids.append(track_id)
            else:
                retained_entries[track_id] = float(refined_entry)

        if refined_ids:
            refined_frames += 1
            arm["x68_object_local_lateral_dequantization_used"] = True
            arm["x68_object_local_lateral_dequantized_track_ids"] = sorted(refined_ids)
            arm["x68_object_local_metric_source_track_ids"] = sorted(metric_source_ids)
        if not released_ids:
            continue

        retained_ids = confirmed_ids - set(released_ids)
        candidate_ids = {
            str(track_id) for track_id in arm.get("candidate_risk_track_ids", [])
        } - set(released_ids)
        all_rows = {str(row["track_id"]): row for row in frame["tracks"]}
        arm.update(
            {
                "route_risk": bool(retained_ids),
                "minimum_entry_s": (
                    min(retained_entries[track_id] for track_id in retained_ids)
                    if retained_ids
                    else None
                ),
                "candidate_risk_track_ids": sorted(candidate_ids),
                "confirmed_risk_track_ids": sorted(retained_ids),
                "candidate_risk_parent_track_ids": sorted(
                    {
                        str(all_rows[track_id]["parent_track_id"])
                        for track_id in candidate_ids
                        if track_id in all_rows and all_rows[track_id].get("parent_track_id")
                    }
                ),
                "confirmed_risk_parent_track_ids": sorted(
                    {
                        str(all_rows[track_id]["parent_track_id"])
                        for track_id in retained_ids
                        if track_id in all_rows and all_rows[track_id].get("parent_track_id")
                    }
                ),
                "x68_object_local_lateral_released_track_ids": sorted(released_ids),
            }
        )
        released_frames += 1
        released_tracks += len(released_ids)

    value["arms"][ARM_X68] = value["arms"].pop(x67.ARM_X67)
    value["diagnostics"]["x68_route_mode_counts"] = value["diagnostics"].pop(
        "x67_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x68_object_local_lateral_dequantization_frames": refined_frames,
            "x68_object_local_lateral_release_frames": released_frames,
            "x68_object_local_lateral_released_tracks": released_tracks,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X68] = frame["arms"].pop(x67.ARM_X67)
    return value


def self_check() -> dict[str, Any]:
    surface = {"velocity_forward_mps": -3.0, "velocity_right_mps": 2.0}
    x24.require(
        _metric_refines_lateral(
            surface,
            {"velocity_forward_mps": -2.0, "velocity_right_mps": 0.5},
        )
        and not _metric_refines_lateral(
            surface,
            {"velocity_forward_mps": 2.0, "velocity_right_mps": 0.5},
        )
        and not _metric_refines_lateral(
            surface,
            {"velocity_forward_mps": -2.0, "velocity_right_mps": 3.0},
        ),
        "x68_directional_lateral_partition",
    )
    return {
        "status": "X68_OBJECT_LOCAL_LATERAL_DEQUANTIZATION_FALSIFIER_MET",
        "retained_core": "X67",
        "surface_footprint_retained": True,
        "same_direction_required": True,
        "metric_cannot_introduce_more_lateral_motion": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
