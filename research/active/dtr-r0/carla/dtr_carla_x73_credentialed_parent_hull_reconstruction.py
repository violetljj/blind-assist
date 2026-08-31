"""X73 reconstructs a credentialed surface parent before route prediction.

X72 can complete a fragmented surface with a current confirmed X25 collision
footprint. X73 covers the complementary case where the same credentialed
surface parent remains currently measured but none of its individual fragments
retains collision geometry. It reconstructs the parent convex hull, transports
it with the area-weighted current fragment velocity, and applies the inherited
X25 footprint-to-route test.

Any current measured X25 rigid center contained by a parent fragment vetoes
reconstruction. In that case the rigid representation already explains the
surface support and parent-wide filling is ambiguous. X69 explicit release
retains precedence. No detector, route, class, distance, time, or score
threshold is added. Consumed cohorts are Development evidence only.
"""

from __future__ import annotations

import copy
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import dtr_carla_x24_plan_adherent_predictor as x24  # noqa: E402
import dtr_carla_x25_rigid_footprint_predictor as x25  # noqa: E402
import dtr_carla_x27_occupancy_authority_predictor as x27  # noqa: E402
import dtr_carla_x72_credentialed_surface_boundary_completion as x72  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X73_CREDENTIALED_PARENT_HULL_RECONSTRUCTION"
ARM_X73 = "X73_ISSUED_PLAN_CREDENTIALED_PARENT_HULL_RECONSTRUCTION"
SUPPORT_MODE = "CREDENTIALED_PARENT_CURRENT_FRAGMENT_HULL"
SURFACE_CLASS = x72.SURFACE_CLASS


def fixed_constants() -> dict[str, Any]:
    return {
        **x72.fixed_constants(),
        "representation": "X72_WITH_CREDENTIALED_PARENT_HULL_RECONSTRUCTION",
        "retained_core": "X72",
        "credential_rule": "EXISTING_X72_SURFACE_PARENT_COLLISION_CREDENTIAL",
        "reconstruction_rule": (
            "CONVEX_HULL_OF_ALL_CURRENT_MEASURED_FRAGMENTS_FOR_ONE_"
            "CREDENTIALED_SURFACE_PARENT"
        ),
        "transport_rule": "CURRENT_FRAGMENT_AREA_WEIGHTED_VELOCITY",
        "route_geometry_rule": "INHERITED_X25_FOOTPRINT_ROUTE_ENTRY",
        "rigid_veto_rule": (
            "REJECT_WHEN_ANY_CURRENT_MEASURED_X25_CENTER_IS_CONTAINED_BY_"
            "ANY_CURRENT_PARENT_FRAGMENT"
        ),
        "release_precedence": "X69_EXPLICIT_RIGID_CONTRADICTION_RELEASE_FIRST",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "new_numeric_threshold_added": False,
        "class_specific_prior_used": False,
    }


def convex_hull(points: Sequence[Sequence[float]]) -> np.ndarray:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 1:
        return np.asarray(unique, dtype=np.float64)

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    return abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    ) * 0.5


def rigid_center_contained(
    fragments: Sequence[Mapping[str, Any]],
    rigid_rows: Sequence[Mapping[str, Any]],
) -> bool:
    fragment_polygons = [x72._polygon(fragment) for fragment in fragments]
    return any(
        x25.point_in_convex_polygon(x72._position(rigid_row), fragment_polygon)
        for rigid_row in rigid_rows
        for fragment_polygon in fragment_polygons
    )


def _parent_velocity(fragments: Sequence[Mapping[str, Any]]) -> np.ndarray:
    weights = np.asarray(
        [float(fragment["footprint_area_m2"]) for fragment in fragments],
        dtype=np.float64,
    )
    velocities = np.asarray(
        [
            [
                float(fragment["velocity_forward_mps"]),
                float(fragment["velocity_right_mps"]),
            ]
            for fragment in fragments
        ],
        dtype=np.float64,
    )
    return np.average(velocities, axis=0, weights=weights)


def _carrier(
    parent_id: str,
    fragments: Sequence[Mapping[str, Any]],
    hull: np.ndarray,
    velocity: np.ndarray,
) -> dict[str, Any]:
    value = copy.deepcopy(dict(fragments[0]))
    center = np.mean(hull, axis=0)
    value.update(
        {
            "track_id": f"x73-parent-hull::{parent_id}",
            "parent_track_id": parent_id,
            "position_forward_m": float(center[0]),
            "position_right_m": float(center[1]),
            "velocity_forward_mps": float(velocity[0]),
            "velocity_right_mps": float(velocity[1]),
            "footprint_xy": hull.tolist(),
            "footprint_area_m2": _polygon_area(hull),
            "disposition": "MEASURED",
            "evidence_age_s": 0.0,
            "risk_eligible": True,
            "motion_authority": x27.RIGID_DYNAMIC,
            "support_footprint_mode": SUPPORT_MODE,
            "x73_credentialed_parent_hull_reconstruction": True,
            "x73_source_fragment_track_ids": sorted(
                str(fragment["track_id"]) for fragment in fragments
            ),
        }
    )
    return value


def apply_credentialed_parent_hull_reconstruction_episode(
    core: dict[str, Any],
    rigid: dict[str, Any],
    episode: Any,
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_surface_parents: set[str] = set()
    receipt_cache: dict[Path, Mapping[str, Any]] = {}
    previous_mode: str | None = None
    credential_births = 0
    release_clears = 0
    reconstruction_frames = 0
    reconstruction_tracks = 0
    rigid_containment_rejections = 0
    route_absent_rejections = 0

    x24.require(
        len(value["frames"]) == len(rigid["frames"]) == len(episode.observations),
        "x73_frame_count",
    )
    for frame, rigid_frame, observation in zip(
        value["frames"], rigid["frames"], episode.observations, strict=True
    ):
        x24.require(
            int(frame["sample_index"])
            == int(rigid_frame["sample_index"])
            == int(observation.sample_index),
            "x73_frame_alignment",
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

        arm = frame["arms"][x72.ARM_X72]
        arm["x73_credentialed_parent_hull_reconstruction_used"] = False
        arm["x73_credentialed_surface_parent_ids"] = []
        arm["x73_reconstructed_surface_parent_ids"] = []
        arm["x73_source_fragment_track_ids"] = []
        arm["x73_parent_hull_track_ids"] = []

        surface_rows = [
            row for row in frame["tracks"] if row.get("class_name") == SURFACE_CLASS
        ]
        live_parents = {
            str(row.get("parent_track_id") or row["track_id"])
            for row in surface_rows
        }
        credentialed_surface_parents.intersection_update(live_parents)
        if bool(arm.get("x69_mature_cross_route_rigid_contradiction_release_used")):
            release_clears += len(credentialed_surface_parents)
            credentialed_surface_parents.clear()
            continue

        if bool(arm.get("route_risk")):
            before = len(credentialed_surface_parents)
            credentialed_surface_parents.update(
                str(parent_id)
                for parent_id in arm.get("confirmed_risk_parent_track_ids", [])
                if str(parent_id).startswith("surface-cone-")
            )
            credential_births += max(0, len(credentialed_surface_parents) - before)
            continue
        if not credentialed_surface_parents:
            continue

        fragments_by_parent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in surface_rows:
            parent_id = str(row.get("parent_track_id") or row["track_id"])
            if (
                parent_id in credentialed_surface_parents
                and row.get("disposition") == "MEASURED"
                and float(row.get("footprint_area_m2", 0.0)) > 0.0
            ):
                fragments_by_parent[parent_id].append(row)
        if not fragments_by_parent:
            continue

        measured_rigid = [
            row
            for row in rigid_frame["tracks"]
            if row.get("disposition") == "MEASURED"
        ]
        reconstructed: list[tuple[str, list[Mapping[str, Any]], np.ndarray, np.ndarray, float]] = []
        for parent_id, fragments in sorted(fragments_by_parent.items()):
            if rigid_center_contained(fragments, measured_rigid):
                rigid_containment_rejections += 1
                continue
            hull = convex_hull(
                [point for fragment in fragments for point in fragment["footprint_xy"]]
            )
            x24.require(len(hull) >= 3, "x73_parent_hull")
            velocity = _parent_velocity(fragments)
            entry = x25.first_footprint_route_entry_s(hull, velocity, segments)
            if entry is None:
                route_absent_rejections += 1
                continue
            reconstructed.append((parent_id, fragments, hull, velocity, float(entry)))
        if not reconstructed:
            continue

        carriers = [
            _carrier(parent_id, fragments, hull, velocity)
            for parent_id, fragments, hull, velocity, _entry in reconstructed
        ]
        existing_ids = {str(row["track_id"]) for row in frame["tracks"]}
        for carrier in carriers:
            x24.require(
                str(carrier["track_id"]) not in existing_ids,
                "x73_parent_hull_track_id_collision",
            )
            frame["tracks"].append(carrier)
            frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
        track_ids = sorted(str(carrier["track_id"]) for carrier in carriers)
        parent_ids = sorted(str(carrier["parent_track_id"]) for carrier in carriers)
        source_ids = sorted(
            str(fragment["track_id"])
            for _parent, fragments, _hull, _velocity, _entry in reconstructed
            for fragment in fragments
        )
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": min(entry for *_rest, entry in reconstructed),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x73_credentialed_parent_hull_reconstruction_used": True,
                "x73_credentialed_surface_parent_ids": sorted(
                    credentialed_surface_parents
                ),
                "x73_reconstructed_surface_parent_ids": parent_ids,
                "x73_source_fragment_track_ids": source_ids,
                "x73_parent_hull_track_ids": track_ids,
            }
        )
        reconstruction_frames += 1
        reconstruction_tracks += len(carriers)

    value["arms"][ARM_X73] = value["arms"].pop(x72.ARM_X72)
    value["diagnostics"]["x73_route_mode_counts"] = value["diagnostics"].pop(
        "x72_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x73_surface_credential_births": credential_births,
            "x73_surface_credential_release_clears": release_clears,
            "x73_parent_hull_reconstruction_frames": reconstruction_frames,
            "x73_parent_hull_reconstruction_tracks": reconstruction_tracks,
            "x73_rigid_center_containment_rejections": rigid_containment_rejections,
            "x73_parent_hull_route_absent_rejections": route_absent_rejections,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X73] = frame["arms"].pop(x72.ARM_X72)
    return value


def self_check() -> dict[str, Any]:
    fragments = [
        {
            "footprint_xy": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        },
        {
            "footprint_xy": [[2.0, 0.0], [3.0, 0.0], [3.0, 1.0], [2.0, 1.0]],
        },
    ]
    hull = convex_hull(
        [point for fragment in fragments for point in fragment["footprint_xy"]]
    )
    x24.require(_polygon_area(hull) == 3.0, "x73_convex_hull_area")
    contained = {
        "position_forward_m": 0.5,
        "position_right_m": 0.5,
    }
    outside = {
        "position_forward_m": 4.0,
        "position_right_m": 4.0,
    }
    x24.require(
        rigid_center_contained(fragments, [contained]),
        "x73_contained_rigid_veto",
    )
    x24.require(
        not rigid_center_contained(fragments, [outside]),
        "x73_outside_rigid_retained",
    )
    return {
        "status": "X73_CREDENTIALED_PARENT_HULL_FALSIFIER_MET",
        "current_credentialed_fragments_required": True,
        "current_rigid_center_containment_veto": True,
        "x69_release_precedence": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
