"""X72 completes a credentialed fragmented surface with an X25 footprint.

A surface parent earns a collision credential only through an existing X71
route-risk decision.  While that same parent remains observable, X72 may use a
current confirmed X25 route-risk footprint to complete its object extent only
when the footprint intersects at least one current measured surface fragment
but the rigid center lies inside none of that parent's fragments.  This
boundary-only relation distinguishes fragmented support from a contained
same-object near miss.

An explicit X69 rigid contradiction release clears surface credentials before
completion.  Polygon overlap is exact convex geometry with only inherited
floating-point epsilon; no detector, route, class, distance, or time threshold
is added.  Consumed cohorts are Development evidence only.
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
import dtr_carla_x71_entry_cotransport_occupancy_birth as x71  # noqa: E402


EXPERIMENT_ID = "DTR_CARLA_X72_CREDENTIALED_SURFACE_BOUNDARY_COMPLETION"
ARM_X72 = "X72_ISSUED_PLAN_CREDENTIALED_SURFACE_BOUNDARY_COMPLETION"
SUPPORT_MODE = "X25_CREDENTIALED_SURFACE_BOUNDARY_COMPLETION"
SURFACE_CLASS = "WORLD_OCCUPANCY_COMPONENT"


def fixed_constants() -> dict[str, Any]:
    return {
        **x71.fixed_constants(),
        "representation": "X71_WITH_CREDENTIALED_SURFACE_BOUNDARY_COMPLETION",
        "retained_core": "X71",
        "credential_birth_rule": "EXISTING_X71_SURFACE_PARENT_ROUTE_RISK",
        "completion_rule": (
            "CURRENT_X25_CONFIRMED_ROUTE_RISK_FOOTPRINT_OVERLAPS_CURRENT_"
            "MEASURED_CREDENTIALED_PARENT_FRAGMENT_WITH_RIGID_CENTER_OUTSIDE_"
            "ALL_PARENT_FRAGMENTS"
        ),
        "release_precedence": "X69_EXPLICIT_RIGID_CONTRADICTION_RELEASE_FIRST",
        "polygon_overlap": "CONVEX_SEPARATING_AXIS_THEOREM",
        "detector_threshold_change": False,
        "route_threshold_change": False,
        "new_numeric_threshold_added": False,
        "class_specific_prior_used": False,
    }


def _position(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(
        [row["position_forward_m"], row["position_right_m"]], dtype=np.float64
    )


def _polygon(row: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(row["footprint_xy"], dtype=np.float64).reshape(-1, 2)


def convex_polygons_overlap(
    left: Sequence[Sequence[float]] | np.ndarray,
    right: Sequence[Sequence[float]] | np.ndarray,
) -> bool:
    left_polygon = np.asarray(left, dtype=np.float64).reshape(-1, 2)
    right_polygon = np.asarray(right, dtype=np.float64).reshape(-1, 2)
    x24.require(len(left_polygon) >= 3, "x72_left_polygon")
    x24.require(len(right_polygon) >= 3, "x72_right_polygon")
    for polygon in (left_polygon, right_polygon):
        for index in range(len(polygon)):
            edge = polygon[(index + 1) % len(polygon)] - polygon[index]
            axis = np.asarray([-edge[1], edge[0]], dtype=np.float64)
            if float(np.linalg.norm(axis)) <= x24.EPSILON:
                continue
            left_projection = left_polygon @ axis
            right_projection = right_polygon @ axis
            if float(np.max(left_projection)) < (
                float(np.min(right_projection)) - x24.EPSILON
            ):
                return False
            if float(np.max(right_projection)) < (
                float(np.min(left_projection)) - x24.EPSILON
            ):
                return False
    return True


def boundary_fragment_parent_ids(
    rigid_row: Mapping[str, Any],
    fragments_by_parent: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    rigid_polygon = _polygon(rigid_row)
    rigid_center = _position(rigid_row)
    matched: list[str] = []
    for parent_id, fragments in fragments_by_parent.items():
        fragment_polygons = [_polygon(fragment) for fragment in fragments]
        if not any(
            convex_polygons_overlap(rigid_polygon, fragment_polygon)
            for fragment_polygon in fragment_polygons
        ):
            continue
        if any(
            x25.point_in_convex_polygon(rigid_center, fragment_polygon)
            for fragment_polygon in fragment_polygons
        ):
            continue
        matched.append(str(parent_id))
    return sorted(matched)


def _carrier(row: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    source_id = str(row["track_id"])
    value.update(
        {
            "track_id": f"x72-completion::{source_id}",
            "parent_track_id": source_id,
            "risk_eligible": True,
            "motion_authority": x27.RIGID_DYNAMIC,
            "support_footprint_mode": SUPPORT_MODE,
            "x72_credentialed_surface_boundary_completion": True,
            "x72_x25_source_track_id": source_id,
        }
    )
    return value


def apply_credentialed_surface_boundary_completion_episode(
    core: dict[str, Any],
    rigid: dict[str, Any],
) -> dict[str, Any]:
    value = copy.deepcopy(core)
    credentialed_surface_parents: set[str] = set()
    credential_births = 0
    release_clears = 0
    completion_frames = 0
    completion_tracks = 0
    center_contained_rejections = 0
    overlap_absent_rejections = 0

    x24.require(len(value["frames"]) == len(rigid["frames"]), "x72_frame_count")
    for frame, rigid_frame in zip(value["frames"], rigid["frames"], strict=True):
        x24.require(
            int(frame["sample_index"]) == int(rigid_frame["sample_index"]),
            "x72_frame_alignment",
        )
        arm = frame["arms"][x71.ARM_X71]
        arm["x72_credentialed_surface_boundary_completion_used"] = False
        arm["x72_credentialed_surface_parent_ids"] = []
        arm["x72_boundary_overlap_surface_parent_ids"] = []
        arm["x72_x25_source_track_ids"] = []
        arm["x72_completion_track_ids"] = []

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

        rigid_arm = rigid_frame["arms"][x25.ARM_X25]
        if not bool(rigid_arm.get("route_risk")):
            continue
        rigid_rows = {str(row["track_id"]): row for row in rigid_frame["tracks"]}
        rigid_confirmed = [
            rigid_rows[str(track_id)]
            for track_id in rigid_arm.get("confirmed_risk_track_ids", [])
            if str(track_id) in rigid_rows
        ]
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

        matched_rigid: dict[str, Mapping[str, Any]] = {}
        matched_parents: set[str] = set()
        for rigid_row in rigid_confirmed:
            rigid_polygon = _polygon(rigid_row)
            rigid_center = _position(rigid_row)
            has_overlap = False
            has_containment = False
            for fragments in fragments_by_parent.values():
                fragment_polygons = [_polygon(fragment) for fragment in fragments]
                has_overlap = has_overlap or any(
                    convex_polygons_overlap(rigid_polygon, fragment_polygon)
                    for fragment_polygon in fragment_polygons
                )
                has_containment = has_containment or any(
                    x25.point_in_convex_polygon(rigid_center, fragment_polygon)
                    for fragment_polygon in fragment_polygons
                )
            overlap_absent_rejections += int(not has_overlap)
            center_contained_rejections += int(has_containment)
            parent_ids = boundary_fragment_parent_ids(rigid_row, fragments_by_parent)
            if parent_ids:
                matched_rigid[str(rigid_row["track_id"])] = rigid_row
                matched_parents.update(parent_ids)
        if not matched_rigid:
            continue

        carriers = [_carrier(matched_rigid[key]) for key in sorted(matched_rigid)]
        existing_ids = {str(row["track_id"]) for row in frame["tracks"]}
        for carrier in carriers:
            x24.require(
                str(carrier["track_id"]) not in existing_ids,
                "x72_completion_track_id_collision",
            )
            frame["tracks"].append(carrier)
            frame["risk_eligible_tracks"] = int(frame["risk_eligible_tracks"]) + 1
        track_ids = sorted(str(row["track_id"]) for row in carriers)
        parent_ids = sorted(str(row["parent_track_id"]) for row in carriers)
        source_ids = sorted(matched_rigid)
        arm.update(
            {
                "route_risk": True,
                "minimum_entry_s": rigid_arm.get("minimum_entry_s"),
                "candidate_risk_track_ids": track_ids,
                "confirmed_risk_track_ids": track_ids,
                "candidate_risk_parent_track_ids": parent_ids,
                "confirmed_risk_parent_track_ids": parent_ids,
                "x72_credentialed_surface_boundary_completion_used": True,
                "x72_credentialed_surface_parent_ids": sorted(
                    credentialed_surface_parents
                ),
                "x72_boundary_overlap_surface_parent_ids": sorted(matched_parents),
                "x72_x25_source_track_ids": source_ids,
                "x72_completion_track_ids": track_ids,
            }
        )
        completion_frames += 1
        completion_tracks += len(carriers)

    value["arms"][ARM_X72] = value["arms"].pop(x71.ARM_X71)
    value["diagnostics"]["x72_route_mode_counts"] = value["diagnostics"].pop(
        "x71_route_mode_counts"
    )
    value["diagnostics"].update(
        {
            "x72_surface_credential_births": credential_births,
            "x72_surface_credential_release_clears": release_clears,
            "x72_boundary_completion_frames": completion_frames,
            "x72_boundary_completion_tracks": completion_tracks,
            "x72_center_contained_rejections": center_contained_rejections,
            "x72_overlap_absent_rejections": overlap_absent_rejections,
        }
    )
    for frame in value["frames"]:
        frame["arms"][ARM_X72] = frame["arms"].pop(x71.ARM_X71)
    return value


def self_check() -> dict[str, Any]:
    rigid = {
        "track_id": "rigid-1",
        "position_forward_m": 2.0,
        "position_right_m": 0.0,
        "footprint_xy": [[1.5, -0.5], [2.5, -0.5], [2.5, 0.5], [1.5, 0.5]],
    }
    boundary_fragment = {
        "footprint_xy": [[2.4, 0.2], [2.8, 0.2], [2.8, 0.8], [2.4, 0.8]]
    }
    containing_fragment = {
        "footprint_xy": [[1.8, -0.2], [2.2, -0.2], [2.2, 0.2], [1.8, 0.2]]
    }
    separated = {
        "footprint_xy": [[4.0, 4.0], [5.0, 4.0], [5.0, 5.0], [4.0, 5.0]]
    }
    x24.require(
        convex_polygons_overlap(_polygon(rigid), _polygon(boundary_fragment)),
        "x72_boundary_overlap",
    )
    x24.require(
        boundary_fragment_parent_ids(
            rigid, {"surface-parent": [boundary_fragment]}
        )
        == ["surface-parent"],
        "x72_boundary_completion_admitted",
    )
    x24.require(
        not boundary_fragment_parent_ids(
            rigid, {"surface-parent": [boundary_fragment, containing_fragment]}
        ),
        "x72_any_parent_fragment_center_containment_rejected",
    )
    x24.require(
        not convex_polygons_overlap(_polygon(rigid), _polygon(separated)),
        "x72_separated_rejected",
    )
    return {
        "status": "X72_CREDENTIALED_SURFACE_BOUNDARY_COMPLETION_FALSIFIER_MET",
        "surface_parent_credential_required": True,
        "current_measured_fragment_required": True,
        "boundary_overlap_required": True,
        "any_fragment_center_containment_rejected": True,
        "x69_release_precedence": True,
        "new_numeric_threshold_added": False,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(self_check(), sort_keys=True))
