"""Anchor-conditioned, RGB-independent portal-interior geometry teacher.

The primary representation is a soft field of rays that cross an anchor's
support plane and reach valid mesh farther behind it.  Image boundary lines are
derived only at the legacy evaluator seam.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .two_view_observation import ImageLine, _project_world_line


@dataclass(frozen=True)
class SupportPlane:
    point_world_m: np.ndarray
    normal_world: np.ndarray
    axis_u_world: np.ndarray
    axis_v_world: np.ndarray
    anchor_uv_m: tuple[float, float]
    fit_rmse_m: float
    support_count: int


@dataclass(frozen=True)
class PortalViewField:
    soft_mask: np.ndarray
    plane_hit_mask: np.ndarray
    behind_free_mask: np.ndarray
    unknown_mask: np.ndarray
    component_mask: np.ndarray
    component_uv_bounds_m: tuple[float, float, float, float] | None
    diagnostics: dict


@dataclass(frozen=True)
class PortalInteriorPrediction:
    views: tuple[PortalViewField, PortalViewField]
    support_plane: SupportPlane | None
    center_bearing_rad: float | None
    range_m: float | None
    width_m: float | None
    center_x_m: float | None
    target_front_waypoint_world_m: tuple[float, float, float] | None
    derived_boundary_lines: tuple[tuple[ImageLine, ImageLine], tuple[ImageLine, ImageLine]] | None
    confidence: float
    diagnostics: dict


def _camera_points(depth_m: np.ndarray, intrinsics) -> np.ndarray:
    y, x = np.indices(depth_m.shape, dtype=np.float64)
    return np.stack(
        ((x - intrinsics.cx) * depth_m / intrinsics.fx, (y - intrinsics.cy) * depth_m / intrinsics.fy, depth_m),
        axis=2,
    )


def fit_anchor_support_plane(depth_m: np.ndarray, pose, intrinsics, anchor_bbox: tuple[int, int, int, int]) -> SupportPlane | None:
    """Fit the nearest coherent mesh plane around the semantic anchor."""

    depth = np.asarray(depth_m, dtype=np.float64)
    x1, y1, x2, y2 = anchor_bbox
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    radius = max(24, round(max(x2 - x1, y2 - y1) * 1.6))
    xa, xb = max(0, round(cx) - radius), min(depth.shape[1], round(cx) + radius + 1)
    ya, yb = max(0, round(cy) - radius), min(depth.shape[0], round(cy) + radius + 1)
    patch = depth[ya:yb, xa:xb]
    valid = np.isfinite(patch) & (patch >= 0.35) & (patch <= 8.0)
    if int(valid.sum()) < 80:
        return None
    # The anchor may be composited over an aperture.  The nearest substantial
    # surface cluster is the support wall; deeper returns belong to the portal.
    near_limit = float(np.percentile(patch[valid], 42)) + 0.16
    valid &= patch <= near_limit
    yy, xx = np.nonzero(valid)
    if len(xx) < 60:
        return None
    sample_depth = patch[yy, xx]
    px = xx + xa
    py = yy + ya
    camera = np.stack(
        ((px - intrinsics.cx) * sample_depth / intrinsics.fx, (py - intrinsics.cy) * sample_depth / intrinsics.fy, sample_depth),
        axis=1,
    )
    world = camera @ pose.rotation.T + pose.position
    center = np.median(world, axis=0)
    centered = world - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    residual = np.abs(centered @ normal)
    keep = residual <= max(0.035, float(np.percentile(residual, 75)))
    if int(keep.sum()) < 50:
        return None
    center = world[keep].mean(axis=0)
    _, _, vh = np.linalg.svd(world[keep] - center, full_matrices=False)
    normal = vh[-1]
    if float(normal @ (pose.position - center)) < 0:
        normal *= -1.0
    camera_x_world = pose.rotation[:, 0]
    axis_u = camera_x_world - normal * float(camera_x_world @ normal)
    if np.linalg.norm(axis_u) < 1e-6:
        return None
    axis_u /= np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    anchor_ray_camera = np.asarray([(cx - intrinsics.cx) / intrinsics.fx, (cy - intrinsics.cy) / intrinsics.fy, 1.0])
    anchor_ray_world = pose.rotation @ anchor_ray_camera
    denom = float(normal @ anchor_ray_world)
    if abs(denom) < 1e-6:
        return None
    t = float(normal @ (center - pose.position) / denom)
    anchor_point = pose.position + t * anchor_ray_world
    delta = anchor_point - center
    rmse = float(np.sqrt(np.mean(((world[keep] - center) @ normal) ** 2)))
    return SupportPlane(center, normal, axis_u, axis_v, (float(delta @ axis_u), float(delta @ axis_v)), rmse, int(keep.sum()))


def _view_field(depth_m: np.ndarray, pose, intrinsics, plane: SupportPlane) -> tuple[PortalViewField, list[dict]]:
    depth = np.asarray(depth_m, dtype=np.float64)
    h, w = depth.shape
    y, x = np.indices((h, w), dtype=np.float64)
    rays_camera = np.stack(((x - intrinsics.cx) / intrinsics.fx, (y - intrinsics.cy) / intrinsics.fy, np.ones_like(x)), axis=2)
    rays_world = rays_camera @ pose.rotation.T
    denom = rays_world @ plane.normal_world
    numer = float(plane.normal_world @ (plane.point_world_m - pose.position))
    plane_depth = np.divide(numer, denom, out=np.full_like(denom, np.nan), where=np.abs(denom) > 1e-6)
    mesh_valid = np.isfinite(depth) & (depth >= 0.35) & (depth <= 8.0)
    plane_valid = np.isfinite(plane_depth) & (plane_depth >= 0.35) & (plane_depth <= 8.0)
    delta = depth - plane_depth
    plane_hit = mesh_valid & plane_valid & (np.abs(delta) <= 0.12)
    behind = mesh_valid & plane_valid & (delta >= 0.22)
    unknown = ~mesh_valid | ~plane_valid
    plane_points = pose.position + plane_depth[..., None] * rays_world
    rel = plane_points - plane.point_world_m
    uu = rel @ plane.axis_u_world
    vv = rel @ plane.axis_v_world
    roi = (np.abs(uu - plane.anchor_uv_m[0]) <= 2.2) & (np.abs(vv - plane.anchor_uv_m[1]) <= 1.55)
    behind &= roi
    kernel = np.ones((5, 5), np.uint8)
    cleaned = cv2.morphologyEx(behind.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    components: list[dict] = []
    for label in range(1, count):
        mask = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < max(55, int(0.0015 * h * w)):
            continue
        u_values, v_values = uu[mask], vv[mask]
        u0, u1 = float(np.percentile(u_values, 5)), float(np.percentile(u_values, 95))
        v0, v1 = float(np.percentile(v_values, 5)), float(np.percentile(v_values, 95))
        width, height = u1 - u0, v1 - v0
        if not (0.38 <= width <= 2.0 and 0.42 <= height <= 3.0):
            continue
        anchor_distance = 0.0 if u0 <= plane.anchor_uv_m[0] <= u1 else min(abs(plane.anchor_uv_m[0] - u0), abs(plane.anchor_uv_m[0] - u1))
        components.append({"label": label, "mask": mask, "bounds": (u0, u1, v0, v1), "area": area, "anchor_distance_m": anchor_distance})
    # Meshes frequently fragment a real portal vertically.  The representation
    # is still the behind-plane field, but a plane-horizontal column run is a
    # more stable connected-set summary than requiring one pixel-connected blob.
    column_support = behind.mean(axis=0)
    supported_columns = (column_support >= 0.045).astype(np.uint8)[None, :]
    supported_columns = cv2.morphologyEx(supported_columns, cv2.MORPH_CLOSE, np.ones((1, 13), np.uint8))[0]
    padded = np.pad(supported_columns, (1, 1))
    transitions = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
    for start, end in zip(starts, ends):
        if end - start < 8:
            continue
        mask = behind.copy()
        mask[:, :start] = False
        mask[:, end:] = False
        if int(mask.sum()) < 45:
            continue
        u_values, v_values = uu[mask], vv[mask]
        u0, u1 = float(np.percentile(u_values, 4)), float(np.percentile(u_values, 96))
        v0, v1 = float(np.percentile(v_values, 4)), float(np.percentile(v_values, 96))
        width, height = u1 - u0, v1 - v0
        if not (0.30 <= width <= 2.25 and 0.28 <= height <= 3.1):
            continue
        anchor_distance = 0.0 if u0 <= plane.anchor_uv_m[0] <= u1 else min(abs(plane.anchor_uv_m[0] - u0), abs(plane.anchor_uv_m[0] - u1))
        if any(abs(u0 - row["bounds"][0]) < 0.08 and abs(u1 - row["bounds"][1]) < 0.08 for row in components):
            continue
        components.append({"label": -1, "mask": mask, "bounds": (u0, u1, v0, v1), "area": int(mask.sum()), "anchor_distance_m": anchor_distance})
    empty = np.zeros((h, w), dtype=np.uint8)
    field = PortalViewField(
        np.zeros((h, w), dtype=np.float32),
        plane_hit,
        behind,
        unknown,
        empty,
        None,
        {
            "plane_hit_fraction": float(np.mean(plane_hit)),
            "behind_valid_space_fraction": float(np.mean(behind)),
            "mesh_unknown_fraction": float(np.mean(unknown)),
            "component_candidate_count": len(components),
        },
    )
    return field, components


def infer_portal_interior(depths: tuple[np.ndarray, np.ndarray], poses: tuple, intrinsics, anchor_bbox: tuple[int, int, int, int]) -> PortalInteriorPrediction:
    plane = fit_anchor_support_plane(depths[0], poses[0], intrinsics, anchor_bbox)
    if plane is None:
        return PortalInteriorPrediction(tuple(), None, None, None, None, None, None, None, 0.0, {"failure": "SUPPORT_PLANE_MISSING"})
    base_fields, pools = zip(*[_view_field(depth, pose, intrinsics, plane) for depth, pose in zip(depths, poses)])
    matches = []
    for a in pools[0]:
        for b in pools[1]:
            a0, a1, *_ = a["bounds"]
            b0, b1, *_ = b["bounds"]
            overlap = max(0.0, min(a1, b1) - max(a0, b0))
            union = max(a1, b1) - min(a0, b0)
            iou = overlap / max(union, 1e-6)
            center_delta = abs((a0 + a1) * 0.5 - (b0 + b1) * 0.5)
            if iou < 0.16 or center_delta > 0.62:
                continue
            score = 2.0 * iou - 0.55 * (a["anchor_distance_m"] + b["anchor_distance_m"]) + 0.0002 * min(a["area"], b["area"])
            matches.append((score, iou, a, b))
    if not matches:
        diagnostics = {"failure": "CROSS_VIEW_PORTAL_COMPONENT_MISSING", "support_plane_fit_rmse_m": plane.fit_rmse_m, "support_plane_count": plane.support_count}
        return PortalInteriorPrediction(base_fields, plane, None, None, None, None, None, None, 0.0, diagnostics)
    _, iou, a, b = max(matches, key=lambda row: row[0])
    selected_fields = []
    for base, component in zip(base_fields, (a, b)):
        mask = component["mask"].astype(np.uint8)
        soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)
        if float(soft.max()) > 0:
            soft /= float(soft.max())
        selected_fields.append(PortalViewField(soft, base.plane_hit_mask, base.behind_free_mask, base.unknown_mask, mask, component["bounds"], base.diagnostics))
    left_u = 0.5 * (a["bounds"][0] + b["bounds"][0])
    right_u = 0.5 * (a["bounds"][1] + b["bounds"][1])
    center_u = 0.5 * (left_u + right_u)
    center_v = 0.5 * (a["bounds"][2] + a["bounds"][3] + b["bounds"][2] + b["bounds"][3]) * 0.5
    center_world = plane.point_world_m + center_u * plane.axis_u_world + center_v * plane.axis_v_world
    center_camera = poses[0].rotation.T @ (center_world - poses[0].position)
    width = float(right_u - left_u)
    range_m = float(center_camera[2])
    center_x = float(center_camera[0])
    bearing = float(np.arctan2(center_x, range_m))
    waypoint = center_world + 0.45 * plane.normal_world
    direction = plane.axis_v_world
    boundary_world = [plane.point_world_m + value * plane.axis_u_world for value in (left_u, right_u)]
    derived = tuple(tuple(_project_world_line(point, direction, pose, np.asarray([[intrinsics.fx, 0, intrinsics.cx], [0, intrinsics.fy, intrinsics.cy], [0, 0, 1]], dtype=np.float64)) for point in boundary_world) for pose in poses)
    confidence = float(np.clip(iou * np.exp(-plane.fit_rmse_m / 0.08), 0.0, 1.0))
    return PortalInteriorPrediction(
        tuple(selected_fields), plane, bearing, range_m, width, center_x,
        tuple(float(value) for value in waypoint), derived, confidence,
        {
            "cross_view_plane_interval_iou": float(iou),
            "support_plane_fit_rmse_m": plane.fit_rmse_m,
            "support_plane_count": plane.support_count,
            "selected_plane_u_bounds_m": [left_u, right_u],
            "primary_output": "PORTAL_INTERIOR_SOFT_MASK_AND_TARGET_FRONT_GEOMETRY",
            "legacy_boundaries": "DERIVED_FOR_EVALUATOR_ONLY",
        },
    )
