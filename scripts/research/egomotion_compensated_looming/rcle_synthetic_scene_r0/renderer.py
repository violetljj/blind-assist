"""Deterministic analytic renderer for the RCLE synthetic-scene protocol.

The renderer uses a right-handed camera frame (+X right, +Y up, +Z forward).
Image row coordinates increase downwards, so the corresponding camera ray has
``y = -(v - cy) / fy``.  Rays deliberately have camera Z equal to one; their
intersection parameter is therefore the authoritative camera +Z depth.

Only finite axis-aligned room planes and bounded frontoparallel panels are
rendered.  All geometry and transforms are evaluated in float64.  OpenCV is
kept single-threaded to avoid host-dependent worker scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


SURFACE_FLOOR = 1
SURFACE_CEILING = 2
SURFACE_LEFT_WALL = 3
SURFACE_RIGHT_WALL = 4
SURFACE_BACK_WALL = 5
SURFACE_PANEL_BASE = 100

_EPS = np.float64(1.0e-12)

# (z, x_min, x_max, y_min, y_max).  This table is part of the deterministic
# renderer definition: a panel_variant selects geometry, while the scene seed
# selects texture only.  All rectangles remain within the frozen room bounds.
_PANEL_VARIANTS: dict[int, tuple[tuple[float, float, float, float, float], ...]] = {
    0: (
        (5.5, -1.15, 1.15, 0.40, 2.60),
        (8.3, -2.15, -1.35, 0.25, 2.75),
        (9.1, 1.30, 2.15, 0.55, 2.45),
    ),
    1: (
        (4.8, -1.95, -0.35, 0.30, 2.25),
        (6.7, 0.20, 1.90, 0.75, 2.70),
        (9.4, -0.70, 0.75, 0.15, 1.45),
    ),
    2: (
        (5.2, -2.20, -1.25, 0.15, 2.85),
        (5.2, -0.55, 0.55, 0.15, 2.85),
        (5.2, 1.25, 2.20, 0.15, 2.85),
        (8.7, -1.55, -0.75, 0.35, 2.65),
        (8.7, 0.75, 1.55, 0.35, 2.65),
    ),
    3: (
        (4.6, 0.35, 2.10, 0.25, 2.35),
        (6.4, -2.10, -0.45, 0.65, 2.80),
        (8.2, -0.25, 1.45, 0.20, 1.70),
        (10.0, -1.65, -0.75, 1.15, 2.70),
    ),
}


@dataclass(frozen=True)
class RenderedFrame:
    """One rendered frame and its geometry truth."""

    rgb: np.ndarray
    depth_m: np.ndarray
    surface_id: np.ndarray
    t_world_camera: np.ndarray
    t_camera_world: np.ndarray


def camera_intrinsics(spec: Mapping[str, Any]) -> np.ndarray:
    """Return the frozen 3x3 pinhole intrinsic matrix as float64."""

    intrinsics = spec["rendering"]["intrinsics"]
    return np.array(
        [
            [float(intrinsics["fx_pixels"]), 0.0, float(intrinsics["cx_pixels"])],
            [0.0, -float(intrinsics["fy_pixels"]), float(intrinsics["cy_pixels"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _camera_pose(
    spec: Mapping[str, Any],
    motion: Mapping[str, Any],
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(frame_index, bool) or not isinstance(frame_index, (int, np.integer)):
        raise TypeError("frame_index must be an integer")
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")

    fps = float(spec["rendering"]["fps"])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("rendering.fps must be finite and positive")
    time_s = np.float64(frame_index) / np.float64(fps)

    initial = np.asarray(spec["scene_geometry"]["camera_initial_world_m"], dtype=np.float64)
    velocity = np.asarray(motion["translation_world_m_per_s"], dtype=np.float64)
    if initial.shape != (3,) or velocity.shape != (3,):
        raise ValueError("camera position and translation velocity must have three elements")
    position = initial + velocity * time_s

    yaw = np.deg2rad(np.float64(motion["yaw_deg_per_s"]) * time_s)
    cosine = np.cos(yaw)
    sine = np.sin(yaw)
    rotation_world_camera = np.array(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=np.float64,
    )

    t_world_camera = np.eye(4, dtype=np.float64)
    t_world_camera[:3, :3] = rotation_world_camera
    t_world_camera[:3, 3] = position

    t_camera_world = np.eye(4, dtype=np.float64)
    t_camera_world[:3, :3] = rotation_world_camera.T
    t_camera_world[:3, 3] = -(rotation_world_camera.T @ position)
    return t_world_camera, t_camera_world


def _camera_rays(spec: Mapping[str, Any]) -> np.ndarray:
    rendering = spec["rendering"]
    width = int(rendering["width"])
    height = int(rendering["height"])
    if width <= 0 or height <= 0:
        raise ValueError("rendering width and height must be positive")

    intrinsic = camera_intrinsics(spec)
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    x = (columns - intrinsic[0, 2]) / intrinsic[0, 0]
    y = (rows - intrinsic[1, 2]) / intrinsic[1, 1]
    return np.stack((x, y, np.ones_like(x)), axis=-1)


def _consider_plane(
    best_depth: np.ndarray,
    best_surface: np.ndarray,
    origin: np.ndarray,
    rays_world: np.ndarray,
    *,
    axis: int,
    coordinate: float,
    bounds: Sequence[tuple[int, float, float]],
    surface: int,
) -> None:
    denominator = rays_world[..., axis]
    with np.errstate(divide="ignore", invalid="ignore"):
        candidate = (np.float64(coordinate) - origin[axis]) / denominator
    valid = np.isfinite(candidate) & (candidate > _EPS)

    for bounded_axis, lower, upper in bounds:
        hit_coordinate = origin[bounded_axis] + candidate * rays_world[..., bounded_axis]
        valid &= hit_coordinate >= np.float64(lower) - _EPS
        valid &= hit_coordinate <= np.float64(upper) + _EPS

    closer = valid & (candidate < best_depth)
    best_depth[closer] = candidate[closer]
    best_surface[closer] = np.int16(surface)


def _texture(
    points_world: np.ndarray,
    surface_id: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Generate a seeded coordinate texture without mutable RNG state."""

    height, width = surface_id.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    palettes = np.array(
        [
            [92, 106, 121],
            [183, 188, 194],
            [116, 137, 159],
            [139, 118, 101],
            [107, 144, 119],
        ],
        dtype=np.int64,
    )

    for surface in np.unique(surface_id):
        mask = surface_id == surface
        if not np.any(mask):
            continue
        points = points_world[mask]
        surface_value = int(surface)
        if surface_value in (SURFACE_FLOOR, SURFACE_CEILING):
            u, v = points[:, 0], points[:, 2]
        elif surface_value in (SURFACE_LEFT_WALL, SURFACE_RIGHT_WALL):
            u, v = points[:, 2], points[:, 1]
        else:
            u, v = points[:, 0], points[:, 1]

        frequency = 4.0 + float((seed + surface_value) % 3)
        lattice_u = np.floor((u + 20.0) * frequency).astype(np.int64)
        lattice_v = np.floor((v + 20.0) * frequency).astype(np.int64)
        hashed = (
            lattice_u * np.int64(73856093)
            ^ lattice_v * np.int64(19349663)
            ^ np.int64(seed) * np.int64(83492791)
            ^ np.int64(surface_value) * np.int64(2654435761)
        )
        modulation = (hashed & np.int64(63)) - 31
        checker = ((lattice_u + lattice_v) & 1) * 18 - 9

        palette_index = (
            surface_value - SURFACE_PANEL_BASE
            if surface_value >= SURFACE_PANEL_BASE
            else surface_value - 1
        ) % len(palettes)
        base = palettes[palette_index]
        colors = np.clip(
            base[None, :]
            + modulation[:, None]
            + checker[:, None]
            + np.array([0, (surface_value * 7) % 19, (seed + surface_value) % 23]),
            0,
            255,
        )
        rgb[mask] = colors.astype(np.uint8)
    return rgb


def render_frame(
    spec: Mapping[str, Any],
    scene: Mapping[str, Any],
    motion: Mapping[str, Any],
    frame_index: int,
) -> RenderedFrame:
    """Render one deterministic RGB/depth/surface/pose frame."""

    cv2.setNumThreads(int(spec["rendering"].get("opencv_threads", 1)))
    t_world_camera, t_camera_world = _camera_pose(spec, motion, frame_index)
    rays_camera = _camera_rays(spec)
    rotation_world_camera = t_world_camera[:3, :3]
    rays_world = rays_camera @ rotation_world_camera.T
    origin = t_world_camera[:3, 3]
    bounds = spec["scene_geometry"]["room_bounds_m"]
    for axis, name in enumerate(("x", "y", "z")):
        lower, upper = map(float, bounds[name])
        clearance = float(
            spec["scene_geometry"]["minimum_camera_surface_clearance_m"]
        )
        if not lower + clearance <= origin[axis] <= upper - clearance:
            raise ValueError(f"CAMERA_OUTSIDE_CLEARANCE:{name}")

    height, width = rays_camera.shape[:2]
    depth = np.full((height, width), np.inf, dtype=np.float64)
    surfaces = np.zeros((height, width), dtype=np.int16)

    x_min, x_max = map(float, bounds["x"])
    y_min, y_max = map(float, bounds["y"])
    z_min, z_max = map(float, bounds["z"])

    _consider_plane(
        depth,
        surfaces,
        origin,
        rays_world,
        axis=1,
        coordinate=y_min,
        bounds=((0, x_min, x_max), (2, z_min, z_max)),
        surface=SURFACE_FLOOR,
    )
    _consider_plane(
        depth,
        surfaces,
        origin,
        rays_world,
        axis=1,
        coordinate=y_max,
        bounds=((0, x_min, x_max), (2, z_min, z_max)),
        surface=SURFACE_CEILING,
    )
    _consider_plane(
        depth,
        surfaces,
        origin,
        rays_world,
        axis=0,
        coordinate=x_min,
        bounds=((1, y_min, y_max), (2, z_min, z_max)),
        surface=SURFACE_LEFT_WALL,
    )
    _consider_plane(
        depth,
        surfaces,
        origin,
        rays_world,
        axis=0,
        coordinate=x_max,
        bounds=((1, y_min, y_max), (2, z_min, z_max)),
        surface=SURFACE_RIGHT_WALL,
    )
    _consider_plane(
        depth,
        surfaces,
        origin,
        rays_world,
        axis=2,
        coordinate=z_max,
        bounds=((0, x_min, x_max), (1, y_min, y_max)),
        surface=SURFACE_BACK_WALL,
    )

    variant = int(scene["panel_variant"])
    if variant not in _PANEL_VARIANTS:
        raise ValueError(f"unsupported panel_variant: {variant}")
    for index, (z, panel_x_min, panel_x_max, panel_y_min, panel_y_max) in enumerate(
        _PANEL_VARIANTS[variant]
    ):
        _consider_plane(
            depth,
            surfaces,
            origin,
            rays_world,
            axis=2,
            coordinate=z,
            bounds=((0, panel_x_min, panel_x_max), (1, panel_y_min, panel_y_max)),
            surface=SURFACE_PANEL_BASE + index,
        )

    if np.any(~np.isfinite(depth)) or np.any(surfaces == 0):
        raise RuntimeError("scene geometry does not enclose every forward camera ray")
    near = float(spec["rendering"]["near_plane_m"])
    far = float(spec["rendering"]["far_plane_m"])
    if np.any(depth < near) or np.any(depth > far):
        raise RuntimeError("rendered depth lies outside frozen near/far planes")

    points_world = origin[None, None, :] + depth[..., None] * rays_world
    rgb = _texture(points_world, surfaces, int(scene["seed"]))
    return RenderedFrame(
        rgb=rgb,
        depth_m=depth,
        surface_id=surfaces,
        t_world_camera=t_world_camera,
        t_camera_world=t_camera_world,
    )
