"""Mesh-only privileged aperture-boundary teacher for SAGE-LM V1-E0.

RGB is deliberately absent from the teacher path.  An ARKitScenes scene mesh,
official camera pose and camera intrinsics produce metric depth, surface
normals, a signed depth-jump field, label-valid mask and LEFT/RIGHT heatmaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .task_boundary_field_experiment import TOP_K_PER_ROLE
from .two_view_observation import (
    ImageLine,
    _image_line_from_points,
    _project_world_line,
)

MIN_DEPTH_M = 0.35
MAX_DEPTH_M = 8.0
MIN_JUMP_M = 0.16
JUMP_GAP_PX = 2


@dataclass(frozen=True)
class PrivilegedGeometryFrame:
    depth_m: np.ndarray
    normals_camera: np.ndarray
    signed_depth_jump_m: np.ndarray
    valid_mask: np.ndarray
    boundary_heatmap: np.ndarray
    lines: tuple[tuple[ImageLine, ...], tuple[ImageLine, ...]]
    diagnostics: dict


def geometry_fields(depth_m: np.ndarray, normals_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Create LEFT/RIGHT geometry labels without consulting RGB or evaluator truth."""

    depth = np.asarray(depth_m, dtype=np.float32)
    normals = np.asarray(normals_camera, dtype=np.float32)
    if depth.ndim != 2 or normals.shape != (*depth.shape, 3):
        raise ValueError("depth/normals shape mismatch")
    h, w = depth.shape
    finite = np.isfinite(depth) & (depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M)
    normal_valid = np.all(np.isfinite(normals), axis=2) & (np.linalg.norm(normals, axis=2) > 0.5)
    # A small median filter suppresses isolated mesh triangles but preserves the
    # metric discontinuity that defines an aperture boundary.
    filtered = np.where(finite, depth, 0.0).astype(np.float32)
    filtered = cv2.medianBlur(filtered, 5)
    left_depth = np.zeros_like(filtered)
    right_depth = np.zeros_like(filtered)
    left_depth[:, JUMP_GAP_PX : w - JUMP_GAP_PX] = filtered[:, : w - 2 * JUMP_GAP_PX]
    right_depth[:, JUMP_GAP_PX : w - JUMP_GAP_PX] = filtered[:, 2 * JUMP_GAP_PX :]
    paired_valid = np.zeros_like(finite)
    paired_valid[:, JUMP_GAP_PX : w - JUMP_GAP_PX] = (
        finite[:, : w - 2 * JUMP_GAP_PX]
        & finite[:, 2 * JUMP_GAP_PX :]
        & normal_valid[:, : w - 2 * JUMP_GAP_PX]
        & normal_valid[:, 2 * JUMP_GAP_PX :]
    )
    jump = np.where(paired_valid, right_depth - left_depth, 0.0)
    magnitude = np.abs(jump)
    # Require both an absolute metric jump and a relative separation.  This
    # rejects shallow mesh tessellation while retaining weak/textureless edges.
    near = np.minimum(left_depth, right_depth)
    edge_valid = paired_valid & (magnitude >= MIN_JUMP_M) & (magnitude >= 0.10 * np.maximum(near, 0.5))
    signed = np.where(edge_valid, jump, 0.0).astype(np.float32)
    raw = np.stack((np.maximum(signed, 0.0), np.maximum(-signed, 0.0)))
    # Vertical support is the dominant-front-surface test: an isolated depth
    # edge is not an aperture boundary, while a coherent near-to-far transition
    # survives accumulation along the image column.
    heatmap = np.empty_like(raw)
    for role in range(2):
        coherence = cv2.boxFilter((raw[role] > 0).astype(np.float32), -1, (3, 21), normalize=True)
        heatmap[role] = raw[role] * np.clip(coherence / 0.28, 0.0, 1.0)
        scale = float(np.percentile(heatmap[role][heatmap[role] > 0], 95)) if np.any(heatmap[role] > 0) else 0.0
        if scale > 0:
            heatmap[role] = np.clip(heatmap[role] / scale, 0.0, 1.0)
    border = max(5, round(h * 0.04))
    heatmap[:, :border] = 0
    heatmap[:, h - border :] = 0
    diagnostics = {
        "metric_depth_valid_fraction": float(np.mean(finite)),
        "label_valid_fraction": float(np.mean(paired_valid)),
        "depth_jump_positive_fraction": float(np.mean(edge_valid)),
        "left_heatmap_nonzero_fraction": float(np.mean(heatmap[0] > 0)),
        "right_heatmap_nonzero_fraction": float(np.mean(heatmap[1] > 0)),
        "maximum_signed_depth_jump_m": float(np.max(np.abs(signed))),
    }
    return signed, paired_valid, heatmap, diagnostics


def _select_peaks(profile: np.ndarray, count: int = TOP_K_PER_ROLE, radius: int = 7) -> list[int]:
    work = np.asarray(profile, dtype=np.float32).copy()
    peaks: list[int] = []
    for _ in range(count):
        peak = int(np.argmax(work))
        if float(work[peak]) <= 0:
            break
        peaks.append(peak)
        work[max(0, peak - radius) : min(len(work), peak + radius + 1)] = 0
    return peaks


def heatmap_line_candidates(heatmap: np.ndarray) -> tuple[ImageLine, ...]:
    """Convert one geometry heatmap into the frozen top-8 line interface."""

    h, w = heatmap.shape
    profile = 0.75 * heatmap.max(axis=0) + 0.25 * heatmap.mean(axis=0)
    lines: list[ImageLine] = []
    for peak in _select_peaks(profile):
        points: list[tuple[float, float]] = []
        weights: list[float] = []
        for y in range(h):
            xa, xb = max(0, peak - 5), min(w, peak + 6)
            local = heatmap[y, xa:xb]
            if local.size == 0 or float(local.max()) < 0.04:
                continue
            x = xa + int(np.argmax(local))
            points.append((float(x), float(y)))
            weights.append(float(local.max()))
        if len(points) >= max(8, int(0.12 * h)):
            line = _image_line_from_points(points, sum(weights), len(points))
        else:
            line = _image_line_from_points([(float(peak), 0.0), (float(peak), float(h - 1))], float(profile[peak]), 1)
        lines.append(ImageLine(line.coefficients, float(profile[peak] * h), line.segment_count))
    return tuple(lines)


class MeshDepthRenderer:
    """CPU raycaster with one cached Open3D scene per ARKitScenes mesh."""

    def __init__(self) -> None:
        try:
            import open3d as o3d
        except ImportError as error:  # pragma: no cover - runtime dependency check
            raise RuntimeError("V1-E0 mesh rendering requires open3d") from error
        self.o3d = o3d
        self.cache: dict[Path, object] = {}

    def _scene(self, mesh_path: Path):
        mesh_path = mesh_path.resolve()
        if mesh_path not in self.cache:
            legacy = self.o3d.io.read_triangle_mesh(str(mesh_path))
            if legacy.is_empty() or not legacy.has_triangles():
                raise ValueError(f"empty ARKitScenes mesh: {mesh_path}")
            scene = self.o3d.t.geometry.RaycastingScene()
            scene.add_triangles(self.o3d.t.geometry.TriangleMesh.from_legacy(legacy))
            self.cache[mesh_path] = scene
        return self.cache[mesh_path]

    def render(self, mesh_path: Path, pose, intrinsics) -> tuple[np.ndarray, np.ndarray]:
        scene = self._scene(mesh_path)
        intrinsic = np.asarray(
            [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        camera_to_world = np.eye(4, dtype=np.float64)
        camera_to_world[:3, :3] = pose.rotation
        camera_to_world[:3, 3] = pose.position
        world_to_camera = np.linalg.inv(camera_to_world)
        rays = self.o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic, world_to_camera, intrinsics.width, intrinsics.height
        )
        result = scene.cast_rays(rays)
        depth = result["t_hit"].numpy().astype(np.float32)
        normals_world = result["primitive_normals"].numpy().astype(np.float32)
        normals_camera = normals_world @ pose.rotation
        return depth, normals_camera

    def teacher_frame(self, mesh_path: Path, pose, intrinsics) -> PrivilegedGeometryFrame:
        depth, normals = self.render(mesh_path, pose, intrinsics)
        signed, valid, heatmap, diagnostics = geometry_fields(depth, normals)
        lines = (heatmap_line_candidates(heatmap[0]), heatmap_line_candidates(heatmap[1]))
        diagnostics = {
            **diagnostics,
            "top_k_per_role": TOP_K_PER_ROLE,
            "role_candidate_counts": [len(lines[0]), len(lines[1])],
        }
        return PrivilegedGeometryFrame(depth, normals, signed, valid, heatmap, lines, diagnostics)


def link_boundary_lines(
    frame: PrivilegedGeometryFrame,
    pose_a,
    pose_b,
    intrinsics,
) -> tuple[tuple[ImageLine, ...], tuple[ImageLine, ...], dict]:
    """Lift first-view depth boundaries to 3D and project them into view B."""

    intrinsic = np.asarray(
        [[intrinsics.fx, 0.0, intrinsics.cx], [0.0, intrinsics.fy, intrinsics.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    projected: list[list[ImageLine]] = [[], []]
    ranges: list[list[float]] = [[], []]
    h, w = frame.depth_m.shape
    for role, pool in enumerate(frame.lines):
        for line in pool:
            samples = []
            for y in np.linspace(0.12 * h, 0.88 * h, 25):
                x = round(line.x_at(float(y)))
                # The frozen geometry defines range on the deeper aperture
                # interior, not on the foreground wall.  LEFT opens to +x and
                # RIGHT opens to -x by construction of the signed jump roles.
                if role == 0:
                    values = frame.depth_m[round(y), min(w, x + 3) : min(w, x + 13)]
                else:
                    values = frame.depth_m[round(y), max(0, x - 12) : max(0, x - 2)]
                values = values[np.isfinite(values) & (values >= MIN_DEPTH_M) & (values <= MAX_DEPTH_M)]
                if values.size:
                    samples.append(float(np.median(values)))
            if len(samples) < 5:
                continue
            range_m = float(np.percentile(samples, 60))
            x_mid = line.x_at(intrinsics.cy)
            point_camera = np.asarray(
                [(x_mid - intrinsics.cx) * range_m / intrinsics.fx, 0.0, range_m], dtype=np.float64
            )
            point_world = pose_a.rotation @ point_camera + pose_a.position
            direction_world = pose_a.rotation @ np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
            projected[role].append(_project_world_line(point_world, direction_world, pose_b, intrinsic))
            ranges[role].append(range_m)
    return tuple(projected[0]), tuple(projected[1]), {
        "linked_boundary_range_m": ranges,
        "linked_role_candidate_counts": [len(projected[0]), len(projected[1])],
    }
