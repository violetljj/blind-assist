#!/usr/bin/env python3
"""Natural-3D GRAIL M0 teacher transfer on ARKitScenes meshes.

The source supplies metric meshes and oriented semantic instance boxes, but no
functional-front or navmesh truth.  This module therefore names its outputs as
derived proxies and returns AMBIGUOUS rather than guessing when the two object
faces have indistinguishable interaction-pose support.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage


INTERACTION_LABELS = {
    "cabinet", "dishwasher", "oven", "refrigerator", "shelf", "sink", "stove"
}


@dataclass(frozen=True)
class Object3D:
    uid: str
    label: str
    center: tuple[float, float, float]
    lengths: tuple[float, float, float]
    rotation: tuple[float, ...]


@dataclass(frozen=True)
class NaturalPose:
    x: float
    y: float
    yaw_rad: float
    face: int


@dataclass
class SceneGrid:
    scene_id: str
    floor_z: float
    step_m: float
    origin_x: float
    origin_y: float
    free: np.ndarray
    occupied: np.ndarray
    component: np.ndarray
    largest_component: int

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return round((y - self.origin_y) / self.step_m), round((x - self.origin_x) / self.step_m)

    def point(self, row: int, col: int) -> tuple[float, float]:
        return self.origin_x + col * self.step_m, self.origin_y + row * self.step_m

    def inside(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.free.shape[0] and 0 <= col < self.free.shape[1]

    def reachable(self, x: float, y: float) -> bool:
        cell = self.cell(x, y)
        return (
            self.largest_component != 0
            and self.inside(cell)
            and self.free[cell]
            and self.component[cell] == self.largest_component
        )


def load_binary_ply_vertices(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        vertex_count = None
        properties: list[str] = []
        in_vertex = False
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"PLY header is incomplete: {path}")
            line = raw.decode("ascii").strip()
            if line == "format binary_little_endian 1.0":
                continue
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertex = True
                continue
            if line.startswith("element ") and not line.startswith("element vertex "):
                in_vertex = False
            if in_vertex and line.startswith("property "):
                properties.append(line)
            if line == "end_header":
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {path}")
        expected = [
            "property float x", "property float y", "property float z",
            "property uchar red", "property uchar green", "property uchar blue", "property uchar alpha",
        ]
        if properties != expected:
            raise ValueError(f"unsupported PLY vertex schema: {properties}")
        dtype = np.dtype([
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgba", "u1", (4,)),
        ])
        vertices = np.fromfile(handle, dtype=dtype, count=vertex_count)
    return np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(np.float64)


def load_objects(path: Path) -> tuple[Object3D, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    objects: list[Object3D] = []
    for row in payload.get("data", []):
        if row.get("label") not in INTERACTION_LABELS:
            continue
        aligned = row["segments"]["obbAligned"]
        objects.append(Object3D(
            uid=str(row["uid"]),
            label=str(row["label"]),
            center=tuple(float(value) for value in aligned["centroid"]),
            lengths=tuple(float(value) for value in aligned["axesLengths"]),
            rotation=tuple(float(value) for value in aligned["normalizedAxes"]),
        ))
    return tuple(objects)


def estimate_floor_z(vertices: np.ndarray) -> float:
    z = vertices[:, 2]
    lower = z[z <= np.quantile(z, 0.35)]
    low, high = float(np.quantile(lower, 0.01)), float(np.quantile(lower, 0.99))
    bins = max(10, int(math.ceil((high - low) / 0.025)))
    counts, edges = np.histogram(lower, bins=bins, range=(low, high))
    index = int(np.argmax(counts))
    selected = lower[(lower >= edges[index]) & (lower <= edges[index + 1])]
    return float(np.median(selected))


def build_scene_grid(scene_id: str, vertices: np.ndarray, step_m: float = 0.12) -> SceneGrid:
    floor_z = estimate_floor_z(vertices)
    min_x, min_y = np.quantile(vertices[:, :2], 0.005, axis=0) - 0.2
    max_x, max_y = np.quantile(vertices[:, :2], 0.995, axis=0) + 0.2
    cols = int(math.ceil((max_x - min_x) / step_m)) + 1
    rows = int(math.ceil((max_y - min_y) / step_m)) + 1
    if rows * cols > 2_000_000:
        raise ValueError(f"implausible grid size for {scene_id}: {rows}x{cols}")

    def raster(points: np.ndarray) -> np.ndarray:
        mask = np.zeros((rows, cols), dtype=bool)
        rr = np.rint((points[:, 1] - min_y) / step_m).astype(int)
        cc = np.rint((points[:, 0] - min_x) / step_m).astype(int)
        valid = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        mask[rr[valid], cc[valid]] = True
        return mask

    ground_points = vertices[np.abs(vertices[:, 2] - floor_z) <= 0.10]
    obstacle_points = vertices[
        (vertices[:, 2] >= floor_z + 0.16) & (vertices[:, 2] <= floor_z + 1.75)
    ]
    supported = raster(ground_points)
    support_radius = max(2, int(math.ceil(0.45 / step_m)))
    supported = ndimage.binary_closing(
        ndimage.binary_dilation(supported, iterations=support_radius), iterations=2
    )
    occupied = raster(obstacle_points)
    clearance_cells = max(1, int(math.ceil(0.28 / step_m)))
    obstacles = ndimage.binary_dilation(occupied, iterations=clearance_cells)
    free = supported & ~obstacles
    component, count = ndimage.label(free)
    if count == 0:
        largest = 0
    else:
        sizes = np.bincount(component.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
    return SceneGrid(
        scene_id, floor_z, step_m, float(min_x), float(min_y), free, occupied, component, largest
    )


def _horizontal_axes(obj: Object3D) -> tuple[np.ndarray, np.ndarray, float, float]:
    rotation = np.asarray(obj.rotation, dtype=np.float64).reshape(3, 3)
    axes = []
    for index in (0, 1):
        vector = rotation[index, :2]
        norm = np.linalg.norm(vector)
        if norm > 1e-6:
            axes.append((vector / norm, obj.lengths[index]))
    if len(axes) != 2:
        raise ValueError(f"object {obj.uid} has invalid horizontal OBB axes")
    axes.sort(key=lambda item: item[1])
    depth_axis, depth_length = axes[0]
    tangent_axis, width_length = axes[1]
    return depth_axis, tangent_axis, depth_length, width_length


def _line_clear(grid: SceneGrid, start: tuple[float, float], end: tuple[float, float]) -> bool:
    distance = math.dist(start, end)
    steps = max(2, int(math.ceil(distance / (grid.step_m * 0.5))))
    # Stop before the target surface; the target itself is expected occupancy.
    stop_distance_m = 0.42
    for index in range(1, steps - 1):
        ratio = index / steps
        if distance * (1.0 - ratio) <= stop_distance_m:
            break
        x = start[0] + (end[0] - start[0]) * ratio
        y = start[1] + (end[1] - start[1]) * ratio
        cell = grid.cell(x, y)
        if not grid.inside(cell) or grid.occupied[cell]:
            return False
    return True


def sample_face_poses(grid: SceneGrid, obj: Object3D, face: int) -> tuple[NaturalPose, ...]:
    depth_axis, tangent_axis, depth_length, width_length = _horizontal_axes(obj)
    normal = depth_axis * float(face)
    poses: list[NaturalPose] = []
    for clearance in (0.65, 0.80, 0.95, 1.10):
        distance = depth_length / 2 + clearance
        maximum_lateral = min(0.55, width_length / 2 + 0.12)
        for lateral in np.linspace(-maximum_lateral, maximum_lateral, 7):
            xy = np.asarray(obj.center[:2]) + normal * distance + tangent_axis * float(lateral)
            x, y = float(xy[0]), float(xy[1])
            if not grid.reachable(x, y):
                continue
            if not _line_clear(grid, (x, y), obj.center[:2]):
                continue
            poses.append(NaturalPose(x, y, math.atan2(obj.center[1] - y, obj.center[0] - x), face))
    return tuple(poses)


@dataclass(frozen=True)
class TeacherOutput:
    state: str
    poses: tuple[NaturalPose, ...]
    chosen_face: int | None
    face_counts: tuple[int, int]


def interaction_pose_teacher(grid: SceneGrid, obj: Object3D) -> TeacherOutput:
    negative = sample_face_poses(grid, obj, -1)
    positive = sample_face_poses(grid, obj, 1)
    counts = len(negative), len(positive)
    best = max(counts)
    if best < 3:
        return TeacherOutput("NONE", (), None, counts)
    gap = abs(counts[0] - counts[1])
    if gap < max(2, int(math.ceil(best * 0.25))):
        return TeacherOutput("AMBIGUOUS", (), None, counts)
    face = -1 if counts[0] > counts[1] else 1
    return TeacherOutput("VALID_SET", negative if face == -1 else positive, face, counts)


def pose_matches(candidate: NaturalPose | None, truth: tuple[NaturalPose, ...]) -> bool:
    if candidate is None:
        return False
    for pose in truth:
        yaw_error = abs((candidate.yaw_rad - pose.yaw_rad + math.pi) % (2 * math.pi) - math.pi)
        if math.hypot(candidate.x - pose.x, candidate.y - pose.y) <= 0.50 and yaw_error <= math.radians(20):
            return True
    return False


def shortest_path(grid: SceneGrid, goal: tuple[float, float]) -> list[tuple[float, float]] | None:
    goal_cell = grid.cell(*goal)
    if not grid.inside(goal_cell) or grid.component[goal_cell] != grid.largest_component:
        return None
    cells = np.argwhere(grid.component == grid.largest_component)
    if len(cells) == 0:
        return None
    distances = (cells[:, 0] - goal_cell[0]) ** 2 + (cells[:, 1] - goal_cell[1]) ** 2
    start = tuple(int(value) for value in cells[int(np.argmax(distances))])
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal_cell:
            path = []
            cursor: tuple[int, int] | None = cell
            while cursor is not None:
                path.append(grid.point(*cursor))
                cursor = parent[cursor]
            return list(reversed(path))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = cell[0] + dr, cell[1] + dc
            if nxt in parent or not grid.inside(nxt):
                continue
            if grid.component[nxt] != grid.largest_component:
                continue
            parent[nxt] = cell
            queue.append(nxt)
    return None
