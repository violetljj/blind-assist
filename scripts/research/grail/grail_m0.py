#!/usr/bin/env python3
"""GRAIL M0: analytic set-valued interaction-pose teacher and oracle.

This is a metric 2.5D geometry canary.  It tests whether the new task admits
stable, reachable, target-visible terminal poses before any student is trained.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import random
from typing import Iterable


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def expanded(self, margin: float) -> "Rect":
        return Rect(self.x0 - margin, self.y0 - margin, self.x1 + margin, self.y1 + margin)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class Target:
    instance_id: str
    category: str
    cx: float
    cy: float
    width_m: float
    depth_m: float
    front_x: float
    front_y: float

    @property
    def footprint(self) -> Rect:
        return Rect(
            self.cx - self.width_m / 2,
            self.cy - self.depth_m / 2,
            self.cx + self.width_m / 2,
            self.cy + self.depth_m / 2,
        )


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw_rad: float


@dataclass(frozen=True)
class Scene:
    scene_id: str
    split: str
    width_m: float
    height_m: float
    start_x: float
    start_y: float
    target: Target
    distractor: Target
    obstacles: tuple[Rect, ...]
    expected_pose_state: str
    no_pose_reason: str | None


@dataclass(frozen=True)
class PoseJudgement:
    valid: bool
    reasons: tuple[str, ...]


def angle_delta(left: float, right: float) -> float:
    return abs((left - right + math.pi) % (2 * math.pi) - math.pi)


def point_free(scene: Scene, x: float, y: float, clearance_m: float = 0.28) -> bool:
    if not (clearance_m <= x <= scene.width_m - clearance_m):
        return False
    if not (clearance_m <= y <= scene.height_m - clearance_m):
        return False
    occupied = scene.obstacles + (scene.target.footprint, scene.distractor.footprint)
    return not any(rect.expanded(clearance_m).contains(x, y) for rect in occupied)


def line_clear(scene: Scene, start: tuple[float, float], end: tuple[float, float]) -> bool:
    """Visibility excluding the intended target footprint at the terminal ray."""
    distance = math.dist(start, end)
    steps = max(2, int(math.ceil(distance / 0.05)))
    for index in range(1, steps):
        ratio = index / steps
        x = start[0] + (end[0] - start[0]) * ratio
        y = start[1] + (end[1] - start[1]) * ratio
        if any(rect.contains(x, y) for rect in scene.obstacles + (scene.distractor.footprint,)):
            return False
    return True


def _grid_point(scene: Scene, cell: tuple[int, int], step_m: float) -> tuple[float, float]:
    return cell[0] * step_m, cell[1] * step_m


def shortest_path(
    scene: Scene, goal: tuple[float, float], step_m: float = 0.20
) -> list[tuple[float, float]] | None:
    start = (round(scene.start_x / step_m), round(scene.start_y / step_m))
    finish = (round(goal[0] / step_m), round(goal[1] / step_m))
    queue = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    limit_x, limit_y = round(scene.width_m / step_m), round(scene.height_m / step_m)
    while queue:
        cell = queue.popleft()
        if cell == finish:
            path: list[tuple[float, float]] = []
            cursor: tuple[int, int] | None = cell
            while cursor is not None:
                path.append(_grid_point(scene, cursor, step_m))
                cursor = parent[cursor]
            return list(reversed(path))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = cell[0] + dx, cell[1] + dy
            if nxt in parent or not (0 <= nxt[0] <= limit_x and 0 <= nxt[1] <= limit_y):
                continue
            if not point_free(scene, *_grid_point(scene, nxt, step_m)):
                continue
            parent[nxt] = cell
            queue.append(nxt)
    return None


def judge_pose(scene: Scene, target: Target, pose: Pose) -> PoseJudgement:
    reasons: list[str] = []
    if not point_free(scene, pose.x, pose.y):
        reasons.append("COLLISION_OR_OUTSIDE")
    offset_x, offset_y = pose.x - target.cx, pose.y - target.cy
    forward = offset_x * target.front_x + offset_y * target.front_y
    tangent_x, tangent_y = -target.front_y, target.front_x
    lateral = abs(offset_x * tangent_x + offset_y * tangent_y)
    distance = math.hypot(offset_x, offset_y)
    if forward < 0.62 or forward > 1.18 or lateral > target.width_m * 0.55 + 0.18:
        reasons.append("NOT_FUNCTIONAL_SIDE")
    if not (0.62 <= distance <= 1.30):
        reasons.append("BAD_INTERACTION_DISTANCE")
    desired_yaw = math.atan2(target.cy - pose.y, target.cx - pose.x)
    if angle_delta(pose.yaw_rad, desired_yaw) > math.radians(20):
        reasons.append("BAD_ORIENTATION")
    if not line_clear(scene, (pose.x, pose.y), (target.cx, target.cy)):
        reasons.append("TARGET_NOT_VISIBLE")
    if shortest_path(scene, (pose.x, pose.y)) is None:
        reasons.append("UNREACHABLE")
    return PoseJudgement(not reasons, tuple(reasons))


def interaction_pose_set(scene: Scene, target: Target | None = None) -> tuple[Pose, ...]:
    target = target or scene.target
    tangent_x, tangent_y = -target.front_y, target.front_x
    poses: list[Pose] = []
    for distance in (0.70, 0.85, 1.00, 1.15):
        for lateral in (-0.45, -0.30, -0.15, 0.0, 0.15, 0.30, 0.45):
            x = target.cx + target.front_x * distance + tangent_x * lateral
            y = target.cy + target.front_y * distance + tangent_y * lateral
            yaw = math.atan2(target.cy - y, target.cx - x)
            pose = Pose(x, y, yaw)
            if judge_pose(scene, target, pose).valid:
                poses.append(pose)
    return tuple(poses)


def pose_matches(candidate: Pose | None, truth: Iterable[Pose]) -> bool:
    if candidate is None:
        return False
    return any(
        math.hypot(candidate.x - pose.x, candidate.y - pose.y) <= 0.50
        and angle_delta(candidate.yaw_rad, pose.yaw_rad) <= math.radians(20)
        for pose in truth
    )


def choose_oracle(scene: Scene, truth: tuple[Pose, ...]) -> Pose | None:
    paths = [(len(path), pose) for pose in truth if (path := shortest_path(scene, (pose.x, pose.y)))]
    return min(paths, key=lambda item: item[0])[1] if paths else None


def choose_bbox_fixed_distance(scene: Scene) -> Pose | None:
    target = scene.target
    dx, dy = scene.start_x - target.cx, scene.start_y - target.cy
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    x, y = target.cx + 0.9 * dx / norm, target.cy + 0.9 * dy / norm
    return Pose(x, y, math.atan2(target.cy - y, target.cx - x))


def choose_nearest_free(scene: Scene) -> Pose | None:
    target = scene.target
    candidates: list[tuple[int, Pose]] = []
    for radius in (0.70, 0.85, 1.00, 1.15):
        for index in range(24):
            angle = 2 * math.pi * index / 24
            pose = Pose(
                target.cx + radius * math.cos(angle),
                target.cy + radius * math.sin(angle),
                angle + math.pi,
            )
            path = shortest_path(scene, (pose.x, pose.y))
            if path and line_clear(scene, (pose.x, pose.y), (target.cx, target.cy)):
                candidates.append((len(path), pose))
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def _target_rect(target: Target, margin: float = 0.0) -> Rect:
    return target.footprint.expanded(margin)


def make_scene(split: str, index: int) -> Scene:
    seed_base = 12000 if split == "DEVELOPMENT" else 73000
    rng = random.Random(seed_base + index)
    scene_id = f"{split.lower()}-building-{index:03d}"
    categories = ("door", "counter", "shelf") if split == "DEVELOPMENT" else ("door", "counter", "shelf", "panel")
    category = categories[index % len(categories)]
    no_pose_kind = None if index % 3 else ("BLOCKED_FRONT" if index % 2 else "ISOLATED_BY_WALL")
    # Do not couple target orientation to the every-third-scene NONE schedule.
    front = ((0.0, -1.0), (-1.0, 0.0), (1.0, 0.0))[(index + index // 3) % 3]
    if front == (0.0, -1.0):
        cx, cy = rng.uniform(2.0, 6.0), rng.uniform(6.2, 6.8)
    elif front == (-1.0, 0.0):
        cx, cy = rng.uniform(6.2, 6.8), rng.uniform(4.3, 6.2)
    else:
        cx, cy = rng.uniform(1.2, 1.8), rng.uniform(4.3, 6.2)
    target = Target(f"{scene_id}-target", category, cx, cy, 1.0, 0.32, *front)
    distractor = Target(
        f"{scene_id}-same-class-distractor", category,
        7.0 if cx < 4.0 else 1.0, 2.4, 1.0, 0.32, 0.0, -1.0,
    )
    obstacles: list[Rect] = [
        Rect(2.25, 2.75, 3.00, 3.45),
        Rect(5.15, 1.55, 5.85, 2.25),
    ]
    if no_pose_kind == "BLOCKED_FRONT":
        fx, fy = target.front_x, target.front_y
        tx, ty = -fy, fx
        centre_x, centre_y = target.cx + fx * 0.85, target.cy + fy * 0.85
        half_tangent = target.width_m * 0.75 + 0.32
        # Axis-aligned because all generated normals are axis-aligned.
        if abs(fx) > 0:
            obstacles.append(Rect(centre_x - 0.26, centre_y - half_tangent, centre_x + 0.26, centre_y + half_tangent))
        else:
            obstacles.append(Rect(centre_x - half_tangent, centre_y - 0.26, centre_x + half_tangent, centre_y + 0.26))
    elif no_pose_kind == "ISOLATED_BY_WALL":
        obstacles.append(Rect(0.0, 3.45, 8.0, 3.80))
    return Scene(
        scene_id, split, 8.0, 8.0, 4.0, 0.65, target, distractor,
        tuple(obstacles), "NONE" if no_pose_kind else "VALID_SET", no_pose_kind,
    )


def make_cohort() -> tuple[Scene, ...]:
    return tuple(make_scene("DEVELOPMENT", index) for index in range(12)) + tuple(
        make_scene("HELD_OUT", index) for index in range(36)
    )


def perturb_scene(scene: Scene) -> Scene:
    delta_x = 0.04 if int(scene.scene_id[-1]) % 2 else -0.04
    delta_y = -0.03 if int(scene.scene_id[-1]) % 2 else 0.03
    target = replace(scene.target, cx=scene.target.cx + delta_x, cy=scene.target.cy + delta_y)
    return replace(scene, target=target)


def counterfactual_judgements(scene: Scene, truth: tuple[Pose, ...]) -> dict[str, bool]:
    target = scene.target
    front_pose = truth[0] if truth else Pose(
        target.cx + target.front_x * 0.85,
        target.cy + target.front_y * 0.85,
        math.atan2(-target.front_y, -target.front_x),
    )
    back = Pose(
        target.cx - target.front_x * 0.85,
        target.cy - target.front_y * 0.85,
        math.atan2(target.front_y, target.front_x),
    )
    irrelevant = Pose(scene.start_x, scene.start_y + 0.7, front_pose.yaw_rad)
    distractor_truth = interaction_pose_set(scene, scene.distractor)
    wrong_target = distractor_truth[0] if distractor_truth else Pose(
        scene.distractor.cx, scene.distractor.cy - 0.85, math.pi / 2
    )
    isolated_scene = replace(scene, obstacles=scene.obstacles + (Rect(0.0, 3.45, 8.0, 3.80),))
    unreachable = judge_pose(isolated_scene, target, front_pose)
    return {
        "same_class_wrong_instance_rejected": not pose_matches(wrong_target, truth),
        "correct_target_back_side_rejected": "NOT_FUNCTIONAL_SIDE" in judge_pose(scene, target, back).reasons,
        "free_but_goal_irrelevant_rejected": "BAD_INTERACTION_DISTANCE" in judge_pose(scene, target, irrelevant).reasons,
        "face_target_but_unreachable_rejected": "UNREACHABLE" in unreachable.reasons,
    }
