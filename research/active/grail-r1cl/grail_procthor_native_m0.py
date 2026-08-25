#!/usr/bin/env python3
"""Pure helpers for the ProcTHOR/AI2-THOR native-interaction GRAIL M0."""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


POSITION_TOLERANCE_M = 0.50
YAW_TOLERANCE_DEG = 20.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_action_target(obj: dict[str, Any]) -> bool:
    """Keep stationary targets with an observable simulator action."""
    return (
        not bool(obj.get("pickupable"))
        and not bool(obj.get("moveable"))
        and (bool(obj.get("openable")) or bool(obj.get("toggleable")))
    )


def action_pair(obj: dict[str, Any]) -> tuple[str, str]:
    if obj.get("openable"):
        return ("CloseObject", "OpenObject") if obj.get("isOpen") else ("OpenObject", "CloseObject")
    if obj.get("toggleable"):
        return ("ToggleObjectOff", "ToggleObjectOn") if obj.get("isToggled") else ("ToggleObjectOn", "ToggleObjectOff")
    raise ValueError(f"object is not an action target: {obj.get('objectId')}")


def pose_position(pose: dict[str, Any]) -> tuple[float, float]:
    return float(pose["x"]), float(pose["z"])


def yaw_error_deg(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def interaction_pose_success(
    candidate: dict[str, Any] | None,
    truth: Iterable[dict[str, Any]],
) -> bool:
    if candidate is None:
        return False
    cx, cz = pose_position(candidate)
    for pose in truth:
        px, pz = pose_position(pose)
        if (
            math.hypot(cx - px, cz - pz) <= POSITION_TOLERANCE_M
            and yaw_error_deg(float(candidate["rotation"]), float(pose["rotation"]))
            <= YAW_TOLERANCE_DEG
        ):
            return True
    return False


def yaw_toward(position: dict[str, float], target: dict[str, float]) -> float:
    dx = float(target["x"]) - float(position["x"])
    dz = float(target["z"]) - float(position["z"])
    return math.degrees(math.atan2(dx, dz)) % 360.0


def representative_pose(
    poses: list[dict[str, Any]], target_position: dict[str, float]
) -> dict[str, Any] | None:
    if not poses:
        return None
    return min(
        poses,
        key=lambda pose: (
            math.hypot(
                float(pose["x"]) - float(target_position["x"]),
                float(pose["z"]) - float(target_position["z"]),
            ),
            float(pose["rotation"]),
            float(pose["x"]),
            float(pose["z"]),
        ),
    )


def has_local_stability(candidate: dict[str, Any] | None, poses: list[dict[str, Any]]) -> bool:
    if candidate is None:
        return False
    cx, cz = pose_position(candidate)
    for pose in poses:
        if pose is candidate:
            continue
        px, pz = pose_position(pose)
        distance = math.hypot(cx - px, cz - pz)
        yaw_error = yaw_error_deg(float(candidate["rotation"]), float(pose["rotation"]))
        if (0.0 < distance <= 0.26 and yaw_error <= 30.0) or (
            distance <= 1e-6 and 0.0 < yaw_error <= 30.0
        ):
            return True
    return False


def reachable_path_exists(
    reachable: list[dict[str, float]], goal_pose: dict[str, Any] | None, grid_m: float = 0.25
) -> bool:
    if goal_pose is None or not reachable:
        return False

    def key(position: dict[str, Any]) -> tuple[int, int]:
        return round(float(position["x"]) / grid_m), round(float(position["z"]) / grid_m)

    cells = {key(position) for position in reachable}
    goal = key(goal_pose)
    if goal not in cells:
        return False
    start = max(cells, key=lambda cell: (cell[0] - goal[0]) ** 2 + (cell[1] - goal[1]) ** 2)
    queue = deque([start])
    seen = {start}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            return True
        for delta in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = cell[0] + delta[0], cell[1] + delta[1]
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False


def counterfactuals(
    candidate: dict[str, Any] | None,
    poses: list[dict[str, Any]],
    reachable: list[dict[str, float]],
    target_position: dict[str, float],
) -> list[dict[str, Any]]:
    if candidate is None:
        return []
    rows: list[dict[str, Any]] = []
    back = dict(candidate)
    back["rotation"] = (float(candidate["rotation"]) + 180.0) % 360.0
    if not interaction_pose_success(back, poses):
        rows.append({"family": "BACK_FACING", "rejected": True})

    if reachable:
        far = max(
            reachable,
            key=lambda point: math.hypot(
                float(point["x"]) - float(target_position["x"]),
                float(point["z"]) - float(target_position["z"]),
            ),
        )
        unrelated = {
            "x": float(far["x"]),
            "y": float(far["y"]),
            "z": float(far["z"]),
            "rotation": yaw_toward(far, target_position),
            "standing": True,
            "horizon": 0.0,
        }
        if not interaction_pose_success(unrelated, poses):
            rows.append({"family": "FREE_BUT_UNRELATED", "rejected": True})

    unreachable = dict(candidate)
    unreachable["x"] = float(candidate["x"]) + 10.0
    unreachable["z"] = float(candidate["z"]) + 10.0
    unreachable["rotation"] = yaw_toward(unreachable, target_position)
    rows.append({
        "family": "OUTSIDE_REACHABLE_NAVMESH",
        "rejected": not interaction_pose_success(unreachable, poses),
    })
    return rows
