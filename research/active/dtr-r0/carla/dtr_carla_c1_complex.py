"""Pure protocol and evidence helpers for the DTR-CARLA-C1 scene pack.

This module intentionally has no CARLA dependency.  It defines the immutable
route-receipt boundary, scripted local trajectories, geometric contact helpers,
and the checks that can be exercised before paying for a simulator run.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


EXPERIMENT_ID = "DTR_CARLA_C1_COMPLEX_SCENE_ASSET_CANARY_V1"
FORBIDDEN_MODEL_KEYS = {
    "actual_future_trajectory",
    "adherence_truth",
    "collision_polygons_xy",
    "contact_label",
    "current_contact",
    "executed_route",
    "expected_contact",
    "first_contact_time_s",
    "future_pose",
    "future_contact_within_horizon",
    "instance_visibility",
    "minimum_distance_m",
    "realized_path",
    "realized_time_to_contact_seconds",
    "responsible_asset",
    "responsible_actor",
    "scenario_role",
    "truth",
    "twin_role",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def segments(trajectory: dict[str, Any]) -> list[dict[str, float]]:
    values = sorted(trajectory["segments"], key=lambda item: float(item["start_s"]))
    if not values or abs(float(values[0]["start_s"])) > 1e-9:
        raise ValueError("every trajectory must begin at t=0")
    starts = [float(item["start_s"]) for item in values]
    if starts != sorted(set(starts)):
        raise ValueError("trajectory segment start times must be unique and ordered")
    return values


def trajectory_position(trajectory: dict[str, Any], time_s: float) -> tuple[float, float]:
    forward = float(trajectory["start_forward_m"])
    right = float(trajectory["start_right_m"])
    values = segments(trajectory)
    for index, segment in enumerate(values):
        start_s = float(segment["start_s"])
        if time_s <= start_s:
            break
        end_s = (
            min(time_s, float(values[index + 1]["start_s"]))
            if index + 1 < len(values)
            else time_s
        )
        if end_s <= start_s:
            continue
        elapsed = end_s - start_s
        forward += float(segment["velocity_forward_mps"]) * elapsed
        right += float(segment["velocity_right_mps"]) * elapsed
        if end_s >= time_s:
            break
    return forward, right


def trajectory_velocity(trajectory: dict[str, Any], time_s: float) -> tuple[float, float]:
    selected = segments(trajectory)[0]
    for segment in segments(trajectory):
        if float(segment["start_s"]) <= time_s + 1e-9:
            selected = segment
        else:
            break
    return (
        float(selected["velocity_forward_mps"]),
        float(selected["velocity_right_mps"]),
    )


def build_plan_receipt(plan: dict[str, Any] | None) -> dict[str, Any] | None:
    """Seal only the issued plan; realized execution is deliberately absent."""

    if plan is None:
        return None
    payload = {
        "schema_version": "dtr-c1-plan-receipt-v1",
        "plan_id": str(plan["plan_id"]),
        "session_id": str(plan["session_id"]),
        "issued_at_s": float(plan["issued_at_s"]),
        "valid_from_s": float(plan.get("valid_from_s", plan["issued_at_s"])),
        "expires_at_s": float(plan["expires_at_s"]),
        "coordinate_frame": str(plan.get("coordinate_frame", "ANCHOR_FORWARD_RIGHT")),
        "time_parameterized_waypoints": [
            {
                "time_s": float(item["time_s"]),
                "forward_m": float(item["forward_m"]),
                "right_m": float(item["right_m"]),
            }
            for item in plan["time_parameterized_waypoints"]
        ],
    }
    return {**payload, "receipt_sha256": sha256_json(payload)}


def validate_plan_receipt(receipt: dict[str, Any]) -> None:
    supplied = str(receipt.get("receipt_sha256", ""))
    payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    expected = sha256_json(payload)
    if supplied != expected:
        raise ValueError(f"plan receipt hash mismatch: expected {expected}, got {supplied}")


def plan_authority(
    receipt: dict[str, Any] | None,
    *,
    session_id: str,
    time_s: float,
) -> str:
    if receipt is None:
        return "NO_PLAN"
    validate_plan_receipt(receipt)
    if str(receipt["session_id"]) != session_id:
        return "WRONG_SESSION"
    if time_s + 1e-9 < float(receipt["valid_from_s"]):
        return "FUTURE_ISSUED"
    if time_s > float(receipt["expires_at_s"]) + 1e-9:
        return "EXPIRED"
    return "VALID"


def trajectory_prefix_equal(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    end_s: float,
    sample_s: float,
    tolerance: float = 1e-9,
) -> bool:
    steps = int(round(end_s / sample_s))
    for index in range(steps + 1):
        time_s = index * sample_s
        a = trajectory_position(first, time_s)
        b = trajectory_position(second, time_s)
        if math.dist(a, b) > tolerance:
            return False
        av = trajectory_velocity(first, time_s)
        bv = trajectory_velocity(second, time_s)
        if time_s < end_s - tolerance and math.dist(av, bv) > tolerance:
            return False
    return True


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    sx, sy = start
    ex, ey = end
    px, py = point
    dx, dy = ex - sx, ey - sy
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.dist(point, start)
    ratio = ((px - sx) * dx + (py - sy) * dy) / denominator
    ratio = min(1.0, max(0.0, ratio))
    return math.hypot(px - (sx + ratio * dx), py - (sy + ratio * dy))


def _point_in_convex_polygon(
    point: tuple[float, float], polygon: list[tuple[float, float]]
) -> bool:
    signs: list[float] = []
    for index, start in enumerate(polygon):
        end = polygon[(index + 1) % len(polygon)]
        edge_x, edge_y = end[0] - start[0], end[1] - start[1]
        rel_x, rel_y = point[0] - start[0], point[1] - start[1]
        signs.append(edge_x * rel_y - edge_y * rel_x)
    return all(value >= -1e-7 for value in signs) or all(
        value <= 1e-7 for value in signs
    )


def point_polygon_distance(
    point: tuple[float, float], polygon: list[list[float]]
) -> float:
    values = [(float(item[0]), float(item[1])) for item in polygon]
    if len(values) < 3:
        raise ValueError("a contact polygon requires at least three vertices")
    if _point_in_convex_polygon(point, values):
        return 0.0
    return min(
        _point_segment_distance(point, values[index], values[(index + 1) % len(values)])
        for index in range(len(values))
    )


def contact_union(
    wearer_xy: tuple[float, float],
    polygons: dict[str, list[list[float]]],
    *,
    wearer_radius_m: float,
) -> tuple[bool, float, list[str]]:
    distances = {
        key: point_polygon_distance(wearer_xy, polygon)
        for key, polygon in polygons.items()
    }
    if not distances:
        return False, math.inf, []
    minimum = min(distances.values())
    contacts = sorted(
        key for key, distance in distances.items() if distance <= wearer_radius_m + 1e-9
    )
    return bool(contacts), minimum, contacts


def forbidden_model_paths(value: Any, prefix: str = "$") -> list[str]:
    failures: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if str(key).lower() in FORBIDDEN_MODEL_KEYS:
                failures.append(path)
            failures.extend(forbidden_model_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            failures.extend(forbidden_model_paths(child, f"{prefix}[{index}]"))
    return failures


def validate_protocol(protocol: dict[str, Any]) -> None:
    if protocol.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("unexpected C1 protocol identity")
    scenarios = list(protocol.get("scenarios", []))
    ids = [str(item["episode_id"]) for item in scenarios]
    if len(ids) != 8 or len(set(ids)) != 8:
        raise ValueError("C1 requires exactly eight unique opaque episode ids")
    assets = list(protocol.get("asset_cluster", []))
    keys = [str(item["asset_key"]) for item in assets]
    if len(keys) != len(set(keys)):
        raise ValueError("asset keys must be unique")
    if "wearer" not in keys:
        raise ValueError("asset cluster must contain the wearer")
    known_trajectories = set(protocol.get("trajectory_library", {}))
    for scenario in scenarios:
        references = [str(scenario["wearer_trajectory"])] + [
            str(value) for value in scenario["asset_trajectories"].values()
        ]
        missing = sorted(set(references) - known_trajectories)
        if missing:
            raise ValueError(f"{scenario['episode_id']} references missing trajectories: {missing}")
        receipt = build_plan_receipt(scenario.get("issued_plan"))
        if receipt is not None:
            validate_plan_receipt(receipt)
    known_ids = set(ids)
    for pair in protocol.get("twin_contracts", []):
        if str(pair["a"]) not in known_ids or str(pair["b"]) not in known_ids:
            raise ValueError(f"unknown twin endpoint: {pair}")


def scenario_by_id(protocol: dict[str, Any], episode_id: str) -> dict[str, Any]:
    for scenario in protocol["scenarios"]:
        if str(scenario["episode_id"]) == episode_id:
            return scenario
    raise KeyError(episode_id)
