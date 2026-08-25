#!/usr/bin/env python3
"""Privileged ProcTHOR relation signatures for the GRAIL-R0 oracle probe."""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Any


STABLE_TYPES = {
    "ArmChair", "Bathtub", "Bed", "Bookcase", "Bookshelf", "Cabinet",
    "CoffeeTable", "CounterTop", "Desk", "DiningTable", "Dresser",
    "Fridge", "LaundryHamper", "Shelf", "ShowerDoor", "SideTable",
    "Sink", "Sofa", "StoveBurner", "Television", "Toilet", "TVStand",
}

RELATION_FIELD_GROUPS = (
    "semantic_type", "support", "room_types", "height_band", "sibling_ordinal",
    "nearby_type", "nearby_direction", "nearby_distance", "nearby_height",
)


def object_type(object_id: str) -> str:
    return object_id.split("|", 1)[0]


def flatten_objects(objects: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    index: dict[str, dict[str, Any]] = {}
    roots: list[dict[str, Any]] = []

    def visit(node: dict[str, Any], parent_type: str | None, root_id: str) -> None:
        record = {
            "id": node["id"],
            "type": object_type(node["id"]),
            "position": node.get("position"),
            "rotation_y": float(node.get("rotation", {}).get("y", 0.0)),
            "parent_type": parent_type,
            "root_id": root_id,
        }
        index[node["id"]] = record
        for child in node.get("children", []):
            visit(child, record["type"], root_id)

    for node in objects:
        visit(node, None, node["id"])
        roots.append(index[node["id"]])
    return index, roots


def _wall_endpoints(wall_id: str) -> tuple[tuple[float, float], tuple[float, float]]:
    fields = wall_id.split("|")
    return (float(fields[-4]), float(fields[-3])), (float(fields[-2]), float(fields[-1]))


def door_record(door: dict[str, Any]) -> dict[str, Any]:
    start, end = _wall_endpoints(door["wall0"])
    dx, dz = end[0] - start[0], end[1] - start[1]
    length = max(math.hypot(dx, dz), 1e-6)
    offset = float(door["assetPosition"]["x"])
    return {
        "id": door["id"],
        "type": "Doorway",
        "position": {
            "x": start[0] + dx / length * offset,
            "y": float(door["assetPosition"].get("y", 1.0)),
            "z": start[1] + dz / length * offset,
        },
        "rotation_y": math.degrees(math.atan2(dx, dz)),
        "parent_type": None,
        "root_id": door["id"],
        "room_ids": tuple(sorted({door.get("room0", ""), door.get("room1", "")} - {""})),
    }


def load_houses(dataset: Path, house_indices: set[int]) -> dict[int, dict[str, Any]]:
    houses: dict[int, dict[str, Any]] = {}
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in house_indices:
                houses[index] = json.loads(line)
            if len(houses) == len(house_indices):
                break
    if set(houses) != house_indices:
        raise ValueError("dataset lacks required GRAIL-R0 houses")
    return houses


def _point_in_polygon(x: float, z: float, polygon: list[dict[str, float]]) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x0, z0 = float(previous["x"]), float(previous["z"])
        x1, z1 = float(current["x"]), float(current["z"])
        if (z0 > z) != (z1 > z):
            crossing = (x1 - x0) * (z - z0) / (z1 - z0) + x0
            if x < crossing:
                inside = not inside
        previous = current
    return inside


def _room_types(house: dict[str, Any], record: dict[str, Any]) -> tuple[str, ...]:
    by_id = {room["id"]: room.get("roomType", "Unknown") for room in house.get("rooms", [])}
    if record.get("room_ids"):
        return tuple(sorted(by_id.get(room_id, "Exterior") for room_id in record["room_ids"]))
    position = record["position"]
    return tuple(sorted(
        room.get("roomType", "Unknown")
        for room in house.get("rooms", [])
        if _point_in_polygon(float(position["x"]), float(position["z"]), room["floorPolygon"])
    )) or ("Unknown",)


def _rank_bin(value: float, values: list[float], labels: tuple[str, str, str]) -> str:
    if len(values) < 2 or max(values) - min(values) < 1e-6:
        return "SINGLE"
    fraction = (value - min(values)) / (max(values) - min(values))
    return labels[0] if fraction <= 1 / 3 else labels[1] if fraction <= 2 / 3 else labels[2]


def _height_band(y: float) -> str:
    return "LOW" if y < 0.45 else "MID" if y < 1.15 else "HIGH"


def _distance_band(distance: float) -> str:
    return "NEAR" if distance <= 1.25 else "MID" if distance <= 2.5 else "FAR"


def _direction(dx: float, dz: float, yaw_deg: float) -> str:
    yaw = math.radians(yaw_deg)
    right = dx * math.cos(yaw) - dz * math.sin(yaw)
    front = dx * math.sin(yaw) + dz * math.cos(yaw)
    if abs(right) > abs(front):
        return "RIGHT" if right >= 0 else "LEFT"
    return "FRONT" if front >= 0 else "BEHIND"


def _candidate_record(candidate: dict[str, Any], object_index: dict[str, dict[str, Any]],
                      doors: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate_id = candidate["object_id"]
    if candidate_id in doors:
        return doors[candidate_id]
    if candidate_id in object_index:
        return object_index[candidate_id]
    base_id = candidate_id.split("___", 1)[0]
    if base_id not in object_index:
        raise ValueError(f"unmapped ProcTHOR object id: {candidate_id}")
    base = dict(object_index[base_id])
    base["id"] = candidate_id
    base["type"] = candidate["object_type"]
    base["parent_type"] = object_index[base_id]["type"]
    return base


def relation_signatures(house: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    object_index, roots = flatten_objects(house.get("objects", []))
    doors = {door["id"]: door_record(door) for door in house.get("doors", [])}
    records = [_candidate_record(candidate, object_index, doors) for candidate in candidates]
    anchors = [record for record in roots if record["type"] in STABLE_TYPES and record.get("position")]

    group_centers: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for candidate, record in zip(candidates, records):
        bbox = candidate["bbox"]
        group_centers.setdefault((record["root_id"], candidate["object_type"]), []).append(
            ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        )

    signatures = []
    for candidate, record in zip(candidates, records):
        position = record["position"]
        if position is None:
            raise ValueError(f"object has no native position: {candidate['object_id']}")
        bbox = candidate["bbox"]
        center_x, center_y = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
        centers = group_centers[(record["root_id"], candidate["object_type"])]
        neighbors = []
        for anchor in anchors:
            if anchor["root_id"] == record["root_id"]:
                continue
            dx = float(anchor["position"]["x"]) - float(position["x"])
            dz = float(anchor["position"]["z"]) - float(position["z"])
            distance = math.hypot(dx, dz)
            if distance <= 4.5:
                height_delta = float(anchor["position"]["y"]) - float(position["y"])
                height_relation = "ABOVE" if height_delta > 0.4 else "BELOW" if height_delta < -0.4 else "LEVEL"
                neighbors.append((distance, anchor["type"], _direction(dx, dz, record["rotation_y"]),
                                  _distance_band(distance), height_relation))
        neighbors.sort(key=lambda item: (item[0], item[1], item[2]))
        signatures.append({
            "semantic_type": candidate["object_type"],
            "support": f"PART_OF:{record['parent_type']}" if record.get("parent_type") else "FLOOR_OR_STRUCTURE",
            "room_types": _room_types(house, record),
            "height_band": _height_band(float(position["y"])),
            "part_horizontal": _rank_bin(center_x, [value[0] for value in centers], ("LEFT", "CENTER", "RIGHT")),
            "part_vertical": _rank_bin(center_y, [value[1] for value in centers], ("TOP", "MIDDLE", "BOTTOM")),
            "nearby": tuple((kind, direction, distance, height) for _, kind, direction, distance, height in neighbors[:3]),
        })
    return signatures


def canonical_signature(signature: dict[str, Any]) -> tuple[Any, ...]:
    return (
        signature["semantic_type"], signature["support"], tuple(signature["room_types"]),
        signature["height_band"], signature["part_horizontal"], signature["part_vertical"],
        tuple(signature["nearby"]),
    )


def projected_signature(signature: dict[str, Any], field_groups: tuple[str, ...]) -> tuple[Any, ...]:
    """Project an R0 signature onto named, independently ablatable relation groups."""
    unknown = set(field_groups) - set(RELATION_FIELD_GROUPS)
    if unknown:
        raise ValueError(f"unknown relation field groups: {sorted(unknown)}")
    selected = set(field_groups)
    projected: list[Any] = []
    if "semantic_type" in selected:
        projected.append(("semantic_type", signature["semantic_type"]))
    if "support" in selected:
        projected.append(("support", signature["support"]))
    if "room_types" in selected:
        projected.append(("room_types", tuple(signature["room_types"])))
    if "height_band" in selected:
        projected.append(("height_band", signature["height_band"]))
    if "sibling_ordinal" in selected:
        projected.append(("sibling_ordinal", signature["part_horizontal"], signature["part_vertical"]))
    nearby_indices = {
        "nearby_type": 0,
        "nearby_direction": 1,
        "nearby_distance": 2,
        "nearby_height": 3,
    }
    active_nearby = [(name, index) for name, index in nearby_indices.items() if name in selected]
    if active_nearby:
        projected.append((
            "nearby",
            tuple(tuple(neighbor[index] for _, index in active_nearby) for neighbor in signature["nearby"]),
        ))
    return tuple(projected)


def select_with_projected_relations(
    target_signature: dict[str, Any],
    candidate_signatures: list[dict[str, Any]],
    appearance_scores: list[float],
    spatial_keys: list[Any],
    field_groups: tuple[str, ...],
) -> tuple[int | None, float, str]:
    """R0 selection with only the requested signature field groups visible."""
    target = projected_signature(target_signature, field_groups)
    exact = [
        index for index, signature in enumerate(candidate_signatures)
        if projected_signature(signature, field_groups) == target
    ]
    if not exact:
        return None, 0.0, "NO_EXACT_RELATION_MATCH"
    selected = max(exact, key=lambda index: (appearance_scores[index], spatial_keys[index]))
    if len(exact) == 1:
        return selected, 1.0, "UNIQUE_RELATION_MATCH"
    return selected, float(appearance_scores[selected]), "RELATION_COLLISION_APPEARANCE_TIEBREAK"


def select_with_relational_oracle(target_signature: dict[str, Any], candidate_signatures: list[dict[str, Any]],
                                  appearance_scores: list[float], spatial_keys: list[Any]) -> tuple[int | None, float, str]:
    """Require an exact coarse relation signature; use frozen appearance only to break collisions."""
    target = canonical_signature(target_signature)
    exact = [index for index, signature in enumerate(candidate_signatures) if canonical_signature(signature) == target]
    if not exact:
        return None, 0.0, "NO_EXACT_RELATION_MATCH"
    selected = max(exact, key=lambda index: (appearance_scores[index], spatial_keys[index]))
    if len(exact) == 1:
        return selected, 1.0, "UNIQUE_RELATION_MATCH"
    return selected, float(appearance_scores[selected]), "RELATION_COLLISION_APPEARANCE_TIEBREAK"
