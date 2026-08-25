#!/usr/bin/env python3
"""Privileged owner-local coordinate helpers for GRAIL-R1C-O."""

from __future__ import annotations

import math
from typing import Any


def owner_id_for(record: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> tuple[str | None, str]:
    """Resolve a source-native immediate owner without using camera pose or pixels."""
    object_id = record["objectId"]
    encoded_owner = object_id.split("___", 1)[0]
    if encoded_owner != object_id and encoded_owner in by_id:
        return encoded_owner, "OBJECT_ID_NATIVE_COMPONENT_OWNER"
    parents = sorted(parent for parent in (record.get("parentReceptacles") or []) if parent in by_id)
    if parents:
        return parents[0], "PARENT_RECEPTACLE"
    return object_id, "STANDALONE_SELF"


def owner_local_coordinate(record: dict[str, Any], owner: dict[str, Any]) -> tuple[float, float, float]:
    """Return canonical (right, up, front) using the owner's source-native yaw."""
    position = record.get("position")
    owner_position = owner.get("position")
    owner_rotation = owner.get("rotation")
    if not position or not owner_position or not owner_rotation or "y" not in owner_rotation:
        raise ValueError("native owner frame lacks position or yaw")
    dx = float(position["x"]) - float(owner_position["x"])
    dy = float(position["y"]) - float(owner_position["y"])
    dz = float(position["z"]) - float(owner_position["z"])
    yaw = math.radians(float(owner_rotation["y"]))
    return (
        dx * math.cos(yaw) - dz * math.sin(yaw),
        dy,
        dx * math.sin(yaw) + dz * math.cos(yaw),
    )


def rank_bin(value: float, values: list[float], labels: tuple[str, str, str]) -> str:
    if len(values) < 2 or max(values) - min(values) < 1e-6:
        return "SINGLE"
    fraction = (value - min(values)) / (max(values) - min(values))
    return labels[0] if fraction <= 1 / 3 else labels[1] if fraction <= 2 / 3 else labels[2]


def canonicalize_scene(objects: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Create view-invariant owner-local slots for every runtime object."""
    by_id = {record["objectId"]: record for record in objects}
    resolved: dict[str, tuple[str, str, tuple[float, float, float]]] = {}
    failures: dict[str, str] = {}
    for object_id, record in by_id.items():
        owner_id, source = owner_id_for(record, by_id)
        try:
            if owner_id is None or owner_id not in by_id:
                raise ValueError("native owner is unavailable")
            coordinate = owner_local_coordinate(record, by_id[owner_id])
            resolved[object_id] = (owner_id, source, coordinate)
        except (KeyError, TypeError, ValueError) as error:
            failures[object_id] = str(error)

    groups: dict[tuple[str, str], list[tuple[float, float, float]]] = {}
    for object_id, (owner_id, _, coordinate) in resolved.items():
        groups.setdefault((owner_id, by_id[object_id]["objectType"]), []).append(coordinate)

    output: dict[str, dict[str, Any]] = {}
    for object_id, record in by_id.items():
        if object_id in failures:
            output[object_id] = {"evaluable": False, "reason": failures[object_id]}
            continue
        owner_id, source, coordinate = resolved[object_id]
        siblings = groups[(owner_id, record["objectType"])]
        right, up, front = coordinate
        output[object_id] = {
            "evaluable": True,
            "owner_id": owner_id,
            "owner_source": source,
            "owner_yaw_degrees": float(by_id[owner_id]["rotation"]["y"]),
            "local_right": right,
            "local_up": up,
            "local_front": front,
            "horizontal": rank_bin(right, [value[0] for value in siblings], ("LEFT", "CENTER", "RIGHT")),
            "vertical": rank_bin(-up, [-value[1] for value in siblings], ("TOP", "MIDDLE", "BOTTOM")),
            "native_sibling_count": len(siblings),
        }
    return output

