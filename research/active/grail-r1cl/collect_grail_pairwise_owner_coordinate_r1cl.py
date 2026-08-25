#!/usr/bin/env python3
"""Render house-disjoint RGB/mask owner-group pairs for GRAIL-R1C-L."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from collect_grail_m1 import bbox_for, start_controller
from grail_canonical_coordinates_r1c import canonicalize_scene
from grail_procthor_native_m0 import canonical_sha256, is_action_target, sha256_file, yaw_toward


TARGET_TYPES = {"Drawer", "Doorway"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_houses(path: Path, roster: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    wanted = {int(row["house_index"]) for row in roster}
    houses: dict[int, dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                houses[index] = json.loads(line)
            if len(houses) == len(wanted):
                break
    if set(houses) != wanted:
        raise ValueError("R1C-L dataset lacks frozen houses")
    for row in roster:
        index = int(row["house_index"])
        if canonical_sha256(houses[index]) != row["house_sha256"]:
            raise ValueError(f"R1C-L house hash mismatch at {index}")
    return houses


def _rank(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _crop_box(detections: list[tuple[list[int], np.ndarray]], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x0 = min(item[0][0] for item in detections)
    y0 = min(item[0][1] for item in detections)
    x1 = max(item[0][2] for item in detections)
    y1 = max(item[0][3] for item in detections)
    width, height = max(x1 - x0, 1), max(y1 - y0, 1)
    pad_x, pad_y = max(8, round(width * 0.20)), max(8, round(height * 0.20))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(shape[1], x1 + pad_x), min(shape[0], y1 + pad_y)


def _save_view(event: Any, members: list[dict[str, Any]], root: Path, relative_stem: str,
               group_id: str, camera_quadrant: int, distance_m: float) -> dict[str, Any] | None:
    visible: list[tuple[dict[str, Any], list[int], np.ndarray]] = []
    for member in members:
        detected = bbox_for(event, member["objectId"])
        if detected is not None:
            visible.append((member, detected[0], detected[1]))
    if not 2 <= len(visible) <= 4:
        return None
    box = _crop_box([(bbox, mask) for _, bbox, mask in visible], event.frame.shape)
    x0, y0, x1, y1 = box
    rgb = Image.fromarray(event.frame[y0:y1, x0:x1])
    union = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    member_rows = []
    for member, bbox, mask in visible:
        local_mask = mask[y0:y1, x0:x1]
        union[local_mask] = 255
        ys, xs = np.nonzero(local_mask)
        member_rows.append({
            "object_id": member["objectId"],
            "bbox": [bbox[0] - x0, bbox[1] - y0, bbox[2] - x0, bbox[3] - y0],
            "centroid": [float(xs.mean()), float(ys.mean())],
            "visible_area": int(local_mask.sum()),
        })
    centroid = Image.new("L", rgb.size, 0)
    draw = ImageDraw.Draw(centroid)
    radius = max(2, round(min(rgb.size) * 0.025))
    for row in member_rows:
        cx, cy = row["centroid"]
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=255)
    rgb_path = f"{relative_stem}-rgb.png"
    union_path = f"{relative_stem}-union.png"
    centroid_path = f"{relative_stem}-centroids.png"
    (root / rgb_path).parent.mkdir(parents=True, exist_ok=True)
    rgb.save(root / rgb_path)
    Image.fromarray(union).save(root / union_path)
    centroid.save(root / centroid_path)
    camera = event.metadata["agent"]
    return {
        "view_id": Path(relative_stem).name,
        "group_id": group_id,
        "rgb": rgb_path,
        "owner_union_mask": union_path,
        "sibling_centroid_mask": centroid_path,
        "width": rgb.width,
        "height": rgb.height,
        "members": sorted(member_rows, key=lambda row: row["object_id"]),
        "camera_yaw_label_degrees": float(camera["rotation"]["y"]),
        "camera_quadrant_label": camera_quadrant,
        "distance_label_m": distance_m,
    }


def _axis(centroids: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(centroids, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0)
    if len(matrix) < 2 or float(np.linalg.norm(centered)) < 1e-8:
        return np.asarray([1.0, 0.0])
    _, vectors = np.linalg.eigh(centered.T @ centered)
    options = [vectors[:, 0], vectors[:, 1]]
    value = max(options, key=lambda item: (abs(float(item[0])), -abs(float(item[1]))))
    value = value / max(float(np.linalg.norm(value)), 1e-8)
    if value[0] < 0 or (abs(float(value[0])) < 1e-12 and value[1] < 0):
        value = -value
    return value


def _rank_bin(value: float, values: list[float], labels: tuple[str, str, str]) -> str:
    if len(values) < 2 or max(values) - min(values) < 1e-6:
        return "SINGLE"
    fraction = (value - min(values)) / (max(values) - min(values))
    return labels[0] if fraction <= 1 / 3 else labels[1] if fraction <= 2 / 3 else labels[2]


def view_ordinals(view: dict[str, Any], direction: float = 1.0) -> dict[str, tuple[str, str]]:
    axis = _axis([row["centroid"] for row in view["members"]]) * direction
    horizontal = [float(np.dot(row["centroid"], axis)) for row in view["members"]]
    vertical = [float(row["centroid"][1]) for row in view["members"]]
    return {
        row["object_id"]: (
            _rank_bin(float(np.dot(row["centroid"], axis)), horizontal, ("LEFT", "CENTER", "RIGHT")),
            _rank_bin(float(row["centroid"][1]), vertical, ("TOP", "MIDDLE", "BOTTOM")),
        )
        for row in view["members"]
    }


def valid_bins(reference: dict[str, Any], query: dict[str, Any]) -> tuple[list[int], list[str]]:
    reference_ids = {row["object_id"] for row in reference["members"]}
    query_ids = {row["object_id"] for row in query["members"]}
    shared = reference_ids & query_ids
    if len(shared) < 2:
        return [], []
    expected = view_ordinals(reference, 1.0)
    modes = []
    query_preserve = view_ordinals(query, 1.0)
    query_flip = view_ordinals(query, -1.0)
    if all(query_preserve[object_id] == expected[object_id] for object_id in shared):
        modes.append("PRESERVE")
    if all(query_flip[object_id] == expected[object_id] for object_id in shared):
        modes.append("FLIP")
    bins = []
    for index in range(36):
        cosine = math.cos(math.radians(index * 10.0))
        if cosine > 1e-8 and "PRESERVE" in modes:
            bins.append(index)
        elif cosine < -1e-8 and "FLIP" in modes:
            bins.append(index)
    return bins, modes


def _ranked_positions(reachable: list[dict[str, float]], center: dict[str, float], owner_yaw: float,
                      group_key: str) -> list[tuple[dict[str, float], int, float]]:
    buckets: dict[tuple[int, str], list[tuple[str, dict[str, float], float]]] = {}
    for position in reachable:
        dx = float(position["x"]) - center["x"]
        dz = float(position["z"]) - center["z"]
        distance = math.hypot(dx, dz)
        if not 1.5 <= distance <= 4.25:
            continue
        angle = (math.degrees(math.atan2(dx, dz)) - owner_yaw) % 360.0
        quadrant = int(angle // 90.0) % 4
        distance_band = "NEAR" if distance < 2.75 else "FAR"
        key = f"{group_key}:{position['x']:.3f}:{position['z']:.3f}"
        buckets.setdefault((quadrant, distance_band), []).append((_rank(key), position, distance))
    for values in buckets.values():
        values.sort(key=lambda item: item[0])
    ordered = []
    for round_index in range(12):
        for quadrant in range(4):
            for band in ("NEAR", "FAR"):
                values = buckets.get((quadrant, band), [])
                if round_index < len(values):
                    _, position, distance = values[round_index]
                    ordered.append((position, quadrant, distance))
    return ordered


def _group_rows(objects: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]], float]]:
    canonical = canonicalize_scene(objects)
    by_id = {row["objectId"]: row for row in objects}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in objects:
        coordinate = canonical.get(row["objectId"], {})
        if not is_action_target(row) or row.get("objectType") not in TARGET_TYPES or not coordinate.get("evaluable"):
            continue
        grouped.setdefault((coordinate["owner_id"], row["objectType"]), []).append(row)
    output = []
    for (owner_id, object_type), members in grouped.items():
        if len(members) < 2 or owner_id not in by_id:
            continue
        members = sorted(members, key=lambda row: row["objectId"])
        neighborhoods: dict[tuple[str, ...], list[dict[str, Any]]] = {}
        if len(members) <= 4:
            neighborhoods[tuple(row["objectId"] for row in members)] = members
        else:
            for anchor in members:
                position = anchor["position"]
                nearest = sorted(members, key=lambda row: (
                    math.sqrt(
                        (float(row["position"]["x"]) - float(position["x"])) ** 2
                        + (float(row["position"]["y"]) - float(position["y"])) ** 2
                        + (float(row["position"]["z"]) - float(position["z"])) ** 2
                    ),
                    row["objectId"],
                ))[:4]
                key = tuple(sorted(row["objectId"] for row in nearest))
                neighborhoods[key] = sorted(nearest, key=lambda row: row["objectId"])
        for neighborhood in neighborhoods.values():
            output.append((owner_id, object_type, neighborhood, float(by_id[owner_id]["rotation"]["y"])))
    return sorted(output, key=lambda item: (item[1], item[0]))


def collect(dataset: Path, manifest_path: Path, role: str, output: Path,
            house_limit: int | None = None, allow_under_minimum: bool = False,
            shard_index: int = 0, shard_count: int = 1) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_key = "test_sha256" if role == "final_test" else "train_sha256"
    if sha256_file(dataset) != manifest["source"][dataset_key]:
        raise ValueError("R1C-L dataset/manifest identity mismatch")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("R1C-L shard identity is invalid")
    full_roster = manifest["rosters"][role]
    if house_limit is not None:
        full_roster = full_roster[:house_limit]
    roster = full_roster[shard_index::shard_count]
    houses = _load_houses(dataset, roster)
    root = output / role
    partial_path = root / "collection.partial.json"
    progress_path = root / "progress.json"
    views: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial["manifest_sha256"] != sha256_file(manifest_path) or partial["dataset_sha256"] != sha256_file(dataset):
            raise ValueError("R1C-L partial identity mismatch")
        if partial.get("shard_index", 0) != shard_index or partial.get("shard_count", 1) != shard_count:
            raise ValueError("R1C-L partial shard identity mismatch")
        views, pairs, receipts = partial["views"], partial["pairs"], partial["scene_receipts"]
    completed = {int(row["house_index"]) for row in receipts}
    started = time.monotonic()
    controller = None
    try:
        for sequence, roster_row in enumerate(roster, 1):
            house_index = int(roster_row["house_index"])
            if house_index in completed:
                continue
            # AI2-THOR software rendering can accumulate FIFO instability after
            # many scene resets/actions. A per-house controller bounds recovery
            # and ensures every completed receipt owns a released Unity process.
            if controller is not None:
                controller.stop()
            controller = start_controller(houses[house_index])
            event = controller.last_event
            if not event.metadata.get("lastActionSuccess"):
                raise RuntimeError(f"R1C-L scene reset failed {house_index}")
            reachable_event = controller.step(action="GetReachablePositions")
            reachable = reachable_event.metadata.get("actionReturn") or []
            groups = _group_rows(reachable_event.metadata.get("objects", []))
            house_views_before, house_pairs_before = len(views), len(pairs)
            runtime_timeout = False
            for group_number, (owner_id, object_type, members, owner_yaw) in enumerate(groups):
                center = {
                    "x": float(np.mean([row["position"]["x"] for row in members])),
                    "z": float(np.mean([row["position"]["z"] for row in members])),
                }
                member_identity = ":".join(row["objectId"] for row in members)
                group_id = f"r1cl-{role}-h{house_index:04d}-{hashlib.sha256((owner_id + ':' + object_type + ':' + member_identity).encode()).hexdigest()[:12]}"
                group_views = []
                for candidate_number, (position, quadrant, distance) in enumerate(
                    _ranked_positions(reachable, center, owner_yaw, group_id)[:48]
                ):
                    offset = (-20.0, 0.0, 20.0)[int(_rank(f"{group_id}:{candidate_number}")[:2], 16) % 3]
                    pose = {**position, "rotation": (yaw_toward(position, center) + offset) % 360.0,
                            "horizon": 0.0, "standing": True}
                    try:
                        view_event = controller.step(action="TeleportFull", **pose)
                    except TimeoutError:
                        runtime_timeout = True
                        break
                    if not view_event.metadata.get("lastActionSuccess"):
                        continue
                    stem = f"views/{group_id}-v{len(group_views):02d}"
                    view = _save_view(view_event, members, root, stem, group_id, quadrant, distance)
                    if view is None:
                        continue
                    view["house_index"] = house_index
                    view["object_type"] = object_type
                    group_views.append(view)
                    views.append(view)
                    completed_count = len(receipts)
                    elapsed = max(time.monotonic() - started, 1e-6)
                    throughput = completed_count / elapsed
                    _atomic_json(progress_path, {
                        "phase": "collection", "completed_units": completed_count,
                        "total_units": len(roster), "throughput": throughput,
                        "eta_seconds": ((len(roster) - completed_count) / throughput) if throughput > 0 else None,
                        "last_progress_at": _now(), "status": "running", "views": len(views),
                        "pairs": len(pairs), "active_house": house_index, "active_group": group_number,
                    })
                    if len(group_views) >= int(manifest["collection"]["views_per_owner_group_maximum"]):
                        break
                for reference, query in itertools.permutations(group_views, 2):
                    bins, modes = valid_bins(reference, query)
                    if not bins:
                        continue
                    pair_id = f"{reference['view_id']}--{query['view_id']}"
                    relative_yaw = (query["camera_yaw_label_degrees"] - reference["camera_yaw_label_degrees"] + 180) % 360 - 180
                    pairs.append({
                        "pair_id": pair_id, "house_index": house_index, "group_id": group_id,
                        "object_type": object_type, "reference_view_id": reference["view_id"],
                        "query_view_id": query["view_id"], "valid_bins": bins,
                        "valid_slot_modes": modes, "relative_yaw_label_degrees": relative_yaw,
                    })
                if runtime_timeout:
                    break
            receipts.append({
                "house_index": house_index, "owner_groups": len(groups),
                "views": len(views) - house_views_before, "pairs": len(pairs) - house_pairs_before,
                "runtime_timeout": runtime_timeout,
            })
            checkpoint = {
                "schema": "blindassist_grail_r1c_l_collection_checkpoint_v1",
                "manifest_sha256": sha256_file(manifest_path), "dataset_sha256": sha256_file(dataset),
                "role": role, "shard_index": shard_index, "shard_count": shard_count,
                "scene_receipts": receipts, "views": views, "pairs": pairs,
            }
            _atomic_json(partial_path, checkpoint)
            elapsed = max(time.monotonic() - started, 1e-6)
            completed_count = len(receipts)
            throughput = completed_count / elapsed
            _atomic_json(progress_path, {
                "phase": "collection", "completed_units": completed_count, "total_units": len(roster),
                "throughput": throughput, "eta_seconds": (len(roster) - completed_count) / max(throughput, 1e-9),
                "last_progress_at": _now(), "status": "running", "views": len(views), "pairs": len(pairs),
            })
            print(json.dumps({"role": role, **receipts[-1], "total_pairs": len(pairs)}), flush=True)
            controller.stop()
            controller = None
    finally:
        if controller is not None:
            controller.stop()
    maximum = manifest["collection"]["validation_pair_range" if role == "validation" else "train_pair_range"][1] \
        if role != "final_test" else len(pairs)
    if shard_count == 1 and len(pairs) > maximum:
        pairs = sorted(pairs, key=lambda row: _rank(row["pair_id"]))[:maximum]
    minimum = manifest["collection"]["validation_pair_range" if role == "validation" else "train_pair_range"][0] \
        if role != "final_test" else 0
    if shard_count == 1 and len(pairs) < minimum and not allow_under_minimum:
        raise RuntimeError(f"R1C-L_NOT_EVALUABLE_PAIR_QUOTA role={role} pairs={len(pairs)}/{minimum}")
    result = {
        "schema": "blindassist_grail_r1c_l_collection_v1", "role": role,
        "manifest_sha256": sha256_file(manifest_path), "dataset_sha256": sha256_file(dataset),
        "shard_index": shard_index, "shard_count": shard_count,
        "houses": len(receipts), "views": views, "pairs": pairs, "scene_receipts": receipts,
        "summary": {
            "views": len(views), "pairs": len(pairs),
            "drawer_pairs": sum(row["object_type"] == "Drawer" for row in pairs),
            "doorway_pairs": sum(row["object_type"] == "Doorway" for row in pairs),
            "preserve_pairs": sum("PRESERVE" in row["valid_slot_modes"] for row in pairs),
            "flip_pairs": sum("FLIP" in row["valid_slot_modes"] for row in pairs),
        },
    }
    _atomic_json(root / "collection.json", result)
    _atomic_json(progress_path, {
        "phase": "collection", "completed_units": len(receipts), "total_units": len(roster),
        "throughput": len(receipts) / max(time.monotonic() - started, 1e-6), "eta_seconds": 0.0,
        "last_progress_at": _now(), "status": "complete", **result["summary"],
    })
    print(json.dumps(result["summary"], indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation", "final_test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--house-limit", type=int)
    parser.add_argument("--allow-under-minimum", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    collect(args.dataset, args.manifest, args.role, args.output, args.house_limit,
            args.allow_under_minimum, args.shard_index, args.shard_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
