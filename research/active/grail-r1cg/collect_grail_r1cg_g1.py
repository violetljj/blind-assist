#!/usr/bin/env python3
"""Collect fixed anchor-camera-frame triplets for R1C-G1."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


SCHEMA = "blindassist_grail_r1c_g1_collection_v1"


def _load_r1cl(path: Path) -> Any:
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("grail_r1cl_collector_for_g1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load frozen R1C-L mechanics: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _rank(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _load_houses(dataset: Path, roster: list[dict[str, Any]], r1cl: Any) -> dict[int, dict[str, Any]]:
    wanted = {int(row["house_index"]) for row in roster}
    houses: dict[int, dict[str, Any]] = {}
    with gzip.open(dataset, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                houses[index] = json.loads(line)
            if len(houses) == len(wanted):
                break
    if set(houses) != wanted:
        raise ValueError("R1C-G1 dataset lacks frozen houses")
    for row in roster:
        index = int(row["house_index"])
        if r1cl.canonical_sha256(houses[index]) != row["house_sha256"]:
            raise ValueError(f"R1C-G1 house hash mismatch at {index}")
    return houses


def _groups_without_orientation(objects: list[dict[str, Any]], r1cl: Any) -> list[tuple[str, str, list[dict[str, Any]]]]:
    """Build evaluator owner groups without reading the owner's rotation."""
    canonical = r1cl.canonicalize_scene(objects)
    by_id = {row["objectId"]: row for row in objects}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in objects:
        coordinate = canonical.get(row["objectId"], {})
        if (not r1cl.is_action_target(row) or row.get("objectType") not in r1cl.TARGET_TYPES
                or not coordinate.get("evaluable")):
            continue
        grouped.setdefault((coordinate["owner_id"], row["objectType"]), []).append(row)
    output: list[tuple[str, str, list[dict[str, Any]]]] = []
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
            output.append((owner_id, object_type, neighborhood))
    return sorted(output, key=lambda item: (item[1], item[0], tuple(row["objectId"] for row in item[2])))


def _candidate_anchors(reachable: list[dict[str, float]], center: dict[str, float],
                       group_id: str) -> list[dict[str, float]]:
    """Diverse world-frame anchors; owner orientation is intentionally absent."""
    buckets: dict[tuple[int, str], list[tuple[str, dict[str, float]]]] = {}
    for position in reachable:
        dx = float(position["x"]) - center["x"]
        dz = float(position["z"]) - center["z"]
        distance = math.hypot(dx, dz)
        if not 1.5 <= distance <= 4.0:
            continue
        world_sector = int((math.degrees(math.atan2(dx, dz)) % 360.0) // 45.0)
        band = "NEAR" if distance < 2.75 else "FAR"
        rank = _rank(f"{group_id}:anchor:{position['x']:.3f}:{position['z']:.3f}")
        buckets.setdefault((world_sector, band), []).append((rank, position))
    for rows in buckets.values():
        rows.sort(key=lambda item: item[0])
    ordered: list[dict[str, float]] = []
    for round_index in range(16):
        for sector in range(8):
            for band in ("NEAR", "FAR"):
                rows = buckets.get((sector, band), [])
                if round_index < len(rows):
                    ordered.append(rows[round_index][1])
    return ordered


def _side_position(anchor: dict[str, float], reachable: list[dict[str, float]],
                   center: dict[str, float], side: int, group_id: str) -> tuple[dict[str, float], float, float] | None:
    forward_x = center["x"] - float(anchor["x"])
    forward_z = center["z"] - float(anchor["z"])
    length = math.hypot(forward_x, forward_z)
    if length < 1e-8:
        return None
    forward_x, forward_z = forward_x / length, forward_z / length
    right_x, right_z = forward_z, -forward_x
    options: list[tuple[float, str, dict[str, float], float, float]] = []
    for position in reachable:
        rel_x = float(position["x"]) - float(anchor["x"])
        rel_z = float(position["z"]) - float(anchor["z"])
        lateral = rel_x * right_x + rel_z * right_z
        longitudinal = rel_x * forward_x + rel_z * forward_z
        signed = lateral * side
        if not 0.20 <= signed <= 0.45 or abs(longitudinal) > 0.20:
            continue
        score = abs(signed - 0.30) + 0.5 * abs(longitudinal)
        rank = _rank(f"{group_id}:side:{side}:{position['x']:.3f}:{position['z']:.3f}")
        options.append((score, rank, position, lateral, longitudinal))
    if not options:
        return None
    _, _, position, lateral, longitudinal = min(options, key=lambda row: (row[0], row[1]))
    return position, lateral, longitudinal


def _crop_box(detections: list[tuple[list[int], np.ndarray]], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    x0 = min(item[0][0] for item in detections)
    y0 = min(item[0][1] for item in detections)
    x1 = max(item[0][2] for item in detections)
    y1 = max(item[0][3] for item in detections)
    width, height = max(x1 - x0, 1), max(y1 - y0, 1)
    pad_x, pad_y = max(8, round(width * 0.20)), max(8, round(height * 0.20))
    return max(0, x0 - pad_x), max(0, y0 - pad_y), min(shape[1], x1 + pad_x), min(shape[0], y1 + pad_y)


def _save_view(event: Any, members: list[dict[str, Any]], role_root: Path, relative_stem: str,
               group_id: str, scan_id: str, scan_role: str, r1cl: Any) -> dict[str, Any] | None:
    visible: list[tuple[dict[str, Any], list[int], np.ndarray]] = []
    for member in members:
        detected = r1cl.bbox_for(event, member["objectId"])
        if detected is not None:
            visible.append((member, detected[0], detected[1]))
    if not 2 <= len(visible) <= 4:
        return None
    x0, y0, x1, y1 = _crop_box([(bbox, mask) for _, bbox, mask in visible], event.frame.shape)
    rgb = Image.fromarray(event.frame[y0:y1, x0:x1])
    union = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    member_rows: list[dict[str, Any]] = []
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
    (role_root / rgb_path).parent.mkdir(parents=True, exist_ok=True)
    rgb.save(role_root / rgb_path)
    Image.fromarray(union).save(role_root / union_path)
    centroid.save(role_root / centroid_path)
    return {
        "view_id": Path(relative_stem).name,
        "group_id": group_id,
        "scan_id": scan_id,
        "scan_role": scan_role,
        "rgb": rgb_path,
        "owner_union_mask": union_path,
        "sibling_centroid_mask": centroid_path,
        "width": rgb.width,
        "height": rgb.height,
        "members": sorted(member_rows, key=lambda row: row["object_id"]),
    }


def _teleport_view(controller: Any, position: dict[str, float], center: dict[str, float],
                   members: list[dict[str, Any]], role_root: Path, stem: str, group_id: str,
                   scan_id: str, scan_role: str, r1cl: Any) -> dict[str, Any] | None:
    pose = {
        **position,
        "rotation": r1cl.yaw_toward(position, center) % 360.0,
        "horizon": 0.0,
        "standing": True,
    }
    event = controller.step(action="TeleportFull", **pose)
    if not event.metadata.get("lastActionSuccess"):
        return None
    return _save_view(event, members, role_root, stem, group_id, scan_id, scan_role, r1cl)


def collect(dataset: Path, manifest_path: Path, role: str, output: Path, r1cl_path: Path,
            shard_index: int, shard_count: int, house_limit: int | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blindassist_grail_r1c_g1_manifest_v1":
        raise ValueError("R1C-G1 manifest schema mismatch")
    r1cl = _load_r1cl(r1cl_path.resolve())
    if r1cl.sha256_file(dataset) != manifest["source"]["train_sha256"]:
        raise ValueError("R1C-G1 dataset/manifest identity mismatch")
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("R1C-G1 shard identity is invalid")
    full_roster = manifest["rosters"][role]
    if house_limit is not None:
        full_roster = full_roster[:house_limit]
    roster = full_roster[shard_index::shard_count]
    houses = _load_houses(dataset, roster, r1cl)
    role_root = output / role
    shard_name = f"shard-{shard_index:02d}-of-{shard_count:02d}"
    shard_root = role_root / shard_name
    partial_path = shard_root / "collection.partial.json"
    progress_path = shard_root / "progress.json"
    views: list[dict[str, Any]] = []
    scan_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if (partial["manifest_sha256"] != r1cl.sha256_file(manifest_path)
                or partial["dataset_sha256"] != r1cl.sha256_file(dataset)
                or partial["shard_index"] != shard_index or partial["shard_count"] != shard_count):
            raise ValueError("R1C-G1 partial identity mismatch")
        views = partial["views"]
        scan_rows = partial.get("scans", [])
        samples, receipts = partial["samples"], partial["scene_receipts"]
    completed = {int(row["house_index"]) for row in receipts}
    started = time.monotonic()
    controller = None
    try:
        for roster_row in roster:
            house_index = int(roster_row["house_index"])
            if house_index in completed:
                continue
            if controller is not None:
                controller.stop()
            controller = r1cl.start_controller(houses[house_index])
            reachable_event = controller.step(action="GetReachablePositions")
            reachable = reachable_event.metadata.get("actionReturn") or []
            groups = _groups_without_orientation(reachable_event.metadata.get("objects", []), r1cl)
            before_views, before_samples = len(views), len(samples)
            attempted_scans = valid_scans = omitted_scans = 0
            runtime_timeout = False
            for owner_id, object_type, members in groups:
                center = {
                    "x": float(np.mean([row["position"]["x"] for row in members])),
                    "z": float(np.mean([row["position"]["z"] for row in members])),
                }
                identities = ":".join(row["objectId"] for row in members)
                digest = hashlib.sha256((owner_id + ":" + object_type + ":" + identities).encode()).hexdigest()[:12]
                group_id = f"r1cg1-{role}-h{house_index:04d}-{digest}"
                scans: list[dict[str, Any]] = []
                for anchor_number, anchor in enumerate(_candidate_anchors(reachable, center, group_id)[:64]):
                    if len(scans) >= int(manifest["collection"]["scans_per_owner_group_maximum"]):
                        break
                    left = _side_position(anchor, reachable, center, -1, group_id)
                    right = _side_position(anchor, reachable, center, 1, group_id)
                    if left is None or right is None or left[0] == right[0]:
                        continue
                    attempted_scans += 1
                    scan_id = f"{group_id}-s{len(scans):02d}"
                    prefix = f"{shard_name}/views/{scan_id}"
                    created: list[dict[str, Any]] = []
                    try:
                        for scan_role, position in (("anchor", anchor), ("left", left[0]), ("right", right[0])):
                            view = _teleport_view(
                                controller, position, center, members, role_root,
                                f"{prefix}-{scan_role}", group_id, scan_id, scan_role, r1cl,
                            )
                            if view is None:
                                break
                            view["house_index"] = house_index
                            view["object_type"] = object_type
                            created.append(view)
                    except TimeoutError:
                        runtime_timeout = True
                        break
                    if len(created) != 3:
                        omitted_scans += 1
                        continue
                    scan = {
                        "scan_id": scan_id,
                        "group_id": group_id,
                        "house_index": house_index,
                        "object_type": object_type,
                        "reference_view_ids": [row["view_id"] for row in created],
                        "anchor_view_id": created[0]["view_id"],
                        "acquisition_geometry_audit_only": {
                            "left_lateral_m": left[1], "left_longitudinal_m": left[2],
                            "right_lateral_m": right[1], "right_longitudinal_m": right[2],
                        },
                    }
                    scans.append(scan)
                    scan_rows.append(scan)
                    views.extend(created)
                    valid_scans += 1
                for reference_scan, query_scan in itertools.permutations(scans, 2):
                    anchor_view = next(row for row in views if row["view_id"] == reference_scan["anchor_view_id"])
                    query_view = next(row for row in views if row["view_id"] == query_scan["anchor_view_id"])
                    _, modes = r1cl.valid_bins(anchor_view, query_view)
                    if not modes:
                        continue
                    sample_id = f"{reference_scan['scan_id']}--q-{query_view['view_id']}"
                    samples.append({
                        "sample_id": sample_id,
                        "house_index": house_index,
                        "group_id": group_id,
                        "object_type": object_type,
                        "reference_scan_id": reference_scan["scan_id"],
                        "reference_view_ids": reference_scan["reference_view_ids"],
                        "anchor_view_id": reference_scan["anchor_view_id"],
                        "query_view_id": query_view["view_id"],
                        "valid_slot_modes": modes,
                    })
                if runtime_timeout:
                    break
            receipts.append({
                "house_index": house_index,
                "owner_groups": len(groups),
                "views": len(views) - before_views,
                "samples": len(samples) - before_samples,
                "attempted_scans": attempted_scans,
                "valid_scans": valid_scans,
                "omitted_scans": omitted_scans,
                "runtime_timeout": runtime_timeout,
            })
            checkpoint = {
                "schema": "blindassist_grail_r1c_g1_collection_checkpoint_v1",
                "manifest_sha256": r1cl.sha256_file(manifest_path),
                "dataset_sha256": r1cl.sha256_file(dataset),
                "role": role, "shard_index": shard_index, "shard_count": shard_count,
                "scene_receipts": receipts, "views": views, "scans": scan_rows, "samples": samples,
            }
            _atomic_json(partial_path, checkpoint)
            elapsed = max(time.monotonic() - started, 1e-6)
            _atomic_json(progress_path, {
                "phase": "collection", "completed_units": len(receipts), "total_units": len(roster),
                "throughput": len(receipts) / elapsed,
                "eta_seconds": (len(roster) - len(receipts)) / max(len(receipts) / elapsed, 1e-9),
                "last_progress_at": _now(), "status": "running", "views": len(views), "samples": len(samples),
            })
            print(json.dumps({"role": role, "shard": shard_index, **receipts[-1],
                              "total_samples": len(samples)}), flush=True)
            controller.stop()
            controller = None
    finally:
        if controller is not None:
            controller.stop()
    result = {
        "schema": SCHEMA,
        "manifest_sha256": r1cl.sha256_file(manifest_path),
        "dataset_sha256": r1cl.sha256_file(dataset),
        "role": role, "shard_index": shard_index, "shard_count": shard_count,
        "houses": len(receipts), "views": views, "scans": scan_rows,
        "samples": samples, "scene_receipts": receipts,
        "summary": {
            "views": len(views), "samples": len(samples),
            "discriminative_samples": sum(len(row["valid_slot_modes"]) == 1 for row in samples),
            "ambiguous_samples": sum(len(row["valid_slot_modes"]) > 1 for row in samples),
            "flip_only_samples": sum(row["valid_slot_modes"] == ["FLIP"] for row in samples),
            "preserve_only_samples": sum(row["valid_slot_modes"] == ["PRESERVE"] for row in samples),
            "drawer_samples": sum(row["object_type"] == "Drawer" for row in samples),
            "doorway_samples": sum(row["object_type"] == "Doorway" for row in samples),
            "runtime_timeouts": sum(row["runtime_timeout"] for row in receipts),
        },
        "leakage_audit": {
            "owner_yaw_read_for_view_selection": False,
            "canonical_sign_read_for_view_selection": False,
            "camera_or_owner_pose_in_model_input": False,
            "side_views_selected_in_anchor_camera_frame": True,
            "duplicate_anchor_substitution": False,
        },
    }
    _atomic_json(shard_root / "collection.json", result)
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
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r1cl-collector", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--house-limit", type=int)
    args = parser.parse_args()
    collect(args.dataset, args.manifest, args.role, args.output, args.r1cl_collector,
            args.shard_index, args.shard_count, args.house_limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
