#!/usr/bin/env python3
"""Collect RGB/reference/candidate/truth records from frozen ProcTHOR M1 houses."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from grail_procthor_native_m0 import canonical_sha256, is_action_target, sha256_file, yaw_toward
from run_grail_procthor_native_m0 import nearby_positions


def load_houses(path: Path, roster: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    wanted = {int(row["house_index"]) for row in roster}
    houses = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index in wanted:
                houses[index] = json.loads(line)
            if len(houses) == len(wanted):
                break
    if set(houses) != wanted:
        raise ValueError("dataset lacks frozen houses")
    for row in roster:
        if canonical_sha256(houses[int(row["house_index"])]) != row["house_sha256"]:
            raise ValueError(f"house hash mismatch at {row['house_index']}")
    return houses


def start_controller(scene: dict[str, Any]) -> Any:
    from ai2thor.controller import Controller
    from ai2thor.platform import Linux64
    from ai2thor.util.lock import Lock
    Lock.lock = lambda self: None
    return Controller(
        scene=scene, platform=Linux64, width=320, height=240, fieldOfView=90,
        gridSize=0.25, snapToGrid=False, rotateStepDegrees=30, visibilityDistance=1.5,
        renderDepthImage=True, renderInstanceSegmentation=True,
    )


def local_pose(pose: dict[str, Any], camera: dict[str, Any]) -> dict[str, float]:
    yaw = math.radians(float(camera["rotation"]["y"]))
    dx = float(pose["x"]) - float(camera["position"]["x"])
    dz = float(pose["z"]) - float(camera["position"]["z"])
    return {
        "x": dx * math.cos(yaw) - dz * math.sin(yaw),
        "z": dx * math.sin(yaw) + dz * math.cos(yaw),
        "yaw": (float(pose["rotation"]) - float(camera["rotation"]["y"]) + 180.0) % 360.0 - 180.0,
    }


def bbox_for(event: Any, object_id: str) -> tuple[list[int], np.ndarray] | None:
    mask = event.instance_masks.get(object_id) if event.instance_masks else None
    if mask is None or int(mask.sum()) < 36:
        return None
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1], mask


def save_crop(frame: np.ndarray, bbox: list[int], path: Path) -> None:
    x0, y0, x1, y1 = bbox
    # Tiny actionable parts (especially drawers) need stable object context rather
    # than an upscaled handful of pixels. The same expansion is used at feature time.
    pad_x, pad_y = max(32, x1 - x0), max(32, y1 - y0)
    crop = frame[max(0, y0-pad_y):min(frame.shape[0], y1+pad_y), max(0, x0-pad_x):min(frame.shape[1], x1+pad_x)]
    Image.fromarray(crop).save(path)


def ranked_query_positions(reachable: list[dict[str, float]], target: dict[str, float], key: str) -> list[dict[str, float]]:
    candidates = [p for p in reachable if 1.75 <= math.hypot(float(p["x"])-float(target["x"]), float(p["z"])-float(target["z"])) <= 4.0]
    return sorted(candidates, key=lambda p: hashlib.sha256(f"{key}:{p['x']:.3f}:{p['z']:.3f}".encode()).hexdigest())


def collect(dataset: Path, manifest_path: Path, role: str, output: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest["source"][f"{role if role != 'train' and role != 'dev' else 'val'}_sha256"]
    if sha256_file(dataset) != expected_hash:
        raise ValueError("dataset hash mismatch")
    roster = manifest["rosters"][role]
    houses = load_houses(dataset, roster)
    image_dir = output / role / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output / role / "collection.partial.json"
    manifest_hash, dataset_hash = sha256_file(manifest_path), sha256_file(dataset)
    rows, scene_receipts = [], []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("manifest_sha256") != manifest_hash or partial.get("dataset_sha256") != dataset_hash:
            raise ValueError("partial collection identity mismatch")
        rows, scene_receipts = partial["rows"], partial["scene_receipts"]
    completed_houses = {int(receipt["house_index"]) for receipt in scene_receipts}
    controller = None
    try:
        for roster_row in roster:
            house_index = int(roster_row["house_index"])
            if house_index in completed_houses:
                print(json.dumps({"role": role, "house_index": house_index, "state": "RESUME_SKIP"}), flush=True)
                continue
            event = controller.reset(scene=houses[house_index]) if controller else None
            if controller is None:
                controller = start_controller(houses[house_index]); event = controller.last_event
            if not event.metadata.get("lastActionSuccess"):
                raise RuntimeError(f"scene reset failed {house_index}")
            reachable_event = controller.step(action="GetReachablePositions")
            reachable = reachable_event.metadata.get("actionReturn") or []
            objects = [o for o in reachable_event.metadata.get("objects", []) if is_action_target(o)]
            objects.sort(key=lambda o: o["objectId"])
            before = len(rows)
            for obj in objects:
                positions = nearby_positions(reachable, obj["position"])
                if not positions:
                    continue
                pose_event = controller.step(action="GetInteractablePoses", objectId=obj["objectId"], positions=positions,
                                             rotations=list(range(0, 360, 30)), horizons=[0], standings=[True])
                truth = pose_event.metadata.get("actionReturn") or []
                if not truth:
                    continue
                ref_pose = sorted(truth, key=lambda p: (float(p["x"]), float(p["z"]), float(p["rotation"])))[0]
                ref_event = controller.step(action="TeleportFull", **ref_pose)
                ref_bbox_mask = bbox_for(ref_event, obj["objectId"])
                if ref_bbox_mask is None:
                    continue
                query_event = None
                for query_position in ranked_query_positions(reachable, obj["position"], f"{role}:{house_index}:{obj['objectId']}")[:32]:
                    query_pose = {**query_position, "rotation": yaw_toward(query_position, obj["position"]), "horizon": 0.0, "standing": True}
                    candidate_event = controller.step(action="TeleportFull", **query_pose)
                    if bbox_for(candidate_event, obj["objectId"]) is not None:
                        query_event = candidate_event
                        break
                if query_event is None:
                    continue
                sample_id = f"{role}-h{house_index:04d}-{hashlib.sha256(obj['objectId'].encode()).hexdigest()[:12]}"
                query_rel = f"{role}/images/{sample_id}-query.png"
                ref_rel = f"{role}/images/{sample_id}-reference.png"
                Image.fromarray(query_event.frame).save(output / query_rel)
                save_crop(ref_event.frame, ref_bbox_mask[0], output / ref_rel)
                visible_candidates = []
                for candidate in query_event.metadata.get("objects", []):
                    if not is_action_target(candidate):
                        continue
                    detected = bbox_for(query_event, candidate["objectId"])
                    if detected is None:
                        continue
                    bbox, mask = detected
                    depth_values = query_event.depth_frame[mask]
                    visible_candidates.append({
                        "object_id": candidate["objectId"], "object_type": candidate["objectType"],
                        "bbox": bbox, "mask_area": int(mask.sum()),
                        "source_depth_median_m": float(np.median(depth_values)),
                        "is_target": candidate["objectId"] == obj["objectId"],
                    })
                if not any(c["is_target"] for c in visible_candidates):
                    continue
                camera = query_event.metadata["agent"]
                rows.append({
                    "sample_id": sample_id, "role": role, "house_index": house_index,
                    "target_object_id": obj["objectId"], "target_type": obj["objectType"],
                    "query_image": query_rel, "reference_image": ref_rel,
                    "camera": camera, "candidates": visible_candidates,
                    "truth_local_poses": [local_pose(p, camera) for p in truth],
                    "same_type_visible_candidates": sum(c["object_type"] == obj["objectType"] for c in visible_candidates),
                })
            scene_receipts.append({"house_index": house_index, "examples": len(rows)-before, "actionable_objects": len(objects)})
            checkpoint = {
                "schema": "blindassist_grail_m1_collection_checkpoint_v1", "role": role,
                "manifest_sha256": manifest_hash, "dataset_sha256": dataset_hash,
                "scene_receipts": scene_receipts, "rows": rows,
            }
            temporary = partial_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
            temporary.replace(partial_path)
            print(json.dumps({"role": role, "house_index": house_index, "state": "CHECKPOINTED", "examples": len(rows)}), flush=True)
    finally:
        if controller is not None:
            controller.stop()
    report = {
        "schema": "blindassist_grail_m1_collection_v1", "role": role,
        "manifest_sha256": manifest_hash, "dataset_sha256": dataset_hash,
        "examples": len(rows), "wrong_target_examples": sum(r["same_type_visible_candidates"] >= 2 for r in rows),
        "scene_receipts": scene_receipts, "rows": rows,
    }
    (output / role).mkdir(parents=True, exist_ok=True)
    (output / role / "collection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("role", "examples", "wrong_target_examples", "scene_receipts")}, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.dataset, args.manifest, args.role, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
