#!/usr/bin/env python3
"""Collect and admit the frozen fresh bilateral GRAIL-R1C-P cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from collect_grail_m1 import (
    bbox_for,
    expanded_crop_array,
    load_houses,
    local_pose,
    ranked_query_positions,
    start_controller,
)
from grail_procthor_native_m0 import is_action_target, sha256_file, yaw_toward
from run_grail_procthor_native_m0 import nearby_positions


def _rank(sample_id: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{sample_id}".encode("utf-8")).hexdigest()


def _save_candidates(event: Any, target_id: str, output: Path, prefix: str) -> list[dict[str, Any]]:
    candidates = []
    for candidate in sorted(event.metadata.get("objects", []), key=lambda item: item["objectId"]):
        if not is_action_target(candidate):
            continue
        detected = bbox_for(event, candidate["objectId"])
        if detected is None:
            continue
        bbox, mask = detected
        relative = f"masks/{prefix}-{len(candidates):03d}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / relative)
        depth_values = event.depth_frame[mask]
        candidates.append({
            "object_id": candidate["objectId"],
            "object_type": candidate["objectType"],
            "bbox": bbox,
            "mask_area": int(mask.sum()),
            "mask_image": relative,
            "source_depth_median_m": float(np.median(depth_values)),
            "is_target": candidate["objectId"] == target_id,
        })
    return candidates


def _admit(rows: list[dict[str, Any]], salt: str) -> list[dict[str, Any]]:
    distractor = sorted((row for row in rows if row["same_type_visible_candidates"] >= 2),
                        key=lambda row: _rank(row["sample_id"], salt))
    clean = sorted((row for row in rows if row["same_type_visible_candidates"] < 2),
                   key=lambda row: _rank(row["sample_id"], salt))
    if len(distractor) < 43 or len(clean) < 35:
        raise RuntimeError(
            f"R1C-P_NOT_EVALUABLE_ADMISSION_QUOTA distractor={len(distractor)}/43 clean={len(clean)}/35"
        )
    return sorted(distractor[:43] + clean[:35], key=lambda row: row["sample_id"])


def collect(dataset: Path, manifest_path: Path, output: Path, docker_image_id: str) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != manifest["source"]["val_sha256"]:
        raise ValueError("R1C-P dataset identity mismatch")
    roster = manifest["source"]["fresh_house_roster"]
    houses = load_houses(dataset, roster)
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "masks").mkdir(parents=True, exist_ok=True)
    partial_path = output / "collection.partial.json"
    manifest_hash = sha256_file(manifest_path)
    rows, receipts = [], []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial["manifest_sha256"] != manifest_hash or partial["dataset_sha256"] != sha256_file(dataset):
            raise ValueError("R1C-P partial identity mismatch")
        rows, receipts = partial["rows"], partial["scene_receipts"]
    completed = {int(receipt["house_index"]) for receipt in receipts}
    controller = None
    try:
        for roster_row in roster:
            house_index = int(roster_row["house_index"])
            if house_index in completed:
                print(json.dumps({"house_index": house_index, "state": "RESUME_SKIP"}), flush=True)
                continue
            event = controller.reset(scene=houses[house_index]) if controller else None
            if controller is None:
                controller = start_controller(houses[house_index])
                event = controller.last_event
            if not event.metadata.get("lastActionSuccess"):
                raise RuntimeError(f"scene reset failed {house_index}")
            reachable_event = controller.step(action="GetReachablePositions")
            reachable = reachable_event.metadata.get("actionReturn") or []
            objects = sorted(
                (record for record in reachable_event.metadata.get("objects", []) if is_action_target(record)),
                key=lambda record: record["objectId"],
            )
            before = len(rows)
            for obj in objects:
                positions = nearby_positions(reachable, obj["position"])
                if not positions:
                    continue
                pose_event = controller.step(
                    action="GetInteractablePoses", objectId=obj["objectId"], positions=positions,
                    rotations=list(range(0, 360, 30)), horizons=[0], standings=[True],
                )
                truth = pose_event.metadata.get("actionReturn") or []
                if not truth:
                    continue
                reference_pose = sorted(
                    truth, key=lambda pose: (float(pose["x"]), float(pose["z"]), float(pose["rotation"]))
                )[0]
                reference_event = controller.step(action="TeleportFull", **reference_pose)
                reference_target = bbox_for(reference_event, obj["objectId"])
                if reference_target is None:
                    continue
                sample_key = f"r1cp:{house_index}:{obj['objectId']}"
                query_event = None
                for position in ranked_query_positions(reachable, obj["position"], sample_key)[:32]:
                    center_yaw = round(yaw_toward(position, obj["position"]) / 30.0) * 30.0
                    rotations = [(center_yaw + delta) % 360.0 for delta in (-30.0, 0.0, 30.0)]
                    rotations.sort(key=lambda yaw: hashlib.sha256(
                        f"{sample_key}:{position['x']:.3f}:{position['z']:.3f}:{yaw:.1f}".encode()
                    ).hexdigest())
                    query_pose = {**position, "rotation": rotations[0], "horizon": 0.0, "standing": True}
                    candidate_event = controller.step(action="TeleportFull", **query_pose)
                    if bbox_for(candidate_event, obj["objectId"]) is not None:
                        query_event = candidate_event
                        break
                if query_event is None:
                    continue
                sample_id = f"r1cp-h{house_index:04d}-{hashlib.sha256(obj['objectId'].encode()).hexdigest()[:12]}"
                query_relative = f"images/{sample_id}-query.png"
                reference_full_relative = f"images/{sample_id}-reference-full.png"
                reference_crop_relative = f"images/{sample_id}-reference.png"
                Image.fromarray(query_event.frame).save(output / query_relative)
                Image.fromarray(reference_event.frame).save(output / reference_full_relative)
                Image.fromarray(expanded_crop_array(reference_event.frame, reference_target[0])).save(
                    output / reference_crop_relative
                )
                query_candidates = _save_candidates(query_event, obj["objectId"], output, f"{sample_id}-query")
                reference_candidates = _save_candidates(
                    reference_event, obj["objectId"], output, f"{sample_id}-reference"
                )
                if sum(candidate["is_target"] for candidate in query_candidates) != 1 or \
                        sum(candidate["is_target"] for candidate in reference_candidates) != 1:
                    continue
                camera = query_event.metadata["agent"]
                same_type = sum(candidate["object_type"] == obj["objectType"] for candidate in query_candidates)
                rows.append({
                    "sample_id": sample_id,
                    "house_index": house_index,
                    "target_object_id": obj["objectId"],
                    "target_type": obj["objectType"],
                    "query_image": query_relative,
                    "reference_image": reference_crop_relative,
                    "reference_full_image": reference_full_relative,
                    "camera": camera,
                    "candidates": query_candidates,
                    "reference_candidates": reference_candidates,
                    "truth_local_poses": [local_pose(pose, camera) for pose in truth],
                    "same_type_visible_candidates": same_type,
                })
            receipts.append({"house_index": house_index, "examples": len(rows) - before,
                             "actionable_objects": len(objects)})
            checkpoint = {
                "schema": "blindassist_grail_r1c_p_collection_checkpoint_v1",
                "manifest_sha256": manifest_hash,
                "dataset_sha256": sha256_file(dataset),
                "scene_receipts": receipts,
                "rows": rows,
            }
            temporary = partial_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8")
            temporary.replace(partial_path)
            print(json.dumps({"house_index": house_index, "state": "CHECKPOINTED", "examples": len(rows)}),
                  flush=True)
    finally:
        if controller is not None:
            controller.stop()
    admitted = _admit(rows, manifest["collection"]["admission_salt"])
    result = {
        "schema": "blindassist_grail_r1c_p_fresh_collection_v1",
        "manifest_sha256": manifest_hash,
        "dataset_sha256": sha256_file(dataset),
        "runtime": {"docker_image_id": docker_image_id, "ai2thor_release": "f0825767cd50d69f666c7f282e54abfe58f1e917"},
        "source_rows": len(rows),
        "scene_receipts": receipts,
        "examples": len(admitted),
        "wrong_target_examples": sum(row["same_type_visible_candidates"] >= 2 for row in admitted),
        "rows": admitted,
    }
    (output / "collection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("source_rows", "examples", "wrong_target_examples")}, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-image-id", required=True)
    args = parser.parse_args()
    collect(args.dataset, args.manifest, args.output, args.docker_image_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
