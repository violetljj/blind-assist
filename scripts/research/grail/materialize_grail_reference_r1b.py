#!/usr/bin/env python3
"""Re-materialize full-scene reference RGB and oracle proposals for GRAIL-R1B."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from collect_grail_m1 import bbox_for, expanded_crop_array, start_controller
from grail_procthor_native_m0 import is_action_target, sha256_file
from grail_relational_r0 import load_houses
from run_grail_procthor_native_m0 import nearby_positions


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def materialize(
    dataset: Path, collection_path: Path, collection_root: Path, output: Path,
    docker_image_id: str, dockerfile_sha256: str,
) -> dict[str, Any]:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != collection["dataset_sha256"]:
        raise ValueError("ProcTHOR val identity mismatch")
    rows = collection["rows"]
    if len(rows) != 78:
        raise ValueError(f"R1B requires frozen 78-case Development cohort, got {len(rows)}")
    houses = load_houses(dataset, {int(row["house_index"]) for row in rows})
    image_dir, mask_dir = output / "images", output / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    partial_path = output / "reference-supplement.partial.json"
    completed: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial.get("collection_sha256") != sha256_file(collection_path):
            raise ValueError("R1B partial collection identity mismatch")
        completed = {row["sample_id"]: row for row in partial["rows"]}

    controller = None
    output_rows: list[dict[str, Any]] = []
    try:
        current_house = None
        reachable: list[dict[str, float]] = []
        objects_by_id: dict[str, dict[str, Any]] = {}
        for number, row in enumerate(rows, 1):
            if row["sample_id"] in completed:
                output_rows.append(completed[row["sample_id"]])
                continue
            house_index = int(row["house_index"])
            if current_house != house_index:
                event = controller.reset(scene=houses[house_index]) if controller else None
                if controller is None:
                    controller = start_controller(houses[house_index])
                    event = controller.last_event
                if not event.metadata.get("lastActionSuccess"):
                    raise RuntimeError(f"scene reset failed {house_index}")
                reachable_event = controller.step(action="GetReachablePositions")
                reachable = reachable_event.metadata.get("actionReturn") or []
                objects_by_id = {obj["objectId"]: obj for obj in reachable_event.metadata.get("objects", [])}
                current_house = house_index
            target_id = row["target_object_id"]
            if target_id not in objects_by_id:
                raise ValueError(f"target absent after scene reset: {target_id}")
            target = objects_by_id[target_id]
            positions = nearby_positions(reachable, target["position"])
            pose_event = controller.step(
                action="GetInteractablePoses", objectId=target_id, positions=positions,
                rotations=list(range(0, 360, 30)), horizons=[0], standings=[True],
            )
            truth = pose_event.metadata.get("actionReturn") or []
            if not truth:
                raise ValueError(f"no reference pose for {row['sample_id']}")
            reference_pose = sorted(
                truth, key=lambda pose: (float(pose["x"]), float(pose["z"]), float(pose["rotation"]))
            )[0]
            reference_event = controller.step(action="TeleportFull", **reference_pose)
            target_detection = bbox_for(reference_event, target_id)
            if target_detection is None:
                raise ValueError(f"target not visible at frozen reference pose: {row['sample_id']}")
            target_bbox, _ = target_detection
            frozen_crop = np.asarray(Image.open(collection_root / row["reference_image"]).convert("RGB"))
            reproduced_crop = expanded_crop_array(reference_event.frame, target_bbox)
            crop_equal = bool(frozen_crop.shape == reproduced_crop.shape and np.array_equal(frozen_crop, reproduced_crop))
            crop_mean_abs_error = None
            crop_max_abs_error = None
            crop_within_replay_tolerance = False
            if frozen_crop.shape == reproduced_crop.shape:
                crop_delta = np.abs(frozen_crop.astype(np.int16) - reproduced_crop.astype(np.int16))
                crop_mean_abs_error = float(crop_delta.mean())
                crop_max_abs_error = int(crop_delta.max())
                crop_within_replay_tolerance = crop_mean_abs_error <= 0.5 and crop_max_abs_error <= 3
            if not crop_within_replay_tolerance:
                drift_dir = output / "reproduction-drift"
                drift_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(reproduced_crop).save(drift_dir / f"{row['sample_id']}-reproduced.png")
                Image.fromarray(reference_event.frame).save(drift_dir / f"{row['sample_id']}-full.png")
                raise ValueError(
                    f"reference reproduction drift: {row['sample_id']} "
                    f"frozen_shape={frozen_crop.shape} reproduced_shape={reproduced_crop.shape} "
                    f"frozen_sha256={_array_sha256(frozen_crop)} "
                    f"reproduced_sha256={_array_sha256(reproduced_crop)} "
                    f"mean_abs_error={crop_mean_abs_error} max_abs_error={crop_max_abs_error}"
                )

            full_relative = f"images/{row['sample_id']}-reference-full.png"
            Image.fromarray(reference_event.frame).save(output / full_relative)
            candidates = []
            visible_objects = sorted(reference_event.metadata.get("objects", []), key=lambda item: item["objectId"])
            for candidate in visible_objects:
                if not is_action_target(candidate):
                    continue
                detected = bbox_for(reference_event, candidate["objectId"])
                if detected is None:
                    continue
                bbox, mask = detected
                mask_relative = f"masks/{row['sample_id']}-{len(candidates):03d}.png"
                Image.fromarray(mask.astype(np.uint8) * 255).save(output / mask_relative)
                candidates.append({
                    "object_id": candidate["objectId"],
                    "object_type": candidate["objectType"],
                    "bbox": bbox,
                    "mask_area": int(mask.sum()),
                    "mask_image": mask_relative,
                    "mask_array_sha256": _array_sha256(mask),
                    "is_target": candidate["objectId"] == target_id,
                })
            if sum(candidate["is_target"] for candidate in candidates) != 1:
                raise ValueError(f"reference target proposal count != 1: {row['sample_id']}")
            output_rows.append({
                "sample_id": row["sample_id"],
                "house_index": house_index,
                "target_object_id": target_id,
                "target_type": row["target_type"],
                "reference_full_image": full_relative,
                "reference_frame_array_sha256": _array_sha256(reference_event.frame),
                "frozen_reference_crop_pixel_equal": crop_equal,
                "frozen_reference_crop_within_replay_tolerance": crop_within_replay_tolerance,
                "frozen_reference_crop_mean_abs_error": crop_mean_abs_error,
                "frozen_reference_crop_max_abs_error": crop_max_abs_error,
                "target_reference_bbox": target_bbox,
                "candidates": candidates,
            })
            partial = {
                "schema": "blindassist_grail_r1b_reference_supplement_checkpoint_v1",
                "collection_sha256": sha256_file(collection_path),
                "rows": output_rows,
            }
            temporary = partial_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
            temporary.replace(partial_path)
            print(json.dumps({"state": "REFERENCE_SUPPLEMENT", "completed": number, "total": len(rows)}), flush=True)
    finally:
        if controller is not None:
            controller.stop()

    result = {
        "schema": "blindassist_grail_r1b_reference_supplement_v1",
        "mode": "PROJECT_CONSUMED_DEVELOPMENT_SOURCE_AUGMENTATION",
        "runtime": {
            "docker_image_id": docker_image_id,
            "dockerfile_sha256": dockerfile_sha256,
            "ai2thor_release": "f0825767cd50d69f666c7f282e54abfe58f1e917",
            "platform": "Linux64/Xvfb/Mesa software GL/FIFO",
            "official_release_zip_sha256": "4712a43a510d1f4e50a388958d046acd31a87f1c9d8d5ed116f73dc4b17265ad",
            "replay_equivalence_gate": "same RGB crop shape, mean absolute channel error <=0.5/255, max absolute channel error <=3/255",
        },
        "dataset_sha256": sha256_file(dataset),
        "collection_sha256": sha256_file(collection_path),
        "rows": output_rows,
        "summary": {
            "rows": len(output_rows),
            "crop_pixel_equal": sum(row["frozen_reference_crop_pixel_equal"] for row in output_rows),
            "crop_within_replay_tolerance": sum(row["frozen_reference_crop_within_replay_tolerance"] for row in output_rows),
            "crop_max_mean_abs_error": max(row["frozen_reference_crop_mean_abs_error"] for row in output_rows),
            "crop_max_abs_error": max(row["frozen_reference_crop_max_abs_error"] for row in output_rows),
            "reference_candidates": sum(len(row["candidates"]) for row in output_rows),
            "reference_masks": sum(len(row["candidates"]) for row in output_rows),
        },
    }
    (output / "reference-supplement.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--collection-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--dockerfile-sha256", required=True)
    args = parser.parse_args()
    materialize(
        args.dataset, args.collection, args.collection_root, args.output,
        args.docker_image_id, args.dockerfile_sha256,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
