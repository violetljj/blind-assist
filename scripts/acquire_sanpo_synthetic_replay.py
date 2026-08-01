#!/usr/bin/env python3
"""Acquire a minimal, hash-bound SANPO-Synthetic replay package.

The package is deliberately *not* a canonical training dataset or an assistive
event benchmark.  It preserves official RGB, panoptic masks, metric-depth
bytes, camera poses and split receipts so that a later, separately authorized
SANPO-Synthetic pretraining experiment can be reproduced without inventing
independent GPT/Codex consensus event truth or device-safety evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from build_sanpo_sequence_evalset import (
    DATASET_PAGE,
    DATASET_REPO,
    GCS_PREFIX,
    LICENSE_NAME,
    LICENSE_URL,
    SANPO_CITATION,
    download,
    fetch_json,
    fetch_text,
    get_gcs_object,
    list_gcs_objects,
    media_url,
    object_inventory,
    resample_indices,
    sha256_file,
    verify_gcs_md5,
)


SOURCE_ID = "sanpo_synthetic_v0"
DATASET_NAME = "SANPO-Synthetic v0"
DEFAULT_SESSION_ID = "e1ae36e040a53837dbe40879ddca1fbc47d47752a563e1117629cde73e7de856"
DEFAULT_CAMERA = "camera_chest"
DEFAULT_LENS = "left"
OFFICIAL_SPLITS = ("train", "test")


class ReplayError(ValueError):
    """A source contract or local replay receipt is incomplete."""


def indexed_objects(objects: list[dict[str, Any]], suffix: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in objects:
        name = str(item.get("name", ""))
        if not name.endswith(suffix):
            continue
        stem = Path(name).name.removesuffix(suffix)
        if not stem.isdigit():
            raise ReplayError(f"source frame filename is not numeric: {name}")
        index = int(stem)
        if index in result:
            raise ReplayError(f"duplicate source frame index {index} for {suffix}")
        result[index] = item
    return result


def select_aligned_indices(
    rgb: dict[int, dict[str, Any]],
    masks: dict[int, dict[str, Any]],
    depth: dict[int, dict[str, Any]],
    *,
    source_fps: float,
    target_fps: float,
    start_frame: int,
    frame_count: int,
) -> list[int]:
    """Pick only frames with all three official modalities present."""
    if start_frame < 0 or frame_count <= 0:
        raise ReplayError("start_frame must be non-negative and frame_count must be positive")
    if source_fps <= 0 or target_fps <= 0 or target_fps > source_fps:
        raise ReplayError("target_fps must be positive and cannot exceed source_fps")
    available = sorted(set(rgb) & set(masks) & set(depth))
    selected = resample_indices(available, source_fps, target_fps, start_frame, frame_count)
    if len(selected) != frame_count:
        raise ReplayError(
            f"only {len(selected)} RGB/mask/depth-aligned frames available; requested {frame_count}"
        )
    return selected


def camera_metadata(description: dict[str, Any], camera: str, lens: str) -> tuple[float, dict[str, Any]]:
    locations = description.get("session_camera_location")
    details = description.get("session_camera_details")
    if not isinstance(locations, list) or not isinstance(details, list) or camera not in locations:
        raise ReplayError(f"session does not expose requested camera {camera!r}")
    index = locations.index(camera)
    if index >= len(details) or not isinstance(details[index], dict):
        raise ReplayError("camera metadata does not align with session_camera_location")
    item = details[index]
    dimensions = item.get(f"{lens}_camera_params")
    if not isinstance(dimensions, dict):
        raise ReplayError(f"session has no {lens} camera parameters")
    fps = float(item.get("fps", 0.0))
    if not math.isfinite(fps) or fps <= 0:
        raise ReplayError("session camera fps must be finite and positive")
    for field in ("image_width", "image_height", "fx", "fy", "cx", "cy"):
        if field not in dimensions:
            raise ReplayError(f"session camera parameters missing {field}")
    return fps, dimensions


def validate_downloaded_frame(image_path: Path, mask_path: Path, dimensions: dict[str, Any]) -> tuple[int, int]:
    with Image.open(image_path) as image:
        size = image.size
    with Image.open(mask_path) as mask:
        if mask.size != size:
            raise ReplayError(f"RGB/mask dimensions differ: {image_path.name}")
    expected = (int(dimensions["image_width"]), int(dimensions["image_height"]))
    if size != expected:
        raise ReplayError(f"source description/image dimensions differ: {size} != {expected}")
    return size


def require_fresh_output(path: Path) -> None:
    if path.exists():
        raise ReplayError(f"refusing to overwrite existing replay output: {path}")


def split_contract(official_split: str) -> dict[str, Any]:
    if official_split not in OFFICIAL_SPLITS:
        raise ReplayError(
            f"official_split must be one of {OFFICIAL_SPLITS}: "
            f"{official_split!r}"
        )
    is_train = official_split == "train"
    return {
        "official_split": official_split,
        "split_object_name": (
            f"{GCS_PREFIX}/sanpo-synthetic/splits/"
            f"{official_split}_session_ids.txt"
        ),
        "label_authority": (
            "official_panoptic_ground_truth_pretraining_only"
            if is_train
            else "official_panoptic_heldout_geometry_proxy_only"
        ),
        "pretraining_candidate": is_train,
        "synthetic_heldout_evaluation_candidate": not is_train,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--camera", choices=(DEFAULT_CAMERA,), default=DEFAULT_CAMERA)
    parser.add_argument("--lens", choices=(DEFAULT_LENS,), default=DEFAULT_LENS)
    parser.add_argument(
        "--official-split",
        choices=OFFICIAL_SPLITS,
        default="train",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--frame-count", type=int, default=3)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    try:
        if args.retries <= 0:
            raise ReplayError("retries must be positive")
        output_root = args.output_root.resolve()
        require_fresh_output(output_root)
        split = split_contract(args.official_split)
        session_prefix = f"{GCS_PREFIX}/sanpo-synthetic/{args.session_id}"
        description_name = f"{session_prefix}/description.json"
        labelmap_name = f"{GCS_PREFIX}/labelmap.json"
        annotation_name = f"{session_prefix}/{args.camera}/{args.lens}/frame_segmentation_annotation_type.json"
        poses_name = f"{session_prefix}/{args.camera}/camera_poses.csv"
        objects = {
            "description": get_gcs_object(description_name, args.retries),
            "labelmap": get_gcs_object(labelmap_name, args.retries),
            "annotation_types": get_gcs_object(annotation_name, args.retries),
            "camera_poses": get_gcs_object(poses_name, args.retries),
        }
        description = fetch_json(media_url(description_name, objects["description"].get("generation")), args.retries)
        if description.get("session_type") != "synthetic":
            raise ReplayError("official session description does not identify as synthetic")
        source_fps, dimensions = camera_metadata(description, args.camera, args.lens)

        split_name = str(split["split_object_name"])
        split_object = get_gcs_object(split_name, args.retries)
        split_ids = {
            line.strip()
            for line in fetch_text(media_url(split_name, split_object.get("generation")), args.retries).splitlines()
            if line.strip()
        }
        if args.session_id not in split_ids:
            raise ReplayError(
                "SANPO-Synthetic replay session is not in the requested "
                f"official {args.official_split} split"
            )

        frame_prefix = f"{session_prefix}/{args.camera}/{args.lens}/video_frames/"
        mask_prefix = f"{session_prefix}/{args.camera}/{args.lens}/segmentation_masks/"
        depth_prefix = f"{session_prefix}/{args.camera}/{args.lens}/depth_maps/"
        rgb = indexed_objects(list_gcs_objects(frame_prefix, args.retries), ".png")
        masks = indexed_objects(list_gcs_objects(mask_prefix, args.retries), ".png")
        depth = indexed_objects(list_gcs_objects(depth_prefix, args.retries), ".float16.gz")
        selected = select_aligned_indices(
            rgb, masks, depth, source_fps=source_fps, target_fps=args.target_fps,
            start_frame=args.start_frame, frame_count=args.frame_count,
        )

        for relative in (
            f"images/{args.official_split}",
            f"source_masks/{args.official_split}",
            f"source_depth/{args.official_split}",
            "source_metadata",
            "qa",
        ):
            (output_root / relative).mkdir(parents=True, exist_ok=True)
        for local_name, object_name in (
            ("source_metadata/source_session_description.json", description_name),
            ("source_metadata/source_labelmap.json", labelmap_name),
            ("source_metadata/source_annotation_types.json", annotation_name),
            ("source_metadata/camera_poses.csv", poses_name),
        ):
            item = objects[{"source_session_description.json": "description", "source_labelmap.json": "labelmap", "source_annotation_types.json": "annotation_types", "camera_poses.csv": "camera_poses"}[Path(local_name).name]]
            local_path = output_root / local_name
            download(media_url(object_name, item.get("generation")), local_path, args.retries)
            verify_gcs_md5(local_path, item)

        annotation_types = fetch_json(media_url(annotation_name, objects["annotation_types"].get("generation")), args.retries)
        sequence_id = f"sanpo_synthetic_{args.session_id}_{args.camera}_{args.lens}_{args.start_frame:06d}_{int(args.target_fps)}fps"
        rows: list[dict[str, Any]] = []
        for timeline_index, source_index in enumerate(selected):
            sample_id = f"{sequence_id}_{timeline_index:06d}"
            image_rel = Path("images") / args.official_split / f"{sample_id}.png"
            mask_rel = Path("source_masks") / args.official_split / f"{sample_id}.png"
            depth_rel = Path("source_depth") / args.official_split / f"{sample_id}.float16.gz"
            image_path, mask_path, depth_path = output_root / image_rel, output_root / mask_rel, output_root / depth_rel
            for item, path in ((rgb[source_index], image_path), (masks[source_index], mask_path), (depth[source_index], depth_path)):
                download(media_url(str(item["name"]), item.get("generation")), path, args.retries)
                verify_gcs_md5(path, item)
            width, height = validate_downloaded_frame(image_path, mask_path, dimensions)
            rows.append({
                "id": sample_id,
                "image_path": image_rel.as_posix(),
                "image_sha256": sha256_file(image_path),
                "source_mask_path": mask_rel.as_posix(),
                "source_mask_sha256": sha256_file(mask_path),
                "source_depth_path": depth_rel.as_posix(),
                "source_depth_sha256": sha256_file(depth_path),
                "width": width,
                "height": height,
                "session_id": args.session_id,
                "sequence_id": sequence_id,
                "frame_index": timeline_index,
                "source_frame_index": source_index,
                "source_timestamp_ms": int(round(source_index * 1000.0 / source_fps)),
                "source_annotation_quality": str(annotation_types.get(str(source_index), "UNKNOWN")),
                "label_authority": split["label_authority"],
                "event_truth": None,
                "source": {
                    "source_id": SOURCE_ID,
                    "dataset": DATASET_NAME,
                    "dataset_page": DATASET_PAGE,
                    "repository": DATASET_REPO,
                    "license": LICENSE_NAME,
                    "license_url": LICENSE_URL,
                    "official_split": args.official_split,
                    "session_id": args.session_id,
                    "camera": args.camera,
                    "lens": args.lens,
                    "privacy_status": "synthetic_source_no_personal_data_claimed_by_importer",
                },
                "modalities": {
                    "rgb": object_inventory(rgb[source_index]),
                    "panoptic_mask": object_inventory(masks[source_index]),
                    "metric_depth": object_inventory(depth[source_index]),
                    "camera_poses": {"path": "source_metadata/camera_poses.csv", "sha256": sha256_file(output_root / "source_metadata/camera_poses.csv")},
                    "imu": {"status": "not_present_in_this_published_session_inventory", "usable_for_replay": False},
                },
                "authorization": {
                    "offline_replay": True,
                    "pretraining_candidate": split[
                        "pretraining_candidate"
                    ],
                    "synthetic_heldout_evaluation_candidate": split[
                        "synthetic_heldout_evaluation_candidate"
                    ],
                    "real_finetune_or_eval": False,
                    "human_event_truth": False,
                    "calibration": False,
                    "blind_evaluation": False,
                    "android_runtime": False,
                    "production_model_replacement": False,
                },
            })

        (output_root / "manifest.replay.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        spec = {
            "schema": "blindassist_sanpo_synthetic_replay_v1",
            "purpose": (
                "official SANPO-Synthetic source-contract replay and "
                + (
                    "pretraining candidate intake"
                    if args.official_split == "train"
                    else "synthetic heldout geometry-proxy evaluation intake"
                )
            ),
            "source": {"source_id": SOURCE_ID, "dataset": DATASET_NAME, "official_split": args.official_split, "session_id": args.session_id},
            "sampling": {"source_fps": source_fps, "target_fps": args.target_fps, "selected_source_frames": selected},
            "camera": dimensions,
            "source_inventory": {**{name: object_inventory(item) for name, item in objects.items()}, "official_split_receipt": object_inventory(split_object), "rgb": [object_inventory(rgb[i]) for i in selected], "masks": [object_inventory(masks[i]) for i in selected], "depth": [object_inventory(depth[i]) for i in selected]},
            "required_downstream_order": (
                ["SANPO-Synthetic pretraining candidate", "separately gated SANPO-Real finetune", "independent offline/INT8/device gates"]
                if args.official_split == "train"
                else ["frozen synthetic heldout geometry-proxy evaluation only"]
            ),
            "prohibited_claims": ["independent GPT/Codex consensus event truth", "calibration evidence", "blind evaluation", "Android runtime authorization", "production model replacement", "user safety proof"],
        }
        (output_root / "dataset_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (output_root / "source_licenses.md").write_text(
            f"# SANPO-Synthetic source and license\n\n- Dataset: {DATASET_NAME}\n- Dataset page: {DATASET_PAGE}\n- Repository: {DATASET_REPO}\n- License: [{LICENSE_NAME}]({LICENSE_URL})\n- Session: `{args.session_id}` / official {args.official_split} split\n- Attribution: {SANPO_CITATION}\n- Local policy: original source assets remain local-only under test-artifacts.local and are not committed.\n- Boundary: this package is source-contract replay intake only; it carries no human assistive-event truth and cannot authorize calibration, blind evaluation, Android runtime, or default-model replacement. Official-test packages are synthetic heldout-only and may not be used for training or development selection.\n",
            encoding="utf-8",
        )
        validation = {
            "ok": True,
            "frame_count": len(rows),
            "required_modalities_hash_bound": True,
            "all_rgb_mask_dimensions_match": True,
            "official_split": args.official_split,
            "all_frames_official_split_match": True,
            "all_frames_official_train_split": (
                args.official_split == "train"
            ),
            "pretraining_candidate": split["pretraining_candidate"],
            "synthetic_heldout_evaluation_candidate": split[
                "synthetic_heldout_evaluation_candidate"
            ],
            "imu_status": "absent_in_published_session_inventory_not_synthesized",
            "production_authorized": False,
        }
        (output_root / "qa" / "replay_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "output_root": str(output_root), "frame_count": len(rows), "production_authorized": False}, ensure_ascii=False))
    except (ReplayError, OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
