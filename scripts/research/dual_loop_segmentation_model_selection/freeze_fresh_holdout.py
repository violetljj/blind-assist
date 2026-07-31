"""Freeze a fresh SANPO source-native holdout without interpreting mask pixels.

The existing SANPO sequence builder derives semantic regions and preview images.
That is intentionally unsuitable before model-selection settings are frozen. This
freezer only reads public object metadata, downloads exact RGB/mask bytes, checks
byte-level integrity, and emits a source-native manifest with no derived labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.build_sanpo_sequence_evalset import (
    DEFAULT_LENS,
    GCS_PREFIX,
    LICENSE_NAME,
    LICENSE_URL,
    DATASET_PAGE,
    DATASET_REPO,
    download,
    fetch_json,
    fetch_text,
    frame_number,
    get_gcs_object,
    list_gcs_objects,
    media_url,
    object_inventory,
    resample_indices,
    sha256_file,
    verify_gcs_md5,
)

from . import PROTOCOL_ID


class HoldoutFreezeError(ValueError):
    """Raised when the fresh formal identity cannot be frozen fail-closed."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HoldoutFreezeError(f"cannot read JSON {path}") from error
    if not isinstance(value, dict):
        raise HoldoutFreezeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _sha256_bytes(path: Path) -> str:
    return sha256_file(path).lower()


def _protocol_session_ids(protocol: dict[str, Any]) -> list[str]:
    if protocol.get("protocol_id") != PROTOCOL_ID:
        raise HoldoutFreezeError(f"unexpected protocol id: {protocol.get('protocol_id')!r}")
    if protocol.get("status") != "DESIGN_FROZEN":
        raise HoldoutFreezeError(f"protocol is not frozen: {protocol.get('status')!r}")
    role = protocol.get("data_roles", {}).get("fresh_source_native_pixel_truth", {})
    if role.get("status_before_acquisition") != "FROZEN_IDENTITY_PIXEL_TRUTH_NOT_ACCESSED":
        raise HoldoutFreezeError("fresh holdout role is not in pre-acquisition state")
    if role.get("official_split") != "test":
        raise HoldoutFreezeError("fresh holdout must come from official test split")
    if role.get("camera") != "camera_chest" or role.get("lens") != DEFAULT_LENS:
        raise HoldoutFreezeError("fresh holdout camera/lens contract mismatch")
    ids = role.get("session_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) for item in ids):
        raise HoldoutFreezeError("fresh holdout session identity list is missing")
    if len(ids) != len(set(ids)):
        raise HoldoutFreezeError("fresh holdout session identity list contains duplicates")
    return list(ids)


def _official_session_ids(split: str, retries: int) -> list[str]:
    object_name = f"{GCS_PREFIX}/sanpo-real/splits/{split}_session_ids.txt"
    item = get_gcs_object(object_name, retries)
    values = [
        line.strip()
        for line in fetch_text(media_url(object_name, item.get("generation")), retries).splitlines()
        if line.strip()
    ]
    if len(values) != len(set(values)):
        raise HoldoutFreezeError(f"official {split} session list contains duplicates")
    return values


def _session_description(session_id: str, camera: str, retries: int) -> tuple[dict[str, Any], float, int, int]:
    prefix = f"{GCS_PREFIX}/sanpo-real/{session_id}"
    description_name = f"{prefix}/description.json"
    description_object = get_gcs_object(description_name, retries)
    description = fetch_json(media_url(description_name, description_object.get("generation")), retries)
    locations = list(description.get("session_camera_location", []))
    if camera not in locations:
        raise HoldoutFreezeError(f"{session_id}: missing required camera {camera}")
    camera_index = locations.index(camera)
    details = list(description.get("session_camera_details", []))
    if camera_index >= len(details):
        raise HoldoutFreezeError(f"{session_id}: camera details are incomplete")
    camera_detail = details[camera_index]
    fps = float(camera_detail["fps"])
    params = camera_detail[f"{DEFAULT_LENS}_camera_params"]
    width = int(params["image_width"])
    height = int(params["image_height"])
    if fps <= 0 or width <= 0 or height <= 0:
        raise HoldoutFreezeError(f"{session_id}: invalid camera metadata")
    return object_inventory(description_object), fps, width, height


def _aligned_objects(session_id: str, camera: str, retries: int) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    prefix = f"{GCS_PREFIX}/sanpo-real/{session_id}/{camera}/{DEFAULT_LENS}"
    frame_items = {
        frame_number(item["name"]): item
        for item in list_gcs_objects(f"{prefix}/video_frames/", retries)
        if item.get("name", "").endswith(".png")
    }
    mask_items = {
        frame_number(item["name"]): item
        for item in list_gcs_objects(f"{prefix}/segmentation_masks/", retries)
        if item.get("name", "").endswith(".png")
    }
    if not frame_items or not mask_items:
        raise HoldoutFreezeError(f"{session_id}: missing RGB or segmentation-mask objects")
    return frame_items, mask_items


def _download_and_hash(url: str, item: dict[str, Any], target: Path, retries: int) -> str:
    download(url, target, retries)
    verify_gcs_md5(target, item)
    digest = _sha256_bytes(target)
    if not digest:
        raise HoldoutFreezeError(f"empty SHA256 for downloaded object {target}")
    return digest


def freeze_holdout(
    *,
    protocol_path: Path,
    output_root: Path,
    retries: int = 3,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    session_ids = _protocol_session_ids(protocol)
    role = protocol["data_roles"]["fresh_source_native_pixel_truth"]
    target_fps = float(role["target_fps"])
    start_frame = int(role["start_frame"])
    frame_count = int(role["frame_count_per_session"])
    camera = str(role["camera"])
    lens = str(role["lens"])
    if output_root.exists() and any(output_root.iterdir()):
        raise HoldoutFreezeError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    official_test = _official_session_ids("test", retries)
    old_blind = {
        "5LlqRK-hWoDLSW5MmoLjKj6uQtZMKjb9",
        "i2jglnBfoIqIIA7ojQGe-4vK07hUm4T3",
    }
    if any(item not in official_test for item in session_ids):
        raise HoldoutFreezeError("fresh session is absent from the official test split")
    if old_blind.intersection(session_ids):
        raise HoldoutFreezeError("fresh holdout overlaps consumed old blind session")
    official_positions = [official_test.index(item) for item in session_ids]
    if official_positions != sorted(official_positions):
        raise HoldoutFreezeError("fresh session order is not official test split order")

    manifest_rows: list[dict[str, Any]] = []
    session_receipts: list[dict[str, Any]] = []
    for session_id in session_ids:
        description_inventory, source_fps, width, height = _session_description(session_id, camera, retries)
        frame_items, mask_items = _aligned_objects(session_id, camera, retries)
        selected = resample_indices(
            sorted(set(frame_items).intersection(mask_items)),
            source_fps,
            target_fps,
            start_frame,
            frame_count,
        )
        if len(selected) != frame_count:
            raise HoldoutFreezeError(
                f"{session_id}: expected {frame_count} aligned resampled frames, got {len(selected)}"
            )
        source_id = f"sanpo_real_v0:{session_id}"
        sequence_id = f"sanpo_real_v0_{session_id}_{camera}_{lens}_{start_frame:06d}_{int(target_fps)}fps"
        downloads: list[dict[str, Any]] = []
        for sequence_frame_index, source_frame_index in enumerate(selected):
            frame_item = frame_items[source_frame_index]
            mask_item = mask_items[source_frame_index]
            sample_id = f"{sequence_id}_{sequence_frame_index:06d}"
            image_rel = Path("sessions") / session_id / "images" / f"{sample_id}.png"
            mask_rel = Path("sessions") / session_id / "masks" / f"{sample_id}.png"
            image_path = output_root / image_rel
            mask_path = output_root / mask_rel
            image_sha = _download_and_hash(
                media_url(frame_item["name"], frame_item.get("generation")),
                frame_item,
                image_path,
                retries,
            )
            mask_sha = _download_and_hash(
                media_url(mask_item["name"], mask_item.get("generation")),
                mask_item,
                mask_path,
                retries,
            )
            timestamp_ns = int(round(source_frame_index / source_fps * 1_000_000_000))
            manifest_rows.append({
                "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.fresh_holdout_row.v1",
                "id": sample_id,
                "source_id": source_id,
                "session_id": source_id,
                "sequence_id": sequence_id,
                "frame_id": int(source_frame_index),
                "sequence_frame_index": sequence_frame_index,
                "source_capture_timestamp_ns": timestamp_ns,
                "image_path": image_rel.as_posix(),
                "image_sha256": image_sha,
                "semantic_mask_path": mask_rel.as_posix(),
                "semantic_mask_sha256": mask_sha,
                "width": width,
                "height": height,
                "split": "fresh_formal",
                "scene_bucket": "fresh_official_test_unknown_until_formal",
                "label_authority": "source_ground_truth",
                "source": {
                    "dataset": "SANPO-Real v0",
                    "official_split": "test",
                    "session_id": session_id,
                    "camera": camera,
                    "lens": lens,
                    "license": LICENSE_NAME,
                    "license_url": LICENSE_URL,
                    "dataset_page": DATASET_PAGE,
                    "repository": DATASET_REPO,
                    "rgb_object": object_inventory(frame_item),
                    "mask_object": object_inventory(mask_item),
                    "pixel_truth_interpretation": "forbidden_before_formal_freeze",
                },
            })
            downloads.append({
                "sequence_frame_index": sequence_frame_index,
                "source_frame_index": source_frame_index,
                "image_path": image_rel.as_posix(),
                "image_sha256": image_sha,
                "mask_path": mask_rel.as_posix(),
                "mask_sha256": mask_sha,
                "rgb_object": object_inventory(frame_item),
                "mask_object": object_inventory(mask_item),
            })
        session_receipts.append({
            "source_id": source_id,
            "official_split": "test",
            "camera": camera,
            "lens": lens,
            "source_fps": source_fps,
            "target_fps": target_fps,
            "start_frame": start_frame,
            "frame_count": len(selected),
            "selected_source_frames": selected,
            "image_dimensions": [width, height],
            "description_inventory": description_inventory,
            "downloads": downloads,
        })

    manifest_path = output_root / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in manifest_rows),
        encoding="utf-8",
        newline="\n",
    )
    receipt = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.fresh_holdout_receipt.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "FRESH_FORMAL_HOLDOUT_FROZEN",
        "pixel_truth_status": "DOWNLOADED_HASHED_NOT_INTERPRETED",
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "output_root": str(output_root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "row_count": len(manifest_rows),
        "source_ids": [item["source_id"] for item in session_receipts],
        "session_count": len(session_receipts),
        "session_receipts": session_receipts,
        "truth_access_boundary": "No mask pixel values, class counts, components, model outputs or truth-derived selection were read by this freezer.",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_root / "freeze_receipt.json", receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = freeze_holdout(
        protocol_path=args.protocol.resolve(),
        output_root=args.output_root.resolve(),
        retries=args.retries,
    )
    print(json.dumps({"status": receipt["status"], "rows": receipt["row_count"], "manifest": receipt["manifest"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
