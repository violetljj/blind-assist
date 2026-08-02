#!/usr/bin/env python3
"""Download a bounded, hash-verified SANPO-Real RGB/depth/mask window.

The command is source-intake only.  It never assigns an HFTF event label and
never reads existing development annotations.  A fresh receipt run is
required; downloads are limited to one explicitly named source session,
camera/view, and contiguous frame range.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pipeline import ContractError, sha256_file, utc_now, write_json, write_jsonl


BUCKET = "gresearch"
API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
MEDIA = "https://storage.googleapis.com/download/storage/v1/b/gresearch/o"
PREFIX = "sanpo_dataset/v0/sanpo-real/"
FRAME_RE = re.compile(r"/(?P<kind>video_frames|depth_maps|segmentation_masks)/(?P<index>\d{6})(?P<suffix>\.float16\.gz|\.png)$")


def _request_json(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "blindassist-hftf-d7/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except OSError as exc:
        raise ContractError(f"SANPO GCS metadata request failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("SANPO GCS metadata response is not an object")
    return payload


def _list_objects(prefix: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        params = {"prefix": prefix, "maxResults": "1000"}
        if token:
            params["pageToken"] = token
        payload = _request_json(params)
        for item in payload.get("items", []):
            if isinstance(item, dict) and item.get("name"):
                items.append(item)
        token = payload.get("nextPageToken")
        if not token:
            return items


def _md5_base64(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return base64.b64encode(digest.digest()).decode("ascii")


def _download_object(item: dict[str, Any], destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != int(item.get("size", -1)) or _md5_base64(destination) != str(item.get("md5Hash")):
            raise ContractError(f"existing SANPO file does not match provider metadata: {destination}")
    else:
        encoded_name = urllib.parse.quote(str(item["name"]), safe="")
        url = f"{MEDIA}/{encoded_name}?alt=media"
        request = urllib.request.Request(url, headers={"User-Agent": "blindassist-hftf-d7/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
        except OSError as exc:
            raise ContractError(f"SANPO media download failed: {item['name']}: {exc}") from exc
    actual_md5 = _md5_base64(destination)
    expected_md5 = str(item.get("md5Hash") or "")
    if actual_md5 != expected_md5:
        raise ContractError(f"SANPO MD5 mismatch: {destination}")
    return {
        "remote_name": item["name"],
        "generation": item.get("generation"),
        "size": int(item.get("size", destination.stat().st_size)),
        "provider_md5_base64": expected_md5,
        "local_path": str(destination.resolve()),
        "local_sha256": sha256_file(destination),
        "md5_verified": True,
    }


def _frame_index(item: dict[str, Any]) -> tuple[str, int] | None:
    match = FRAME_RE.search(str(item.get("name", "")))
    if not match:
        return None
    kind = {"video_frames": "rgb", "depth_maps": "depth", "segmentation_masks": "mask"}[match.group("kind")]
    return kind, int(match.group("index"))


def _nominal_time_ns(frame_index: int, fps: float | None) -> int | None:
    if fps is None:
        return None
    if not math.isfinite(fps) or fps <= 0:
        raise ContractError("fps must be finite and positive")
    return round(frame_index * 1_000_000_000 / fps)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.session_id):
        raise ContractError("unsafe SANPO session id")
    if args.start_frame < 0 or args.frame_count <= 0:
        raise ContractError("start-frame must be non-negative and frame-count must be positive")
    if args.max_bytes <= 0:
        raise ContractError("max-bytes must be positive")
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        raise ContractError("fps must be finite and positive")
    if args.camera not in {"chest", "head"} or args.view not in {"left", "right"}:
        raise ContractError("camera must be chest/head and view must be left/right")
    root = Path(args.output_root).resolve()
    receipt_path = root / "receipts" / f"sanpo_media_receipt_{args.run_id}.json"
    if receipt_path.exists():
        raise ContractError(f"receipt already exists; refusing overwrite: {receipt_path}")
    session_prefix = f"{PREFIX}{args.session_id}/"
    camera_prefix = f"{session_prefix}camera_{args.camera}/"
    view_prefix = f"{camera_prefix}{args.view}/"
    objects = _list_objects(view_prefix)
    indexed: dict[str, dict[int, dict[str, Any]]] = {"rgb": {}, "depth": {}, "mask": {}}
    for item in objects:
        parsed = _frame_index(item)
        if parsed is not None:
            kind, index = parsed
            indexed[kind][index] = item
    required_indices = list(range(args.start_frame, args.start_frame + args.frame_count))
    missing_rgb = [index for index in required_indices if index not in indexed["rgb"]]
    missing_depth = [index for index in required_indices if index not in indexed["depth"]]
    if missing_rgb or missing_depth:
        raise ContractError(f"requested range is not synchronized: missing_rgb={missing_rgb[:5]}, missing_depth={missing_depth[:5]}")
    selected_mask = [index for index in required_indices if index in indexed["mask"]]
    selected_items: list[tuple[str, dict[str, Any], Path]] = []
    session_root = root / "raw" / "sanpo-real" / args.session_id / f"camera_{args.camera}_{args.view}"
    for index in required_indices:
        selected_items.append(("rgb", indexed["rgb"][index], session_root / "rgb" / f"{index:06d}.png"))
        selected_items.append(("depth", indexed["depth"][index], session_root / "depth" / f"{index:06d}.float16.gz"))
        if index in indexed["mask"]:
            selected_items.append(("mask", indexed["mask"][index], session_root / "mask" / f"{index:06d}.png"))
    pose_items = [item for item in _list_objects(camera_prefix) if str(item.get("name", "")).endswith(("camera_poses.csv", "fixed_camera_poses.csv"))]
    selected_items.extend(("pose", item, session_root / Path(str(item["name"])).name) for item in pose_items)
    intrinsics_items = [
        item
        for item in _list_objects(session_prefix)
        if str(item.get("name", "")).endswith("/description.json")
    ]
    if len(intrinsics_items) != 1:
        raise ContractError(f"expected one SANPO session description.json, found {len(intrinsics_items)}")
    selected_items.append(("intrinsics", intrinsics_items[0], session_root / "description.json"))
    total_bytes = sum(int(item.get("size", 0)) for _, item, _ in selected_items)
    if total_bytes > args.max_bytes:
        raise ContractError(f"bounded SANPO download exceeds max-bytes: {total_bytes} > {args.max_bytes}")

    materialized: list[dict[str, Any]] = []
    for kind, item, destination in selected_items:
        record = _download_object(item, destination)
        record.update({"kind": kind, "frame_index": next((index for index in required_indices if indexed[kind].get(index) is item), None) if kind in {"rgb", "depth", "mask"} else None})
        materialized.append(record)
    manifest_path = root / "manifests" / f"sanpo_media_manifest_{args.run_id}.jsonl"
    manifest_rows = [
        {
            "schema": "hftf_d7_public_real_sanpo_media_frame_v1",
            "dataset_id": "SANPO-Real",
            "source_session_id": args.session_id,
            "camera": args.camera,
            "view": args.view,
            "frame_index": item.get("frame_index"),
            "timestamp_ns": None,
            "nominal_time_ns": _nominal_time_ns(int(item["frame_index"]), args.fps),
            "time_semantics": "DERIVED_RELATIVE_NOMINAL" if args.fps is not None else "NOT_EVALUABLE",
            "capture_timestamp_authoritative": False,
            "pose_row_binding": "NOT_EVALUABLE",
            "rgb_depth_mask_binding": "INDEX_KEYED",
            "nominal_time_contract": {
                "kind": "FRAME_INDEX_DIVIDED_BY_EXPLICIT_FPS",
                "fps": args.fps,
                "official_url": "https://research.google/blog/sanpo-a-scene-understanding-accessibility-navigation-pathfinding-obstacle-avoidance-dataset/",
            } if args.fps is not None else None,
            "rgb_local_path": next((entry["local_path"] for entry in materialized if entry.get("kind") == "rgb" and entry.get("frame_index") == item.get("frame_index")), None),
            "depth_local_path": next((entry["local_path"] for entry in materialized if entry.get("kind") == "depth" and entry.get("frame_index") == item.get("frame_index")), None),
            "mask_local_path": next((entry["local_path"] for entry in materialized if entry.get("kind") == "mask" and entry.get("frame_index") == item.get("frame_index")), None),
            "intrinsics_local_path": next((entry["local_path"] for entry in materialized if entry.get("kind") == "intrinsics"), None),
            "pose_local_paths": [entry["local_path"] for entry in materialized if entry.get("kind") == "pose"],
            "source_native_geometry": "DEPTH_AND_CAMERA_POSE_WITH_INTRINSICS",
            "source_license": "CC-BY-4.0",
            "event_truth_authority": False,
        }
        for item in materialized
        if item.get("kind") == "rgb"
    ]
    write_jsonl(manifest_path, manifest_rows)
    receipt = {
        "schema": "hftf_d7_public_real_sanpo_media_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "dataset_id": "SANPO-Real",
        "source_session_id": args.session_id,
        "camera": args.camera,
        "view": args.view,
        "frame_range": {"start": args.start_frame, "count": args.frame_count},
        "fps": args.fps,
        "time_semantics": "DERIVED_RELATIVE_NOMINAL" if args.fps is not None else "NOT_EVALUABLE",
        "capture_timestamp_authoritative": False,
        "pose_row_binding": "NOT_EVALUABLE",
        "rgb_depth_mask_binding": "INDEX_KEYED",
        "nominal_time_contract": {
            "kind": "FRAME_INDEX_DIVIDED_BY_EXPLICIT_FPS",
            "fps": args.fps,
            "official_url": "https://research.google/blog/sanpo-a-scene-understanding-accessibility-navigation-pathfinding-obstacle-avoidance-dataset/",
        } if args.fps is not None else None,
        "object_count": len(materialized),
        "frame_count": len(required_indices),
        "rgb_frame_count": sum(1 for item in materialized if item.get("kind") == "rgb"),
        "depth_frame_count": sum(1 for item in materialized if item.get("kind") == "depth"),
        "mask_frame_count": sum(1 for item in materialized if item.get("kind") == "mask"),
        "intrinsics_file_count": sum(1 for item in materialized if item.get("kind") == "intrinsics"),
        "bytes_from_provider_metadata": total_bytes,
        "media_status": "PUBLIC_GCS_RGB_DEPTH_MASK_WINDOW_DOWNLOADED",
        "license": "CC-BY-4.0",
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "objects": materialized,
        "event_truth_authority": False,
        "training_authorized": False,
        "confirmation_authorized": False,
        "production_authorized": False,
    }
    write_json(receipt_path, receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--camera", default="chest")
    parser.add_argument("--view", default="left")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-count", type=int, default=20)
    parser.add_argument("--max-bytes", type=int, default=200_000_000)
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Explicit published FPS for derived nominal times; omitted means no nominal times are inferred.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
