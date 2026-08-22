"""Audit the pinned ABotN-POIBench tree for pre-rendered observation pixels.

This is a source-availability check only. It never starts a renderer, teacher,
provider, or baseline, and it does not reopen a sealed episode.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import struct
from typing import Any, Iterable, Mapping
from urllib.parse import quote

import requests

from .audit_abotn_poibench_truth_source import DATASET_ID, DATASET_REVISION


SCHEMA = "blindassist_abotn_official_pixel_availability_audit_v0"
MEDIA_SUFFIXES = {".avi", ".jpeg", ".jpg", ".mkv", ".mov", ".mp4", ".png", ".webp"}
TRAJECTORY_PNG = re.compile(r"^annotations/[^/]+/png/traj_\d+_poi_\d+_.+\.png$")
FAILED_TRAJECTORY_PNG = re.compile(r"^annotations/[^/]+/png_failed/failed_\d+_poi_\d+_.+\.png$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _png_size(payload: bytes) -> tuple[int, int]:
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        raise ValueError("downloaded trajectory visualization is not a valid PNG header")
    width, height = struct.unpack(">II", payload[16:24])
    if width <= 0 or height <= 0:
        raise ValueError("downloaded trajectory visualization has invalid dimensions")
    return width, height


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def classify_files(entries: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(entries)
    files = [row for row in materialized if row.get("type") != "directory"]
    extensions = Counter(Path(str(row["path"])).suffix.lower() or "<none>" for row in files)
    media = [row for row in files if Path(str(row["path"])).suffix.lower() in MEDIA_SUFFIXES]
    categories: Counter[str] = Counter()
    observation_candidates: list[str] = []
    for row in media:
        path = str(row["path"])
        if TRAJECTORY_PNG.fullmatch(path):
            categories["annotation_trajectory_visualization"] += 1
        elif FAILED_TRAJECTORY_PNG.fullmatch(path):
            categories["annotation_failed_trajectory_visualization"] += 1
        elif path.startswith("occmaps/") and path.endswith("/map/occ_map.png"):
            categories["occupancy_map"] += 1
        else:
            categories["unclassified_media"] += 1
            observation_candidates.append(path)
    return {
        "entry_count": len(materialized),
        "file_count": len(files),
        "reported_bytes": sum(int(row.get("size") or 0) for row in files),
        "extensions": dict(sorted(extensions.items())),
        "media_count": len(media),
        "media_categories": dict(sorted(categories.items())),
        "pre_rendered_observation_candidate_count": len(observation_candidates),
        "pre_rendered_observation_candidates": sorted(observation_candidates),
    }


def _trajectory_visualization_path(entries: Iterable[Mapping[str, Any]], annotation_path: str) -> str:
    annotation = Path(annotation_path)
    if len(annotation.parts) != 3 or annotation.parts[0] != "annotations":
        raise ValueError("action-graph annotation path is not an ABotN task annotation")
    prefix = f"annotations/{annotation.parts[1]}/png/{annotation.stem}_poi_"
    matches = sorted(
        str(row["path"])
        for row in entries
        if row.get("type") != "directory" and str(row.get("path", "")).startswith(prefix)
    )
    if len(matches) != 1:
        raise ValueError(f"expected one official trajectory visualization for {annotation_path}, got {matches}")
    return matches[0]


def run_audit(*, action_graph_receipt: Path, output_dir: Path, timeout_s: float = 60.0) -> dict[str, Any]:
    freeze = json.loads(action_graph_receipt.read_text(encoding="utf-8"))
    annotation_path = str(freeze["inputs"]["annotation_path"])
    annotation_payload = Path(annotation_path).read_bytes()
    if _sha256(annotation_payload) != freeze["inputs"]["annotation_sha256"]:
        raise ValueError("sealed annotation hash does not match the action-graph receipt")
    annotation_relative = "annotations/" + Path(annotation_path).relative_to(
        next(parent for parent in Path(annotation_path).parents if parent.name == "annotations")
    ).as_posix()

    tree_url = (
        f"https://huggingface.co/api/datasets/{DATASET_ID}/tree/{DATASET_REVISION}"
        "?recursive=true&expand=false&limit=1000"
    )
    response = requests.get(tree_url, timeout=timeout_s, headers={"User-Agent": "BlindAssist-ABotN-pixel-audit/0"})
    response.raise_for_status()
    if "next" in response.links:
        raise ValueError("pinned release tree exceeds the single-page audit bound")
    entries = response.json()
    if not isinstance(entries, list):
        raise ValueError("unexpected Hugging Face tree response")
    inventory = classify_files(entries)
    visualization_path = _trajectory_visualization_path(entries, annotation_relative)
    visualization_url = (
        f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{DATASET_REVISION}/"
        f"{quote(visualization_path, safe='/')}?download=true"
    )
    image_response = requests.get(
        visualization_url,
        timeout=timeout_s,
        headers={"User-Agent": "BlindAssist-ABotN-pixel-audit/0"},
    )
    image_response.raise_for_status()
    image_payload = image_response.content
    width, height = _png_size(image_payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    visualization_file = output_dir / "official_trajectory_visualization.png"
    _atomic_bytes(visualization_file, image_payload)

    no_observation_rgb = inventory["pre_rendered_observation_candidate_count"] == 0
    result = {
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "sources": {
            "dataset_id": DATASET_ID,
            "dataset_revision": DATASET_REVISION,
            "tree_url": tree_url,
            "action_graph_receipt": str(action_graph_receipt.resolve()),
            "action_graph_receipt_sha256": _sha256(action_graph_receipt.read_bytes()),
            "sealed_annotation_path": annotation_relative,
            "sealed_annotation_sha256": freeze["inputs"]["annotation_sha256"],
        },
        "inventory": inventory,
        "sealed_task_official_png": {
            "classification": "TRAJECTORY_VISUALIZATION_NOT_CAMERA_RGB",
            "source_path": visualization_path,
            "local_path": str(visualization_file.resolve()),
            "bytes": len(image_payload),
            "sha256": _sha256(image_payload),
            "width": width,
            "height": height,
        },
        "terminal": (
            "OFFICIAL_PRE_RENDERED_OBSERVATION_RGB_NOT_RELEASED"
            if no_observation_rgb
            else "OFFICIAL_MEDIA_REQUIRES_MANUAL_CLASSIFICATION"
        ),
        "render_calls": 0,
        "teacher_calls": 0,
        "provider_calls": 0,
        "baseline_calls": 0,
        "sealed_episode_reruns": 0,
        "claim_ceiling": "PINNED_OFFICIAL_RELEASE_TREE_PIXEL_AVAILABILITY_ONLY",
        "next_action": "REQUIRE_OFFICIAL_RENDER_SERVER_HOST_FOR_SOURCE_NATIVE_OBSERVATION_RGB",
        "forbidden_inferences": [
            "Do not use trajectory visualizations or occupancy maps as camera observations.",
            "Do not expose this private source audit to the provider.",
            "Do not rerun or relabel the sealed episode from this audit.",
            "Do not attribute the sealed failure to an algorithm component until renderer fidelity is deconfounded.",
        ],
    }
    _atomic_json(output_dir / "receipt.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action-graph-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(
        action_graph_receipt=args.action_graph_receipt.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({
        "terminal": result["terminal"],
        "media_categories": result["inventory"]["media_categories"],
        "provider_calls": result["provider_calls"],
        "next_action": result["next_action"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
