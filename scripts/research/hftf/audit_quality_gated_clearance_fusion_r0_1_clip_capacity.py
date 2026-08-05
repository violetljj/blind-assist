#!/usr/bin/env python3
"""Label-blind clip-capacity audit for quality-gated clearance fusion R0.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_clip_capacity_result"
MEDIA_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_arkit_media_manifest"
PNG_RE = re.compile(r"_(\d+(?:\.\d+)?)\.png$", re.IGNORECASE)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def zip_png_stems(path: Path) -> list[float]:
    with zipfile.ZipFile(path) as archive:
        values = []
        for name in archive.namelist():
            if not name.lower().endswith(".png"):
                continue
            match = PNG_RE.search(Path(name).name)
            require(match is not None, f"timestamp stem missing: {name}")
            values.append(float(match.group(1)))
    require(len(values) == len(set(values)), f"duplicate timestamp stems: {path}")
    return sorted(values)


def nonoverlap_clip_count(times: list[float], length: int, max_gap: float) -> tuple[int, list[list[float]]]:
    clips: list[list[float]] = []
    index = 0
    while index + length <= len(times):
        window = times[index:index + length]
        if all(0.0 < right - left <= max_gap for left, right in zip(window, window[1:])):
            clips.append(window)
            index += length
        else:
            index += 1
    return len(clips), clips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.media_manifest.read_text(encoding="utf-8"))
    require(manifest.get("schema") == MEDIA_SCHEMA, "media manifest schema drift")
    require(manifest.get("labels_opened") is False and manifest.get("model_outputs_read") is False, "media boundary violated")
    require(manifest.get("model_loaded") is False and manifest.get("training_started") is False, "runtime boundary violated")
    require(manifest.get("file_count") == 15 and len(manifest.get("files", [])) == 15, "media file count drift")
    by_video: dict[str, dict[str, dict[str, Any]]] = {}
    for row in manifest["files"]:
        by_video.setdefault(str(row["video_id"]), {})[str(row["asset"])] = row
    require(len(by_video) == 3, "parent count drift")
    parents = []
    for video_id, assets in sorted(by_video.items()):
        for asset in ("lowres_wide.zip", "lowres_depth.zip", "confidence.zip"):
            require(asset in assets, f"missing asset {video_id}/{asset}")
            bound = (args.repo_root / "artifacts.local" / "datasets" / "quality-gated-clearance-fusion-r0-1-arkit-381644-20260806" / "raw" / "Validation" / video_id / asset).resolve()
            require(bound.is_file() and sha256_file(bound) == assets[asset]["sha256"], f"asset SHA mismatch: {bound}")
        rgb = set(zip_png_stems((args.repo_root / "artifacts.local" / "datasets" / "quality-gated-clearance-fusion-r0-1-arkit-381644-20260806" / "raw" / "Validation" / video_id / "lowres_wide.zip").resolve()))
        depth = set(zip_png_stems((args.repo_root / "artifacts.local" / "datasets" / "quality-gated-clearance-fusion-r0-1-arkit-381644-20260806" / "raw" / "Validation" / video_id / "lowres_depth.zip").resolve()))
        confidence = set(zip_png_stems((args.repo_root / "artifacts.local" / "datasets" / "quality-gated-clearance-fusion-r0-1-arkit-381644-20260806" / "raw" / "Validation" / video_id / "confidence.zip").resolve()))
        common = sorted(rgb & depth & confidence)
        clip_count, clips = nonoverlap_clip_count(common, 4, 0.5)
        parents.append({"parent_id": "381644", "video_id": video_id, "rgb_stem_count": len(rgb), "depth_stem_count": len(depth), "confidence_stem_count": len(confidence), "common_timestamp_stem_count": len(common), "evaluable_clip_count": clip_count, "clip_timestamp_examples": clips[:2]})
    minimum = int(protocol["capacity_gate"]["minimum_clips_per_parent"])
    terminal = "QUALITY_GATED_CLEARANCE_FUSION_R0_1_CLIP_CAPACITY_READY" if all(row["evaluable_clip_count"] >= minimum for row in parents) else "QUALITY_GATED_CLEARANCE_FUSION_R0_1_CLIP_CAPACITY_INSUFFICIENT"
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "media_manifest_sha256": sha256_file(args.media_manifest),
        "parent_count": len(parents),
        "minimum_clips_per_parent": minimum,
        "maximum_adjacent_gap_seconds": 0.5,
        "frame_body_bytes_read": False,
        "labels_opened": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
        "parents": parents,
        "terminal": terminal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if terminal.endswith("READY") else 1)


if __name__ == "__main__":
    main()
