#!/usr/bin/env python3
"""Generate fail-closed cross-parent RGB identity candidates for R1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from validate_protocol import DEFAULT_PROTOCOL, REPO_ROOT, sha256

DEFAULT_AUDIT_PROTOCOL = REPO_ROOT / "docs/research/hftf/SPATIAL_CALIBRATION_HEAD_R1_RGB_IDENTITY_AUDIT_PROTOCOL_2026-08-04.json"


def variants(gray: np.ndarray) -> dict[str, np.ndarray]:
    if gray.ndim != 2:
        raise ValueError("identity audit expects grayscale input")
    output = {"original": gray, "horizontal_mirror": np.fliplr(gray)}
    height, width = gray.shape
    for fraction, label in ((0.90, "90"), (0.80, "80")):
        crop_height, crop_width = int(round(height * fraction)), int(round(width * fraction))
        y0, x0 = (height - crop_height) // 2, (width - crop_width) // 2
        crop = gray[y0:y0 + crop_height, x0:x0 + crop_width]
        output[f"center_crop_{label}"] = crop
        output[f"center_crop_{label}_horizontal_mirror"] = np.fliplr(crop)
    return output


def phash64(gray: np.ndarray) -> int:
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    coefficients = cv2.dct(resized)[:8, :8].reshape(-1)
    threshold = float(np.median(coefficients[1:]))
    value = 0
    for bit in coefficients > threshold:
        value = (value << 1) | int(bit)
    return value


def candidate_edges(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], list[tuple[int, str, int]]] = {}
    edges: dict[tuple[int, int], dict[str, Any]] = {}
    for frame_index, frame in enumerate(frames):
        for variant_name, value in frame["phash_variants"].items():
            hash_value = int(value, 16)
            candidate_items: set[tuple[int, str, int]] = set()
            for partition in range(4):
                segment = (hash_value >> (partition * 16)) & 0xFFFF
                neighbor_segments = [segment] + [segment ^ (1 << bit) for bit in range(16)]
                for neighbor in neighbor_segments:
                    candidate_items.update(buckets.get((partition, neighbor), ()))
            for other_index, other_variant, other_hash in candidate_items:
                if frames[other_index]["visit_id"] == frame["visit_id"]:
                    continue
                distance = (hash_value ^ other_hash).bit_count()
                if distance > 6:
                    continue
                key = (min(frame_index, other_index), max(frame_index, other_index))
                current = edges.get(key)
                detail = {
                    "left_index": key[0], "right_index": key[1], "phash_distance": distance,
                    "left_variant": other_variant if other_index == key[0] else variant_name,
                    "right_variant": variant_name if frame_index == key[1] else other_variant,
                }
                if current is None or (distance, detail["left_variant"], detail["right_variant"]) < (current["phash_distance"], current["left_variant"], current["right_variant"]):
                    edges[key] = detail
            for partition in range(4):
                segment = (hash_value >> (partition * 16)) & 0xFFFF
                buckets.setdefault((partition, segment), []).append((frame_index, variant_name, hash_value))

    payload_groups: dict[str, list[int]] = {}
    for index, frame in enumerate(frames):
        payload_groups.setdefault(frame["payload_sha256"], []).append(index)
    for group in payload_groups.values():
        for position, left in enumerate(group):
            for right in group[position + 1:]:
                if frames[left]["visit_id"] == frames[right]["visit_id"]:
                    continue
                key = (min(left, right), max(left, right))
                edges.setdefault(key, {"left_index": key[0], "right_index": key[1], "phash_distance": 0, "left_variant": "payload_exact", "right_variant": "payload_exact"})
                edges[key]["payload_exact"] = True
    return [edges[key] for key in sorted(edges)]


def load_frames(media_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    frames = []
    for video in media_manifest["videos"]:
        role = video["role"]
        if role not in ("train", "validation", "sealed_identity_only"):
            raise ValueError(f"unexpected media role: {role}")
        for row in video["extracted"]["lowres_wide"]:
            path = Path(row["path"])
            if sha256(path) != row["sha256"]:
                raise ValueError(f"RGB payload hash mismatch: {path}")
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise OSError(f"cannot decode RGB identity frame: {path}")
            frame = {
                "role": role, "visit_id": str(video["visit_id"]), "video_id": str(video["video_id"]),
                "frame_path": str(path.resolve()), "payload_sha256": row["sha256"],
                "phash_variants": {name: f"{phash64(image):016X}" for name, image in variants(gray).items()},
            }
            frames.append(frame)
    if len(frames) != 3600:
        raise ValueError("identity audit requires exactly 3600 RGB frames")
    return frames


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--audit-protocol", type=Path, default=DEFAULT_AUDIT_PROTOCOL)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_protocol = json.loads(args.audit_protocol.read_text(encoding="utf-8"))
    if audit_protocol.get("status") != "FROZEN_BEFORE_RGB_BODY_ACCESS" or audit_protocol.get("main_protocol_sha256") != sha256(args.main_protocol):
        raise ValueError("identity audit protocol authority mismatch")
    media = json.loads(args.media_manifest.read_text(encoding="utf-8"))
    if media.get("protocol_sha256") != sha256(args.main_protocol) or media.get("sealed_metric_assets_read") is not False:
        raise ValueError("media manifest violates identity firewall")
    frames = load_frames(media)
    edges = candidate_edges(frames)
    output_edges = []
    for edge_index, edge in enumerate(edges):
        left, right = frames[edge["left_index"]], frames[edge["right_index"]]
        output_edges.append({
            "edge_id": f"RGBID-{edge_index:06d}",
            "left": {key: left[key] for key in ("role", "visit_id", "video_id", "frame_path", "payload_sha256")},
            "right": {key: right[key] for key in ("role", "visit_id", "video_id", "frame_path", "payload_sha256")},
            **{key: value for key, value in edge.items() if key not in ("left_index", "right_index")},
            "review_status": "UNREVIEWED",
        })
    result = {
        "schema": "blindassist_spatial_calibration_head_r1_rgb_identity_candidates",
        "main_protocol_sha256": sha256(args.main_protocol),
        "audit_protocol_sha256": sha256(args.audit_protocol),
        "media_manifest_sha256": sha256(args.media_manifest),
        "frame_count": len(frames),
        "candidate_edge_count": len(output_edges),
        "sealed_metric_assets_read": False,
        "semantic_review_performed": False,
        "edges": output_edges,
        "terminal": audit_protocol["terminals"]["review_required"] if output_edges else audit_protocol["terminals"]["no_candidates"],
    }
    write_json_new(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "edges"}, indent=2))


if __name__ == "__main__":
    main()
