#!/usr/bin/env python3
"""Audit a frozen JRDB short window for background affine availability."""
from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import math
import os
import struct
import time
import zipfile
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from materialize_jrdb_single_frame_rgb_time_transform_canary_r1 import (
    RangeClient,
    atomic_write,
    canonical_bytes,
    jpeg_geometry,
    load_json,
    require,
    safe_member,
    sha256_file,
    zip64_values,
)


SCHEMA = "blindassist_ustrf_jrdb_rgb_continuity_egomotion_availability_r0"
CONFIG_SCHEMA = f"{SCHEMA}_config"
STAGE = "JRDB_RGB_CONTINUITY_EGOMOTION_AVAILABILITY_R0"
TERMINALS = (
    "FAIL_CLOSED_AUDIT_INCOMPLETE",
    "RGB_CONTINUITY_INSUFFICIENT",
    "EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT",
    "SHORT_WINDOW_EGOMOTION_AVAILABILITY_PRESENT",
)
IMPLEMENTATIONS = {
    "producer": "scripts/research/ustrf_route_target_evidence_closure/audit_jrdb_rgb_continuity_egomotion_availability_r0.py",
    "validator": "scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_rgb_continuity_egomotion_availability_r0.py",
    "range_library": "scripts/research/ustrf_route_target_evidence_closure/materialize_jrdb_single_frame_rgb_time_transform_canary_r1.py",
}


def load_config(repo: Path, path: Path) -> dict[str, Any]:
    config = load_json(path)
    require(config["schema"] == CONFIG_SCHEMA, "config_schema_drift")
    require(config["stage"] == STAGE, "stage_drift")
    require(config["status"] == "frozen_before_execution", "config_not_frozen")
    require(tuple(config["terminal_states"]) == TERMINALS, "terminal_order_drift")
    for label, binding in config["bindings"].items():
        source = repo / binding["path"]
        require(source.is_file(), f"{label}_missing")
        require(sha256_file(source) == binding["sha256"], f"{label}_sha256_drift")
    digests = config["research_implementation_digests"]
    require(set(digests) == set(IMPLEMENTATIONS), "implementation_digest_keys_drift")
    for label, relative in IMPLEMENTATIONS.items():
        require(sha256_file(repo / relative) == digests[label], f"{label}_implementation_drift")
    return config


def central_members(central: bytes, wanted: set[str]) -> dict[str, dict[str, Any]]:
    cursor = 0
    found: dict[str, dict[str, Any]] = {}
    while cursor < len(central):
        require(central[cursor : cursor + 4] == b"PK\x01\x02", f"central_signature_drift:{cursor}")
        values = struct.unpack_from("<4s6H3L5H2L", central, cursor)
        (
            _signature, _made, _needed, flags, compression, _mtime, _mdate,
            crc32, compressed, uncompressed, name_len, extra_len, comment_len,
            _disk, _internal, _external, local_offset,
        ) = values
        name_start = cursor + 46
        name = central[name_start : name_start + name_len].decode("utf-8")
        extra = central[name_start + name_len : name_start + name_len + extra_len]
        compressed, uncompressed, local_offset = zip64_values(
            extra, compressed, uncompressed, local_offset
        )
        require(safe_member(name), f"unsafe_member:{name}")
        if name in wanted:
            require(name not in found, f"duplicate_member:{name}")
            found[name] = {
                "name": name,
                "flags": flags,
                "compression": compression,
                "crc32": crc32,
                "compressed_size": compressed,
                "uncompressed_size": uncompressed,
                "local_offset": local_offset,
            }
        cursor = name_start + name_len + extra_len + comment_len
    require(cursor == len(central), "central_directory_length_drift")
    require(set(found) == wanted, f"missing_target_members:{len(wanted) - len(found)}")
    return found


def fetch_member(client: RangeClient, member: dict[str, Any], max_bytes: int) -> bytes:
    require(member["flags"] & 0x1 == 0, "encrypted_member")
    require(member["compressed_size"] <= max_bytes, "member_over_budget")
    offset = member["local_offset"]
    local = client.get(offset, offset + 29)
    values = struct.unpack("<4s5H3L2H", local)
    require(values[0] == b"PK\x03\x04", "local_header_signature_drift")
    name_len, extra_len = values[-2], values[-1]
    tail = client.get(offset + 30, offset + 30 + name_len + extra_len - 1)
    require(tail[:name_len].decode("utf-8") == member["name"], "local_name_drift")
    start = offset + 30 + name_len + extra_len
    compressed = client.get(start, start + member["compressed_size"] - 1)
    if member["compression"] == 0:
        payload = compressed
    elif member["compression"] == 8:
        payload = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError(f"unsupported_compression:{member['compression']}")
    require(len(payload) == member["uncompressed_size"], "uncompressed_size_drift")
    require(binascii.crc32(payload) & 0xFFFFFFFF == member["crc32"], "crc_drift")
    jpeg_geometry(payload)
    return payload


def load_window_truth(
    labels_path: Path,
    timestamps_path: Path,
    sequence: str,
    frames: list[str],
) -> tuple[dict[str, list[list[float]]], dict[str, float]]:
    with zipfile.ZipFile(labels_path) as bundle:
        payload = json.loads(bundle.read(f"labels_2d_stitched/{sequence}.json"))
    boxes: dict[str, list[list[float]]] = {}
    for frame in frames:
        require(frame in payload["labels"], f"label_frame_missing:{frame}")
        boxes[frame] = [row["box"] for row in payload["labels"][frame]]
    with zipfile.ZipFile(timestamps_path) as bundle:
        ts = json.loads(bundle.read(f"timestamps/{sequence}/frames_img.json"))
    timestamps: dict[str, float] = {}
    for row in ts["data"]:
        for camera in row.get("cameras", []):
            if camera.get("name") == "stitched_image0":
                frame = str(camera["url"]).rsplit("/", 1)[-1]
                if frame in frames:
                    require(frame not in timestamps, f"duplicate_timestamp:{frame}")
                    timestamps[frame] = float(camera["timestamp"])
    require(set(timestamps) == set(frames), "timestamp_window_incomplete")
    return boxes, timestamps


def person_mask(shape: tuple[int, int], boxes: list[list[float]], expansion: int) -> np.ndarray:
    height, width = shape
    mask = np.full((height, width), 255, dtype=np.uint8)
    for x, y, w, h in boxes:
        x0 = max(0, math.floor(x) - expansion)
        y0 = max(0, math.floor(y) - expansion)
        x1 = min(width, math.ceil(x + w) + expansion)
        y1 = min(height, math.ceil(y + h) + expansion)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0
    return mask


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def pair_audit(
    previous: np.ndarray,
    current: np.ndarray,
    previous_mask: np.ndarray,
    current_mask: np.ndarray,
    gap: float,
    method: dict[str, Any],
    gates: dict[str, Any],
) -> dict[str, Any]:
    features = cv2.goodFeaturesToTrack(
        previous,
        mask=previous_mask,
        maxCorners=method["max_corners"],
        qualityLevel=method["quality_level"],
        minDistance=method["min_distance"],
        blockSize=method["block_size"],
    )
    detected = 0 if features is None else len(features)
    reasons: list[str] = []
    if features is None:
        return {"passed": False, "reasons": ["no_features"], "detected_features": 0, "gap_seconds": gap}
    tracked, status, _error = cv2.calcOpticalFlowPyrLK(
        previous,
        current,
        features,
        None,
        winSize=tuple(method["lk_window"]),
        maxLevel=method["lk_max_level"],
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, method["lk_iterations"], method["lk_epsilon"]),
    )
    status = status.reshape(-1).astype(bool)
    p0 = features.reshape(-1, 2)[status]
    p1 = tracked.reshape(-1, 2)[status]
    height, width = previous.shape
    inside = (
        (p1[:, 0] >= 0) & (p1[:, 0] < width) & (p1[:, 1] >= 0) & (p1[:, 1] < height)
    )
    p0, p1 = p0[inside], p1[inside]
    if len(p0):
        ix0, iy0 = np.floor(p0[:, 0]).astype(int), np.floor(p0[:, 1]).astype(int)
        ix1, iy1 = np.floor(p1[:, 0]).astype(int), np.floor(p1[:, 1]).astype(int)
        background = (previous_mask[iy0, ix0] > 0) & (current_mask[iy1, ix1] > 0)
        p0, p1 = p0[background], p1[background]
    valid = len(p0)
    occupied = 0
    matrix = None
    inlier_ratio = 0.0
    median_residual = float("inf")
    p95_residual = float("inf")
    condition = float("inf")
    determinant = float("nan")
    if valid >= 3:
        matrix, inliers = cv2.estimateAffine2D(
            p0,
            p1,
            method=cv2.RANSAC,
            ransacReprojThreshold=method["ransac_threshold_px"],
            maxIters=method["ransac_max_iterations"],
            confidence=method["ransac_confidence"],
            refineIters=method["ransac_refine_iterations"],
        )
        cells = Counter(
            (
                min(method["grid_columns"] - 1, int(x * method["grid_columns"] / width)),
                min(method["grid_rows"] - 1, int(y * method["grid_rows"] / height)),
            )
            for x, y in p0
        )
        occupied = sum(count >= method["minimum_points_per_grid_cell"] for count in cells.values())
        if matrix is not None and inliers is not None:
            inliers = inliers.reshape(-1).astype(bool)
            inlier_ratio = float(np.mean(inliers))
            predicted = (matrix[:, :2] @ p0[inliers].T).T + matrix[:, 2]
            residual = np.linalg.norm(predicted - p1[inliers], axis=1)
            if len(residual):
                median_residual = float(np.median(residual))
                p95_residual = percentile(residual, 95)
            condition = float(np.linalg.cond(matrix[:, :2]))
            determinant = float(np.linalg.det(matrix[:, :2]))
    checks = {
        "timestamp_gap": 0 < gap <= gates["maximum_gap_seconds"],
        "detected_features": detected >= gates["minimum_detected_features"],
        "valid_tracks": valid >= gates["minimum_valid_tracks"],
        "spatial_distribution": occupied >= gates["minimum_occupied_grid_cells"],
        "inlier_ratio": inlier_ratio >= gates["minimum_inlier_ratio"],
        "median_residual": median_residual <= gates["maximum_median_residual_px"],
        "p95_residual": p95_residual <= gates["maximum_p95_residual_px"],
        "affine_condition": condition <= gates["maximum_affine_condition"],
        "affine_determinant": gates["minimum_affine_determinant"] <= determinant <= gates["maximum_affine_determinant"],
    }
    reasons.extend(key for key, value in checks.items() if not value)
    return {
        "passed": all(checks.values()),
        "reasons": reasons,
        "gap_seconds": gap,
        "detected_features": detected,
        "valid_tracks": valid,
        "occupied_grid_cells": occupied,
        "inlier_ratio": inlier_ratio,
        "median_residual_px": median_residual,
        "p95_residual_px": p95_residual,
        "affine_condition": condition,
        "affine_determinant": determinant,
        "affine": None if matrix is None else matrix.tolist(),
        "checks": checks,
    }


def audit(repo: Path, config_path: Path, persist_frames: bool) -> dict[str, Any]:
    config = load_config(repo, config_path)
    sequence = config["window"]["sequence"]
    frame_numbers = range(config["window"]["first_frame"], config["window"]["last_frame"] + 1)
    frames = [f"{number:06d}.jpg" for number in frame_numbers]
    paths = [f"images/image_stitched/{sequence}/{frame}" for frame in frames]
    boxes, timestamps = load_window_truth(
        repo / config["bindings"]["test_labels"]["path"],
        repo / config["bindings"]["test_timestamps"]["path"],
        sequence,
        frames,
    )
    remote = config["remote_archive"]
    client = RangeClient(remote["url"], config["resource_gate"]["maximum_network_bytes"])
    start = remote["central_directory_offset"]
    size = remote["central_directory_size"]
    central = client.get(start, start + size - 1)
    members = central_members(central, set(paths))
    jpegs: dict[str, bytes] = {}
    frame_receipts = []
    for frame, path in zip(frames, paths, strict=True):
        jpeg = fetch_member(client, members[path], config["resource_gate"]["maximum_member_bytes"])
        jpegs[frame] = jpeg
        frame_receipts.append(
            {
                "frame": frame,
                "bytes": len(jpeg),
                "sha256": hashlib.sha256(jpeg).hexdigest(),
                "capture_timestamp": timestamps[frame],
                "label_object_count": len(boxes[frame]),
            }
        )
        if persist_frames:
            atomic_write(repo / config["outputs"]["frame_directory"] / frame, jpeg)

    grayscale: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for frame in frames:
        decoded = cv2.imdecode(np.frombuffer(jpegs[frame], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        require(decoded is not None, f"jpeg_decode_failed:{frame}")
        grayscale[frame] = decoded
        masks[frame] = person_mask(decoded.shape, boxes[frame], config["method"]["person_mask_expansion_px"])
    pairs = []
    for previous_frame, current_frame in zip(frames[:-1], frames[1:], strict=True):
        result = pair_audit(
            grayscale[previous_frame],
            grayscale[current_frame],
            masks[previous_frame],
            masks[current_frame],
            timestamps[current_frame] - timestamps[previous_frame],
            config["method"],
            config["quality_gates"],
        )
        pairs.append({"previous_frame": previous_frame, "current_frame": current_frame, **result})
    passing = sum(pair["passed"] for pair in pairs)
    required = math.ceil(len(pairs) * config["quality_gates"]["minimum_passing_pair_fraction"])
    continuity = all(0 < pair["gap_seconds"] <= config["quality_gates"]["maximum_gap_seconds"] for pair in pairs)
    if not continuity:
        terminal = "RGB_CONTINUITY_INSUFFICIENT"
    elif passing < required:
        terminal = "EGOMOTION_QUALITY_AVAILABILITY_INSUFFICIENT"
    else:
        terminal = "SHORT_WINDOW_EGOMOTION_AVAILABILITY_PRESENT"
    return {
        "schema": SCHEMA,
        "stage": STAGE,
        "status": "AUDIT_COMPLETE",
        "terminal_state": terminal,
        "process_id": os.getpid(),
        "config_sha256": sha256_file(config_path),
        "window": {
            "sequence": sequence,
            "frames": len(frames),
            "pairs": len(pairs),
            "passing_pairs": passing,
            "required_passing_pairs": required,
            "passing_fraction": passing / len(pairs),
        },
        "frames": frame_receipts,
        "pairs": pairs,
        "network": {
            "bytes_read": client.bytes_read,
            "budget_bytes": client.budget,
            "request_count": len(client.requests),
            "full_archive_downloaded": False,
        },
        "claim_boundary": {
            "pre_g3_source_availability_only": True,
            "g3_authorized": False,
            "g4_authorized": False,
            "signal_authorized": False,
            "route_truth_authorized": False,
            "android_authorized": False,
            "human_authorized": False,
            "production_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve()
    result = audit(repo, config_path, persist_frames=True)
    atomic_write(repo / load_json(config_path)["outputs"]["receipt"], canonical_bytes(result))
    print(json.dumps({"terminal_state": result["terminal_state"], "process_id": result["process_id"], "passing_pairs": result["window"]["passing_pairs"], "pairs": result["window"]["pairs"], "network_bytes": result["network"]["bytes_read"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
