#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np


EXPECTED_MEMBER = "rgbd_bonn_groundtruth_1mm_section.ply"
EXPECTED_VERTEX_COUNT = 54_676_774
EXPECTED_PROPERTIES = (
    "property float x",
    "property float y",
    "property float z",
    "property uchar red",
    "property uchar green",
    "property uchar blue",
    "property float scalar_Scalar_field",
)
VALUES_PER_VERTEX = len(EXPECTED_PROPERTIES)
HASH_MODULUS = 64
READ_CHUNK_BYTES = 16 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_header(stream: BinaryIO) -> dict[str, Any]:
    lines: list[str] = []
    while True:
        raw = stream.readline()
        if not raw:
            raise ValueError("PLY header ended before end_header")
        line = raw.decode("ascii").rstrip("\r\n")
        lines.append(line)
        if line == "end_header":
            break
        if len(lines) > 64:
            raise ValueError("unexpectedly long PLY header")
    if lines[:2] != ["ply", "format ascii 1.0"]:
        raise ValueError("expected ASCII PLY 1.0")
    vertex_rows = [line for line in lines if line.startswith("element vertex ")]
    if vertex_rows != [f"element vertex {EXPECTED_VERTEX_COUNT}"]:
        raise ValueError("unexpected PLY vertex count")
    properties = tuple(line for line in lines if line.startswith("property "))
    if properties != EXPECTED_PROPERTIES:
        raise ValueError("unexpected PLY vertex properties")
    return {
        "header_lines": lines,
        "vertex_count": EXPECTED_VERTEX_COUNT,
        "properties": list(properties),
    }


def complete_line_blocks(
    stream: BinaryIO, chunk_bytes: int = READ_CHUNK_BYTES
) -> Iterator[bytes]:
    carry = b""
    while True:
        chunk = stream.read(chunk_bytes)
        if not chunk:
            break
        data = carry + chunk
        split = data.rfind(b"\n")
        if split < 0:
            carry = data
            continue
        yield data[: split + 1]
        carry = data[split + 1 :]
    if carry.strip():
        yield carry


def deterministic_keep_mask(xyz: np.ndarray) -> np.ndarray:
    millimeters = np.rint(xyz * 1000.0).astype(np.int64, copy=False)
    unsigned = millimeters.astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        mixed = (
            unsigned[:, 0] * np.uint64(0x9E3779B185EBCA87)
            ^ unsigned[:, 1] * np.uint64(0xC2B2AE3D27D4EB4F)
            ^ unsigned[:, 2] * np.uint64(0x165667B19E3779F9)
        )
        mixed ^= mixed >> np.uint64(33)
        mixed *= np.uint64(0xFF51AFD7ED558CCD)
        mixed ^= mixed >> np.uint64(33)
    return (mixed % np.uint64(HASH_MODULUS)) == 0


def stream_geometry(
    stream: BinaryIO, expected_count: int = EXPECTED_VERTEX_COUNT
) -> tuple[np.ndarray, dict[str, Any]]:
    selected: list[np.ndarray] = []
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    count = 0
    for block in complete_line_blocks(stream):
        values = np.fromstring(block, dtype=np.float64, sep=" ")
        if values.size % VALUES_PER_VERTEX:
            raise ValueError("PLY data block does not contain complete vertex rows")
        rows = values.reshape(-1, VALUES_PER_VERTEX)
        xyz = rows[:, :3]
        if not np.isfinite(xyz).all():
            raise ValueError("non-finite PLY coordinate")
        count += len(xyz)
        minimum = np.minimum(minimum, xyz.min(axis=0))
        maximum = np.maximum(maximum, xyz.max(axis=0))
        kept = xyz[deterministic_keep_mask(xyz)]
        if len(kept):
            selected.append(kept.astype(np.float32, copy=True))
    if count != expected_count:
        raise ValueError(f"PLY record count mismatch: {count} != {expected_count}")
    points = np.concatenate(selected, axis=0)
    return points, {
        "point_record_read_count": count,
        "minimum_xyz_meters": minimum.tolist(),
        "maximum_xyz_meters": maximum.tolist(),
        "selected_point_count": len(points),
    }


def build(
    acquisition: dict[str, Any],
    archive_path: Path,
    official_page_path: Path,
    official_script_path: Path,
    output_points: Path,
) -> dict[str, Any]:
    if archive_path.name != acquisition["archive_filename"]:
        raise ValueError("static-map archive filename mismatch")
    if sha256(archive_path) != acquisition["archive_sha256"]:
        raise ValueError("static-map archive SHA-256 mismatch")
    if not official_page_path.is_file() or not official_script_path.is_file():
        raise ValueError("official transform sources missing")
    if sha256(official_script_path) != (
        "913cb25ac0502bf3933d5e4881cac5a864265dfba1c2b753c581f31b13e25868"
    ):
        raise ValueError("official transform script SHA-256 mismatch")

    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != EXPECTED_MEMBER:
            raise ValueError("unexpected static-map ZIP inventory")
        with archive.open(infos[0]) as stream:
            header = parse_header(stream)
            points, geometry = stream_geometry(stream)

    output_points.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_points,
        xyz_meters=points,
        source_vertex_count=np.asarray(
            [geometry["point_record_read_count"]], dtype=np.int64
        ),
        hash_modulus=np.asarray([HASH_MODULUS], dtype=np.int64),
    )
    return {
        "schema_version": "bonn_static_map_geometry_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "source_family": "BONN_RGBD_DYNAMIC",
        "source": {
            "official_page_path": official_page_path.as_posix(),
            "official_page_sha256": sha256(official_page_path),
            "official_transform_script_path": official_script_path.as_posix(),
            "official_transform_script_sha256": sha256(official_script_path),
            "archive_path": archive_path.as_posix(),
            "archive_sha256": acquisition["archive_sha256"],
            "member": EXPECTED_MEMBER,
            "member_crc32": acquisition["members"][0]["crc32"],
        },
        "header": header,
        "stream_audit": geometry,
        "deterministic_geometry_reduction": {
            "method": "COORDINATE_HASH_MODULO",
            "coordinate_quantization_meters": 0.001,
            "hash_modulus": HASH_MODULUS,
            "selection_independent_of_rgb_depth_pose_or_candidate_signal": True,
            "output_path": output_points.as_posix(),
            "output_bytes": output_points.stat().st_size,
            "output_sha256": sha256(output_points),
        },
        "transform_contract": {
            "official_formula": (
                "T_g = inverse(T_ROS) * T_0 * T_ROS * T_m"
            ),
            "official_script_uses_T_ROS_without_explicit_inverse": True,
            "T_ROS_is_self_inverse": True,
            "per_frame_extension_is_inference_pending_geometry_validation": True,
            "transform_chain_status": "PENDING_DEPTH_TO_MAP_GEOMETRY_VALIDATION",
        },
        "read_firewall": {
            "validation_or_holdout_read_count": 0,
            "rgb_member_read_or_decode_count": 0,
            "depth_member_read_or_decode_count": 0,
            "old_window_selection_tuning_acceptance_reads": 0,
            "candidate_signal_computed": False,
        },
        "terminal": (
            "BONN_STATIC_MAP_STREAM_AUDITED_DOWNSAMPLE_AVAILABLE_"
            "TRANSFORM_VALIDATION_PENDING"
        ),
        "status": "VALID",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquisition", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--official-page", required=True, type=Path)
    parser.add_argument("--official-transform-script", required=True, type=Path)
    parser.add_argument("--output-points", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    acquisition = json.loads(args.acquisition.read_text(encoding="utf-8"))
    receipt = build(
        acquisition,
        args.archive,
        args.official_page,
        args.official_transform_script,
        args.output_points,
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                **receipt["stream_audit"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
