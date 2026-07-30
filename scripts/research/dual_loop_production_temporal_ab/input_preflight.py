#!/usr/bin/env python3
"""Outcome-blind input identity preflight for the production temporal A/B route."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL_ID = "DUAL_LOOP_PRODUCTION_TEMPORAL_GEOMETRY_FACTORIAL_AB_R0"
SESSIONS = (
    "defaced_2021-03-27-11-51-18_filtered_lidar_odom",
    "defaced_2021-03-27-11-55-00_filtered_lidar_odom",
)
EXPECTED_FRAME_LEDGER_SHA256 = {
    SESSIONS[0]: "5b99e32b4d2ccbb75089cf1b1796c2e1d0a29c3ca0164feb3844e051051cc748",
    SESSIONS[1]: "88115772b70e0498875925aa909302e5662e0e0edd614f64b68ce1c433a39073",
}
EXPECTED_FRAME_COUNTS = {SESSIONS[0]: 2239, SESSIONS[1]: 2183}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CANDIDATE_OUTPUT_DIRECTORY_NAMES = (
    "device-producer",
    "sealed-producer",
    "evaluation",
    "confirmation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"not a canonical PNG with leading IHDR: {path}")
    width, height = struct.unpack(">II", header[16:24])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions: {path}")
    return width, height


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f"{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dataset_root = (
        repo_root
        / "artifacts.local"
        / "camera-source-prescreen-r1"
        / "dataset"
        / "crowdbot_0327_shared_control"
        / "sequences"
    )
    evidence_root = (
        repo_root
        / "artifacts.local"
        / "evidence"
        / "dual-loop"
        / "production-temporal-geometry-factorial-ab-r0"
    )
    candidate_namespace_state = {
        name: (evidence_root / name).exists() for name in CANDIDATE_OUTPUT_DIRECTORY_NAMES
    }
    if any(candidate_namespace_state.values()):
        raise ValueError(
            "candidate output namespace is not empty: "
            + ",".join(name for name, exists in candidate_namespace_state.items() if exists)
        )
    canonical_inventory = hashlib.sha256()
    session_receipts: list[dict[str, Any]] = []
    total_frames = 0
    total_bytes = 0

    for session_id in SESSIONS:
        session_root = dataset_root / session_id
        ledger_path = session_root / "frames.jsonl"
        ledger_sha256 = sha256_file(ledger_path)
        if ledger_sha256 != EXPECTED_FRAME_LEDGER_SHA256[session_id]:
            raise ValueError(f"frame-ledger hash mismatch: {session_id}")

        rows: list[dict[str, Any]] = []
        with ledger_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{ledger_path}:{line_number}: invalid JSON") from error
                rows.append(row)
        if len(rows) != EXPECTED_FRAME_COUNTS[session_id]:
            raise ValueError(f"frame count mismatch: {session_id}: {len(rows)}")

        frame_ids: set[str] = set()
        timestamps: set[int] = set()
        previous_timestamp: int | None = None
        dimensions: Counter[str] = Counter()
        session_bytes = 0
        session_inventory = hashlib.sha256()

        for index, row in enumerate(rows):
            frame_id = str(row["frame_id"])
            timestamp_ns = int(row["source_capture_timestamp_ns"])
            rgb_path = str(row["rgb_path"])
            expected_rgb_sha256 = str(row["rgb_sha256"]).lower()
            if frame_id in frame_ids:
                raise ValueError(f"duplicate frame_id: {session_id}/{frame_id}")
            if timestamp_ns in timestamps:
                raise ValueError(f"duplicate timestamp: {session_id}/{timestamp_ns}")
            if previous_timestamp is not None and timestamp_ns <= previous_timestamp:
                raise ValueError(f"non-increasing timestamp: {session_id}/{frame_id}")
            if Path(rgb_path).is_absolute() or ".." in Path(rgb_path).parts:
                raise ValueError(f"unsafe rgb_path: {rgb_path}")

            image_path = session_root / rgb_path
            actual_rgb_sha256 = sha256_file(image_path)
            if actual_rgb_sha256 != expected_rgb_sha256:
                raise ValueError(f"RGB hash mismatch: {session_id}/{frame_id}")
            width, height = png_dimensions(image_path)
            image_bytes = image_path.stat().st_size
            canonical = (
                f"{session_id}\t{frame_id}\t{timestamp_ns}\t{rgb_path}\t"
                f"{actual_rgb_sha256}\t{image_bytes}\t{width}\t{height}\n"
            ).encode("utf-8")
            session_inventory.update(canonical)
            canonical_inventory.update(canonical)
            dimensions[f"{width}x{height}"] += 1
            session_bytes += image_bytes
            frame_ids.add(frame_id)
            timestamps.add(timestamp_ns)
            previous_timestamp = timestamp_ns

            completed = total_frames + index + 1
            if completed % 250 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "INPUT_PREFLIGHT",
                            "completed_frames": completed,
                            "total_frames": sum(EXPECTED_FRAME_COUNTS.values()),
                            "session_id": session_id,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        total_frames += len(rows)
        total_bytes += session_bytes
        session_receipts.append(
            {
                "session_id": session_id,
                "frame_ledger_path": str(ledger_path.relative_to(repo_root)).replace("\\", "/"),
                "frame_ledger_sha256": ledger_sha256,
                "frame_count": len(rows),
                "first_source_capture_timestamp_ns": int(rows[0]["source_capture_timestamp_ns"]),
                "last_source_capture_timestamp_ns": int(rows[-1]["source_capture_timestamp_ns"]),
                "rgb_total_bytes": session_bytes,
                "dimension_counts": dict(sorted(dimensions.items())),
                "canonical_rgb_inventory_sha256": session_inventory.hexdigest(),
            }
        )

    receipt = {
        "schema_version": "blindassist.dual_loop_input_preflight.v1",
        "protocol_id": PROTOCOL_ID,
        "generator": {
            "path": str(Path(__file__).resolve().relative_to(repo_root)).replace("\\", "/"),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "status": "VALID",
        "outcome_blind": True,
        "truth_opened": False,
        "candidate_output_opened": False,
        "candidate_output_namespace_audit": {
            "root": str(evidence_root.relative_to(repo_root)).replace("\\", "/"),
            "required_absent_directories": candidate_namespace_state,
            "all_absent": True,
        },
        "decode_policy": "PNG_STORED_PIXELS_NO_EXIF_TRANSFORM",
        "session_count": len(SESSIONS),
        "frame_count": total_frames,
        "rgb_total_bytes": total_bytes,
        "canonical_rgb_inventory_sha256": canonical_inventory.hexdigest(),
        "sessions": session_receipts,
        "errors": [],
    }
    if total_frames != 4422:
        raise ValueError(f"global frame count mismatch: {total_frames}")
    atomic_json(args.output.resolve(), receipt)
    print(json.dumps({"status": "VALID", "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "INVALID", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise
