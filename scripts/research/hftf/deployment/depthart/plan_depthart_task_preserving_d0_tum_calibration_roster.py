#!/usr/bin/env python3
"""Freeze an outcome-free TUM RGB roster shared by the D0 quantized arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "blindassist_depthart_task_preserving_d0_tum_calibration_roster_v1"
INTRINSICS = [535.4, 539.2, 320.1, 247.6]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def parse_rgb_index(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"invalid rgb index row: {line}")
        rows.append((parts[0], parts[1]))
    if not rows:
        raise ValueError(f"empty rgb index: {path}")
    return rows


def select_rows(
    source_root: Path,
    sequences: list[str],
    excluded_rgb_paths: set[tuple[str, str]],
    per_sequence: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for sequence in sequences:
        index_path = source_root / sequence / "rgb.txt"
        candidates = []
        for timestamp, relative in parse_rgb_index(index_path):
            if (sequence, relative) in excluded_rgb_paths:
                continue
            image_path = source_root / sequence / relative
            if not image_path.is_file():
                continue
            rank = hashlib.sha256(f"{sequence}:{timestamp}".encode("utf-8")).hexdigest()
            candidates.append((rank, timestamp, relative, image_path))
        candidates.sort()
        if len(candidates) < per_sequence:
            raise ValueError(f"insufficient non-excluded RGB frames for {sequence}")
        for rank, timestamp, relative, image_path in candidates[:per_sequence]:
            selected.append({
                "calibration_id": f"{sequence}:{timestamp}",
                "sequence_root": sequence,
                "timestamp": float(timestamp),
                "rgb_path": relative,
                "rgb_bytes": image_path.stat().st_size,
                "rgb_sha256": sha256(image_path),
                "intrinsics_fx_fy_cx_cy": INTRINSICS,
                "selection_rank_sha256": rank.upper(),
            })
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--exclusion-roster", required=True, type=Path)
    parser.add_argument("--per-sequence", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite roster: {args.output}")
    exclusion = json.loads(args.exclusion_roster.read_text(encoding="utf-8"))
    excluded = {
        (str(row["sequence_root"]), str(row["rgb_path"]))
        for row in exclusion["rows"]
    }
    sequences = sorted({sequence for sequence, _ in excluded})
    rows = select_rows(args.source_root.resolve(), sequences, excluded, args.per_sequence)
    payload = {
        "schema": SCHEMA,
        "protocol_id": "DEPTHART_TASK_PRESERVING_D0_PRECISION_SCREEN",
        "frozen_at": "2026-08-09",
        "status": "FROZEN_BEFORE_CALIBRATION_MATERIALIZATION",
        "data_role": "DEVELOPMENT_QUANTIZATION_CALIBRATION_ONLY_NO_TASK_OUTCOMES",
        "selection": {
            "algorithm": "SHA256_ASCENDING_PER_SEQUENCE",
            "per_sequence": args.per_sequence,
            "sequence_count": len(sequences),
            "frame_count": len(rows),
            "image_bytes_read_only_after_identity_selection": True
        },
        "source": {
            "dataset": "TUM_RGBD_LOCAL_DEVELOPMENT_COPY",
            "source_root": "artifacts.local/datasets/model-variant-gate-r0",
            "rgb_index_sha256": {
                sequence: sha256(args.source_root / sequence / "rgb.txt")
                for sequence in sequences
            },
            "exclusion_roster": str(args.exclusion_roster).replace("\\", "/"),
            "exclusion_roster_sha256": sha256(args.exclusion_roster),
            "excluded_rows": len(excluded),
            "overlap_with_exclusion_roster": 0
        },
        "arms": ["D0_W8A16_R0", "D0_INT8_R0"],
        "r2_arkit_roster_accessed": False,
        "task_truth_accessed": False,
        "model_outcomes_accessed": False,
        "rows": rows,
        "authority": "Quantizer calibration inputs only. No D0 task-quality, R2, model-selection, production or safety authority."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "frames": len(rows), "sha256": sha256(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
