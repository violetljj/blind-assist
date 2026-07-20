#!/usr/bin/env python3
"""Audit raw metric-depth integrity in a SANPO-Synthetic replay package.

This audit deliberately separates *published metric-depth bytes are structurally
usable for offline replay* from *a pose is admitted as a USTRF safety receipt*.
SANPO's camera-pose CSV has no explicit frame ID or timestamp column, so this
tool records the pose source as unavailable for USTRF pose-warp admission.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_sanpo_synthetic_replay import load_rows, safe_file


def inspect_depth(path: Path, width: int, height: int) -> dict[str, Any]:
    payload = np.frombuffer(gzip.decompress(path.read_bytes()), dtype=np.float16)
    if payload.size < 2:
        raise ValueError(f"depth header missing: {path.name}")
    declared_height, declared_width = (float(payload[0]), float(payload[1]))
    if not declared_height.is_integer() or not declared_width.is_integer():
        raise ValueError(f"depth header must contain integer dimensions: {path.name}")
    if (int(declared_width), int(declared_height)) != (width, height):
        raise ValueError(f"depth header dimensions disagree with manifest: {path.name}")
    values = payload[2:]
    if values.size != width * height:
        raise ValueError(f"depth payload size disagrees with manifest: {path.name}")
    finite_positive = np.isfinite(values) & (values > 0)
    if not finite_positive.any():
        raise ValueError(f"depth has no finite positive samples: {path.name}")
    samples = values[finite_positive].astype(np.float32)
    return {
        "path": path.name,
        "declared_width": int(declared_width),
        "declared_height": int(declared_height),
        "payload_samples": int(values.size),
        "finite_positive_fraction": float(finite_positive.mean()),
        "min": float(samples.min()),
        "p50": float(np.quantile(samples, .50)),
        "p95": float(np.quantile(samples, .95)),
        "max": float(samples.max()),
    }


def audit(root: Path) -> dict[str, Any]:
    rows = load_rows(root / "manifest.replay.jsonl")
    errors: list[str] = []
    depth_summaries: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        sample = str(row.get("id", index))
        try:
            width, height = int(row["width"]), int(row["height"])
            depth_summaries.append(inspect_depth(safe_file(root, row.get("source_depth_path")), width, height))
        except (KeyError, TypeError, ValueError, OSError) as error:
            errors.append(f"{sample}: {error}")
    pose_path = root / "source_metadata" / "camera_poses.csv"
    pose_columns: list[str] = []
    pose_row_count = 0
    if not pose_path.is_file():
        errors.append("missing camera_poses.csv")
    else:
        with pose_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            pose_columns = reader.fieldnames or []
            pose_row_count = sum(1 for _ in reader)
    requested_pose_indices = [int(row["source_frame_index"]) for row in rows if "source_frame_index" in row]
    pose_has_explicit_binding = any(column.lower() in {"frame_id", "frame_index", "timestamp", "timestamp_ns", "timestamp_ms"} for column in pose_columns)
    pose_row_count_covers_window = bool(requested_pose_indices) and pose_row_count > max(requested_pose_indices)
    return {
        "schema": "blindassist_sanpo_synthetic_metric_replay_audit_v1",
        "ok": not errors,
        "frame_count": len(rows),
        "metric_depth_source_integrity": not errors and len(depth_summaries) == len(rows),
        "depth_summaries": depth_summaries,
        "camera_pose_source": {
            "row_count": pose_row_count,
            "columns": pose_columns,
            "row_count_covers_requested_source_indices": pose_row_count_covers_window,
            "has_explicit_frame_or_timestamp_binding": pose_has_explicit_binding,
            "ustrf_pose_warp_admitted": False,
            "ustrf_pose_warp_blockers": [
                "SANPO camera_poses.csv lacks an explicit frame-index/timestamp binding",
                "dataset camera pose is not an independently verified device body-frame receipt",
            ],
        },
        "authorization": "offline_source_integrity_only",
        "production_authorized": False,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report_path = args.report.resolve()
        if report_path.exists():
            raise ValueError(f"refusing to overwrite report: {report_path}")
        report = audit(args.replay_root.resolve())
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"ok": report["ok"], "metric_depth_source_integrity": report["metric_depth_source_integrity"], "pose_warp_admitted": report["camera_pose_source"]["ustrf_pose_warp_admitted"], "errors": report["errors"]}, ensure_ascii=False))
        return 0 if report["ok"] else 1
    except (OSError, ValueError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
