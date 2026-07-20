#!/usr/bin/env python3
"""Build timestamped overview contact sheets for licensed public videos."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import scan_public_video_prompt_free_exit_candidates as discovery


def chunked(rows: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    return [list(rows[index:index + size]) for index in range(0, len(rows), size)]


def decode_samples(video_path: Path, interval_ms: int) -> tuple[list[tuple[int, np.ndarray]], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or frame_count <= 0:
        capture.release()
        raise RuntimeError(f"invalid video timing: {video_path}")
    duration_ms = int(round((frame_count / fps) * 1000.0))
    samples: list[tuple[int, np.ndarray]] = []
    try:
        for timestamp_ms in range(0, duration_ms, interval_ms):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, frame = capture.read()
            if ok and frame is not None:
                samples.append((timestamp_ms, frame))
    finally:
        capture.release()
    if not samples:
        raise RuntimeError(f"no frames decoded: {video_path}")
    return samples, {"fps": fps, "frame_count": frame_count, "duration_ms": duration_ms}


def panel(frame: np.ndarray, timestamp_ms: int) -> np.ndarray:
    output = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
    cv2.rectangle(output, (0, 0), (150, 30), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        f"t={timestamp_ms / 1000.0:.0f}s",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def contact_sheet(rows: Sequence[tuple[int, np.ndarray]], columns: int, rows_per_sheet: int) -> np.ndarray:
    capacity = columns * rows_per_sheet
    if not rows or len(rows) > capacity:
        raise ValueError("contact-sheet row count is invalid")
    panels = [panel(frame, timestamp_ms) for timestamp_ms, frame in rows]
    blank = np.zeros_like(panels[0])
    panels.extend([blank.copy() for _ in range(capacity - len(panels))])
    grid_rows = [np.concatenate(panels[index:index + columns], axis=1) for index in range(0, capacity, columns)]
    return np.concatenate(grid_rows, axis=0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.source_registry, args.output_dir):
        mil.reject_independent_direction(path)
    registry = common.load_json(args.source_registry)
    sources = discovery.validate_registry(registry, args.source_registry.resolve())
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    capacity = args.columns * args.rows_per_sheet
    source_rows: list[dict[str, Any]] = []
    for source in sources:
        video_path = Path(source["local_video_path"])
        samples, video = decode_samples(video_path, args.interval_ms)
        sheets: list[dict[str, Any]] = []
        for index, batch in enumerate(chunked(samples, capacity)):
            image = contact_sheet(batch, args.columns, args.rows_per_sheet)
            destination = args.output_dir / f"{source['source_id']}_{index:02d}.jpg"
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"cannot write contact sheet: {destination}")
            sheets.append({
                "path": str(destination.resolve()),
                "sha256": common.sha256_file(destination),
                "timestamps_ms": [timestamp_ms for timestamp_ms, _frame in batch],
            })
        source_rows.append({
            **source,
            "video_sha256": common.sha256_file(video_path),
            "video": video,
            "sample_count": len(samples),
            "sheets": sheets,
        })
    manifest = {
        "schema": "blindassist_public_video_overview_contact_sheets_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(args.source_registry.resolve()),
        "source_registry_sha256": common.sha256_file(args.source_registry),
        "interval_ms": args.interval_ms,
        "columns": args.columns,
        "rows_per_sheet": args.rows_per_sheet,
        "sources": source_rows,
        "evidence_limit": "Overview images support model/VLM discovery only; no panel is an event label, human truth, calibration evidence, blind evidence, or production evidence.",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(manifest_path) + ".sha256").write_text(common.sha256_file(manifest_path) + "\n", encoding="ascii")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=30000)
    parser.add_argument("--columns", type=int, default=5)
    parser.add_argument("--rows-per-sheet", type=int, default=4)
    args = parser.parse_args()
    if args.interval_ms <= 0 or args.columns <= 0 or args.rows_per_sheet <= 0:
        parser.error("interval, columns, and rows per sheet must be positive")
    return args


def main() -> int:
    try:
        manifest = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "source_count": len(manifest["sources"]),
        "sheet_count": sum(len(source["sheets"]) for source in manifest["sources"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
