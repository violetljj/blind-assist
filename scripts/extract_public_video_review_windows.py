#!/usr/bin/env python3
"""Extract dense contact sheets for model/VLM review of public-video windows."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

import build_public_video_overview_contact_sheets as overview
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import scan_public_video_prompt_free_exit_candidates as discovery


def parse_window(value: str) -> tuple[str, int, int]:
    parts = value.rsplit(":", 2)
    if len(parts) != 3:
        raise ValueError(f"invalid window: {value}")
    source_id, start_text, end_text = parts
    start_ms = int(start_text)
    end_ms = int(end_text)
    if not source_id or start_ms < 0 or end_ms <= start_ms:
        raise ValueError(f"invalid window: {value}")
    return source_id, start_ms, end_ms


def decode_window(video_path: Path, start_ms: int, end_ms: int, interval_ms: int) -> list[tuple[int, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    samples: list[tuple[int, Any]] = []
    try:
        for timestamp_ms in range(start_ms, end_ms + 1, interval_ms):
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, frame = capture.read()
            if ok and frame is not None:
                samples.append((timestamp_ms, frame))
    finally:
        capture.release()
    if not samples:
        raise RuntimeError(f"no frames decoded in window: {video_path} {start_ms}-{end_ms}")
    return samples


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.source_registry, args.output_dir):
        mil.reject_independent_direction(path)
    registry = common.load_json(args.source_registry)
    sources = discovery.validate_registry(registry, args.source_registry.resolve())
    source_by_id = {source["source_id"]: source for source in sources}
    windows = [parse_window(value) for value in args.window]
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    capacity = args.columns * args.rows_per_sheet
    window_rows: list[dict[str, Any]] = []
    for window_index, (source_id, start_ms, end_ms) in enumerate(windows):
        if source_id not in source_by_id:
            raise ValueError(f"window source is not registered: {source_id}")
        source = source_by_id[source_id]
        video_path = Path(source["local_video_path"])
        samples = decode_window(video_path, start_ms, end_ms, args.interval_ms)
        sheets: list[dict[str, Any]] = []
        for sheet_index, batch in enumerate(overview.chunked(samples, capacity)):
            image = overview.contact_sheet(batch, args.columns, args.rows_per_sheet)
            destination = args.output_dir / f"{window_index:02d}_{source_id}_{start_ms}_{end_ms}_{sheet_index:02d}.jpg"
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError(f"cannot write contact sheet: {destination}")
            sheets.append({
                "path": str(destination.resolve()),
                "sha256": common.sha256_file(destination),
                "timestamps_ms": [timestamp_ms for timestamp_ms, _frame in batch],
            })
        window_rows.append({
            "source_id": source_id,
            "video_path": str(video_path),
            "video_sha256": common.sha256_file(video_path),
            "start_ms": start_ms,
            "end_ms": end_ms,
            "interval_ms": args.interval_ms,
            "sheets": sheets,
        })
    manifest = {
        "schema": "blindassist_public_video_dense_review_windows_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_registry": str(args.source_registry.resolve()),
        "source_registry_sha256": common.sha256_file(args.source_registry),
        "windows": window_rows,
        "evidence_limit": "Dense windows are model/VLM review material only, not event truth, calibration evidence, blind evidence, or production evidence.",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(manifest_path) + ".sha256").write_text(common.sha256_file(manifest_path) + "\n", encoding="ascii")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, required=True)
    parser.add_argument("--window", action="append", required=True, help="source_id:start_ms:end_ms")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=2000)
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
        "window_count": len(manifest["windows"]),
        "sheet_count": sum(len(window["sheets"]) for window in manifest["windows"]),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
