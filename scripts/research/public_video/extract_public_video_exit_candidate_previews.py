#!/usr/bin/env python3
"""Extract reproducible contact sheets for public-video exit candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


OFFSETS_MS = (-5000, -2000, 0, 2000, 5000, 8000)


def verify_scan(path: Path) -> dict[str, Any]:
    mil.reject_independent_direction(path)
    sidecar = Path(str(path) + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("scan report or SHA sidecar is missing")
    if sidecar.read_text(encoding="ascii").strip().lower() != common.sha256_file(path):
        raise ValueError("scan report SHA sidecar mismatch")
    report = common.load_json(path)
    if report.get("schema") != "blindassist_public_video_prompt_free_exit_discovery_v1":
        raise ValueError("unexpected scan report schema")
    return report


def select_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    groups: set[str],
    top_per_source_group: int,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        group = str(candidate["semantic_group"])
        if group in groups:
            buckets[(str(candidate["source_id"]), group)].append(candidate)
    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        ordered = sorted(
            buckets[key],
            key=lambda row: (-float(row["present_group_max_confidence"]), int(row["present_timestamp_ms"])),
        )
        selected.extend(ordered[:top_per_source_group])
    return selected


def decode_frame(capture: Any, timestamp_ms: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_MSEC, float(max(timestamp_ms, 0)))
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"cannot decode timestamp {timestamp_ms}ms")
    return cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)


def build_contact_sheet(
    video_path: Path,
    candidate: dict[str, Any],
) -> tuple[np.ndarray, list[int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    present_ms = int(candidate["present_timestamp_ms"])
    timestamps = [max(present_ms + offset, 0) for offset in OFFSETS_MS]
    panels: list[np.ndarray] = []
    try:
        for timestamp_ms in timestamps:
            panel = decode_frame(capture, timestamp_ms)
            label = f"t={timestamp_ms / 1000.0:.1f}s"
            cv2.rectangle(panel, (0, 0), (190, 36), (0, 0, 0), thickness=-1)
            cv2.putText(panel, label, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            panels.append(panel)
    finally:
        capture.release()
    top = np.concatenate(panels[:3], axis=1)
    bottom = np.concatenate(panels[3:], axis=1)
    return np.concatenate([top, bottom], axis=0), timestamps


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.scan_report, args.output_dir):
        mil.reject_independent_direction(path)
    report = verify_scan(args.scan_report.resolve())
    if args.output_dir.exists():
        raise ValueError(f"refusing to overwrite output: {args.output_dir}")
    source_by_id = {source["source_id"]: source for source in report["sources"]}
    groups = {value.strip() for value in args.groups.split(",") if value.strip()}
    selected = select_candidates(
        report["exit_candidates"],
        groups=groups,
        top_per_source_group=args.top_per_source_group,
    )
    if not selected:
        raise ValueError("no matching exit candidates")
    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        source = source_by_id[candidate["source_id"]]
        video_path = Path(source["local_video_path"])
        mil.reject_independent_direction(video_path)
        if common.sha256_file(video_path) != source["video_sha256"]:
            raise ValueError(f"video SHA mismatch: {video_path}")
        sheet, timestamps = build_contact_sheet(video_path, candidate)
        name = (
            f"{index:02d}_{candidate['source_id']}_{candidate['semantic_group']}_"
            f"{int(candidate['present_timestamp_ms'])}.jpg"
        )
        destination = args.output_dir / name
        if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"cannot write contact sheet: {destination}")
        rows.append({
            "source_id": candidate["source_id"],
            "semantic_group": candidate["semantic_group"],
            "present_timestamp_ms": candidate["present_timestamp_ms"],
            "absent_timestamp_ms": candidate["absent_timestamp_ms"],
            "present_group_max_confidence": candidate["present_group_max_confidence"],
            "timestamps_ms": timestamps,
            "preview_path": str(destination.resolve()),
            "preview_sha256": common.sha256_file(destination),
        })
    manifest = {
        "schema": "blindassist_public_video_exit_candidate_preview_v1",
        "scan_report": str(args.scan_report.resolve()),
        "scan_report_sha256": common.sha256_file(args.scan_report),
        "groups": sorted(groups),
        "offsets_ms": list(OFFSETS_MS),
        "previews": rows,
        "evidence_limit": "Contact sheets support model/VLM candidate review only; they are not human event truth or production evidence.",
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(manifest_path) + ".sha256").write_text(common.sha256_file(manifest_path) + "\n", encoding="ascii")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", default="surface_material")
    parser.add_argument("--top-per-source-group", type=int, default=4)
    args = parser.parse_args()
    if args.top_per_source_group <= 0:
        parser.error("top-per-source-group must be positive")
    return args


def main() -> int:
    try:
        manifest = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "preview_count": len(manifest["previews"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
