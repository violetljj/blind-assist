#!/usr/bin/env python3
"""Normalize mechanical AI-review receipt aliases without changing labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_review(raw: dict[str, Any], source_sha256: str) -> dict[str, Any]:
    result = dict(raw)
    result["reviewer_id"] = result.get("reviewer_id") or result.get("reviewer")
    windows = []
    for raw_window in result.get("negative_windows", []):
        window = dict(raw_window)
        window["negative_type"] = window.pop(
            "negative_type", window.pop("negative_category", None)
        )
        window["window_interval_seconds"] = window.pop(
            "window_interval_seconds", window.pop("interval", None)
        )
        windows.append(window)
    result["negative_windows"] = windows

    raw_coverage = result.get("timeline_coverage", [])
    if isinstance(raw_coverage, dict):
        raw_coverage = [raw_coverage]
    coverage = []
    for raw_item in raw_coverage:
        item = dict(raw_item)
        item["full_timeline_coverage"] = item.pop(
            "full_timeline_coverage",
            item.pop("complete_sampled_timeline_reviewed", False),
        )
        item["contact_sheets_reviewed"] = [
            value.get("path", value.get("name", ""))
            if isinstance(value, dict)
            else value
            for value in item.get("contact_sheets_reviewed", [])
        ]
        item["dense_frames_reviewed"] = [
            value.get("path", value.get("name", ""))
            if isinstance(value, dict)
            else value
            for value in item.get("dense_frames_reviewed", [])
        ]
        coverage.append(item)
    result["timeline_coverage"] = coverage
    result["normalization"] = {
        "method": "mechanical_schema_alias_normalization",
        "source_receipt_sha256": source_sha256,
        "label_content_changed": False,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    normalized = normalize_review(raw, sha256_file(args.input))
    args.output.write_text(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "source_sha256": sha256_file(args.input),
                "normalized_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
