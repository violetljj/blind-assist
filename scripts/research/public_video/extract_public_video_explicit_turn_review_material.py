#!/usr/bin/env python3
"""Extract hash-bound dense contact sheets for explicit-turn candidate reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2

import build_public_video_overview_contact_sheets as overview
import extract_public_video_review_windows as review
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_explicit_turn_review_material_v1"


def padded_window(window: list[int], padding: int) -> tuple[int, int]:
    return max(0, int(window[0]) - padding), int(window[1]) + padding


def select_candidates(candidates: list[dict], selection: dict | None) -> list[dict]:
    if not selection:
        return list(candidates)
    if selection.get("mode") == "explicit_candidate_ids":
        wanted = list(selection["candidate_ids"])
        by_id = {row["candidate_id"]: row for row in candidates}
        missing = [candidate_id for candidate_id in wanted if candidate_id not in by_id]
        if missing:
            raise ValueError(f"requested candidates missing from report: {missing}")
        return [by_id[candidate_id] for candidate_id in wanted]
    if selection.get("mode") != "top_per_direction_per_source":
        raise ValueError("unsupported review-material selection mode")
    maximum = int(selection["maximum_per_direction_per_source"])
    grouped: dict[tuple[str, str], list[dict]] = {}
    for candidate in candidates:
        grouped.setdefault((candidate["direction"], candidate["parent_source_id"]), []).append(candidate)
    retained = []
    for key in sorted(grouped):
        ranked = sorted(grouped[key], key=lambda row: (
            -int(row["run_length"]), -float(row.get("mean_absolute_median_dx_norm", 0.0)), row["candidate_id"]
        ))
        retained.extend(ranked[:maximum])
    return sorted(retained, key=lambda row: (row["direction"], row["parent_source_id"], row["candidate_id"]))


def run(contract_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise ValueError("refusing to overwrite explicit-turn review material")
    contract = common.load_json(contract_path)
    binding = contract["bound_search_report"]
    report_path = Path(binding["path"])
    if common.sha256_file(report_path) != binding["sha256"]:
        raise ValueError("turn-candidate search report hash mismatch")
    search = common.load_json(report_path)
    if search.get("review_queue_ready") is not True:
        raise ValueError("turn-candidate review queue did not pass")
    spec = contract["extraction"]
    padding = int(spec["context_padding_ms"])
    interval = int(spec["interval_ms"])
    columns = int(spec["contact_sheet_columns"])
    rows = int(spec["contact_sheet_rows"])
    capacity = columns * rows
    output_dir.mkdir(parents=True)
    windows = []
    candidates = select_candidates(search["candidates"], contract.get("selection"))
    for index, candidate in enumerate(candidates):
        video_path = Path(candidate["local_video_path"])
        actual_video_sha256 = common.sha256_file(video_path)
        if actual_video_sha256 != candidate["source_video_sha256"]:
            raise ValueError(f"candidate video hash mismatch: {candidate['candidate_id']}")
        start, end = padded_window(candidate["window_ms"], padding)
        samples = review.decode_window(video_path, start, end, interval)
        sheets = []
        for sheet_index, batch in enumerate(overview.chunked(samples, capacity)):
            image = overview.contact_sheet(batch, columns, rows)
            destination = output_dir / f"{index:02d}_{candidate['direction'].lower()}_{sheet_index:02d}.jpg"
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, int(spec["jpeg_quality"])]):
                raise RuntimeError(f"cannot write contact sheet: {destination}")
            sheets.append({"path": str(destination.resolve()), "sha256": common.sha256_file(destination),
                           "timestamps_ms": [timestamp for timestamp, _frame in batch]})
        windows.append({
            "candidate_id": candidate["candidate_id"], "parent_source_id": candidate["parent_source_id"],
            "direction": candidate["direction"], "candidate_window_ms": candidate["window_ms"],
            "review_window_ms": [start, end], "video_sha256": actual_video_sha256, "sheets": sheets,
        })
    manifest = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(contract_path),
                   "search_report_sha256": common.sha256_file(report_path)},
        "selection": contract.get("selection", {"mode": "all_candidates"}),
        "review_questions": contract["review_questions"], "windows": windows,
        "authorization": contract["authorization"],
        "evidence_limit": "Contact sheets are model/VLM review material, not event truth or automatic coverage credit."
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(manifest_path) + ".sha256").write_text(common.sha256_file(manifest_path) + "\n", encoding="ascii")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract, args.output_dir)
    print(json.dumps({"window_count": len(result["windows"]),
                      "sheet_count": sum(len(row["sheets"]) for row in result["windows"])}))


if __name__ == "__main__":
    main()
