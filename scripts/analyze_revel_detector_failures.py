#!/usr/bin/env python3
"""Validate and summarize per-frame REveL detector failure receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def analyze(benchmark: dict[str, Any], details_path: Path) -> dict[str, Any]:
    receipt = benchmark.get("details_receipt") or {}
    lines = [line for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if receipt.get("sha256") != _sha256(details_path):
        raise ValueError("details SHA256 does not match benchmark receipt")
    if receipt.get("frame_records") != len(lines) or benchmark.get("dataset", {}).get("evaluated_frames") != len(lines):
        raise ValueError("details frame count does not match benchmark receipt")
    records = [json.loads(line) for line in lines]

    fixed = {name: sum(record["fixed_score_counts"][name] for record in records) for name in ("tp", "fp", "fn")}
    expected_fixed = benchmark.get("fixed_score_metrics", {})
    if any(fixed[name] != expected_fixed.get(name) for name in fixed):
        raise ValueError("per-frame fixed-score totals do not match benchmark")

    strata: dict[str, dict[str, Any]] = {
        name: {"ground_truth": 0, "matched": 0, "missed": 0, "matched_areas": [], "missed_areas": []}
        for name in ("small", "medium", "large")
    }
    frame_failures: list[dict[str, Any]] = []
    small_miss_flags: list[bool] = []
    for record in records:
        missed_by_area = {name: 0 for name in strata}
        for truth in record["ground_truth"]:
            bucket = truth["stratum"]
            area = float(truth["normalized_area"])
            matched = bool(truth["matched_at_fixed_score"])
            strata[bucket]["ground_truth"] += 1
            strata[bucket]["matched"] += int(matched)
            strata[bucket]["missed"] += int(not matched)
            strata[bucket]["matched_areas" if matched else "missed_areas"].append(area)
            missed_by_area[bucket] += int(not matched)
        counts = record["fixed_score_counts"]
        has_failure = counts["fn"] > 0 or counts["fp"] > 0
        small_miss_flags.append(missed_by_area["small"] > 0)
        if has_failure:
            frame_failures.append({
                "selected_index": record["selected_index"],
                "source_timestamp_ns": record["source_timestamp_ns"],
                "image_name": record["image_name"],
                "fixed_score_counts": counts,
                "missed_by_area": missed_by_area,
                "prediction_count_over_score_floor": len(record["predictions_over_score_floor"]),
            })

    expected_strata = benchmark.get("recall_by_normalized_box_area", {})
    for name, values in strata.items():
        expected = expected_strata.get(name, {})
        if values["ground_truth"] != expected.get("ground_truth") or values["matched"] != expected.get("matched"):
            raise ValueError(f"per-frame {name} totals do not match benchmark")

    segments: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(records):
        if not small_miss_flags[cursor]:
            cursor += 1
            continue
        end = cursor
        while (
            end + 1 < len(records)
            and small_miss_flags[end + 1]
            and records[end + 1]["source_timestamp_ns"] - records[end]["source_timestamp_ns"] <= 1_000_000_000
        ):
            end += 1
        segment_records = records[cursor:end + 1]
        segments.append({
            "start_source_timestamp_ns": segment_records[0]["source_timestamp_ns"],
            "end_source_timestamp_ns": segment_records[-1]["source_timestamp_ns"],
            "sampled_frames": len(segment_records),
            "selected_indices": [record["selected_index"] for record in segment_records],
            "small_missed_boxes": sum(
                1
                for record in segment_records
                for truth in record["ground_truth"]
                if truth["stratum"] == "small" and not truth["matched_at_fixed_score"]
            ),
        })
        cursor = end + 1

    summarized_strata = {
        name: {
            "ground_truth": values["ground_truth"],
            "matched": values["matched"],
            "missed": values["missed"],
            "recall": values["matched"] / max(1, values["ground_truth"]),
            "matched_area": _quantiles(values["matched_areas"]),
            "missed_area": _quantiles(values["missed_areas"]),
        }
        for name, values in strata.items()
    }
    top_failures = sorted(
        frame_failures,
        key=lambda item: (-item["missed_by_area"]["small"], -item["fixed_score_counts"]["fn"], -item["fixed_score_counts"]["fp"], item["selected_index"]),
    )[:25]
    return {
        "format": "blindassist_revel_detector_failure_analysis_v1",
        "details_receipt_valid": True,
        "evaluated_frames": len(records),
        "fixed_score_totals": fixed,
        "frames_with_any_fp_or_fn": len(frame_failures),
        "frames_with_no_prediction_over_score_floor": sum(not record["predictions_over_score_floor"] for record in records),
        "recall_by_area": summarized_strata,
        "small_miss_segments": segments,
        "top_failure_frames": top_failures,
        "interpretation": "small-target misses are source-frame-addressable for replay; this remains 2D public-data evidence without distance, TTC, body, or assistive-event authority",
        "production_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.benchmark.read_text(encoding="utf-8")), args.details)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"frames": report["evaluated_frames"], "frames_with_failure": report["frames_with_any_fp_or_fn"], "small_misses": report["recall_by_area"]["small"]["missed"], "small_segments": len(report["small_miss_segments"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
