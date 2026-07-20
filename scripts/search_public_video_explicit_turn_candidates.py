#!/usr/bin/env python3
"""Search label-free route-auxiliary rows for sustained LEFT/RIGHT marker intersections."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_explicit_turn_candidate_search_v1"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def direction(row: dict[str, Any], left_below: float, right_above: float) -> str:
    mean_x = float(np.mean([anchor["point_xy_norm"][0] for anchor in row["future_route_anchors"]]))
    if mean_x < left_below:
        return "LEFT"
    if mean_x > right_above:
        return "RIGHT"
    return "STRAIGHT"


def overlaps_excluded(source_id: str, start: int, end: int,
                      exclusions: dict[str, list[tuple[int, int]]], padding: int) -> bool:
    return any(start < window_end + padding and end > window_start - padding
               for window_start, window_end in exclusions.get(source_id, []))


def find_runs(rows: list[dict[str, Any]], spec: dict[str, Any],
              exclusions: dict[str, list[tuple[int, int]]]) -> list[dict[str, Any]]:
    left = float(spec["left_if_mean_anchor_x_below"])
    right = float(spec["right_if_mean_anchor_x_above"])
    minimum_hit = float(spec["minimum_teacher_marker_hit_fraction"])
    required = int(spec["required_consecutive_samples"])
    step = int(spec["expected_step_ms"])
    padding = int(spec["exclude_r789_window_padding_ms"])
    candidates = []
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row)
    for source_id, source_rows in sorted(by_source.items()):
        ordered = sorted(source_rows, key=lambda row: int(row["timestamp_ms"]))
        run: list[dict[str, Any]] = []
        run_direction: str | None = None
        for row in ordered + [None]:
            current_direction = direction(row, left, right) if row is not None else None
            active = (row is not None and current_direction in {"LEFT", "RIGHT"} and
                      float(row["teacher_marker_hit_fraction_diagnostic_only"]) >= minimum_hit)
            contiguous = (bool(run) and row is not None and
                          int(row["timestamp_ms"]) - int(run[-1]["timestamp_ms"]) == step and
                          current_direction == run_direction)
            if active and (not run or contiguous):
                run.append(row)
                run_direction = current_direction
                continue
            if len(run) >= required:
                start = int(run[0]["timestamp_ms"])
                end = int(run[-1]["timestamp_ms"]) + step
                if not overlaps_excluded(source_id, start, end, exclusions, padding):
                    candidates.append({
                        "candidate_id": f"{source_id}:{run_direction.lower()}:{start}",
                        "parent_source_id": source_id, "direction": run_direction,
                        "window_ms": [start, end], "run_length": len(run),
                        "mean_hit_fraction": float(np.mean([
                            item["teacher_marker_hit_fraction_diagnostic_only"] for item in run
                        ])),
                        "timestamps_ms": [int(item["timestamp_ms"]) for item in run],
                        "route_aux_item_ids": [item["item_id"] for item in run],
                        "local_video_path": run[0]["local_video_path"],
                        "source_video_sha256": run[0]["source_video_sha256"],
                    })
            run = [row] if active else []
            run_direction = current_direction if active else None
    candidates.sort(key=lambda row: (-row["run_length"], -row["mean_hit_fraction"],
                                     row["parent_source_id"], row["window_ms"][0]))
    retained = []
    limit = int(spec["maximum_candidates_per_direction"])
    for direction_name in ("LEFT", "RIGHT"):
        retained.extend([row for row in candidates if row["direction"] == direction_name][:limit])
    return sorted(retained, key=lambda row: (row["direction"], -row["run_length"],
                                              -row["mean_hit_fraction"], row["parent_source_id"],
                                              row["window_ms"][0]))


def run(contract_path: Path, output_path: Path) -> dict[str, Any]:
    if output_path.exists() or Path(str(output_path) + ".sha256").exists():
        raise ValueError("refusing to overwrite turn-candidate search")
    contract = common.load_json(contract_path)
    bound = contract["bound_inputs"]
    paths = {key[:-5]: Path(value) for key, value in bound.items() if key.endswith("_path")}
    for stem, path in paths.items():
        if common.sha256_file(path) != bound[f"{stem}_sha256"]:
            raise ValueError(f"bound input hash mismatch: {stem}")
    coverage = common.load_json(paths["r799a_coverage_report"])
    if coverage.get("full_three_state_provider_supported") is not False:
        raise ValueError("r799a did not expose a direction coverage gap")
    rows = load_jsonl(paths["route_aux_manifest"])
    if any(row.get("event_label") is not None for row in rows):
        raise ValueError("route auxiliary manifest contains event labels")
    video_hashes: dict[str, str] = {}
    for row in rows:
        previous = video_hashes.setdefault(row["source_id"], row["source_video_sha256"])
        if previous != row["source_video_sha256"]:
            raise ValueError(f"inconsistent video hash: {row['source_id']}")
    actionability = common.load_json(paths["actionability_manifest"])
    exclusions: dict[str, list[tuple[int, int]]] = {}
    for item in actionability["items"]:
        exclusions.setdefault(item["parent_source_id"], []).append(tuple(map(int, item["window_ms"])))
    candidates = find_runs(rows, contract["search"], exclusions)
    by_direction = {direction_name: [row for row in candidates if row["direction"] == direction_name]
                    for direction_name in ("LEFT", "RIGHT")}
    gate = contract["gate"]
    checks = {
        "minimum_left_candidate_count": len(by_direction["LEFT"]) >= int(gate["minimum_left_candidate_count"]),
        "minimum_right_candidate_count": len(by_direction["RIGHT"]) >= int(gate["minimum_right_candidate_count"]),
        "minimum_left_source_count": len({row["parent_source_id"] for row in by_direction["LEFT"]}) >= int(gate["minimum_left_source_count"]),
        "minimum_right_source_count": len({row["parent_source_id"] for row in by_direction["RIGHT"]}) >= int(gate["minimum_right_source_count"]),
        "all_parent_video_hashes_consistent": True,
        "all_event_labels_null": True,
        "no_r789_window_overlap": True,
    }
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(contract_path),
                   **{f"{stem}_sha256": common.sha256_file(path) for stem, path in paths.items()}},
        "summary": {"input_row_count": len(rows), "source_count": len(video_hashes),
                    "left_candidate_count": len(by_direction["LEFT"]),
                    "right_candidate_count": len(by_direction["RIGHT"]),
                    "left_source_count": len({row["parent_source_id"] for row in by_direction["LEFT"]}),
                    "right_source_count": len({row["parent_source_id"] for row in by_direction["RIGHT"]})},
        "candidates": candidates, "checks": checks, "review_queue_ready": bool(all(checks.values())),
        "authorization": contract["authorization"],
        "evidence_limit": "Candidates are label-free geometry proposals and require separate continuous-video review."
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(output_path) + ".sha256").write_text(common.sha256_file(output_path) + "\n", encoding="ascii")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.contract, args.output)
    print(json.dumps({"summary": result["summary"], "review_queue_ready": result["review_queue_ready"]}))


if __name__ == "__main__":
    main()
