#!/usr/bin/env python3
"""Aggregate bounded RCLE Discovery chunks without hiding state resets."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import runner


def load_rows(chunk_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = json.loads(
        (chunk_dir / "summary.json").read_text(encoding="utf-8")
    )
    rows = [
        json.loads(line)
        for line in (chunk_dir / "pair_ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    return summary, rows


def aggregate(chunk_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError("OUTPUT_DIRECTORY_EXISTS")
    if not chunk_dirs:
        raise ValueError("NO_CHUNKS")
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for chunk_index, chunk_dir in enumerate(chunk_dirs):
        summary, chunk_rows = load_rows(chunk_dir.resolve())
        summaries.append(summary)
        for row in chunk_rows:
            row["chunk_index"] = chunk_index
            row["chunk_pair_index"] = row["pair_index"]
            rows.append(row)
    rows.sort(key=lambda row: int(row["frame_index_previous_zero_based"]))
    expected_previous = list(range(len(rows)))
    observed_previous = [
        int(row["frame_index_previous_zero_based"]) for row in rows
    ]
    if observed_previous != expected_previous:
        raise ValueError("CHUNK_FRAME_COVERAGE_NOT_EXACT_ZERO_BASED")
    if any(
        int(row["frame_index_current_zero_based"]) != index + 1
        for index, row in enumerate(rows)
    ):
        raise ValueError("CHUNK_PAIR_BOUNDARY_GAP_OR_OVERLAP")
    source_hashes = {
        summary["source"]["video_sha256"] for summary in summaries
    }
    if len(source_hashes) != 1:
        raise ValueError("CHUNK_SOURCE_IDENTITY_DRIFT")
    raw_streak = compensated_streak = image_scale_streak = 0
    for index, row in enumerate(rows):
        row["pair_index"] = index
        raw_streak = runner.update_confirmation(row, raw_streak, "raw")
        compensated_streak = runner.update_confirmation(
            row, compensated_streak, "compensated"
        )
        image_scale = row.get("image_scale_expansion_per_s")
        image_scale_streak = (
            image_scale_streak + 1
            if image_scale is not None and float(image_scale) > runner.THRESHOLD
            else 0
        )
        row["image_scale_above_threshold"] = (
            image_scale is not None
            and float(image_scale) > runner.THRESHOLD
        )
        row[
            "image_scale_consecutive_above_threshold_pair_count"
        ] = image_scale_streak
        row["image_scale_three_pair_trigger"] = (
            image_scale_streak >= runner.REQUIRED_CONSECUTIVE_PAIRS
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    segments = runner.segment_summaries(rows)
    raw_abs_rows = [
        {
            "angular_speed_deg_per_s": row["angular_speed_deg_per_s"],
            "_value": abs(float(row["raw_expansion_median_per_s"])),
        }
        for row in rows
        if row.get("raw_expansion_median_per_s") is not None
    ]
    compensated_abs_rows = [
        {
            "angular_speed_deg_per_s": row["angular_speed_deg_per_s"],
            "_value": abs(float(row["compensated_expansion_median_per_s"])),
        }
        for row in rows
        if row.get("compensated_expansion_median_per_s") is not None
    ]
    abstentions = Counter(
        str(row["reason"])
        for row in rows
        if row.get("evaluable") is not True
    )
    result = {
        "schema": "rcle.ecological_response.discovery.chunk_aggregate.v1",
        "protocol_id": runner.PROTOCOL_ID,
        "governance_policy_id": (
            "DATA_CAPABILITY_DRIVEN_RESEARCH_GOVERNANCE_R2"
        ),
        "research_track": "CAPABILITY_DISCOVERY",
        "outcome_access_state": "OUTPUT_INSPECTED",
        "stage": "CAPABILITY_DISCOVERY",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": summaries[0]["source"],
        "execution": {
            "frame_index_start_zero_based": 0,
            "frame_index_end_exclusive_zero_based": len(rows) + 1,
            "candidate_pair_count": len(rows),
            "evaluable_pair_count": sum(
                row.get("evaluable") is True for row in rows
            ),
            "evaluable_pair_fraction": sum(
                row.get("evaluable") is True for row in rows
            )
            / len(rows),
            "abstention_reasons": dict(sorted(abstentions.items())),
            "duration_s": (
                float(rows[-1]["current_timestamp_s"])
                - float(rows[0]["previous_timestamp_s"])
            ),
            "chunk_count": len(chunk_dirs),
            "chunk_frame_coverage_exact": True,
            "chunk_temporal_core_state_reset": True,
            "affected_boundary_pair_indices": [
                int(chunk_rows["execution"][
                    "frame_index_start_zero_based"
                ])
                for chunk_rows in summaries[1:]
            ],
            "three_pair_state_recomputed_across_chunks": True,
            "runtime_s_sum": sum(
                float(summary["execution"]["runtime_s"])
                for summary in summaries
            ),
            "native_frame_rate_preserved": True,
            "spatial_resize_scale": summaries[0]["execution"][
                "spatial_resize_scale"
            ],
            "threshold_per_s": runner.THRESHOLD,
            "required_consecutive_pairs": runner.REQUIRED_CONSECUTIVE_PAIRS,
        },
        "methods": {
            "raw_local_expansion": runner.method_summary(
                rows,
                "raw_expansion_median_per_s",
                "raw_three_pair_trigger",
            ),
            "source_pose_rotation_compensated_local_expansion": (
                runner.method_summary(
                    rows,
                    "compensated_expansion_median_per_s",
                    "compensated_three_pair_trigger",
                )
            ),
            "global_image_scale_proxy": runner.method_summary(
                rows,
                "image_scale_expansion_per_s",
                "image_scale_three_pair_trigger",
            ),
            "bbox_growth": {
                "status": "NOT_EVALUABLE",
                "reason": "NO_FROZEN_OBJECT_BOXES",
            },
        },
        "diagnostics": {
            "median_angular_speed_deg_per_s": float(
                np.median(
                    [row["angular_speed_deg_per_s"] for row in rows]
                )
            ),
            "p90_angular_speed_deg_per_s": float(
                np.quantile(
                    [row["angular_speed_deg_per_s"] for row in rows], 0.9
                )
            ),
            "median_translation_speed_m_per_s": float(
                np.median(
                    [row["translation_speed_m_per_s"] for row in rows]
                )
            ),
            "p90_translation_speed_m_per_s": float(
                np.quantile(
                    [row["translation_speed_m_per_s"] for row in rows], 0.9
                )
            ),
            "angular_speed_correlation": {
                "raw_expansion": runner.correlation(
                    rows, "raw_expansion_median_per_s"
                ),
                "compensated_expansion": runner.correlation(
                    rows, "compensated_expansion_median_per_s"
                ),
                "raw_abs_expansion": runner.correlation(
                    raw_abs_rows, "_value"
                ),
                "compensated_abs_expansion": runner.correlation(
                    compensated_abs_rows, "_value"
                ),
            },
            "segment_count": len(segments),
        },
        "claim_ceiling": summaries[0]["claim_ceiling"],
        "chunks": [
            {
                "path": chunk_dir.resolve().as_posix(),
                "summary_sha256": runner.sha256_file(
                    chunk_dir.resolve() / "summary.json"
                ),
            }
            for chunk_dir in chunk_dirs
        ],
    }
    result["artifacts"] = {
        "pair_ledger_sha256": runner.write_jsonl(
            output_dir / "pair_ledger.jsonl", rows
        ),
        "segment_summary_sha256": runner.write_jsonl(
            output_dir / "segment_summary.jsonl", segments
        ),
    }
    runner.render_curves(output_dir / "response_curves.png", rows)
    result["artifacts"]["response_curves_sha256"] = runner.sha256_file(
        output_dir / "response_curves.png"
    )
    runner.write_json(output_dir / "summary.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.chunk_dir, args.output_dir.resolve())
    print(
        json.dumps(
            {
                "candidate_pair_count": result["execution"][
                    "candidate_pair_count"
                ],
                "duration_s": result["execution"]["duration_s"],
                "methods": result["methods"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
