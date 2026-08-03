#!/usr/bin/env python3
"""Evaluate the clock-bound sparse-scale clearance sidecar replay."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

BANDS = ("left", "center", "right")
HORIZONS = (1.0, 1.5, 2.0)
EVALUATION_START_FRAME = 10
ANCHOR_FRAME = EVALUATION_START_FRAME - 1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate(sidecar: list[dict[str, Any]], source_report: dict[str, Any]) -> dict[str, Any]:
    truth_groups: dict[str, list[dict[str, Any]]] = {}
    for row in source_report["frames"]:
        truth_groups.setdefault(str(row["sequence_id"]), []).append(row)
    truth = {}
    for sequence, rows in truth_groups.items():
        rows.sort(key=lambda row: float(row["timestamp"]))
        truth.update({(sequence, index): row for index, row in enumerate(rows)})

    premature_valid = []
    clear_errors = []
    decisions = []
    false_clear = []
    delta_errors = []
    previous: dict[tuple[str, str], tuple[float, float]] = {}
    eligible = []
    valid = []
    for row in sidecar:
        sequence = str(row["sequence_id"])
        frame_index = int(row["frame_index"])
        status = str(row["scaled_clearance"]["status"])
        if frame_index < ANCHOR_FRAME and status == "VALID":
            premature_valid.append((sequence, frame_index))
        if frame_index < EVALUATION_START_FRAME:
            continue
        eligible.append(row)
        truth_row = truth.get((sequence, frame_index))
        if truth_row is None:
            raise ValueError(f"missing truth row for {sequence}/{frame_index}")
        if status != "VALID" or truth_row["sensor"]["status"] != "VALID":
            continue
        valid.append(row)
        for band in BANDS:
            predicted = row["scaled_clearance"]["bands"][band]["clearance_m"]
            observed = truth_row["sensor"]["bands"][band]["clearance_m"]
            if predicted is None or observed is None:
                continue
            predicted = float(predicted)
            observed = float(observed)
            clear_errors.append(abs(predicted - observed))
            key = (sequence, band)
            if key in previous:
                old_truth, old_prediction = previous[key]
                delta_errors.append(
                    abs((predicted - old_prediction) - (observed - old_truth))
                )
            previous[key] = (observed, predicted)
            for horizon in HORIZONS:
                truth_occupied = observed <= horizon
                predicted_occupied = predicted <= horizon
                decisions.append(truth_occupied == predicted_occupied)
                false_clear.append(truth_occupied and not predicted_occupied)
    if not eligible or not clear_errors or not delta_errors:
        raise ValueError("replay has insufficient evaluation observations")
    paired_valid_fraction = len(valid) / len(eligible)
    result = {
        "schema": "hftf_sparse_scale_clearance_sidecar_replay_result_r0",
        "records": len(sidecar),
        "sequences": len({str(row["sequence_id"]) for row in sidecar}),
        "evaluation_start_frame": EVALUATION_START_FRAME,
        "eligible_frames": len(eligible),
        "paired_valid_frames": len(valid),
        "paired_valid_fraction": paired_valid_fraction,
        "premature_valid_before_anchor": premature_valid,
        "clearance_mae_m": statistics.fmean(clear_errors),
        "collision_agreement": statistics.fmean(decisions),
        "false_clear_rate": statistics.fmean(false_clear),
        "temporal_clearance_delta_mae_m": statistics.fmean(delta_errors),
        "depth_latency_median_ms": statistics.median(
            float(row["depth_latency_ms"]) for row in sidecar[4:]
        ),
        "geometry_and_scale_latency_median_ms": statistics.median(
            float(row["geometry_and_scale_latency_ms"]) for row in sidecar
        ),
    }
    result["gates"] = {
        "no_premature_valid_before_anchor": not premature_valid,
        "paired_valid_fraction_at_least_0_90": paired_valid_fraction >= 0.90,
        "clearance_mae_at_most_0_25m": result["clearance_mae_m"] <= 0.25,
        "collision_agreement_at_least_0_90": result["collision_agreement"] >= 0.90,
        "false_clear_rate_at_most_0_05": result["false_clear_rate"] <= 0.05,
        "temporal_delta_mae_at_most_0_15m": result[
            "temporal_clearance_delta_mae_m"
        ]
        <= 0.15,
    }
    task_gates = {
        key: value
        for key, value in result["gates"].items()
        if key != "no_premature_valid_before_anchor"
    }
    result["status"] = (
        "SPARSE_SCALE_CLEARANCE_SIDECAR_REPLAY_SUPPORTED_CONSUMED_PROXY"
        if all(result["gates"].values())
        else "SPARSE_SCALE_CLEARANCE_SIDECAR_REPLAY_FAIL"
    )
    result["task_gates_passed"] = sum(task_gates.values())
    result["task_gates_total"] = len(task_gates)
    result["claim_ceiling"] = (
        "consumed sensor-derived anchor proxy and host PyTorch replay only; no real ToF, "
        "HTP end-to-end pipeline, final camera, alert, safety, production, or mainline authority"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        load_jsonl(args.sidecar),
        json.loads(args.source_report.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
