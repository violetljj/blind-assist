#!/usr/bin/env python3
"""Aggregate QAIRT qnn-profile-viewer detailed CSV without treating profile time as App latency."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "mean": mean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def category(identifier: str) -> str:
    operation_path = identifier.split(":OpId_", 1)[0]
    leaf = operation_path.rsplit("/", 1)[-1]
    if operation_path.startswith("Reshape_") or "reshape" in leaf.lower():
        return "reshape"
    if "elementwise" in leaf.lower():
        return "elementwise"
    checks = (
        ("Softmax", "softmax"),
        ("LayerNormalization", "layer_norm"),
        ("MatMul", "matmul"),
        ("Resize", "resize"),
        ("ConvTranspose", "conv_transpose"),
        ("Conv", "conv"),
        ("Transpose", "transpose"),
        ("Reshape", "reshape"),
    )
    for operation, name in checks:
        if leaf == operation or leaf.startswith(operation + "_"):
            return name
    return "other"


def region(identifier: str) -> str:
    if "/blocks." in identifier or "Reshape_post_/blocks." in identifier or "_elementwiseneuron_" in identifier:
        return "transformer_encoder"
    if "/depth_head/" in identifier:
        return "depth_head"
    if "patch_embed" in identifier:
        return "patch_embed"
    return "boundary_or_other"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8-sig", newline="") as handle:
        metadata = [next(handle).rstrip("\r\n") for _ in range(4)]
        rows = list(csv.DictReader(handle, skipinitialspace=True))

    root_cycles: dict[str, float] = {}
    op_runs: dict[str, list[float]] = defaultdict(list)
    category_by_run: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    region_by_run: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    root_us: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        timestamp = row["Msg Timestamp"].strip()
        identifier = row["Event Identifier"].strip()
        level = row["Event Level"].strip()
        unit = row["Unit of Measurement"].strip()
        value = float(row["Time"])
        if level == "ROOT" and identifier == "Accelerator (execute) time (cycles)":
            root_cycles[timestamp] = value
        elif level == "ROOT" and unit == "US" and row["Message"].strip() == "EXECUTE":
            root_us[identifier].append(value)
        elif level == "SUB-EVENT" and unit == "CYCLES":
            op_runs[identifier].append(value)
            category_by_run[timestamp][category(identifier)] += value
            region_by_run[timestamp][region(identifier)] += value

    if not root_cycles:
        raise ValueError("no accelerator execute cycle roots found")
    execution_ids = sorted(root_cycles)
    if any(timestamp not in category_by_run for timestamp in execution_ids):
        raise ValueError("one or more executions have no per-op cycle rows")
    op_counts = {sum(1 for row in rows if row["Msg Timestamp"].strip() == timestamp and row["Event Level"].strip() == "SUB-EVENT" and row["Unit of Measurement"].strip() == "CYCLES") for timestamp in execution_ids}
    if len(op_counts) != 1:
        raise ValueError(f"per-execution op count changed: {sorted(op_counts)}")

    root_values = [root_cycles[timestamp] for timestamp in execution_ids]
    root_mean = mean(root_values)
    top_operators = []
    for identifier, values in op_runs.items():
        item = summary(values)
        item.update({"operator": identifier, "mean_share_of_root_percent": 100.0 * float(item["mean"]) / root_mean})
        top_operators.append(item)
    top_operators.sort(key=lambda item: float(item["mean"]), reverse=True)

    def grouped(per_run: dict[str, dict[str, float]]) -> list[dict[str, float | str]]:
        names = sorted({name for values in per_run.values() for name in values})
        output = []
        for name in names:
            values = [per_run[timestamp].get(name, 0.0) for timestamp in execution_ids]
            item = summary(values)
            item.update({"name": name, "mean_share_of_root_percent": 100.0 * float(item["mean"]) / root_mean})
            output.append(item)
        return sorted(output, key=lambda item: float(item["mean"]), reverse=True)

    category_summary = grouped(category_by_run)
    region_summary = grouped(region_by_run)
    summed_operator_mean = sum(mean(values) for values in op_runs.values())
    report = {
        "schema": "blindassist_dav2_qnn_detailed_operator_profile_analysis_r0",
        "authority": "device profiling diagnostic only; profiled execution time is not App latency",
        "source_metadata": metadata,
        "executions": len(execution_ids),
        "operators_per_execution": next(iter(op_counts)),
        "unique_operator_events": len(op_runs),
        "accelerator_execute_cycles": summary(root_values),
        "summed_operator_mean_cycles": summed_operator_mean,
        "operator_to_root_cycle_closure_error_percent": 100.0 * abs(summed_operator_mean - root_mean) / root_mean,
        "execute_root_us": {name: summary(values) for name, values in sorted(root_us.items())},
        "categories": category_summary,
        "regions": region_summary,
        "top_operators": top_operators[:30],
        "zero_cycle_operator_events": sum(1 for values in op_runs.values() if math.isclose(mean(values), 0.0)),
        "decision_inputs": {
            "softmax_share_percent": next(item["mean_share_of_root_percent"] for item in category_summary if item["name"] == "softmax"),
            "transformer_encoder_share_percent": next(item["mean_share_of_root_percent"] for item in region_summary if item["name"] == "transformer_encoder"),
            "layout_reshape_transpose_share_percent": sum(item["mean_share_of_root_percent"] for item in category_summary if item["name"] in {"reshape", "transpose"}),
            "depth_head_share_percent": next(item["mean_share_of_root_percent"] for item in region_summary if item["name"] == "depth_head"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
