#!/usr/bin/env python3
"""Aggregate QNN HTP linting text while preserving non-additive overlap caveats."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

from analyze_qnn_detailed_operator_profile import category, region, summary


EXECUTE_RE = re.compile(r"^Execute Stat (\d+)$")
ROOT_RE = re.compile(r"^(.*?) : (\d+)  (us|cycles|count)$")
OP_RE = re.compile(r"^    (.+\(cycles\)) : (\d+)  cycles$")
FIELD_RE = re.compile(r"^        (Wait \(Scheduler\) time|Overlap time|Overlap \(wait\) time) : (\d+)  cycles$")
RESOURCE_RE = re.compile(r"^        Resources :\s*(.*)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    lines = args.input.read_text(encoding="utf-8-sig").splitlines()

    runs: list[dict] = []
    current_run: dict | None = None
    current_op: dict | None = None
    for line in lines:
        match = EXECUTE_RE.match(line)
        if match:
            current_run = {"index": int(match.group(1)), "roots": {}, "operators": []}
            runs.append(current_run)
            current_op = None
            continue
        if current_run is None:
            continue
        match = OP_RE.match(line)
        if match:
            current_op = {
                "operator": match.group(1),
                "cycles": float(match.group(2)),
                "wait_cycles": 0.0,
                "overlap_cycles": 0.0,
                "overlap_wait_cycles": 0.0,
                "resources": [],
            }
            current_run["operators"].append(current_op)
            continue
        match = FIELD_RE.match(line)
        if match and current_op is not None:
            key = {
                "Wait (Scheduler) time": "wait_cycles",
                "Overlap time": "overlap_cycles",
                "Overlap (wait) time": "overlap_wait_cycles",
            }[match.group(1)]
            current_op[key] = float(match.group(2))
            continue
        match = RESOURCE_RE.match(line)
        if match and current_op is not None:
            current_op["resources"] = [part.strip() for part in match.group(1).split(",") if part.strip()]
            continue
        match = ROOT_RE.match(line)
        if match and not line.startswith(" "):
            current_run["roots"][match.group(1)] = {"value": float(match.group(2)), "unit": match.group(3)}

    if not runs:
        raise ValueError("no Execute Stat sections found")
    op_counts = {len(run["operators"]) for run in runs}
    if len(op_counts) != 1:
        raise ValueError(f"operator count changed across runs: {sorted(op_counts)}")

    def roots(name: str) -> list[float]:
        values = [run["roots"][name]["value"] for run in runs if name in run["roots"]]
        if len(values) != len(runs):
            raise ValueError(f"missing root metric {name!r}")
        return values

    critical = roots("Accelerator (critical path execute) time (cycles)")
    op_sum = [sum(op["cycles"] for op in run["operators"]) for run in runs]
    wait_sum = [sum(op["wait_cycles"] for op in run["operators"]) for run in runs]
    overlap_sum = [sum(op["overlap_cycles"] for op in run["operators"]) for run in runs]
    overlap_wait_sum = [sum(op["overlap_wait_cycles"] for op in run["operators"]) for run in runs]

    by_operator: dict[str, list[float]] = defaultdict(list)
    resource_combo_by_run: list[dict[str, float]] = []
    inclusive_resource_by_run: list[dict[str, float]] = []
    category_by_run: list[dict[str, float]] = []
    region_by_run: list[dict[str, float]] = []
    for run in runs:
        combos: dict[str, float] = defaultdict(float)
        inclusive: dict[str, float] = defaultdict(float)
        categories: dict[str, float] = defaultdict(float)
        regions: dict[str, float] = defaultdict(float)
        for op in run["operators"]:
            by_operator[op["operator"]].append(op["cycles"])
            combo = "+".join(op["resources"]) if op["resources"] else "none"
            combos[combo] += op["cycles"]
            for resource in op["resources"]:
                inclusive[resource] += op["cycles"]
            categories[category(op["operator"])] += op["cycles"]
            regions[region(op["operator"])] += op["cycles"]
        resource_combo_by_run.append(combos)
        inclusive_resource_by_run.append(inclusive)
        category_by_run.append(categories)
        region_by_run.append(regions)

    def grouped(per_run: list[dict[str, float]], denominator: float) -> list[dict]:
        names = sorted({name for run in per_run for name in run})
        result = []
        for name in names:
            values = [run.get(name, 0.0) for run in per_run]
            item = summary(values)
            item.update({"name": name, "mean_share_of_summed_operator_cycles_percent": 100.0 * float(item["mean"]) / denominator})
            result.append(item)
        return sorted(result, key=lambda item: float(item["mean"]), reverse=True)

    top = []
    for name, values in by_operator.items():
        item = summary(values)
        item.update({"operator": name, "mean_share_of_summed_operator_cycles_percent": 100.0 * float(item["mean"]) / mean(op_sum)})
        top.append(item)
    top.sort(key=lambda item: float(item["mean"]), reverse=True)

    report = {
        "schema": "blindassist_dav2_qnn_htp_linting_profile_analysis_r0",
        "authority": "HTP resource and overlap diagnostic only; linting times are not App latency",
        "executions": len(runs),
        "operators_per_execution": next(iter(op_counts)),
        "critical_path_cycles": summary(critical),
        "summed_operator_cycles": summary(op_sum),
        "summed_operator_to_critical_path_ratio": mean(op_sum) / mean(critical),
        "scheduler_wait_cycles_non_additive": summary(wait_sum),
        "overlap_cycles_non_additive": summary(overlap_sum),
        "overlap_wait_cycles_non_additive": summary(overlap_wait_sum),
        "qnn_accelerator_execute_us_profiled": summary(roots("QNN accelerator (execute) time")),
        "initial_vtcm_acquire_us": summary(roots("Time for initial VTCM acquire")),
        "hvx_hmx_power_acquire_us": summary(roots("Time for HVX + HMX power on and acquire")),
        "resource_combinations": grouped(resource_combo_by_run, mean(op_sum)),
        "inclusive_resources": grouped(inclusive_resource_by_run, mean(op_sum)),
        "categories": grouped(category_by_run, mean(op_sum)),
        "regions": grouped(region_by_run, mean(op_sum)),
        "top_operators": top[:30],
        "explicit_memory_service_mentions": {
            name: sum(line.count(name) for line in lines)
            for name in ("DramToTcm", "TcmToDram", "SystemService", "BlockZapOp")
        },
        "interpretation_limits": [
            "overlap and wait fields are non-additive across concurrently scheduled operators",
            "DMA resource use does not by itself identify DRAM bytes, VTCM pressure, or a spill",
            "absence of named memory services in this high-level log does not prove zero DRAM traffic",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
