#!/usr/bin/env python3
"""Decode frozen frame predictions into train-only two-consecutive open events."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common
from build_public_video_route_conditioned_synthetic_dataset import (
    lifecycle_open, load_json, load_jsonl, reject_independent_direction, sha256_file,
)


SCHEMA = "blindassist_route_conditioned_synthetic_lifecycle_probe_v1"


def build_sequences(examples: Sequence[dict[str, Any]], generation: Sequence[dict[str, Any]],
                    predictions: Sequence[int], *, consecutive: int) -> list[dict[str, Any]]:
    if len(examples) != len(predictions):
        raise ValueError("frame predictions do not align with route examples")
    distance = {row["id"]: int(row["attributes"]["distance_index"]) for row in generation}
    grouped: dict[tuple[str, str, str, str], list[tuple[int, bool, bool, str]]] = {}
    for row, prediction in zip(examples, predictions):
        key = (row["parent_source_id"], str(row.get("asset_name")), row["obstacle_direction"], row["route_choice"])
        grouped.setdefault(key, []).append((distance[row["image_id"]], bool(row["route_blocked"]), bool(prediction), row["example_id"]))
    sequences = []
    for key, values in sorted(grouped.items()):
        values.sort(key=lambda item: item[0])
        if [item[0] for item in values] != [0, 1, 2]:
            raise ValueError(f"lifecycle sequence lacks exactly three ordered frames: {key}")
        expected = [item[1] for item in values]
        predicted = [item[2] for item in values]
        sequences.append({
            "parent_source_id": key[0], "asset_name": key[1], "obstacle_direction": key[2], "route_choice": key[3],
            "example_ids": [item[3] for item in values], "expected_frame_blocked": expected,
            "predicted_frame_blocked": predicted, "expected_intervention_open": lifecycle_open(expected, consecutive),
            "predicted_intervention_open": lifecycle_open(predicted, consecutive),
        })
    return sequences


def metrics_for(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    truth = np.asarray([int(row["expected_intervention_open"]) for row in rows], dtype=np.int64)
    pred = np.asarray([int(row["predicted_intervention_open"]) for row in rows], dtype=np.int64)
    return common.binary_metrics(truth, pred)


def run(contract_path: Path, output: Path) -> dict[str, Any]:
    contract_path, output = contract_path.resolve(), output.resolve()
    for path in (contract_path, output):
        reject_independent_direction(path)
    contract = load_json(contract_path)
    frame_path = (Path.cwd() / contract["bound_frame_probe"]["path"]).resolve()
    dataset = (Path.cwd() / contract["bound_dataset"]["root"]).resolve()
    reject_independent_direction(frame_path)
    reject_independent_direction(dataset)
    if sha256_file(frame_path) != contract["bound_frame_probe"]["sha256"]:
        raise ValueError("frame probe SHA mismatch")
    if sha256_file(dataset / "build_receipt.json") != contract["bound_dataset"]["build_receipt_sha256"]:
        raise ValueError("dataset build receipt SHA mismatch")
    if sha256_file(dataset / "route_examples.jsonl") != contract["bound_dataset"]["route_examples_sha256"]:
        raise ValueError("route examples SHA mismatch")
    frame_report = load_json(frame_path)
    examples = load_jsonl(dataset / "route_examples.jsonl")
    generation = load_jsonl(dataset / "generation_records.jsonl")
    predictions = frame_report["evaluation"]["route_conditioned_readout"]["predictions"]
    consecutive = int(contract["lifecycle"]["open_consecutive_frames"])
    first = build_sequences(examples, generation, predictions, consecutive=consecutive)
    second = build_sequences(examples, generation, predictions, consecutive=consecutive)
    repeat_exact = first == second
    overall = metrics_for(first)
    by_source = {source: metrics_for([row for row in first if row["parent_source_id"] == source])
                 for source in sorted({row["parent_source_id"] for row in first})}
    by_choice = {choice: metrics_for([row for row in first if row["route_choice"] == choice])
                 for choice in ("LEFT", "STRAIGHT", "RIGHT")}
    by_asset = {asset: metrics_for([row for row in first if row["asset_name"] == asset])
                for asset in sorted({row["asset_name"] for row in first})}
    gate_spec = contract["gate"]
    gate = bool(
        repeat_exact
        and overall["balanced_accuracy"] >= float(gate_spec["minimum_balanced_accuracy"])
        and overall["candidate_no_alert_recall"] >= float(gate_spec["minimum_each_class_recall"])
        and overall["candidate_alert_recall"] >= float(gate_spec["minimum_each_class_recall"])
        and min(row["balanced_accuracy"] for row in by_source.values()) >= float(gate_spec["minimum_worst_parent_source_balanced_accuracy"])
    )
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path), "contract_sha256": sha256_file(contract_path),
        "frame_probe_path": str(frame_path), "frame_probe_sha256": sha256_file(frame_path),
        "sequence_count": len(first), "open_consecutive_frames": consecutive,
        "evaluation": {"metrics": overall, "by_parent_source": by_source, "by_route_choice": by_choice,
                       "by_asset_name": by_asset, "repeat_exact": repeat_exact, "sequences": first},
        "open_lifecycle_gate": {"passed": gate, "thresholds": gate_spec},
        "clear_lifecycle_evaluated": False,
        "evidence_limit": "Train-only synthetic open-event diagnosis; no departure/clear sequence and no real event truth.",
        "provider_evaluation_credit": False, "calibration_authorized": False, "blind_authorized": False,
        "android_runtime_change_authorized": False, "production_model_replacement_authorized": False,
    }
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(output) + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        report = run(args.contract, args.output)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, "balanced_accuracy": report["evaluation"]["metrics"]["balanced_accuracy"],
                      "open_lifecycle_gate": report["open_lifecycle_gate"]["passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
