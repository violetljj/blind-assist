#!/usr/bin/env python3
"""Audit frozen r7.64 route-distribution readouts without selecting one."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import train_public_video_temporal_route_head as training


SCHEMA = "blindassist_public_video_temporal_route_uncertainty_audit_v1"


def frame_readouts(predicted: np.ndarray, obstacle: np.ndarray) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    flat_mask = obstacle.reshape(-1)
    for channel in predicted:
        flat = channel.reshape(-1).astype(np.float64)
        order = np.argsort(flat)[::-1]
        values["argmax_hit"].append(float(flat_mask[order[0]]))
        values["top3_any_hit"].append(float(flat_mask[order[:3]].any()))
        values["top5_any_hit"].append(float(flat_mask[order[:5]].any()))
        shifted = flat - float(flat.max())
        probability = np.exp(shifted)
        probability /= probability.sum()
        values["softmax_obstacle_mass"].append(float(probability[flat_mask].sum()))
        values["obstacle_peak_ratio"].append(float(np.exp(flat[flat_mask].max() - flat.max())) if flat_mask.any() else 0.0)
    return {key: float(np.mean(row)) for key, row in values.items()}


def strict_separation(events: list[dict[str, Any]], key: str) -> bool:
    positive = [float(row[key]) for row in events if int(row["label"]) == 1]
    negative = [float(row[key]) for row in events if int(row["label"]) == 0]
    return min(positive) > max(negative)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.audit_contract, args.training_contract, args.training_report, args.cache_report, args.cache, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    audit = json.loads(args.audit_contract.read_text(encoding="utf-8"))
    contract = json.loads(args.training_contract.read_text(encoding="utf-8"))
    baseline = lifecycle.verify_json_sidecar(args.training_report)
    cache_report = lifecycle.verify_json_sidecar(args.cache_report)
    bindings = audit["bound_inputs"]
    for path, key in ((args.training_contract, "training_contract_sha256"), (args.training_report, "training_report_sha256"),
                      (args.cache_report, "cache_report_sha256"), (args.cache, "cache_sha256")):
        if common.sha256_file(path) != bindings[key]:
            raise ValueError(f"bound input mismatch: {path}")
    if common.sha256_file(args.cache) != cache_report["cache"]["sha256"]:
        raise ValueError("cache differs from report")
    cache = np.load(args.cache)
    train_x = cache["train_x"].astype(np.float32)
    train_y = cache["train_y"].astype(np.float32)
    train_sources = cache["train_sources"].astype(str)
    eval_x = cache["eval_x"].astype(np.float32)
    eval_sources = cache["eval_sources"].astype(str)
    eval_events = cache["eval_events"].astype(str)
    eval_labels = cache["eval_labels"].astype(np.int64)
    obstacles = cache["eval_obstacles"].astype(bool)
    probe = contract["single_seed_probe"]
    eval_predictions = np.zeros((len(eval_x), 3, 16, 16), dtype=np.float32)
    seen = np.zeros(len(eval_x), dtype=bool)
    folds = []
    for source in sorted(np.unique(train_sources).tolist()):
        train_indices = np.flatnonzero(train_sources != source)
        source_eval = np.flatnonzero(eval_sources == source)
        model, losses = training.train_fold(train_x[train_indices], train_y[train_indices], train_sources[train_indices], probe, train_x.shape[1])
        if len(source_eval):
            eval_predictions[source_eval] = training.predict(model, eval_x[source_eval], int(probe["batch_size"]))
            seen[source_eval] = True
        folds.append({"held_out_source_id": source, "eval_count": len(source_eval), "final_loss": losses[-1],
                      "finite": bool(np.isfinite(losses).all())})
    if not seen.all():
        raise ValueError("not every evaluation frame received a held-out-source prediction")
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    labels: dict[str, set[int]] = defaultdict(set)
    for event, label, predicted, obstacle in zip(eval_events, eval_labels, eval_predictions, obstacles):
        labels[event].add(int(label))
        for key, value in frame_readouts(predicted, obstacle).items():
            grouped[event][key].append(value)
    event_rows = []
    for event in sorted(grouped):
        if len(labels[event]) != 1:
            raise ValueError(f"inconsistent event labels: {event}")
        event_rows.append({"event_id": event, "label": next(iter(labels[event])), "frame_count": len(next(iter(grouped[event].values()))),
                           **{key: float(np.mean(values)) for key, values in grouped[event].items()}})
    baseline_events = {row["event_id"]: row for row in baseline["event_predictions"]}
    baseline_reproduced = all(abs(row["argmax_hit"] - float(baseline_events[row["event_id"]]["predicted_horizon_hit_fraction"])) < 1e-12 for row in event_rows)
    keys = ["argmax_hit", "top3_any_hit", "top5_any_hit", "softmax_obstacle_mass", "obstacle_peak_ratio"]
    separation = {key: strict_separation(event_rows, key) for key in keys}
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"audit_contract_sha256": common.sha256_file(args.audit_contract),
                   "training_report_sha256": common.sha256_file(args.training_report), "cache_sha256": common.sha256_file(args.cache)},
        "baseline_argmax_exactly_reproduced": baseline_reproduced, "folds": folds,
        "event_readouts": event_rows, "strict_event_separation": separation,
        "diagnostic_complete_separation_any_readout": any(separation.values()),
        "readout_selection_or_promotion_authorized": False,
        "authorization": audit["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-contract", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "baseline_reproduced": value["baseline_argmax_exactly_reproduced"],
                      "any_readout_separates": value["diagnostic_complete_separation_any_readout"],
                      "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
