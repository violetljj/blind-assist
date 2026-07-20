#!/usr/bin/env python3
"""Run the frozen r7.69 paired distance-field auxiliary supervision ablation."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_marker_relation_bootstrap_short_runs as bootstrap
import run_public_video_marker_relation_linear_probe as linear


SCHEMA = "blindassist_public_video_marker_relation_distance_aux_ablation_v1"


class RelationHead(torch.nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.shared = torch.nn.Linear(input_dimension, hidden_dimension)
        self.primary = torch.nn.Linear(hidden_dimension, 1)
        self.distance = torch.nn.Linear(hidden_dimension, 3)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.shared(values))
        return self.primary(hidden).squeeze(-1), self.distance(hidden)


def load_distance_data(contract: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, hit_fraction, sources = bootstrap.load_relation_data(contract)
    bound = contract["bound_inputs"]
    cache = np.load(linear._resolve(bound["r764_feature_cache_path"]), allow_pickle=False)
    manifest = linear._load_manifest(linear._resolve(bound["r763_manifest_path"]))
    detections = linear._load_detection_index(linear._resolve(bound["r754_source_contract_path"]))
    side = int(contract["feature_vector"]["grid_side"])
    expansion = float(contract["feature_vector"]["marker_expansion_object_heights"])
    auxiliary: list[list[float]] = []
    for index, row in enumerate(manifest):
        if int(row["marker_detection_count"]) <= 0:
            continue
        key = (str(row["source_id"]), int(row["timestamp_ms"]))
        mask = linear.marker_grid_mask(detections[key], side, expansion)
        auxiliary.append([float(cache["train_y"][index, horizon][mask].max()) for horizon in range(3)])
    if len(auxiliary) != len(x):
        raise ValueError("distance targets and relation rows differ")
    return x, (hit_fraction > 0.0).astype(np.float64), np.asarray(auxiliary), sources


def train_model(
    x: np.ndarray, primary_y: np.ndarray, distance_y: np.ndarray, weights: np.ndarray,
    initial_state: dict[str, torch.Tensor], spec: dict[str, Any], distance_weight: float,
) -> tuple[RelationHead, list[float]]:
    torch.use_deterministic_algorithms(True)
    model = RelationHead(x.shape[1], int(spec["hidden_dimension"]))
    model.load_state_dict(copy.deepcopy(initial_state))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["learning_rate"]),
                                  weight_decay=float(spec["weight_decay"]))
    tx = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ty = torch.from_numpy(np.asarray(primary_y, dtype=np.float32))
    td = torch.from_numpy(np.asarray(distance_y, dtype=np.float32))
    tw = torch.from_numpy(np.asarray(weights, dtype=np.float32))
    losses: list[float] = []
    for step in range(int(spec["steps"])):
        optimizer.zero_grad(set_to_none=True)
        primary_logit, distance_logit = model(tx)
        primary_loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            primary_logit, ty, reduction="none") * tw).sum()
        distance_loss = (((torch.sigmoid(distance_logit) - td) ** 2).mean(dim=1) * tw).sum()
        loss = primary_loss + float(distance_weight) * distance_loss
        loss.backward()
        optimizer.step()
        if step in {0, int(spec["steps"]) - 1}:
            losses.append(float(loss.detach()))
    return model, losses


def predict(model: RelationHead, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        primary, distance = model(torch.from_numpy(np.asarray(x, dtype=np.float32)))
    return torch.sigmoid(primary).numpy(), torch.sigmoid(distance).numpy()


def run_seed(
    x: np.ndarray, y: np.ndarray, distance_y: np.ndarray, sources: np.ndarray,
    seed: int, spec: dict[str, Any], distance_weight: float,
) -> dict[str, Any]:
    baseline_oof = np.zeros(len(y), dtype=np.float64)
    treatment_oof = np.zeros(len(y), dtype=np.float64)
    treatment_distance = np.zeros_like(distance_y)
    constant_distance = np.zeros_like(distance_y)
    folds = []
    for fold_index, held_source in enumerate(sorted(set(sources.tolist()))):
        test = sources == held_source
        train_x, train_y, train_d, train_sources = x[~test], y[~test], distance_y[~test], sources[~test]
        sampled, sample_weights, draws = bootstrap.bootstrap_source_class_rows(
            train_sources, train_y > 0.0, seed + fold_index * 1009
        )
        mean = np.average(train_x[sampled], axis=0, weights=sample_weights)
        variance = np.average((train_x[sampled] - mean) ** 2, axis=0, weights=sample_weights)
        scale = np.sqrt(variance)
        scale[scale < 1e-8] = 1.0
        normalized_train = (train_x[sampled] - mean) / scale
        normalized_test = (x[test] - mean) / scale
        torch.manual_seed(seed + fold_index * 1009)
        initial = RelationHead(x.shape[1], int(spec["hidden_dimension"])).state_dict()
        baseline, baseline_loss = train_model(normalized_train, train_y[sampled], train_d[sampled],
                                               sample_weights, initial, spec, 0.0)
        treatment, treatment_loss = train_model(normalized_train, train_y[sampled], train_d[sampled],
                                                 sample_weights, initial, spec, distance_weight)
        baseline_oof[test], _ = predict(baseline, normalized_test)
        treatment_oof[test], treatment_distance[test] = predict(treatment, normalized_test)
        constant = np.average(train_d[sampled], axis=0, weights=sample_weights)
        constant_distance[test] = constant
        folds.append({
            "held_out_source_id": held_source,
            "sampled_unique_active_source_count": len(set(draws["active"])),
            "sampled_unique_inactive_source_count": len(set(draws["inactive"])),
            "baseline_loss_first_last": baseline_loss,
            "treatment_loss_first_last": treatment_loss,
        })
    baseline_metrics = bootstrap.source_macro_metrics(y > 0.0, baseline_oof, sources)
    treatment_metrics = bootstrap.source_macro_metrics(y > 0.0, treatment_oof, sources)
    return {
        "seed": seed,
        "baseline_source_macro_metrics": baseline_metrics,
        "treatment_source_macro_metrics": treatment_metrics,
        "treatment_minus_baseline": {
            key: treatment_metrics[key] - baseline_metrics[key] for key in baseline_metrics
        },
        "pooled_oof_auroc_diagnostic": {
            "baseline": linear.roc_auc(y.astype(np.int64), baseline_oof),
            "treatment": linear.roc_auc(y.astype(np.int64), treatment_oof),
        },
        "distance_mae": {
            "constant_training_fold_mean": float(np.abs(constant_distance - distance_y).mean()),
            "treatment": float(np.abs(treatment_distance - distance_y).mean()),
        },
        "folds": folds,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.bootstrap_report, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.bootstrap_report, "r768a_report_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    lifecycle.verify_json_sidecar(args.bootstrap_report)
    linear_contract = common.load_json(args.linear_contract)
    x, y, distance_y, sources = load_distance_data(linear_contract)
    runs = [run_seed(x, y, distance_y, sources, int(seed), contract["optimizer"],
                     float(contract["treatment"]["distance_loss_weight"])) for seed in contract["seeds"]]
    deltas = [row["treatment_minus_baseline"]["source_macro_balanced_accuracy"] for row in runs]
    positive_deltas = [row["treatment_minus_baseline"]["source_macro_positive_recall"] for row in runs]
    negative_deltas = [row["treatment_minus_baseline"]["source_macro_negative_recall"] for row in runs]
    gate = contract["retention_gate"]
    checks = {
        "median_balanced_improvement": float(np.median(deltas)) >= float(gate["median_source_macro_balanced_delta_at_least"]),
        "worst_seed_not_materially_worse": min(deltas) >= -float(gate["maximum_allowed_worst_seed_drop"]),
        "median_positive_recall_not_worse": float(np.median(positive_deltas)) >= 0.0,
        "median_negative_recall_not_worse": float(np.median(negative_deltas)) >= 0.0,
        "distance_beats_constant_all_runs": all(row["distance_mae"]["treatment"] < row["distance_mae"]["constant_training_fold_mean"] for row in runs),
    }
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "r767a_contract_sha256": common.sha256_file(args.linear_contract),
                   "r768a_report_sha256": common.sha256_file(args.bootstrap_report)},
        "data": {"frame_count": len(y), "active_count": int(y.sum()), "source_count": len(set(sources.tolist())),
                 "distance_target_mean_by_horizon": distance_y.mean(axis=0).tolist()},
        "paired_ablation": contract["paired_ablation"], "optimizer": contract["optimizer"],
        "runs": runs,
        "summary": {"balanced_accuracy_deltas": deltas, "median_balanced_accuracy_delta": float(np.median(deltas)),
                    "positive_recall_deltas": positive_deltas, "negative_recall_deltas": negative_deltas},
        "retention_gate": {"checks": checks, "passed": all(checks.values())},
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--bootstrap-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    print(json.dumps({"ok": True, **result["summary"], "retained": result["retention_gate"]["passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
