#!/usr/bin/env python3
"""Train the frozen r7.62 single-seed source-isolated temporal route head."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_temporal_route_head_probe_v1"


class TemporalRouteHead(nn.Module):
    def __init__(self, input_channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def source_balanced_weights(sources: np.ndarray) -> np.ndarray:
    counts = {source: int(np.sum(sources == source)) for source in np.unique(sources)}
    source_count = len(counts)
    total = len(sources)
    return np.asarray([total / (source_count * counts[source]) for source in sources], dtype=np.float32)


def efficient_binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positive = int(labels.sum())
    negative = len(labels) - positive
    if positive == 0 or negative == 0:
        return 0.0
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return float((ranks[labels == 1].sum() - positive * (positive + 1) / 2.0) / (positive * negative))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_fold(train_x: np.ndarray, train_y: np.ndarray, train_sources: np.ndarray, config: dict[str, Any], input_channels: int) -> tuple[TemporalRouteHead, list[float]]:
    seed = int(config["seed"])
    set_seed(seed)
    model = TemporalRouteHead(input_channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]),
                                  weight_decay=float(config["weight_decay"]))
    weights = source_balanced_weights(train_sources)
    dataset = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(weights))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=True, generator=generator, num_workers=0)
    epoch_losses = []
    model.train()
    for _ in range(int(config["epochs"])):
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y, batch_weight in loader:
            optimizer.zero_grad(set_to_none=True)
            predicted = model(batch_x)
            pixel_weight = 1.0 + 4.0 * batch_y
            per_sample = (pixel_weight * (predicted - batch_y).square()).mean(dim=(1, 2, 3))
            loss = (per_sample * batch_weight).mean()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_x)
            total_rows += len(batch_x)
        epoch_losses.append(total_loss / max(1, total_rows))
    return model, epoch_losses


def predict(model: TemporalRouteHead, values: np.ndarray, batch_size: int) -> np.ndarray:
    result = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            result.append(model(torch.from_numpy(values[start:start + batch_size])).cpu().numpy())
    return np.concatenate(result) if result else np.empty((0, 3, 16, 16), dtype=np.float32)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.cache_report, args.cache, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    cache_report = lifecycle.verify_json_sidecar(args.cache_report)
    if common.sha256_file(args.cache) != cache_report["cache"]["sha256"]:
        raise ValueError("feature cache differs from extraction report")
    cache = np.load(args.cache)
    train_x = cache["train_x"].astype(np.float32)
    train_y = cache["train_y"].astype(np.float32)
    train_sources = cache["train_sources"].astype(str)
    eval_x = cache["eval_x"].astype(np.float32)
    eval_sources = cache["eval_sources"].astype(str)
    eval_events = cache["eval_events"].astype(str)
    eval_labels = cache["eval_labels"].astype(np.int64)
    eval_obstacles = cache["eval_obstacles"].astype(bool)
    sources = sorted(np.unique(train_sources).tolist())
    input_channels = int(train_x.shape[1])
    probe = contract["single_seed_probe"]
    oof = np.zeros_like(train_y, dtype=np.float32)
    eval_predictions = np.zeros((len(eval_x), 3, 16, 16), dtype=np.float32)
    eval_seen = np.zeros(len(eval_x), dtype=bool)
    folds = []
    for source in sources:
        train_indices = np.flatnonzero(train_sources != source)
        test_indices = np.flatnonzero(train_sources == source)
        model, losses = train_fold(train_x[train_indices], train_y[train_indices], train_sources[train_indices], probe, input_channels)
        oof[test_indices] = predict(model, train_x[test_indices], int(probe["batch_size"]))
        source_eval = np.flatnonzero(eval_sources == source)
        if len(source_eval):
            eval_predictions[source_eval] = predict(model, eval_x[source_eval], int(probe["batch_size"]))
            eval_seen[source_eval] = True
        folds.append({"held_out_source_id": source, "train_count": len(train_indices), "test_count": len(test_indices),
                      "eval_count": len(source_eval), "epoch_losses": losses,
                      "finite": bool(np.isfinite(losses).all())})
    if not eval_seen.all():
        missing = sorted(set(eval_sources[~eval_seen].tolist()))
        raise ValueError(f"evaluation source missing held-out model: {missing}")
    band = (train_y >= np.exp(-0.5)).astype(np.int64)
    pixel_auroc = efficient_binary_auroc(band, oof)
    localization = []
    for predicted, target in zip(oof, train_y):
        for channel in range(3):
            py, px = np.unravel_index(int(np.argmax(predicted[channel])), predicted[channel].shape)
            ty, tx = np.unravel_index(int(np.argmax(target[channel])), target[channel].shape)
            localization.append(float(np.hypot(px - tx, py - ty) / 16.0))
    frame_scores = []
    for predicted, obstacles in zip(eval_predictions, eval_obstacles):
        hits = []
        for channel in range(3):
            y, x = np.unravel_index(int(np.argmax(predicted[channel])), predicted[channel].shape)
            hits.append(bool(obstacles[y, x]))
        frame_scores.append(sum(hits) / len(hits))
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    grouped_labels: dict[str, set[int]] = defaultdict(set)
    for event, label, score in zip(eval_events, eval_labels, frame_scores):
        grouped_scores[event].append(float(score))
        grouped_labels[event].add(int(label))
    event_predictions = []
    for event in sorted(grouped_scores):
        if len(grouped_labels[event]) != 1:
            raise ValueError(f"event has inconsistent labels: {event}")
        event_predictions.append({"event_id": event, "label": next(iter(grouped_labels[event])),
                                  "frame_count": len(grouped_scores[event]),
                                  "predicted_horizon_hit_fraction": float(np.mean(grouped_scores[event]))})
    positive = [row["predicted_horizon_hit_fraction"] for row in event_predictions if row["label"] == 1]
    negative = [row["predicted_horizon_hit_fraction"] for row in event_predictions if row["label"] == 0]
    parameter_count = sum(parameter.numel() for parameter in TemporalRouteHead(input_channels).parameters())
    gate = contract["gate"]
    localization_mean = float(np.mean(localization))
    checks = {
        "route_band_pixel_auroc": pixel_auroc >= float(gate["route_band_pixel_auroc_at_least"]),
        "mean_argmax_localization_error": localization_mean <= float(gate["mean_argmax_localization_error_norm_at_most"]),
        "strict_event_label_separation": min(positive) > max(negative),
        "all_folds_finite": all(row["finite"] for row in folds),
        "parameter_limit": parameter_count <= int(contract["model"]["trainable_parameter_limit"]),
    }
    passed = all(checks.values())
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract), "cache_report_sha256": common.sha256_file(args.cache_report),
                   "cache_sha256": common.sha256_file(args.cache)},
        "seed": int(probe["seed"]), "source_count": len(sources), "train_frame_count": len(train_x),
        "eval_frame_count": len(eval_x), "trainable_parameter_count": parameter_count, "weights_saved": False,
        "folds": folds, "route_band_pixel_auroc": pixel_auroc,
        "mean_argmax_localization_error_norm": localization_mean,
        "distance_field_mean_absolute_error": float(np.mean(np.abs(oof - train_y))),
        "event_predictions": event_predictions, "checks": checks,
        "single_seed_gate_passed": passed,
        "five_seed_short_runs_authorized": passed,
        "evidence_limit": contract["evidence_role"],
        "authorization": {**contract["authorization"], "five_seed_short_runs": passed},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, "single_seed_gate_passed": value["single_seed_gate_passed"],
                      "route_band_pixel_auroc": value["route_band_pixel_auroc"],
                      "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
