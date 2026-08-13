#!/usr/bin/env python3
"""Test a parent-relative, action-aligned high-precision veto certificate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.research.hftf.deployment.depthart.run_depthart_d3r4_selective_router_canary import (
    CertificateHead,
    STATE_CLEAR,
    STATE_OCCUPIED,
    STATE_UNKNOWN,
    atomic_json,
    balanced_bce,
    metrics,
    require,
    sha256_file,
)


CONTINUOUS_FEATURE_INDICES = tuple(range(4, 13))
THRESHOLD_CANDIDATES = (
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.97,
    0.99,
    0.995,
    0.999,
    0.9995,
    0.9998,
    0.9999,
    0.99995,
    0.99999,
    0.999999,
)


def load_dataset(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"dataset missing: {path}")
    with np.load(path) as archive:
        result = {name: archive[name] for name in archive.files}
    required = {
        "features",
        "truth_state",
        "baseline_state",
        "hard_evidence",
        "source_available",
        "parent_index",
        "frame_index",
        "band_index",
        "horizon_index",
    }
    require(set(result) == required, f"dataset field drift: {path}")
    rows = len(result["truth_state"])
    require(result["features"].shape == (rows, 16), "feature shape drift")
    require(all(len(value) == rows for value in result.values()), "dataset row drift")
    require(np.all(np.isfinite(result["features"])), "non-finite feature")
    return result


def average_rank_scaled(values: np.ndarray) -> np.ndarray:
    """Return deterministic average-tie ranks scaled into (-1, 1)."""

    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 0, "rank input must be non-empty 1D")
    require(np.all(np.isfinite(values)), "rank input must be finite")
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and ordered[end] == ordered[start]:
            end += 1
        average_one_based_rank = ((start + 1) + end) / 2.0
        ranks[order[start:end]] = average_one_based_rank
        start = end
    return 2.0 * ranks / (len(values) + 1.0) - 1.0


def parent_grid_relative_features(dataset: dict[str, np.ndarray]) -> np.ndarray:
    """Remove parent scale while preserving baseline state and grid identity."""

    transformed = np.asarray(dataset["features"], dtype=np.float64).copy()
    parents = dataset["parent_index"]
    bands = dataset["band_index"]
    horizons = dataset["horizon_index"]
    available = dataset["source_available"]
    for parent in sorted(np.unique(parents)):
        for band in range(3):
            for horizon in range(3):
                indices = np.flatnonzero(
                    (parents == parent)
                    & (bands == band)
                    & (horizons == horizon)
                    & available
                )
                require(len(indices) >= 297, "parent/grid source coverage drift")
                for feature_index in CONTINUOUS_FEATURE_INDICES:
                    transformed[indices, feature_index] = average_rank_scaled(
                        transformed[indices, feature_index]
                    )
    require(np.all(np.isfinite(transformed)), "relative feature non-finite")
    return transformed


def train_veto_head(
    dataset: dict[str, np.ndarray],
    features: np.ndarray,
    *,
    steps: int = 1000,
    seed: int = 44,
) -> tuple[CertificateHead, np.ndarray, np.ndarray, dict[str, Any]]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    feature_tensor = torch.as_tensor(features, dtype=torch.float64)
    available = torch.as_tensor(dataset["source_available"], dtype=torch.bool)
    mean = feature_tensor[available].mean(dim=0)
    std = feature_tensor[available].std(dim=0, unbiased=False)
    normalized = (feature_tensor - mean) / (std + 1e-6)
    truth = torch.as_tensor(dataset["truth_state"], dtype=torch.int64)
    baseline = torch.as_tensor(dataset["baseline_state"], dtype=torch.int64)
    hard = torch.as_tensor(dataset["hard_evidence"], dtype=torch.bool)
    action_domain = (truth >= 0) & hard & (baseline == STATE_CLEAR)
    labels = (truth[action_domain] == STATE_OCCUPIED).to(torch.float64)
    require(int(labels.sum()) > 0, "no false-clear positives in TRAIN")
    require(int((labels < 0.5).sum()) > 0, "no correct-clear negatives in TRAIN")
    head = CertificateHead().to(dtype=torch.float64)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.005, weight_decay=0.0001)
    initial_loss: float | None = None
    final_loss = 0.0
    for step in range(steps + 1):
        logits = head(normalized[action_domain])
        loss = balanced_bce(logits, labels)
        if step == 0:
            initial_loss = float(loss.detach())
        if step == steps:
            final_loss = float(loss.detach())
            break
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    require(initial_loss is not None, "initial loss missing")
    return head, mean.numpy(), std.numpy(), {
        "seed": seed,
        "steps": steps,
        "optimizer": "AdamW",
        "learning_rate": 0.005,
        "weight_decay": 0.0001,
        "action_domain": "truth-known AND hard-evidence AND baseline=CLEAR",
        "training_rows": int(action_domain.sum()),
        "positive_false_clear_rows": int(labels.sum()),
        "negative_correct_clear_rows": int((labels < 0.5).sum()),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
    }


def predict(
    features: np.ndarray, head: CertificateHead, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    normalized = torch.as_tensor(
        (features - mean) / (std + 1e-6), dtype=torch.float64
    )
    with torch.inference_mode():
        return torch.sigmoid(head(normalized)).numpy()


def route_veto(
    dataset: dict[str, np.ndarray], probabilities: np.ndarray, threshold: float
) -> tuple[np.ndarray, dict[str, int]]:
    baseline = dataset["baseline_state"]
    truth = dataset["truth_state"]
    action = (
        dataset["hard_evidence"]
        & dataset["source_available"]
        & (baseline == STATE_CLEAR)
        & (probabilities >= threshold)
    )
    result = baseline.copy()
    result[action] = STATE_OCCUPIED
    projected = 0
    parents = dataset["parent_index"]
    frames = dataset["frame_index"]
    bands = dataset["band_index"]
    horizons = dataset["horizon_index"]
    for parent in np.unique(parents):
        for frame in np.unique(frames[parents == parent]):
            frame_mask = (parents == parent) & (frames == frame)
            for band in range(3):
                indices = np.flatnonzero(frame_mask & (bands == band))
                require(len(indices) == 3, "grid row count drift")
                indices = indices[np.argsort(horizons[indices])]
                blocked = False
                for index in indices:
                    if result[index] in (STATE_OCCUPIED, STATE_UNKNOWN):
                        blocked = True
                    elif blocked and result[index] == STATE_CLEAR:
                        result[index] = STATE_UNKNOWN
                        projected += 1
    return result, {
        "direct_veto_actions": int(action.sum()),
        "true_positive_actions": int(np.sum(action & (truth == STATE_OCCUPIED))),
        "false_positive_actions": int(np.sum(action & (truth == STATE_CLEAR))),
        "unknown_truth_actions": int(np.sum(action & (truth == STATE_UNKNOWN))),
        "projection_to_unknown": projected,
    }


def threshold_row(
    dataset: dict[str, np.ndarray], probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    baseline_metrics = metrics(dataset, dataset["baseline_state"])
    states, actions = route_veto(dataset, probabilities, threshold)
    candidate_metrics = metrics(dataset, states)
    base = baseline_metrics["pooled"]
    candidate = candidate_metrics["pooled"]
    return {
        "threshold": threshold,
        "actions": actions,
        "metrics": candidate_metrics,
        "false_clear_all_known_improvement": (
            base["false_clear_all_known"] - candidate["false_clear_all_known"]
        ),
        "false_block_given_clear_improvement": (
            base["false_block_given_clear"] - candidate["false_block_given_clear"]
        ),
        "known_coverage_decrease": (
            base["known_coverage_all_cells"] - candidate["known_coverage_all_cells"]
        ),
    }


def select_zero_false_block_threshold(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [
        row
        for row in rows
        if row["actions"]["false_positive_actions"] == 0
        and row["false_clear_all_known_improvement"] >= 0.01
        and row["known_coverage_decrease"] <= 0.02
    ]
    require(eligible, "TRAIN has no zero-false-block useful threshold")
    return min(eligible, key=lambda row: row["threshold"])


def serialize_head(head: CertificateHead) -> dict[str, Any]:
    return {
        name: value.detach().cpu().numpy().tolist()
        for name, value in head.state_dict().items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d3r4-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()
    require(not args.output_root.exists(), f"fresh output root exists: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    train_path = args.d3r4_root / "train-dataset.npz"
    development_path = args.d3r4_root / "development-dataset.npz"
    d3r4_result_path = args.d3r4_root / "result.json"
    require(d3r4_result_path.is_file(), "D3R4 result missing")
    d3r4_result = json.loads(d3r4_result_path.read_text(encoding="utf-8"))
    require(
        d3r4_result.get("status") == "D3R4_SELECTIVE_ROUTER_MECHANISM_NOT_SUPPORTED",
        "D3R4 failure binding drift",
    )
    datasets = {
        "TRAIN": load_dataset(train_path),
        "DEVELOPMENT": load_dataset(development_path),
    }
    transformed = {
        role: parent_grid_relative_features(dataset)
        for role, dataset in datasets.items()
    }
    head, mean, std, training = train_veto_head(
        datasets["TRAIN"], transformed["TRAIN"], steps=args.steps
    )
    train_probabilities = predict(transformed["TRAIN"], head, mean, std)
    search = [
        threshold_row(datasets["TRAIN"], train_probabilities, threshold)
        for threshold in THRESHOLD_CANDIDATES
    ]
    selected = select_zero_false_block_threshold(search)
    threshold = float(selected["threshold"])
    checkpoint = {
        "schema": "blindassist_depthart_d3r5_parent_relative_veto_checkpoint_v1",
        "mechanism": "parent-grid empirical-rank features plus baseline-CLEAR-only veto",
        "continuous_feature_indices": list(CONTINUOUS_FEATURE_INDICES),
        "rank_scaling": "average ties; 2*rank/(N+1)-1 within parent x band x horizon",
        "threshold_selection": "lowest TRAIN threshold with zero direct false-positive veto, >=0.01 false-clear improvement, and <=0.02 coverage loss",
        "threshold": threshold,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "head": serialize_head(head),
        "training": training,
    }
    checkpoint_path = args.output_root / "veto-checkpoint.json"
    atomic_json(checkpoint_path, checkpoint)
    development_probabilities = predict(
        transformed["DEVELOPMENT"], head, mean, std
    )
    development = threshold_row(
        datasets["DEVELOPMENT"], development_probabilities, threshold
    )
    baseline = metrics(
        datasets["DEVELOPMENT"], datasets["DEVELOPMENT"]["baseline_state"]
    )
    supported = bool(
        development["actions"]["direct_veto_actions"] > 0
        and development["false_clear_all_known_improvement"] >= 0.01
        and development["false_block_given_clear_improvement"] >= -0.01
        and development["known_coverage_decrease"] <= 0.02
    )
    result = {
        "schema": "blindassist_depthart_d3r5_parent_relative_veto_discovery_result_v1",
        "status": (
            "D3R5_PARENT_RELATIVE_ZERO_FALSE_BLOCK_VETO_DISCOVERY_SUPPORTED"
            if supported
            else "D3R5_PARENT_RELATIVE_ZERO_FALSE_BLOCK_VETO_DISCOVERY_NOT_SUPPORTED"
        ),
        "problem": "D3R4 pooled veto generalized as a scene-specific occupied detector and over-blocked clear cells.",
        "hypothesis": "Aligning the action domain to baseline CLEAR, removing parent scale with unlabeled empirical ranks, and requiring zero TRAIN false-positive vetoes yields a transferable high-precision correction.",
        "input_bindings": {
            "d3r4_result": {
                "path": str(d3r4_result_path.resolve()),
                "bytes": d3r4_result_path.stat().st_size,
                "sha256": sha256_file(d3r4_result_path),
            },
            "train_dataset": {
                "path": str(train_path.resolve()),
                "bytes": train_path.stat().st_size,
                "sha256": sha256_file(train_path),
            },
            "development_dataset": {
                "path": str(development_path.resolve()),
                "bytes": development_path.stat().st_size,
                "sha256": sha256_file(development_path),
            },
        },
        "training": training,
        "train_threshold_search": search,
        "selected_threshold": threshold,
        "development": {
            "baseline": baseline,
            "candidate": development["metrics"],
            "actions": development["actions"],
            "false_clear_all_known_improvement": development[
                "false_clear_all_known_improvement"
            ],
            "false_block_given_clear_improvement": development[
                "false_block_given_clear_improvement"
            ],
            "known_coverage_decrease": development["known_coverage_decrease"],
        },
        "decision_rule": {
            "false_clear_improvement_min": 0.01,
            "false_block_improvement_min": -0.01,
            "known_coverage_decrease_max": 0.02,
        },
        "mechanism_supported_on_reused_development": supported,
        "evidence_role": "DISCOVERY_ONLY_DEVELOPMENT_WAS_ALREADY_OPENED_FOR_D3R4_FAILURE_DIAGNOSIS",
        "fresh_parent_confirmation_required": True,
        "development_used_for_training_or_threshold": False,
        "source_unavailable_as_negative": False,
        "far_clear_as_negative": False,
        "release_action_enabled": False,
        "r2_access": "NONE",
        "performance_claim": False,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "bytes": checkpoint_path.stat().st_size,
            "sha256": sha256_file(checkpoint_path),
            "parameter_count": sum(value.numel() for value in head.parameters()),
        },
        "elapsed_seconds_diagnostic_only": time.time() - started,
        "next_action": (
            "FRESH_PARENT_MODEL_OUTPUT_CONFIRMATION"
            if supported
            else "RETHINK_PARENT_SHIFT_MECHANISM"
        ),
    }
    atomic_json(args.output_root / "result.json", result)
    pooled = development["metrics"]["pooled"]
    print(json.dumps({
        "status": result["status"],
        "threshold": threshold,
        "actions": development["actions"],
        "baseline_false_clear": baseline["pooled"]["false_clear_all_known"],
        "candidate_false_clear": pooled["false_clear_all_known"],
        "baseline_false_block": baseline["pooled"]["false_block_given_clear"],
        "candidate_false_block": pooled["false_block_given_clear"],
        "known_coverage_decrease": development["known_coverage_decrease"],
        "next_action": result["next_action"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
