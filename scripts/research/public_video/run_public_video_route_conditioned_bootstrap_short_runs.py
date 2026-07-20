#!/usr/bin/env python3
"""Run five prototype-initialized bootstrap short heads over frozen r816 features."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_video_route_conditioned_synthetic_probe as probe
import sanpo_depth_anything_linear_probe as depth_probe
from build_public_video_route_conditioned_synthetic_dataset import load_json, reject_independent_direction, sha256_file


SCHEMA = "blindassist_route_conditioned_bootstrap_short_runs_v1"


def stratified_source_class_bootstrap(indices: np.ndarray, source_ids: np.ndarray, labels: np.ndarray,
                                      rng: np.random.Generator) -> np.ndarray:
    selected: list[int] = []
    for source in sorted(set(source_ids[indices].tolist())):
        for label in (0, 1):
            cell = indices[(source_ids[indices] == source) & (labels[indices] == label)]
            if not len(cell):
                raise ValueError(f"bootstrap cell is empty: {source}:{label}")
            selected.extend(rng.choice(cell, size=len(cell), replace=True).tolist())
    return np.asarray(selected, dtype=np.int64)


def fit_prototype_short_head(features: np.ndarray, labels: np.ndarray, *, steps: int,
                             learning_rate: float, l2: float) -> dict[str, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or len(x) != len(y) or set(y.tolist()) != {0, 1} or min(steps, learning_rate, l2) <= 0:
        raise ValueError("prototype short head needs two classes and positive optimization settings")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    z = (x - mean) / scale
    prototypes = np.stack([z[y == label].mean(axis=0) for label in (0, 1)])
    kernel = prototypes.T.copy()
    bias = -0.5 * np.sum(prototypes * prototypes, axis=1)
    counts = np.bincount(y, minlength=2)
    weights = np.asarray([len(y) / (2.0 * counts[label]) for label in y], dtype=np.float64)
    targets = np.eye(2, dtype=np.float64)[y]
    for _step in range(steps):
        logits = z @ kernel + bias
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient_logits = (probabilities - targets) * weights[:, None] / weights.sum()
        kernel -= learning_rate * (z.T @ gradient_logits + l2 * kernel)
        bias -= learning_rate * gradient_logits.sum(axis=0)
    return {"mean": mean, "scale": scale, "kernel": kernel, "bias": bias}


def predict_short_head(features: np.ndarray, head: dict[str, np.ndarray]) -> np.ndarray:
    z = (np.asarray(features, dtype=np.float64) - head["mean"]) / head["scale"]
    return np.argmax(z @ head["kernel"] + head["bias"], axis=1).astype(np.int64)


def head_sha256(head: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("mean", "scale", "kernel", "bias"):
        digest.update(np.asarray(head[key], dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def prepare_fold_features(route_examples: Sequence[dict[str, Any]], patch_records: Sequence[dict[str, Any]],
                          feature_maps: dict[str, np.ndarray], source_ids: np.ndarray, *,
                          teacher_ridge: float, distance_sigma_patches: float) -> dict[str, dict[str, Any]]:
    folds = {}
    for held_source in dict.fromkeys(source_ids.tolist()):
        eligible = [row for row in patch_records if row["parent_source_id"] != held_source]
        teacher = probe.fit_distance_teacher(eligible, feature_maps, ridge=teacher_ridge,
                                             sigma_patches=distance_sigma_patches)
        score_maps = {path: probe.distance_score_map(feature_map, teacher) for path, feature_map in feature_maps.items()}
        route_features = np.stack([
            probe.route_conditioned_risk_features(score_maps[row["absolute_image_path"]], row["route_waypoints_xy_norm"])
            for row in route_examples
        ])
        folds[held_source] = {"features": route_features, "teacher_sha256": teacher["coefficient_sha256"],
                              "eligible_parent_source_ids": sorted({row["parent_source_id"] for row in eligible})}
    return folds


def evaluate_once(route_examples: Sequence[dict[str, Any]], fold_features: dict[str, dict[str, Any]], *,
                  seeds: Sequence[int], steps: int, learning_rate: float, l2: float) -> list[dict[str, Any]]:
    labels = np.asarray([int(row["route_blocked"]) for row in route_examples], dtype=np.int64)
    source_ids = np.asarray([row["parent_source_id"] for row in route_examples], dtype=object)
    runs = []
    for seed in seeds:
        predictions = np.full(len(labels), -1, dtype=np.int64)
        fold_rows = []
        for fold_index, held_source in enumerate(dict.fromkeys(source_ids.tolist())):
            train = np.flatnonzero(source_ids != held_source)
            test = np.flatnonzero(source_ids == held_source)
            rng = np.random.default_rng(int(seed) * 1009 + fold_index)
            bootstrap = stratified_source_class_bootstrap(train, source_ids, labels, rng)
            features = fold_features[held_source]["features"]
            head = fit_prototype_short_head(features[bootstrap], labels[bootstrap], steps=steps,
                                            learning_rate=learning_rate, l2=l2)
            fold_predictions = predict_short_head(features[test], head)
            predictions[test] = fold_predictions
            fold_rows.append({
                "held_out_parent_source_id": held_source, "metrics": common.binary_metrics(labels[test], fold_predictions),
                "bootstrap_sample_count": len(bootstrap),
                "bootstrap_unique_example_count": len(set(bootstrap.tolist())),
                "held_out_parent_descendants_excluded": bool(np.all(source_ids[bootstrap] != held_source)),
                "eligible_teacher_parent_source_ids": fold_features[held_source]["eligible_parent_source_ids"],
                "teacher_coefficient_sha256": fold_features[held_source]["teacher_sha256"],
                "head_coefficient_sha256": head_sha256(head),
            })
        if np.any(predictions < 0):
            raise RuntimeError("bootstrap run left examples unscored")
        runs.append({"seed": int(seed), "metrics": common.binary_metrics(labels, predictions),
                     "folds": fold_rows, "predictions": predictions.tolist()})
    return runs


def run(contract_path: Path, src_root: Path, checkpoint: Path, output: Path) -> dict[str, Any]:
    contract_path, src_root, checkpoint, output = (path.resolve() for path in (contract_path, src_root, checkpoint, output))
    for path in (contract_path, src_root, checkpoint, output):
        reject_independent_direction(path)
    contract = load_json(contract_path)
    dataset = (Path.cwd() / contract["bound_dataset"]["root"]).resolve()
    if sha256_file(dataset / "build_receipt.json") != contract["bound_dataset"]["build_receipt_sha256"]:
        raise ValueError("dataset build receipt SHA mismatch")
    if sha256_file(dataset / "qa" / "manual_review.json") != contract["bound_dataset"]["manual_review_sha256"]:
        raise ValueError("dataset review SHA mismatch")
    frozen = contract["frozen_representation"]
    if sha256_file(checkpoint) != frozen["checkpoint_sha256"]:
        raise ValueError("DINO checkpoint SHA mismatch")
    generation, route_examples, patch_records = probe.load_dataset_records(dataset)
    import torch
    torch.manual_seed(int(contract["short_runs"]["seeds"][0]))
    np.random.seed(int(contract["short_runs"]["seeds"][0]))
    torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(src_root, checkpoint, frozen["encoder"])
    model.eval()
    paths = sorted({str((dataset / row["image_path"]).resolve()) for row in generation})
    feature_maps = {}
    with torch.no_grad():
        for path in paths:
            feature_maps[path] = probe.teacher_probe.extract_dino_map(
                model, path, input_size=int(frozen["input_size"]), layer_index=int(frozen["layer_index"]),
            )
    source_ids = np.asarray([row["parent_source_id"] for row in route_examples], dtype=object)
    fold_features = prepare_fold_features(route_examples, patch_records, feature_maps, source_ids,
                                          teacher_ridge=float(frozen["teacher_ridge"]),
                                          distance_sigma_patches=float(frozen["distance_sigma_patches"]))
    settings = contract["short_runs"]
    kwargs = {"seeds": settings["seeds"], "steps": int(settings["steps"]),
              "learning_rate": float(settings["learning_rate"]), "l2": float(settings["l2"])}
    first = evaluate_once(route_examples, fold_features, **kwargs)
    second = evaluate_once(route_examples, fold_features, **kwargs)
    repeat_exact = first == second
    bas = np.asarray([row["metrics"]["balanced_accuracy"] for row in first], dtype=np.float64)
    clear_recalls = np.asarray([row["metrics"]["candidate_no_alert_recall"] for row in first], dtype=np.float64)
    block_recalls = np.asarray([row["metrics"]["candidate_alert_recall"] for row in first], dtype=np.float64)
    worst_fold_ba = min(fold["metrics"]["balanced_accuracy"] for row in first for fold in row["folds"])
    gate_spec = contract["gate"]
    gate = bool(
        repeat_exact
        and bas.min() >= float(gate_spec["minimum_worst_seed_balanced_accuracy"])
        and min(clear_recalls.min(), block_recalls.min()) >= float(gate_spec["minimum_worst_seed_each_class_recall"])
        and bas.mean() >= float(gate_spec["minimum_mean_balanced_accuracy"])
        and bas.std() <= float(gate_spec["maximum_seed_balanced_accuracy_stddev"])
        and worst_fold_ba >= float(gate_spec["minimum_worst_seed_parent_source_balanced_accuracy"])
    )
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path), "contract_sha256": sha256_file(contract_path),
        "dataset": str(dataset), "example_count": len(route_examples), "parent_source_count": len(set(source_ids.tolist())),
        "short_runs": first, "repeat_exact": repeat_exact,
        "stability": {"mean_balanced_accuracy": float(bas.mean()), "stddev_balanced_accuracy": float(bas.std()),
                      "worst_seed_balanced_accuracy": float(bas.min()),
                      "worst_seed_candidate_no_alert_recall": float(clear_recalls.min()),
                      "worst_seed_candidate_alert_recall": float(block_recalls.min()),
                      "worst_seed_parent_source_balanced_accuracy": float(worst_fold_ba)},
        "prototype_bootstrap_gate": {"passed": gate, "thresholds": gate_spec},
        "saved_weights": False, "train_only": True, "real_event_truth": False,
        "provider_evaluation_credit": False, "calibration_authorized": False, "blind_authorized": False,
        "android_runtime_change_authorized": False, "production_model_replacement_authorized": False,
    }
    if output.exists():
        raise ValueError(f"refusing to overwrite output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(output) + ".sha256").write_text(sha256_file(output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        report = run(args.contract, args.src_root, args.checkpoint, args.output)
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"ok": True, **report["stability"],
                      "prototype_bootstrap_gate": report["prototype_bootstrap_gate"]["passed"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
