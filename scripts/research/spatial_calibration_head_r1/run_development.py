#!/usr/bin/env python3
"""Run the frozen four-arm development folds and fixed validation from an R1 cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from core import (
    apply_global_affine,
    evaluate_predictions,
    fit_ridge,
    predict_ridge,
    predict_spatial,
    train_spatial_model,
)
from validate_protocol import DEFAULT_PROTOCOL, REPO_ROOT, sha256, validate

SCHEMA = "blindassist_spatial_calibration_head_r1_development_result"


def load_cache(manifest_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "blindassist_spatial_calibration_head_r1_cache":
        raise ValueError("unexpected cache schema")
    if manifest.get("sealed_truth_included") is not False:
        raise ValueError("development cache must not contain sealed truth")
    array_path = Path(manifest["arrays"]["path"])
    if not array_path.is_absolute():
        array_path = (manifest_path.parent / array_path).resolve()
    if sha256(array_path) != manifest["arrays"]["sha256"]:
        raise ValueError("cache array hash mismatch")
    with np.load(array_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"region_inputs", "raw_clearance", "truth_clearance", "truth_valid", "cls_features", "affine_targets", "affine_valid"}
    if required != set(arrays):
        raise ValueError(f"cache arrays differ: {sorted(set(arrays) ^ required)}")
    count = len(manifest["records"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("cache row counts differ")
    if any(row["role"] == "sealed" for row in manifest["records"]):
        raise ValueError("sealed records forbidden in development cache")
    return manifest, arrays


def arm_predictions(
    arrays: dict[str, np.ndarray],
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    training: dict[str, Any],
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]], dict[str, Any]]:
    raw = arrays["raw_clearance"]
    truth = arrays["truth_clearance"]
    truth_valid = arrays["truth_valid"].astype(bool)
    affine_valid = arrays["affine_valid"].astype(bool)
    affine_train = train_indices[affine_valid[train_indices]]
    if len(affine_train) < 8:
        raise ValueError("insufficient train-parent affine labels")
    constant_parameters = np.median(arrays["affine_targets"][affine_train], axis=0)
    constant_prediction, constant_known = apply_global_affine(raw[test_indices], constant_parameters)
    global_model = fit_ridge(arrays["cls_features"][affine_train], arrays["affine_targets"][affine_train], 10.0)
    global_parameters = predict_ridge(global_model, arrays["cls_features"][test_indices])
    global_prediction, global_known = apply_global_affine(raw[test_indices], global_parameters)
    spatial_model, standardizer, losses = train_spatial_model(
        arrays["region_inputs"], raw, truth, truth_valid, train_indices, training
    )
    spatial_prediction, spatial_known, spatial_confidence = predict_spatial(
        spatial_model, standardizer, arrays["region_inputs"][test_indices], raw[test_indices]
    )
    raw_prediction = raw[test_indices]
    raw_known = np.isfinite(raw_prediction)
    arms = {
        "raw_dav2": (raw_prediction, raw_known, None),
        "train_parent_constant_global_affine": (constant_prediction, constant_known & np.isfinite(raw_prediction), None),
        "global_cls_ridge_770": (global_prediction, global_known & np.isfinite(raw_prediction), None),
        "spatial_shared_mlp_9423_with_confidence_unknown": (spatial_prediction, spatial_known, spatial_confidence),
    }
    model_receipt = {
        "constant_parameters": constant_parameters.tolist(),
        "global_feature_dimension": int(arrays["cls_features"].shape[1]),
        "global_trainable_parameters": 770,
        "global_model": global_model,
        "spatial_trainable_parameters": sum(value.numel() for value in spatial_model.parameters()),
        "spatial_final_epoch_loss": losses[-1],
        "spatial_model": spatial_model,
        "spatial_standardizer": standardizer,
    }
    return arms, model_receipt


def run_development(
    protocol: dict[str, Any], manifest: dict[str, Any], arrays: dict[str, np.ndarray]
) -> tuple[dict[str, Any], dict[str, Any]]:
    records = manifest["records"]
    roles = np.asarray([row["role"] for row in records])
    train_all = np.flatnonzero(roles == "train")
    validation_indices = np.flatnonzero(roles == "validation")
    if len({row["parent_id"] for row in records if row["role"] == "train"}) != 16:
        raise ValueError("development cache must contain 16 train parents")
    if len({row["parent_id"] for row in records if row["role"] == "validation"}) != 4:
        raise ValueError("development cache must contain four validation parents")
    fold_results = []
    positive_folds = 0
    for fold in range(4):
        test = np.asarray([index for index in train_all if int(records[index]["cv_fold"]) == fold])
        train = np.asarray([index for index in train_all if int(records[index]["cv_fold"]) != fold])
        if len({records[index]["parent_id"] for index in test}) != 4:
            raise ValueError(f"fold {fold} must hold four parents")
        arms, _receipt = arm_predictions(arrays, train, test, protocol["training"])
        summaries = {
            name: evaluate_predictions(
                [records[index] for index in test], prediction, known,
                arrays["truth_clearance"][test], arrays["truth_valid"][test], confidence,
            )
            for name, (prediction, known, confidence) in arms.items()
        }
        spatial = summaries["spatial_shared_mlp_9423_with_confidence_unknown"]["parent_macro"]
        constant = summaries["train_parent_constant_global_affine"]["parent_macro"]
        jointly_better = (
            spatial["clearance_mae_m"] < constant["clearance_mae_m"]
            and spatial["false_clear_rate"] <= constant["false_clear_rate"]
        )
        positive_folds += int(jointly_better)
        fold_results.append({"fold": fold, "train_frames": len(train), "test_frames": len(test), "jointly_better_than_constant": jointly_better, "arms": summaries})

    validation_arms, final_receipt = arm_predictions(arrays, train_all, validation_indices, protocol["training"])
    validation_summaries = {
        name: evaluate_predictions(
            [records[index] for index in validation_indices], prediction, known,
            arrays["truth_clearance"][validation_indices], arrays["truth_valid"][validation_indices], confidence,
        )
        for name, (prediction, known, confidence) in validation_arms.items()
    }
    spatial_validation = validation_summaries["spatial_shared_mlp_9423_with_confidence_unknown"]
    constant_validation = validation_summaries["train_parent_constant_global_affine"]
    validation_comparator_gates = {
        "spatial_mae_better_than_constant": spatial_validation["parent_macro"]["clearance_mae_m"] < constant_validation["parent_macro"]["clearance_mae_m"],
        "spatial_false_clear_not_worse_than_constant": spatial_validation["parent_macro"]["false_clear_rate"] <= constant_validation["parent_macro"]["false_clear_rate"],
    }
    development_supported = (
        positive_folds >= 3
        and all(spatial_validation["gates"].values())
        and all(validation_comparator_gates.values())
    )
    result = {
        "schema": SCHEMA,
        "protocol_sha256": manifest["protocol_sha256"],
        "cache_manifest_sha256": None,
        "sealed_truth_opened": False,
        "folds": fold_results,
        "positive_folds_jointly_better_than_constant": positive_folds,
        "fixed_validation": {"arms": validation_summaries, "comparator_gates": validation_comparator_gates},
        "terminal": (
            "SPATIAL_CALIBRATION_HEAD_R1_DEVELOPMENT_SUPPORTED_SEALED_ACTIVATION_PENDING"
            if development_supported
            else protocol["terminals"]["development_not_supported"]
        ),
    }
    return result, final_receipt


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_model_new(path: Path, receipt: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    if partial.exists():
        raise FileExistsError(partial)
    model = receipt.pop("spatial_model")
    standardizer = receipt.pop("spatial_standardizer")
    global_model = receipt.pop("global_model")
    torch.save({
        "state_dict": model.state_dict(),
        "feature_mean": standardizer.mean.astype(np.float32),
        "feature_std": standardizer.std.astype(np.float32),
        "constant_parameters": np.asarray(receipt["constant_parameters"], dtype=np.float32),
        "global_model": {key: np.asarray(value, dtype=np.float32) for key, value in global_model.items()},
    }, partial)
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    errors = validate(protocol)
    if errors:
        raise ValueError(f"protocol invalid: {errors}")
    manifest, arrays = load_cache(args.cache_manifest)
    if manifest.get("protocol_sha256") != sha256(args.protocol):
        raise ValueError("cache protocol mismatch")
    result, receipt = run_development(protocol, manifest, arrays)
    result["cache_manifest_sha256"] = sha256(args.cache_manifest)
    save_model_new(args.model_output, receipt)
    result["final_model"] = {"path": str(args.model_output.resolve()), "sha256": sha256(args.model_output), **receipt}
    write_json_new(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("folds", "fixed_validation")}, indent=2))


if __name__ == "__main__":
    main()
