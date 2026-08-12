#!/usr/bin/env python3
"""Execute the frozen final AG factor canary once and serialize learned factors."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from calibrate_ag_r2_f1_attempt09_final_factors import (  # noqa: E402
    CANONICAL_SEED,
    NON_DEPTH_SEED,
    hybrid_cache,
)
from train_ag_r2_f1_attempt08_depthart_residual_head import DepthArtResidualFactorHead  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    apply_geometry,
    cache_model_outputs,
    evaluate_cached,
    extract_features,
    gate,
    prepare,
    require,
    sha256_file,
)


DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT11_FINAL_CANARY_EXECUTION_LOCK_2026-08-11.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt11-final-canary-r0"


def load_model(checkpoint: Path, baseline: dict[str, Any], device: torch.device) -> DepthArtResidualFactorHead:
    model = DepthArtResidualFactorHead(baseline).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
    model.eval()
    return model


def final_outputs(
    residual_caches: dict[int, list[dict[str, torch.Tensor]]],
    height_cache: list[dict[str, torch.Tensor]],
) -> list[dict[str, torch.Tensor]]:
    rows = hybrid_cache(residual_caches)
    for index, output in enumerate(rows):
        output["camera_height_m"] = height_cache[index]["camera_height_m"]
    return rows


def serialize_factors(
    prepared: list[dict[str, Any]],
    outputs: list[dict[str, torch.Tensor]],
    geometry_config: dict[str, Any],
    output_dir: Path,
    identity: dict[str, str],
    device: torch.device,
) -> list[dict[str, Any]]:
    destination = output_dir / "factor_tensors"
    destination.mkdir(parents=True, exist_ok=False)
    receipts = []
    with torch.no_grad():
        for row, raw in zip(prepared, outputs):
            sample = row["sample"]
            factors, geometry_receipt = apply_geometry(raw, sample, geometry_config, {}, device, None)
            log_depth = factors["predicted_log_depth"][0, 0]
            log_scale = log_depth.mean()
            with np.load(sample.label_path, allow_pickle=False) as source:
                camera_receipt = str(np.asarray(source["camera_geometry_receipt_sha256"]).item())
            payload = {
                "schema": np.asarray("blindassist_assistive_geometry_r2_f1_prediction_v1"),
                "sample_id": np.asarray(sample.sample_id),
                "factor_identity": np.asarray("AG_R2_F1_ATTEMPT11_DEPTHART_RESIDUAL_EXPANDED_SUPERVISION"),
                "model_result_sha256": np.asarray(identity["residual_result_sha256"]),
                "height_result_sha256": np.asarray(identity["height_result_sha256"]),
                "factor_calibration_sha256": np.asarray(identity["factor_calibration_sha256"]),
                "camera_geometry_receipt_sha256": np.asarray(camera_receipt),
                "depth_shape_positive_hw": torch.exp(log_depth - log_scale).cpu().numpy().astype(np.float32),
                "log_metric_scale_m_scalar": np.asarray(float(log_scale), dtype=np.float32),
                "depth_log_sigma_hw": factors["depth_log_sigma"][0, 0].cpu().numpy().astype(np.float32),
                "depth_valid_probability_hw": factors["depth_valid_probability"][0, 0].cpu().numpy().astype(np.float32),
                "metric_scale_valid": np.asarray(bool(factors["depth_valid_probability"].mean() >= 0.5)),
                "support_probability_hw": factors["support_probability"][0, 0].cpu().numpy().astype(np.float32),
                "support_plane_normal_camera_xyz": factors["support_plane_normal_camera_xyz"][0].cpu().numpy().astype(np.float32),
                "camera_height_m": np.asarray(float(factors["camera_height_m"][0]), dtype=np.float32),
                "support_residual_sigma_m": np.asarray(float(factors["support_residual_sigma_m"][0]), dtype=np.float32),
                "support_valid": np.asarray(bool(factors["support_valid_probability"][0] >= 0.5 and geometry_receipt.get("fallback") is None)),
                "obstacle_evidence_probability_hw": factors["obstacle_probability"][0, 0].cpu().numpy().astype(np.float32),
                "boundary_probability_hw": factors["boundary_probability"][0, 0].cpu().numpy().astype(np.float32),
                "boundary_localization_sigma_px_hw": factors["boundary_sigma_px"][0, 0].cpu().numpy().astype(np.float32),
                "evidence_valid_hw": (factors["evidence_valid_probability"][0, 0] >= 0.5).cpu().numpy().astype(np.bool_),
            }
            path = destination / f"{sample.sample_id}.npz"
            np.savez_compressed(path, **payload)
            receipts.append(
                {
                    "sample_id": sample.sample_id,
                    "parent_id": sample.parent_id,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                    "camera_geometry_receipt_sha256": camera_receipt,
                    "metric_scale_valid": bool(payload["metric_scale_valid"].item()),
                    "support_valid": bool(payload["support_valid"].item()),
                    "evidence_valid_pixels": int(payload["evidence_valid_hw"].sum()),
                    "geometry": geometry_receipt,
                }
            )
    return receipts


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    lock = json.loads(args.lock.read_text(encoding="utf-8"))
    require(lock["status"] == "ATTEMPT11_FINAL_CANARY_EXECUTION_AUTHORIZED", "final canary lock invalid")
    bindings = lock["bindings"]
    require(sha256_file(Path(__file__)) == bindings["implementation"]["sha256"], "final canary implementation drift")
    for key, path in (
        ("fresh_label_result", args.fresh_label_result),
        ("residual_result", args.residual_result),
        ("height_result", args.height_result),
        ("factor_calibration_result", args.factor_calibration_result),
        ("baseline_result", args.baseline_result),
    ):
        require(sha256_file(path) == bindings[key]["sha256"], f"{key} drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    fresh = json.loads(args.fresh_label_result.read_text(encoding="utf-8"))
    residual = json.loads(args.residual_result.read_text(encoding="utf-8"))
    height = json.loads(args.height_result.read_text(encoding="utf-8"))
    calibration = json.loads(args.factor_calibration_result.read_text(encoding="utf-8"))
    require(residual["passed"] and height["passed"], "final learned prerequisite failed")
    rows = sorted(
        [{**row, "role": "TRAIN_CANARY"} for row in fresh["frames"] if row["role"] == "TRAIN_CANARY"],
        key=lambda row: row["sample_id"],
    )
    require([row["sample_id"] for row in rows] == lock["canary"]["sample_ids"], "canary sample identity drift")
    require(len(rows) == 6 and len({row["parent_id"] for row in rows}) == 2, "canary denominator drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    started = time.perf_counter()
    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    prepared = prepare(samples, device)
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    baseline_evaluation = evaluate_cached(prepared, None, baseline, None, device)
    residual_caches = {}
    checkpoint_receipts = []
    for seed_row in residual["seed_results"]:
        seed = int(seed_row["seed"])
        checkpoint = Path(seed_row["selected_checkpoint"]["path"])
        expected = str(seed_row["selected_checkpoint"]["sha256"])
        require(sha256_file(checkpoint) == expected, "residual checkpoint drift")
        residual_caches[seed] = cache_model_outputs(load_model(checkpoint, baseline, device), prepared, device)
        checkpoint_receipts.append({"seed": seed, "path": str(checkpoint.resolve()), "sha256": expected})
    require(CANONICAL_SEED in residual_caches and NON_DEPTH_SEED in residual_caches, "final hybrid seed drift")
    height_checkpoint = Path(height["selected_checkpoint"]["path"])
    require(sha256_file(height_checkpoint) == height["selected_checkpoint"]["sha256"], "height checkpoint drift")
    height_cache = cache_model_outputs(load_model(height_checkpoint, baseline, device), prepared, device)
    outputs = final_outputs(residual_caches, height_cache)
    selected_geometry = calibration["geometry_config"]
    geometry_config = {
        "height": {
            **selected_geometry["height"],
            "source": "learned_global",
            "scope": "frame_camera_plane",
            "metric_scale_calibration": "none",
        },
        "support_sigma": selected_geometry["support_sigma"],
    }
    evaluation = evaluate_cached(prepared, outputs, baseline, geometry_config, device)
    final_gate = gate(evaluation, baseline_evaluation, 1111)
    passed = bool(final_gate["all_primary_metrics_passed"] and final_gate["all_uncertainty_families_passed"])
    identity = {
        "residual_result_sha256": bindings["residual_result"]["sha256"],
        "height_result_sha256": bindings["height_result"]["sha256"],
        "factor_calibration_sha256": bindings["factor_calibration_result"]["sha256"],
    }
    factor_receipts = serialize_factors(prepared, outputs, geometry_config, args.output_dir, identity, device) if passed else []
    result = {
        "schema": "blindassist_ag_r2_f1_attempt11_final_canary_result_v1",
        "status": "ATTEMPT11_FINAL_CANARY_PASS_FACTORS_SERIALIZED_ADAPTER_REQUIRED" if passed else "ATTEMPT11_FINAL_CANARY_FAIL_NO_PROMOTION",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "lock": {"path": str(args.lock.resolve()), "sha256": sha256_file(args.lock)},
        "feature_receipt": feature_receipt,
        "checkpoint_receipts": checkpoint_receipts,
        "height_checkpoint": {"path": str(height_checkpoint.resolve()), "sha256": sha256_file(height_checkpoint)},
        "hybrid": {"canonical_depth_seed": CANONICAL_SEED, "non_depth_seed": NON_DEPTH_SEED, "height_source": "attempt10_learned_global"},
        "geometry_config": geometry_config,
        "baseline_evaluation": baseline_evaluation,
        "evaluation": evaluation,
        "gate": final_gate,
        "factor_tensors": factor_receipts,
        "decision": {
            "canary_opened_once": True,
            "factor_tensor_count": len(factor_receipts),
            "next_action_if_pass": "Run FactorTensorAdapter and deterministic reducer on all six serialized tensors.",
        },
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--fresh-label-result", type=Path, default=REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt05-fresh-ag-held-labels-r0/result.json")
    parser.add_argument("--residual-result", type=Path, default=REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-expanded-residual-depth-r2/result.json")
    parser.add_argument("--height-result", type=Path, default=REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt10-camera-height-r0/result.json")
    parser.add_argument("--factor-calibration-result", type=Path, default=REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-final-factor-calibration-r0/result.json")
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "primary": result["gate"]["metric_improvements"], "uncertainty": result["gate"]["uncertainty"], "factor_tensor_count": len(result["factor_tensors"])}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
