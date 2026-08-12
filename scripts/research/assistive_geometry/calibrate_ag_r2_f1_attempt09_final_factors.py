#!/usr/bin/env python3
"""Select final geometry and uncertainty on consumed internal validation only."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt07_point_factor_expansion import load_rows  # noqa: E402
from train_ag_r2_f1_attempt08_depthart_residual_head import DepthArtResidualFactorHead  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    cache_model_outputs,
    evaluate_cached,
    extract_features,
    gate,
    height_candidates,
    prepare,
    require,
    sha256_file,
    sigma_candidates,
)


RESIDUAL_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-expanded-residual-depth-r2/result.json"
EXPECTED_RESIDUAL_RESULT_SHA256 = "9B022091A409D821AE01779FD8E5266A31C619CE1739A85CAC93166E82F37FAE"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-final-factor-calibration-r0"
CANONICAL_SEED = 29
NON_DEPTH_SEED = 17
DEPTH_SIGMA_SOURCES = (
    {"name": "raw", "epistemic_weight": 0.0, "base_disagreement_weight": 0.0, "global_shift_weight": 0.0},
    {"name": "raw_plus_epistemic", "epistemic_weight": 1.0, "base_disagreement_weight": 0.0, "global_shift_weight": 0.0},
    {"name": "raw_plus_global", "epistemic_weight": 0.0, "base_disagreement_weight": 0.0, "global_shift_weight": 0.5},
    {"name": "raw_plus_base", "epistemic_weight": 0.0, "base_disagreement_weight": 0.5, "global_shift_weight": 0.0},
    {"name": "raw_plus_epi_global", "epistemic_weight": 0.5, "base_disagreement_weight": 0.0, "global_shift_weight": 0.5},
    {"name": "raw_plus_all", "epistemic_weight": 0.5, "base_disagreement_weight": 0.25, "global_shift_weight": 0.5},
)
DEPTH_SIGMA_SCALES = (0.75, 1.0, 1.25)


def hybrid_cache(
    caches: dict[int, list[dict[str, torch.Tensor]]],
) -> list[dict[str, torch.Tensor]]:
    rows = []
    for index in range(len(caches[CANONICAL_SEED])):
        output = dict(caches[NON_DEPTH_SEED][index])
        depth = caches[CANONICAL_SEED][index]
        for key in ("predicted_log_depth", "depth_log_sigma", "depth_valid_probability", "depth_gate"):
            output[key] = depth[key]
        rows.append(output)
    return rows


def depth_sigma_cache(
    hybrid: list[dict[str, torch.Tensor]],
    caches: dict[int, list[dict[str, torch.Tensor]]],
    samples: list[Any],
    config: dict[str, Any],
    scale: float,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    rows = []
    for index, (sample, original) in enumerate(zip(samples, hybrid)):
        seed_logs = torch.stack([caches[seed][index]["predicted_log_depth"] for seed in sorted(caches)])
        epistemic = seed_logs.std(dim=0, correction=0)
        base_log = F.interpolate(
            sample.base_depth_feature[None].to(device=device, dtype=torch.float32),
            sample.native_hw,
            mode="bilinear",
            align_corners=False,
        ).clamp_min(0.01).log()
        base_disagreement = (original["predicted_log_depth"] - base_log).abs()
        validity = original["depth_valid_probability"].clamp_min(1.0e-3)
        global_shift = ((original["predicted_log_depth"] - base_log) * validity).sum() / validity.sum().clamp_min(1.0e-6)
        raw = original["depth_log_sigma"].exp()
        variance = (
            raw.square()
            + float(config["epistemic_weight"]) * epistemic.square()
            + float(config["base_disagreement_weight"]) * base_disagreement.square()
            + float(config["global_shift_weight"]) * global_shift.abs().square()
        )
        output = dict(original)
        output["depth_log_sigma"] = (float(scale) * torch.sqrt(variance.clamp_min(1.0e-8))).clamp(0.01, 3.0).log()
        rows.append(output)
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.residual_result) == EXPECTED_RESIDUAL_RESULT_SHA256, "residual result drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    residual = json.loads(args.residual_result.read_text(encoding="utf-8"))
    validation_parents = set(residual["internal_validation_parents"])
    all_rows, data_receipt = load_rows()
    rows = [{**row, "role": "CONSUMED_INTERNAL_VALIDATION"} for row in all_rows if row["parent_id"] in validation_parents]
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
    caches = {}
    checkpoint_receipts = []
    for seed_row in residual["seed_results"]:
        seed = int(seed_row["seed"])
        checkpoint = Path(seed_row["selected_checkpoint"]["path"])
        expected = str(seed_row["selected_checkpoint"]["sha256"])
        require(sha256_file(checkpoint) == expected, "residual checkpoint drift")
        model = DepthArtResidualFactorHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        model.eval()
        caches[seed] = cache_model_outputs(model, prepared, device)
        checkpoint_receipts.append({"seed": seed, "path": str(checkpoint.resolve()), "sha256": expected})
    require(CANONICAL_SEED in caches and NON_DEPTH_SEED in caches, "hybrid seed identity drift")
    hybrid = hybrid_cache(caches)

    height_rows = []
    fixed_sigma = {"source": "coverage_complement", "multiplier": 1.5}
    for height_config in height_candidates():
        if height_config["metric_scale_calibration"] != "none":
            continue
        config = {"height": height_config, "support_sigma": fixed_sigma}
        evaluation = evaluate_cached(prepared, hybrid, baseline, config, device)
        candidate_gate = gate(evaluation, baseline_evaluation, 409)
        height_gate = candidate_gate["metric_improvements"]["camera_height_abs_log_error"]
        scale_gate = candidate_gate["metric_improvements"]["depth_scale_abs_log_error"]
        height_rows.append(
            {
                "config": height_config,
                "height_gate": height_gate,
                "depth_scale_gate": scale_gate,
                "height_overall": evaluation["overall_metrics"]["camera_height_abs_log_error"],
                "eligible": bool(height_gate["passed"] and scale_gate["passed"]),
            }
        )
    selected_height = min(
        height_rows,
        key=lambda row: (
            0 if row["eligible"] else 1,
            -float(row["height_gate"]["bootstrap_95_lower"]),
            -float(row["height_gate"]["favorable_parent_fraction"]),
            float(row["height_overall"]),
            json.dumps(row["config"], sort_keys=True),
        ),
    )

    support_rows = []
    for sigma_config in sigma_candidates():
        config = {"height": selected_height["config"], "support_sigma": sigma_config}
        evaluation = evaluate_cached(prepared, hybrid, baseline, config, device)
        candidate_gate = gate(evaluation, baseline_evaluation, 509)
        support_rows.append(
            {
                "config": sigma_config,
                "support_metric_gate": candidate_gate["metric_improvements"]["support_nll"],
                "support_uncertainty_gate": candidate_gate["uncertainty"]["support"],
                "eligible": bool(candidate_gate["metric_improvements"]["support_nll"]["passed"] and candidate_gate["uncertainty"]["support"]["passed"]),
            }
        )
    selected_support = min(
        support_rows,
        key=lambda row: (
            0 if row["eligible"] else 1,
            -float(row["support_uncertainty_gate"]["proper_score_gain"]),
            json.dumps(row["config"], sort_keys=True),
        ),
    )
    geometry_config = {"height": selected_height["config"], "support_sigma": selected_support["config"]}

    depth_rows = []
    for source in DEPTH_SIGMA_SOURCES:
        for scale in DEPTH_SIGMA_SCALES:
            calibrated = depth_sigma_cache(hybrid, caches, samples, source, scale, device)
            evaluation = evaluate_cached(prepared, calibrated, baseline, geometry_config, device)
            candidate_gate = gate(evaluation, baseline_evaluation, 609)
            depth_rows.append(
                {
                    "source": source,
                    "scale": scale,
                    "depth_metric_gate": candidate_gate["metric_improvements"]["depth_nll"],
                    "depth_uncertainty_gate": candidate_gate["uncertainty"]["depth"],
                    "eligible": bool(candidate_gate["metric_improvements"]["depth_nll"]["passed"] and candidate_gate["uncertainty"]["depth"]["passed"]),
                }
            )
    selected_depth = min(
        depth_rows,
        key=lambda row: (
            0 if row["eligible"] else 1,
            -float(row["depth_uncertainty_gate"]["proper_score_gain"]),
            row["scale"],
            row["source"]["name"],
        ),
    )
    final_cache = depth_sigma_cache(hybrid, caches, samples, selected_depth["source"], selected_depth["scale"], device)
    final_evaluation = evaluate_cached(prepared, final_cache, baseline, geometry_config, device)
    final_gate = gate(final_evaluation, baseline_evaluation, 709)
    passed = bool(final_gate["all_primary_metrics_passed"] and final_gate["all_uncertainty_families_passed"])
    calibration_path = args.output_dir / "final_factor_calibration.pt"
    torch.save(
        {
            "schema": "blindassist_ag_r2_f1_attempt09_final_factor_calibration_v1",
            "residual_result_sha256": EXPECTED_RESIDUAL_RESULT_SHA256,
            "canonical_depth_seed": CANONICAL_SEED,
            "non_depth_seed": NON_DEPTH_SEED,
            "checkpoint_receipts": checkpoint_receipts,
            "geometry_config": geometry_config,
            "depth_sigma_source": selected_depth["source"],
            "depth_sigma_scale": selected_depth["scale"],
        },
        calibration_path,
    )
    result = {
        "schema": "blindassist_ag_r2_f1_attempt09_final_factor_calibration_result_v1",
        "status": "ATTEMPT09_FINAL_FACTORS_INTERNAL_PASS_CANARY_LOCK_REQUIRED" if passed else "ATTEMPT09_FINAL_FACTORS_INTERNAL_FAIL_NO_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "preserved_canary_metrics_opened": False,
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "residual_result": {"path": str(args.residual_result.resolve()), "sha256": EXPECTED_RESIDUAL_RESULT_SHA256},
        "checkpoint_receipts": checkpoint_receipts,
        "hybrid": {"canonical_depth_seed": CANONICAL_SEED, "non_depth_seed": NON_DEPTH_SEED},
        "height_search": {"selected": selected_height, "candidates": height_rows},
        "support_sigma_search": {"selected": selected_support, "candidates": support_rows},
        "depth_sigma_search": {"selected": selected_depth, "candidates": depth_rows},
        "geometry_config": geometry_config,
        "baseline_evaluation": baseline_evaluation,
        "final_evaluation": final_evaluation,
        "final_gate": final_gate,
        "calibration": {"path": str(calibration_path.resolve()), "sha256": sha256_file(calibration_path), "bytes": calibration_path.stat().st_size},
        "decision": {"preserved_canary_metrics_opened": False, "next_action_if_pass": "Freeze exact identities and execute the preserved canary once."},
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual-result", type=Path, default=RESIDUAL_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "height": result["height_search"]["selected"], "support": result["support_sigma_search"]["selected"], "depth": result["depth_sigma_search"]["selected"], "primary_pass": result["final_gate"]["all_primary_metrics_passed"], "uncertainty_pass": result["final_gate"]["all_uncertainty_families_passed"], "calibration": result["calibration"]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
