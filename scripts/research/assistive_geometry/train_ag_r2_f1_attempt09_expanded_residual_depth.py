#!/usr/bin/env python3
"""Retrain the DepthART residual depth head on expanded source-native FIT labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt07_point_factor_expansion import load_rows, parent_split  # noqa: E402
from train_ag_r2_f1_attempt08_depthart_residual_head import train_seed  # noqa: E402
from train_ag_r2_f1_factor_learnability import evaluate, extract_features  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    EXPECTED_DEPTHART_SHA256,
    require,
    sha256_file,
)


POINT_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-point-factor-expansion-r0/result.json"
EXPECTED_POINT_RESULT_SHA256 = "580A94AD71B9C86A706D8FB233BF87AAD9376B503C8E7C35DF8F1102A34AE946"
EXPANSION_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-fit-expansion-labels-r0/result.json"
EXPECTED_EXPANSION_RESULT_SHA256 = "FDC70D9384CD0376265C582E2698B7A1DB7D3E2224D026C83DD7141A5BC96C1F"
EXPANSION_ADMISSION = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-fit-expansion-labels-r0/depth_training_admission_r1.json"
EXPECTED_EXPANSION_ADMISSION_SHA256 = "70117C2BD7472135191697399CFFC2473D6A9300AE96887DD981D4F872309ABC"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt09-expanded-residual-depth-r2"


def expanded_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(sha256_file(EXPANSION_RESULT) == EXPECTED_EXPANSION_RESULT_SHA256, "expansion result drift")
    require(sha256_file(EXPANSION_ADMISSION) == EXPECTED_EXPANSION_ADMISSION_SHA256, "expansion admission drift")
    admission = json.loads(EXPANSION_ADMISSION.read_text(encoding="utf-8"))
    require(admission["passed"], "expansion not admitted")
    current, current_receipt = load_rows()
    parents = {str(row["parent_id"]) for row in current}
    fit_parent_rows, validation_parent_rows = parent_split(parents)
    fit_parents, validation_parents = set(fit_parent_rows), set(validation_parent_rows)
    expansion = json.loads(EXPANSION_RESULT.read_text(encoding="utf-8"))
    expanded_fit = [
        {**row, "role": "FIT_EXPANDED"}
        for row in expansion["frames"]
        if row["parent_id"] in fit_parents
    ]
    expanded_parents = {str(row["parent_id"]) for row in expanded_fit}
    require(len(expanded_fit) == 84 and len(expanded_parents) == 7, "usable expanded FIT identity drift")
    require(not (expanded_parents & validation_parents), "expanded labels leaked into validation")
    retained = [row for row in current if row["parent_id"] not in expanded_parents]
    combined = sorted(expanded_fit + retained, key=lambda row: (str(row["parent_id"]), str(row["sample_id"])))
    require(len(combined) == 138, "expanded combined frame count drift")
    require(len({row["sample_id"] for row in combined}) == len(combined), "expanded sample identity collision")
    return combined, {
        "current": current_receipt,
        "expansion_result": {"path": str(EXPANSION_RESULT.resolve()), "sha256": EXPECTED_EXPANSION_RESULT_SHA256},
        "expansion_admission": {"path": str(EXPANSION_ADMISSION.resolve()), "sha256": EXPECTED_EXPANSION_ADMISSION_SHA256},
        "expanded_fit_parents": sorted(expanded_parents),
        "expanded_fit_frames": len(expanded_fit),
        "retained_frames": len(retained),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available() and args.device.startswith("cuda"), "CUDA required")
    require(not args.output_dir.exists(), f"output exists: {args.output_dir}")
    require(sha256_file(args.point_result) == EXPECTED_POINT_RESULT_SHA256, "point result drift")
    require(sha256_file(args.depthart_checkpoint) == EXPECTED_DEPTHART_SHA256, "DepthART drift")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = torch.device(args.device)
    rows, data_receipt = expanded_rows()
    parents = {str(row["parent_id"]) for row in rows}
    fit_parents, validation_parents = parent_split(parents)
    samples, feature_receipt = extract_features(
        rows,
        args.depthart_source,
        args.depthart_checkpoint,
        args.depthart_extension,
        device,
    )
    fit = [sample for sample in samples if sample.parent_id in fit_parents]
    validation = [sample for sample in samples if sample.parent_id in validation_parents]
    require(len(fit) == 123 and len(validation) == 15, "expanded train/validation frame split drift")
    baseline_result = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    baseline_parameters = baseline_result["baseline_parameters"]
    normalization = baseline_result["optimizer_normalization"]
    validation_baseline = evaluate(None, validation, baseline_parameters, device)
    point = json.loads(args.point_result.read_text(encoding="utf-8"))
    seed_results = []
    for seed_row in point["seed_results"]:
        seed = int(seed_row["seed"])
        source_checkpoint = Path(seed_row["selected_checkpoint"]["path"])
        require(sha256_file(source_checkpoint) == seed_row["selected_checkpoint"]["sha256"], "source checkpoint drift")
        result = train_seed(
            seed,
            fit,
            validation,
            baseline_parameters,
            normalization,
            validation_baseline,
            source_checkpoint,
            args.output_dir,
            device,
            args.optimizer_steps,
        )
        seed_results.append(result)
        print(json.dumps({"seed": seed, "selected_step": result["selected_step"], "eligible": result["selected_evidence"]["eligible"], "metrics": result["selected_evidence"]["metrics"]}), flush=True)
        torch.cuda.empty_cache()
    eligible = [row for row in seed_results if row["selected_evidence"]["eligible"]]
    canonical = min(
        eligible or seed_results,
        key=lambda row: (row["selected_evidence"]["selection_score"], row["seed"], row["selected_checkpoint"]["sha256"]),
    )
    passed = bool(eligible)
    result = {
        "schema": "blindassist_ag_r2_f1_attempt09_expanded_residual_depth_result_v1",
        "status": "ATTEMPT09_EXPANDED_RESIDUAL_DEPTH_INTERNAL_PASS_FINAL_FACTOR_CALIBRATION_REQUIRED" if passed else "ATTEMPT09_EXPANDED_RESIDUAL_DEPTH_INTERNAL_FAIL_NO_CANARY",
        "passed": passed,
        "elapsed_seconds": time.perf_counter() - started,
        "preserved_canary_metrics_opened": False,
        "data_receipt": data_receipt,
        "fit_parents": fit_parents,
        "internal_validation_parents": validation_parents,
        "role_frame_counts": {"FIT": len(fit), "INTERNAL_VALIDATION": len(validation), "PRESERVED_CANARY": 0},
        "feature_receipt": feature_receipt,
        "seed_results": seed_results,
        "canonical_seed": canonical["seed"],
        "canonical_checkpoint": canonical["selected_checkpoint"],
        "decision": {
            "expanded_source_native_fit_used": True,
            "validation_parent_labels_used_for_optimizer": False,
            "preserved_canary_metrics_opened": False,
            "next_action_if_pass": "Calibrate factor uncertainties and geometry on consumed validation, freeze final canary execution, then run once.",
        },
    }
    with (args.output_dir / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-result", type=Path, default=POINT_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--optimizer-steps", type=int, default=2400)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    result = run(args)
    print(json.dumps({"status": result["status"], "passed": result["passed"], "canonical_seed": result["canonical_seed"], "canonical_checkpoint": result["canonical_checkpoint"], "seeds": [{"seed": row["seed"], "step": row["selected_step"], "eligible": row["selected_evidence"]["eligible"]} for row in result["seed_results"]]}, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
