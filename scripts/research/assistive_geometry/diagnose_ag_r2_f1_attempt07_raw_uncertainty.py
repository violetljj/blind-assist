#!/usr/bin/env python3
"""Consumed internal-validation diagnostic for expanded-head raw sigmas."""

from __future__ import annotations

import json
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_attempt07_point_factor_expansion import load_rows  # noqa: E402
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    FactorSplitHead,
    cache_model_outputs,
    evaluate_cached,
    extract_features,
    gate,
    prepare,
    sha256_file,
)
from train_ag_r2_f1_factor_learnability_attempt04 import GEOMETRY_CONFIG  # noqa: E402


POINT_RESULT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-point-factor-expansion-r0/result.json"
OUTPUT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt07-raw-uncertainty-diagnostic-r0/result.json"


def main() -> int:
    device = torch.device("cuda:0")
    point = json.loads(POINT_RESULT.read_text(encoding="utf-8"))
    rows, _ = load_rows()
    validation = set(point["internal_validation_parents"])
    selected_rows = [{**row, "role": "CONSUMED_INTERNAL_VALIDATION"} for row in rows if row["parent_id"] in validation]
    samples, feature_receipt = extract_features(selected_rows, DEFAULT_DEPTHART_SOURCE, DEFAULT_DEPTHART_CHECKPOINT, DEFAULT_DEPTHART_EXTENSION, device)
    prepared = prepare(samples, device)
    baseline = json.loads(DEFAULT_BASELINE_RESULT.read_text(encoding="utf-8"))["baseline_parameters"]
    baseline_evaluation = evaluate_cached(prepared, None, baseline, None, device)
    caches = []
    checkpoints = []
    for seed_row in point["seed_results"]:
        checkpoint = Path(seed_row["selected_checkpoint"]["path"])
        model = FactorSplitHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        caches.append(cache_model_outputs(model, prepared, device))
        checkpoints.append({"seed": seed_row["seed"], "path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)})
    evaluation = evaluate_cached(prepared, caches[0], baseline, GEOMETRY_CONFIG, device)
    result = {
        "schema": "blindassist_ag_r2_f1_attempt07_raw_uncertainty_diagnostic_v1",
        "consumed_internal_validation_only": True,
        "canary_read": False,
        "parents": sorted(validation),
        "feature_receipt": feature_receipt,
        "checkpoints": checkpoints,
        "baseline": baseline_evaluation,
        "evaluation": evaluation,
        "gate": gate(evaluation, baseline_evaluation, 107),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"uncertainty": result["gate"]["uncertainty"], "all_primary": result["gate"]["all_primary_metrics_passed"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
