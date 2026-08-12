#!/usr/bin/env python3
"""Test a deployable DepthART metric-scale anchor on consumed validation only."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

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
OUTPUT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt08-depth-anchor-diagnostic-r0/result.json"
BETAS = (0.0, 0.25, 0.5, 0.75, 1.0)
GAMMAS = (0.0, 0.25, 0.5, 1.0, 2.0)


def anchored_cache(
    samples: list[object],
    cached: list[dict[str, torch.Tensor]],
    beta: float,
    gamma: float,
    device: torch.device,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, float]]]:
    rows: list[dict[str, torch.Tensor]] = []
    receipts: list[dict[str, float]] = []
    for sample, original in zip(samples, cached):
        output = dict(original)
        native_hw = tuple(int(value) for value in sample.native_hw)
        base_log = F.interpolate(
            sample.base_depth_feature[None].to(device=device, dtype=torch.float32),
            native_hw,
            mode="bilinear",
            align_corners=False,
        ).clamp_min(0.01).log()
        weight = original["depth_valid_probability"].clamp_min(1.0e-3)
        shift = ((original["predicted_log_depth"] - base_log) * weight).sum() / weight.sum().clamp_min(1.0e-6)
        output["predicted_log_depth"] = original["predicted_log_depth"] - float(beta) * shift
        raw_sigma = original["depth_log_sigma"].exp()
        anchored_sigma = torch.sqrt(raw_sigma.square() + float(gamma) * shift.abs().square()).clamp(0.01, 3.0)
        output["depth_log_sigma"] = anchored_sigma.log()
        rows.append(output)
        receipts.append({"model_vs_depthart_weighted_log_shift": float(shift), "applied_log_shift": float(beta * shift)})
    return rows, receipts


def main() -> int:
    device = torch.device("cuda:0")
    point = json.loads(POINT_RESULT.read_text(encoding="utf-8"))
    rows, data_receipt = load_rows()
    validation = set(point["internal_validation_parents"])
    selected_rows = [{**row, "role": "CONSUMED_INTERNAL_VALIDATION"} for row in rows if row["parent_id"] in validation]
    samples, feature_receipt = extract_features(
        selected_rows,
        DEFAULT_DEPTHART_SOURCE,
        DEFAULT_DEPTHART_CHECKPOINT,
        DEFAULT_DEPTHART_EXTENSION,
        device,
    )
    prepared = prepare(samples, device)
    baseline = json.loads(DEFAULT_BASELINE_RESULT.read_text(encoding="utf-8"))["baseline_parameters"]
    baseline_evaluation = evaluate_cached(prepared, None, baseline, None, device)
    seed17 = point["seed_results"][0]
    checkpoint = Path(seed17["selected_checkpoint"]["path"])
    model = FactorSplitHead(baseline).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
    cached = cache_model_outputs(model, prepared, device)
    candidates = []
    for beta in BETAS:
        for gamma in GAMMAS:
            candidate_cache, anchor_receipts = anchored_cache(samples, cached, beta, gamma, device)
            evaluation = evaluate_cached(prepared, candidate_cache, baseline, GEOMETRY_CONFIG, device)
            candidate_gate = gate(evaluation, baseline_evaluation, 108)
            primary_pass_count = sum(row["passed"] for row in candidate_gate["metric_improvements"].values())
            uncertainty_pass_count = sum(row["passed"] for row in candidate_gate["uncertainty"].values())
            candidates.append(
                {
                    "beta": beta,
                    "gamma": gamma,
                    "evaluation": evaluation,
                    "gate": candidate_gate,
                    "primary_pass_count": primary_pass_count,
                    "uncertainty_pass_count": uncertainty_pass_count,
                    "anchor_receipts": anchor_receipts,
                }
            )
    selected = min(
        candidates,
        key=lambda row: (
            -row["primary_pass_count"],
            -row["uncertainty_pass_count"],
            -row["gate"]["uncertainty"]["depth"]["proper_score_gain"],
            row["beta"],
            row["gamma"],
        ),
    )
    result = {
        "schema": "blindassist_ag_r2_f1_attempt08_depthart_metric_anchor_diagnostic_v1",
        "consumed_internal_validation_only": True,
        "preserved_canary_metrics_opened": False,
        "parents": sorted(validation),
        "data_receipt": data_receipt,
        "feature_receipt": feature_receipt,
        "point_result": {"path": str(POINT_RESULT.resolve()), "sha256": sha256_file(POINT_RESULT)},
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "baseline": baseline_evaluation,
        "selection_order": ["primary_pass_count_desc", "uncertainty_pass_count_desc", "depth_proper_score_gain_desc", "beta_asc", "gamma_asc"],
        "selected": selected,
        "candidates": candidates,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                "selected_beta": selected["beta"],
                "selected_gamma": selected["gamma"],
                "primary_pass_count": selected["primary_pass_count"],
                "uncertainty_pass_count": selected["uncertainty_pass_count"],
                "depth_uncertainty": selected["gate"]["uncertainty"]["depth"],
                "depth_metrics": {
                    key: value
                    for key, value in selected["gate"]["metric_improvements"].items()
                    if key.startswith("depth_")
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
