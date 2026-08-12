#!/usr/bin/env python3
"""Selection-only deterministic three-seed ensemble diagnostic; canary stays sealed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(MODULE_DIR))

from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    DEFAULT_ATTEMPT02_RESULT,
    DEFAULT_BASELINE_RESULT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_EXTENSION,
    DEFAULT_DEPTHART_SOURCE,
    DEFAULT_FRESH_LABEL_RESULT,
    FactorSplitHead,
    cache_model_outputs,
    choose_geometry_config,
    evaluate_cached,
    extract_features,
    gate,
    prepare,
    require,
    sha256_file,
)

DEFAULT_OUTPUT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt04-selection-ensemble-diagnostic-r0/result.json"


def ensemble_outputs(
    caches: list[list[dict[str, torch.Tensor]]],
    *,
    depth_epistemic_weight: float,
    depth_sigma_scale: float,
    point_source: str,
) -> list[dict[str, torch.Tensor]]:
    rows = []
    for frame_index in range(len(caches[0])):
        members = [cache[frame_index] for cache in caches]
        result: dict[str, torch.Tensor] = {}
        log_depth = torch.stack([row["predicted_log_depth"] for row in members])
        require(point_source in {"seed17", "mean"}, "point source invalid")
        result["predicted_log_depth"] = members[0]["predicted_log_depth"] if point_source == "seed17" else log_depth.mean(dim=0)
        aleatoric_variance = (
            members[0]["depth_log_sigma"].mul(2.0).exp()
            if point_source == "seed17"
            else torch.stack([row["depth_log_sigma"].mul(2.0).exp() for row in members]).mean(dim=0)
        )
        epistemic_variance = log_depth.var(dim=0, correction=0)
        depth_sigma = torch.sqrt(aleatoric_variance + float(depth_epistemic_weight) * epistemic_variance).clamp_min(1.0e-3)
        result["depth_log_sigma"] = (depth_sigma * float(depth_sigma_scale)).clamp_min(1.0e-3).log()
        for key in (
            "depth_valid_probability",
            "support_probability",
            "obstacle_probability",
            "evidence_valid_probability",
            "support_plane_normal_camera_xyz",
            "camera_height_m",
            "support_residual_sigma_m",
            "support_valid_probability",
            "depth_gate",
        ):
            result[key] = members[0][key] if point_source == "seed17" else torch.stack([row[key] for row in members]).mean(dim=0)
        distances = torch.stack([(-3.0 * row["boundary_probability"].clamp_min(1.0e-8).log()).clamp_max(32.0) for row in members])
        if point_source == "seed17":
            result["boundary_probability"] = members[0]["boundary_probability"]
            result["boundary_sigma_px"] = members[0]["boundary_sigma_px"]
        else:
            mean_distance = distances.mean(dim=0)
            result["boundary_probability"] = torch.exp(-mean_distance / 3.0)
            boundary_aleatoric = torch.stack([row["boundary_sigma_px"].square() for row in members]).mean(dim=0)
            result["boundary_sigma_px"] = torch.sqrt(boundary_aleatoric + distances.var(dim=0, correction=0)).clamp(0.05, 64.0)
        rows.append(result)
    return rows


def failed(gate_result: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "primary": [key for key, value in gate_result["metric_improvements"].items() if not value["passed"]],
        "uncertainty": [key for key, value in gate_result["uncertainty"].items() if not value["passed"]],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA required")
    require(not args.output.exists(), f"diagnostic output exists: {args.output}")
    fresh = json.loads(args.fresh_label_result.read_text(encoding="utf-8"))
    selection_rows = [{**row, "role": "CHECKPOINT_SELECTION"} for row in fresh["frames"] if row["role"] == "CHECKPOINT_SELECTION"]
    require(len(selection_rows) == 6, "Attempt-04 selection roster drift")
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    attempt02 = json.loads(args.attempt02_result.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    samples, feature_receipt = extract_features(sorted(selection_rows, key=lambda row: row["sample_id"]), args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device)
    prepared = prepare(samples, device)
    baseline_eval = evaluate_cached(prepared, None, baseline, None, device)
    caches = []
    checkpoints = []
    for seed_row in attempt02["seed_results"]:
        checkpoint = Path(seed_row["composite_checkpoint"]["path"])
        model = FactorSplitHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        caches.append(cache_model_outputs(model, prepared, device))
        checkpoints.append({"seed": int(seed_row["seed"]), "path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)})
        del model
    point_source = "seed17"
    reference = ensemble_outputs(caches, depth_epistemic_weight=1.0, depth_sigma_scale=1.0, point_source=point_source)
    geometry_config, geometry_receipt = choose_geometry_config(prepared, reference, baseline, baseline_eval, device)
    variants = []
    for weight in (0.0, 1.0, 4.0, 16.0):
        for scale in (0.5, 1.0, 2.0):
            outputs = ensemble_outputs(caches, depth_epistemic_weight=weight, depth_sigma_scale=scale, point_source=point_source)
            evaluation = evaluate_cached(prepared, outputs, baseline, geometry_config, device)
            gate_result = gate(evaluation, baseline_eval, 71)
            variants.append({
                "depth_epistemic_weight": weight,
                "depth_sigma_scale": scale,
                "evaluation": evaluation,
                "gate": gate_result,
                "failed": failed(gate_result),
                "pass": gate_result["all_primary_metrics_passed"] and gate_result["all_uncertainty_families_passed"],
            })
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt04_selection_ensemble_diagnostic_v1",
        "selection_only": True,
        "canary_read": False,
        "feature_receipt": feature_receipt,
        "checkpoints": checkpoints,
        "geometry_config": geometry_config,
        "geometry_receipt": geometry_receipt,
        "point_source": point_source,
        "variants": variants,
        "passing_variant_count": sum(row["pass"] for row in variants),
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fresh-label-result", type=Path, default=DEFAULT_FRESH_LABEL_RESULT)
    parser.add_argument("--attempt02-result", type=Path, default=DEFAULT_ATTEMPT02_RESULT)
    parser.add_argument("--baseline-result", type=Path, default=DEFAULT_BASELINE_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--depthart-extension", type=Path, default=DEFAULT_DEPTHART_EXTENSION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in ("fresh_label_result", "attempt02_result", "baseline_result", "depthart_source", "depthart_checkpoint", "depthart_extension", "output"):
        setattr(args, name, getattr(args, name).resolve())
    result = run(args)
    print(json.dumps({"passing_variant_count": result["passing_variant_count"], "variants": [{"weight": row["depth_epistemic_weight"], "scale": row["depth_sigma_scale"], "failed": row["failed"], "pass": row["pass"]} for row in result["variants"]]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
