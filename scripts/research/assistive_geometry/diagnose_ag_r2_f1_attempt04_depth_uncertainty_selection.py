#!/usr/bin/env python3
"""Selection-only search for a physical depth-uncertainty score; canary is never read."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

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
    evaluate_cached,
    extract_features,
    prepare,
    require,
    sha256_file,
)

DEFAULT_OUTPUT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt04-depth-uncertainty-selection-r0/result.json"
GAUSSIAN_CONSTANT = 0.5 * math.log(2.0 * math.pi)


def sampled_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size)
    return np.linspace(0, size - 1, num=limit, dtype=np.int64)


def quantile_means(score: np.ndarray, residual: np.ndarray) -> list[float]:
    order = np.argsort(score, kind="stable")
    return [float(residual[group].mean()) for group in np.array_split(order, 4)]


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA required")
    require(not args.output.exists(), f"output exists: {args.output}")
    fresh = json.loads(args.fresh_label_result.read_text(encoding="utf-8"))
    rows = [{**row, "role": "CHECKPOINT_SELECTION"} for row in fresh["frames"] if row["role"] == "CHECKPOINT_SELECTION"]
    require(len(rows) == 6, "selection roster drift")
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    attempt02 = json.loads(args.attempt02_result.read_text(encoding="utf-8"))
    device = torch.device(args.device)
    samples, feature_receipt = extract_features(sorted(rows, key=lambda row: row["sample_id"]), args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device)
    prepared = prepare(samples, device)
    baseline_eval = evaluate_cached(prepared, None, baseline, None, device)
    caches = []
    checkpoints = []
    for seed_row in attempt02["seed_results"]:
        checkpoint = Path(seed_row["composite_checkpoint"]["path"])
        model = FactorSplitHead(baseline).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
        caches.append(cache_model_outputs(model, prepared, device))
        checkpoints.append({"seed": int(seed_row["seed"]), "sha256": sha256_file(checkpoint)})
        del model

    frames = []
    for index, row in enumerate(prepared):
        sample, target = row["sample"], row["target"]
        valid = target["depth_valid"]
        seed17 = caches[0][index]
        logs = torch.stack([cache[index]["predicted_log_depth"] for cache in caches])
        residual = (seed17["predicted_log_depth"] - target["depth"].clamp_min(0.01).log()).abs()
        aleatoric = seed17["depth_log_sigma"].exp()
        epistemic = logs.std(dim=0, correction=0)
        base_depth = F.interpolate(sample.base_depth_feature[None].to(device=device, dtype=torch.float32), sample.native_hw, mode="bilinear", align_corners=False).clamp_min(0.01)
        base_disagreement = (seed17["predicted_log_depth"] - base_depth.log()).abs()
        invalidity = 1.0 - seed17["depth_valid_probability"]
        boundary = seed17["boundary_probability"]
        arrays = {
            "residual": residual[valid].cpu().numpy().astype(np.float64),
            "aleatoric": aleatoric[valid].cpu().numpy().astype(np.float64),
            "epistemic": epistemic[valid].cpu().numpy().astype(np.float64),
            "base_disagreement": base_disagreement[valid].cpu().numpy().astype(np.float64),
            "invalidity": invalidity[valid].cpu().numpy().astype(np.float64),
            "boundary": boundary[valid].cpu().numpy().astype(np.float64),
        }
        frames.append({"sample_id": sample.sample_id, "parent_id": sample.parent_id, "arrays": arrays})

    sample_features = {key: [] for key in ("residual", "aleatoric", "epistemic", "base_disagreement", "invalidity", "boundary")}
    for frame in frames:
        indices = sampled_indices(frame["arrays"]["residual"].size, 30000)
        for key in sample_features:
            sample_features[key].append(frame["arrays"][key][indices])
    sample_features = {key: np.concatenate(value) for key, value in sample_features.items()}
    candidates = []
    weights = (0.0, 0.5, 1.0, 2.0, 4.0)
    for epistemic_weight in weights:
        for base_weight in weights:
            for invalidity_weight in weights:
                score = (
                    sample_features["aleatoric"]
                    + epistemic_weight * sample_features["epistemic"]
                    + base_weight * sample_features["base_disagreement"]
                    + invalidity_weight * sample_features["invalidity"]
                )
                correlation = float(np.corrcoef(score, sample_features["residual"])[0, 1])
                q = quantile_means(score, sample_features["residual"])
                candidates.append({
                    "epistemic_weight": epistemic_weight,
                    "base_disagreement_weight": base_weight,
                    "invalidity_weight": invalidity_weight,
                    "sample_correlation": correlation,
                    "sample_quantile_residual_means": q,
                    "sample_nondecreasing": all(left <= right + 1.0e-6 for left, right in zip(q, q[1:])),
                })
    shortlisted = sorted(candidates, key=lambda row: (-row["sample_correlation"], row["epistemic_weight"], row["base_disagreement_weight"], row["invalidity_weight"]))[:20]
    exact_rows = []
    for candidate in shortlisted:
        parent_nll: dict[str, list[float]] = defaultdict(list)
        full_score = []
        full_residual = []
        raw_by_frame = []
        for frame in frames:
            arrays = frame["arrays"]
            raw = (
                arrays["aleatoric"]
                + candidate["epistemic_weight"] * arrays["epistemic"]
                + candidate["base_disagreement_weight"] * arrays["base_disagreement"]
                + candidate["invalidity_weight"] * arrays["invalidity"]
            ).clip(1.0e-3)
            raw_by_frame.append((frame, raw))
            full_score.append(raw)
            full_residual.append(arrays["residual"])
        concatenated_raw = np.concatenate(full_score)
        concatenated_residual = np.concatenate(full_residual)
        optimal_scale = float(np.sqrt(np.mean((concatenated_residual / concatenated_raw) ** 2)))
        optimal_scale = min(max(optimal_scale, 0.10), 10.0)
        for frame, raw in raw_by_frame:
            sigma = (raw * optimal_scale).clip(1.0e-3)
            residual = frame["arrays"]["residual"]
            nll = 0.5 * (residual / sigma) ** 2 + np.log(sigma) + GAUSSIAN_CONSTANT
            parent_nll[frame["parent_id"]].append(float(nll.mean()))
        parent_macro = {parent: float(np.mean(value)) for parent, value in parent_nll.items()}
        improvements = {
            parent: float(baseline_eval["parent_metrics"][parent]["depth_nll"] - value)
            for parent, value in parent_macro.items()
        }
        exact_q = quantile_means(concatenated_raw, concatenated_residual)
        exact_rows.append({
            **candidate,
            "scale": optimal_scale,
            "exact_quantile_residual_means": exact_q,
            "exact_nondecreasing": all(left <= right + 1.0e-6 for left, right in zip(exact_q, exact_q[1:])),
            "parent_depth_nll": parent_macro,
            "parent_improvements": improvements,
            "proper_score_pass": all(value > 0.0 for value in improvements.values()),
            "eligible": all(left <= right + 1.0e-6 for left, right in zip(exact_q, exact_q[1:])) and all(value > 0.0 for value in improvements.values()),
        })
    eligible = [row for row in exact_rows if row["eligible"]]
    selected = min(eligible, key=lambda row: (max(row["parent_depth_nll"].values()), -row["sample_correlation"])) if eligible else None
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt04_depth_uncertainty_selection_v1",
        "selection_only": True,
        "canary_read": False,
        "feature_receipt": feature_receipt,
        "checkpoints": checkpoints,
        "candidate_count": len(candidates),
        "shortlisted": exact_rows,
        "eligible_count": len(eligible),
        "selected": selected,
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
    print(json.dumps({"eligible_count": result["eligible_count"], "selected": result["selected"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
