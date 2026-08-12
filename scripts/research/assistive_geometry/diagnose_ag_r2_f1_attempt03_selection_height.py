#!/usr/bin/env python3
"""Selection-only height attribution diagnostic; never reads Attempt-03 canary rows."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
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
    apply_geometry,
    extract_features,
    geometry_height_and_sigma,
    height_candidates,
    parent_vio_height_context,
    prepare,
    require,
    sha256_file,
)

DEFAULT_OUTPUT = REPO_ROOT / "artifacts.local/experiments/ag-r2-f1-attempt03-selection-height-diagnostic-r0/result.json"


def best_configuration(prepared: list[dict[str, Any]], outputs: list[dict[str, torch.Tensor]], device: torch.device) -> dict[str, Any]:
    rows = []
    learned_sigma = {"source": "attempt02_learned", "multiplier": 1.0}
    for config in height_candidates():
        parent_errors: dict[str, list[float]] = defaultdict(list)
        predictions = []
        parent_context = parent_vio_height_context(prepared, outputs, config, device)
        for index, row in enumerate(prepared):
            adjusted, receipt = apply_geometry(
                outputs[index],
                row["sample"],
                {"height": config, "support_sigma": learned_sigma},
                {},
                device,
                parent_context,
            )
            height = adjusted["camera_height_m"]
            target = float(row["target"]["height"])
            error = abs(float(torch.log(height[0])) - float(torch.log(row["target"]["height"])))
            parent_errors[row["sample"].parent_id].append(error)
            predictions.append({"sample_id": row["sample"].sample_id, "predicted_height_m": float(height[0]), "target_height_m": target, "abs_log_error": error, "receipt": receipt})
        macro = {key: float(np.mean(value)) for key, value in parent_errors.items()}
        rows.append({"config": config, "parent_errors": macro, "maximum_parent_error": max(macro.values()), "mean_parent_error": float(np.mean(list(macro.values()))), "predictions": predictions})
    return min(rows, key=lambda row: (row["maximum_parent_error"], row["mean_parent_error"], json.dumps(row["config"], sort_keys=True)))


def substitute(outputs: list[dict[str, torch.Tensor]], prepared: list[dict[str, Any]], *, truth_depth: bool, truth_support: bool) -> list[dict[str, torch.Tensor]]:
    result = []
    for raw, row in zip(outputs, prepared):
        value = dict(raw)
        target = row["target"]
        if truth_depth:
            value["predicted_log_depth"] = target["depth"].clamp_min(0.01).log()
            value["depth_valid_probability"] = target["depth_valid"].float()
        if truth_support:
            value["support_probability"] = target["support"]
            value["depth_valid_probability"] = target["support_valid"].float()
        result.append(value)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA required")
    require(not args.output.exists(), f"diagnostic output exists: {args.output}")
    fresh = json.loads(args.fresh_label_result.read_text(encoding="utf-8"))
    rows = [{**row, "role": "CHECKPOINT_SELECTION"} for row in fresh["frames"] if row["role"] == "CHECKPOINT_SELECTION"]
    require(len(rows) == 6 and not any(row["role"] == "TRAIN_CANARY" for row in rows), "selection-only roster drift")
    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))["baseline_parameters"]
    attempt02 = json.loads(args.attempt02_result.read_text(encoding="utf-8"))
    seed17 = next(row for row in attempt02["seed_results"] if int(row["seed"]) == 17)
    checkpoint = Path(seed17["composite_checkpoint"]["path"])
    device = torch.device(args.device)
    samples, feature_receipt = extract_features(sorted(rows, key=lambda row: row["sample_id"]), args.depthart_source, args.depthart_checkpoint, args.depthart_extension, device)
    prepared = prepare(samples, device)
    model = FactorSplitHead(baseline).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["model"], strict=True)
    outputs = cache_model_outputs(model, prepared, device)
    scenarios = {
        "predicted_depth_predicted_support": outputs,
        "source_depth_predicted_support_upper_bound": substitute(outputs, prepared, truth_depth=True, truth_support=False),
        "predicted_depth_source_support_upper_bound": substitute(outputs, prepared, truth_depth=False, truth_support=True),
        "source_depth_source_support_oracle_mechanics": substitute(outputs, prepared, truth_depth=True, truth_support=True),
    }
    result = {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt03_selection_height_attribution_diagnostic_v1",
        "selection_only": True,
        "canary_read": False,
        "feature_receipt": feature_receipt,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "baseline_camera_height_m": baseline["camera_height_m"],
        "scenarios": {key: best_configuration(prepared, value, device) for key, value in scenarios.items()},
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
    print(json.dumps({key: {"max_parent_error": value["maximum_parent_error"], "mean_parent_error": value["mean_parent_error"], "config": value["config"]} for key, value in result["scenarios"].items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
