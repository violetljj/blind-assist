#!/usr/bin/env python3
"""Compare paired bounded REveL detector runs without expanding GPU load."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _guard_valid(guard: dict[str, Any], benchmark: dict[str, Any]) -> bool:
    limits = guard.get("limits", {})
    observed = guard.get("observed", {})
    return (
        guard.get("format") == "blindassist_guarded_gpu_run_v1"
        and guard.get("exit_code") == 0
        and guard.get("stop_reason") is None
        and guard.get("monitor_samples", 0) > 0
        and limits.get("max_frames") == benchmark.get("dataset", {}).get("evaluated_frames")
        and limits.get("batch") == benchmark.get("model", {}).get("batch") == 1
        and observed.get("relevant_system_events") == 0
        and observed.get("max_temperature_c", float("inf")) < limits.get("max_temperature_c", 0)
    )


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    baseline_guard: dict[str, Any],
    candidate_guard: dict[str, Any],
) -> dict[str, Any]:
    baseline_dataset = baseline.get("dataset", {})
    candidate_dataset = candidate.get("dataset", {})
    baseline_model = baseline.get("model", {})
    candidate_model = candidate.get("model", {})
    paired = (
        baseline.get("format") == candidate.get("format") == "blindassist_revel_yolo11n_person_benchmark_v2"
        and baseline_dataset == candidate_dataset
        and baseline_model.get("weights_sha256") == candidate_model.get("weights_sha256")
        and baseline_model.get("batch") == candidate_model.get("batch") == 1
        and baseline_model.get("half") is candidate_model.get("half") is False
        and baseline_model.get("imgsz") != candidate_model.get("imgsz")
        and _guard_valid(baseline_guard, baseline)
        and _guard_valid(candidate_guard, candidate)
    )
    if not paired:
        raise ValueError("runs are not a valid paired, guarded resolution comparison")

    baseline_metrics = baseline["fixed_score_metrics"]
    candidate_metrics = candidate["fixed_score_metrics"]
    baseline_strata = baseline["recall_by_normalized_box_area"]
    candidate_strata = candidate["recall_by_normalized_box_area"]
    metric_names = ("precision", "recall", "f1")
    strata_names = ("small", "medium", "large")
    metric_delta = {name: candidate_metrics[name] - baseline_metrics[name] for name in metric_names}
    stratum_delta = {name: candidate_strata[name]["recall"] - baseline_strata[name]["recall"] for name in strata_names}
    ap50_delta = candidate["ap50_over_score_floor"] - baseline["ap50_over_score_floor"]

    # Resolution scaling is useful here only if it improves the named failure
    # mode (small targets) without lowering the aggregate F1.
    scale_candidate = stratum_delta["small"] > 0.0 and metric_delta["f1"] >= 0.0
    return {
        "format": "blindassist_revel_detector_resolution_sensitivity_v1",
        "paired_receipt_valid": True,
        "baseline_imgsz": baseline_model["imgsz"],
        "candidate_imgsz": candidate_model["imgsz"],
        "evaluated_frames": baseline_dataset["evaluated_frames"],
        "ground_truth_boxes": baseline_dataset["person_ground_truth_boxes"],
        "baseline": {
            "ap50": baseline["ap50_over_score_floor"],
            **{name: baseline_metrics[name] for name in metric_names},
            "recall_by_area": {name: baseline_strata[name]["recall"] for name in strata_names},
            "guard": baseline_guard["observed"],
        },
        "candidate": {
            "ap50": candidate["ap50_over_score_floor"],
            **{name: candidate_metrics[name] for name in metric_names},
            "recall_by_area": {name: candidate_strata[name]["recall"] for name in strata_names},
            "guard": candidate_guard["observed"],
        },
        "delta_candidate_minus_baseline": {
            "ap50": ap50_delta,
            **metric_delta,
            "recall_by_area": stratum_delta,
        },
        "candidate_qualifies_for_512_frame_scale": scale_candidate,
        "recommendation": "scale_candidate_to_512" if scale_candidate else "do_not_scale_candidate_to_512",
        "reason": "candidate must strictly improve small-target recall without reducing aggregate F1",
        "production_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-guard", type=Path, required=True)
    parser.add_argument("--candidate-guard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(_read(args.baseline), _read(args.candidate), _read(args.baseline_guard), _read(args.candidate_guard))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"recommendation": report["recommendation"], "small_recall_delta": report["delta_candidate_minus_baseline"]["recall_by_area"]["small"], "f1_delta": report["delta_candidate_minus_baseline"]["f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
