#!/usr/bin/env python3
"""Evaluate the frozen r7.73 zero-training pair-prototype risk profile/lifecycle."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_marker_relation_pair_bootstrap_short_runs as pair_bootstrap
import run_public_video_marker_relation_pair_ranking_probe as pair_probe
import run_public_video_marker_relation_linear_probe as linear


SCHEMA = "blindassist_public_video_pair_prototype_risk_profile_lifecycle_v1"


def causal_open_timestamp(timestamps: list[int], relative_scores: np.ndarray, consecutive: int) -> int | None:
    scores = np.asarray(relative_scores, dtype=np.float64)
    if len(timestamps) != len(scores) or consecutive <= 0:
        raise ValueError("invalid causal profile")
    run_length = 0
    for timestamp, score in zip(timestamps, scores):
        run_length = run_length + 1 if score > 0.0 else 0
        if run_length >= consecutive:
            return int(timestamp)
    return None


def event_profile(
    models: list[dict[str, np.ndarray]], vectors: np.ndarray, timestamps: list[int],
    baseline_frames: int, consecutive: int,
) -> dict[str, Any]:
    if len(vectors) < baseline_frames or baseline_frames <= 0:
        raise ValueError("event lacks baseline frames")
    rows = []
    for model in models:
        absolute = pair_bootstrap.projection(model, vectors)
        baseline = float(np.median(absolute[:baseline_frames]))
        rows.append(absolute - baseline)
    model_relative = np.stack(rows)
    ensemble = np.median(model_relative, axis=0)
    eligible_timestamps = timestamps[baseline_frames:]
    eligible_scores = ensemble[baseline_frames:]
    opened = causal_open_timestamp(eligible_timestamps, eligible_scores, consecutive)
    return {
        "timestamps_ms": timestamps,
        "ensemble_relative_scores": ensemble.tolist(),
        "per_model_relative_scores": model_relative.tolist(),
        "baseline_frame_count": baseline_frames,
        "open_timestamp_ms": opened,
        "reminder_timestamps_ms": [] if opened is None else [opened],
        "maximum_relative_score": float(ensemble.max()),
        "minimum_relative_score": float(ensemble.min()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.bootstrap_contract, args.bootstrap_report,
                 args.training_contract, args.bangkok_features, args.negative_result,
                 args.positive_result, args.model_dir, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.bootstrap_contract, "r772_contract_sha256"),
                      (args.bootstrap_report, "r772_report_sha256"),
                      (args.bangkok_features, "bangkok_feature_report_sha256"),
                      (args.negative_result, "bangkok_negative_result_sha256"),
                      (args.positive_result, "bangkok_positive_result_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    bootstrap_report = lifecycle.verify_json_sidecar(args.bootstrap_report)
    if bootstrap_report.get("summary", {}).get("runs_passing") != 5:
        raise ValueError("r7.72 did not preserve all absolute run gates")
    linear_contract = common.load_json(args.linear_contract)
    bootstrap_contract = common.load_json(args.bootstrap_contract)
    x, y, sources, timestamps = pair_probe.load_data(linear_contract)
    active = y > 0.0
    pairs = pair_probe.nearest_time_pairs(active, sources, timestamps)
    deltas = np.stack([x[row["positive_index"]] - x[row["negative_index"]] for row in pairs])
    pair_sources = np.asarray([row["source_id"] for row in pairs])
    models = []
    audits = []
    for seed in bootstrap_contract["seeds"]:
        _, prototype, audit = pair_bootstrap.fit_pair_head(
            deltas, pair_sources, int(seed) + int(contract["prototype_ensemble"]["final_seed_offset"]),
            bootstrap_contract["optimizer"],
        )
        models.append(prototype)
        audits.append({"seed": int(seed), "sampled_unique_source_count": audit["sampled_unique_source_count"],
                       "sampled_source_ids": audit["sampled_source_ids"]})
    training_contract = common.load_json(args.training_contract)
    features = lifecycle.verify_json_sidecar(args.bangkok_features)
    negative = lifecycle.verify_json_sidecar(args.negative_result)
    positive = lifecycle.verify_json_sidecar(args.positive_result)
    expansion = float(linear_contract["feature_vector"]["marker_expansion_object_heights"])
    negative_x, negative_times = linear._bangkok_event_vectors(
        negative, features, training_contract, args.model_dir, expansion, args.batch_size
    )
    positive_x, positive_times = linear._bangkok_event_vectors(
        positive, features, training_contract, args.model_dir, expansion, args.batch_size
    )
    profile_spec = contract["causal_profile"]
    negative_profile = event_profile(models, negative_x, negative_times,
                                     int(profile_spec["baseline_frame_count"]),
                                     int(profile_spec["positive_consecutive_samples_to_open"]))
    positive_profile = event_profile(models, positive_x, positive_times,
                                     int(profile_spec["baseline_frame_count"]),
                                     int(profile_spec["positive_consecutive_samples_to_open"]))
    expected = contract["posthoc_bangkok_checks"]
    positive_open = positive_profile["open_timestamp_ms"]
    inherited_lifecycle = positive["lifecycle"]
    checks = {
        "safe_lateral_never_opens": negative_profile["open_timestamp_ms"] is None,
        "positive_opens": positive_open is not None,
        "positive_open_not_late": positive_open is not None and positive_open <= int(expected["positive_latest_open_timestamp_ms"]),
        "positive_single_reminder": len(positive_profile["reminder_timestamps_ms"]) == 1,
        "inherited_event_local_clear_passed": inherited_lifecycle["positive_timing_gate"]["checks"]["clears_inside_stable_route_clear_window"] is True,
    }
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract),
                   "r772_report_sha256": common.sha256_file(args.bootstrap_report)},
        "prototype_ensemble": {"model_count": len(models), "bootstrap_audits": audits,
                               "trainable_parameters": 0, "saved_weights": False},
        "causal_profile_contract": profile_spec,
        "bangkok_same_source_posthoc_diagnostic": {
            "prospective_credit": False,
            "safe_lateral": negative_profile,
            "positive": positive_profile,
            "positive_inherited_event_local_lifecycle": inherited_lifecycle,
        },
        "checks": checks, "posthoc_diagnostic_passed": all(checks.values()),
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--bootstrap-contract", type=Path, required=True)
    parser.add_argument("--bootstrap-report", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--bangkok-features", type=Path, required=True)
    parser.add_argument("--negative-result", type=Path, required=True)
    parser.add_argument("--positive-result", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    result = run(parsed)
    diagnostic = result["bangkok_same_source_posthoc_diagnostic"]
    print(json.dumps({"ok": True, "safe_open": diagnostic["safe_lateral"]["open_timestamp_ms"],
                      "positive_open": diagnostic["positive"]["open_timestamp_ms"],
                      "passed": result["posthoc_diagnostic_passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
