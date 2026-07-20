#!/usr/bin/env python3
"""Evaluate r7.77 majority-horizon bootstrap prototypes on Bangkok lifecycle pressure."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_pair_prototype_risk_profile_lifecycle as profile
import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_marker_relation_pair_bootstrap_short_runs as pair_bootstrap
import run_public_video_marker_relation_pair_ranking_probe as pair_probe
import run_public_video_marker_relation_linear_probe as linear


SCHEMA = "blindassist_public_video_majority_pair_prototype_lifecycle_v1"


def fit_bootstrap_prototype(
    deltas: np.ndarray, pair_sources: np.ndarray, seed: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sampled, weights, draws = pair_bootstrap.bootstrap_pair_rows(pair_sources, seed)
    values = np.asarray(deltas, dtype=np.float64)
    variance = np.average(values[sampled] ** 2, axis=0, weights=weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-8] = 1.0
    normalized = values[sampled] / scale
    direction = np.average(normalized, axis=0, weights=weights)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("bootstrap prototype is degenerate")
    direction /= norm
    return {"scale": scale, "weight": direction}, {
        "sampled_source_ids": draws, "sampled_unique_source_count": len(set(draws))
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.linear_contract, args.majority_report, args.training_contract,
                 args.bangkok_features, args.negative_result, args.positive_result, args.model_dir, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    for path, key in ((args.linear_contract, "r767a_contract_sha256"),
                      (args.majority_report, "r776_report_sha256"),
                      (args.bangkok_features, "bangkok_feature_report_sha256"),
                      (args.negative_result, "bangkok_negative_result_sha256"),
                      (args.positive_result, "bangkok_positive_result_sha256")):
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    if lifecycle.verify_json_sidecar(args.majority_report).get("diagnostic_gate_passed") is not True:
        raise ValueError("r7.76 majority target gate did not pass")
    linear_contract = common.load_json(args.linear_contract)
    x, y, sources, timestamps = pair_probe.load_data(linear_contract)
    strong = y >= float(contract["target"]["strong_intrusion_fraction_at_least"])
    pairs = pair_probe.nearest_time_pairs(strong, sources, timestamps)
    deltas = np.stack([x[row["positive_index"]] - x[row["negative_index"]] for row in pairs])
    pair_sources = np.asarray([row["source_id"] for row in pairs])
    models = []
    audits = []
    ensemble = contract["prototype_ensemble"]
    for seed in ensemble["seeds"]:
        model, audit = fit_bootstrap_prototype(deltas, pair_sources, int(seed) + int(ensemble["final_seed_offset"]))
        models.append(model)
        audits.append({"seed": int(seed), **audit})
    training_contract = common.load_json(args.training_contract)
    features = lifecycle.verify_json_sidecar(args.bangkok_features)
    negative = lifecycle.verify_json_sidecar(args.negative_result)
    positive = lifecycle.verify_json_sidecar(args.positive_result)
    expansion = float(linear_contract["feature_vector"]["marker_expansion_object_heights"])
    negative_x, negative_times = linear._bangkok_event_vectors(negative, features, training_contract, args.model_dir, expansion, args.batch_size)
    positive_x, positive_times = linear._bangkok_event_vectors(positive, features, training_contract, args.model_dir, expansion, args.batch_size)
    spec = contract["causal_profile"]
    negative_profile = profile.event_profile(models, negative_x, negative_times, int(spec["baseline_frame_count"]),
                                             int(spec["positive_consecutive_samples_to_open"]))
    positive_profile = profile.event_profile(models, positive_x, positive_times, int(spec["baseline_frame_count"]),
                                             int(spec["positive_consecutive_samples_to_open"]))
    positive_open = positive_profile["open_timestamp_ms"]
    inherited = positive["lifecycle"]
    checks = {"safe_lateral_never_opens": negative_profile["open_timestamp_ms"] is None,
              "positive_opens": positive_open is not None,
              "positive_open_not_late": positive_open is not None and positive_open <= int(contract["posthoc_bangkok_checks"]["positive_latest_open_timestamp_ms"]),
              "positive_single_reminder": len(positive_profile["reminder_timestamps_ms"]) == 1,
              "inherited_event_local_clear_passed": inherited["positive_timing_gate"]["checks"]["clears_inside_stable_route_clear_window"] is True}
    report = {"schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": {"contract_sha256": common.sha256_file(args.contract), "r776_report_sha256": common.sha256_file(args.majority_report)},
              "prototype_ensemble": {"model_count": len(models), "bootstrap_audits": audits,
                                     "trainable_parameters": 0, "saved_weights": False},
              "causal_profile_contract": spec,
              "bangkok_same_source_posthoc_diagnostic": {"prospective_credit": False,
                                                          "safe_lateral": negative_profile,
                                                          "positive": positive_profile,
                                                          "positive_inherited_event_local_lifecycle": inherited},
              "checks": checks, "posthoc_diagnostic_passed": all(checks.values()), "authorization": contract["authorization"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--majority-report", type=Path, required=True)
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
