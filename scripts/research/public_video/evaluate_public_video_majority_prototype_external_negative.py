#!/usr/bin/env python3
"""Evaluate the frozen majority-horizon prototype on one bound external negative."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import evaluate_public_video_majority_pair_prototype_lifecycle as majority
import evaluate_public_video_pair_prototype_risk_profile_lifecycle as profile
import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_marker_relation_pair_ranking_probe as pair_probe
import run_public_video_marker_relation_linear_probe as linear
import evaluate_public_video_temporal_risk_profile_prospective as prospective


SCHEMA = "blindassist_public_video_majority_prototype_external_negative_v1"


def fit_models(contract: dict[str, Any], linear_contract: dict[str, Any]) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    x, y, sources, timestamps = pair_probe.load_data(linear_contract)
    strong = y >= float(contract["target"]["strong_intrusion_fraction_at_least"])
    pairs = pair_probe.nearest_time_pairs(strong, sources, timestamps)
    deltas = np.stack([x[row["positive_index"]] - x[row["negative_index"]] for row in pairs])
    pair_sources = np.asarray([row["source_id"] for row in pairs])
    models: list[dict[str, np.ndarray]] = []
    audits: list[dict[str, Any]] = []
    ensemble = contract["prototype_ensemble"]
    for seed in ensemble["seeds"]:
        model, audit = majority.fit_bootstrap_prototype(
            deltas, pair_sources, int(seed) + int(ensemble["final_seed_offset"])
        )
        models.append(model)
        audits.append({"seed": int(seed), **audit})
    return models, audits


def accepted_event_vectors(
    report: dict[str, Any], features: dict[str, Any], training_contract: dict[str, Any],
    model_dir: Path, expansion: float, batch_size: int
) -> tuple[np.ndarray, list[int]]:
    source_id = str(report["source_id"])
    source_rows = [row for row in features.get("sources", []) if row.get("source_id") == source_id]
    if len(source_rows) != 1:
        raise ValueError("external source row is missing")
    event = report["frozen_radial_event"]
    timestamps = list(range(int(event["event_entry_timestamp_ms"]),
                            int(event["last_active_timestamp_ms"]) + 1, 1000))
    grids, samples = prospective._build_features(
        source_rows[0], timestamps, training_contract, model_dir, batch_size
    )
    vectors: list[np.ndarray] = []
    accepted: list[int] = []
    for timestamp, grid, sample in zip(timestamps, grids, samples):
        mask = linear.marker_grid_mask(sample.get("detections", []), grid.shape[-1], expansion)
        if not mask.any():
            continue
        vectors.append(linear.relation_vector(grid, mask))
        accepted.append(timestamp)
    if len(vectors) != int(event["accepted_sample_count"]):
        raise ValueError("accepted marker sample count differs from frozen radial event")
    return np.stack(vectors), accepted


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = (args.contract, args.r777_contract, args.linear_contract, args.majority_report,
             args.training_contract, args.features, args.review, args.negative_result,
             args.model_dir, args.output)
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = common.load_json(args.contract)
    bound = contract["bound_inputs"]
    checks = (
        (args.r777_contract, "r777_contract_sha256"),
        (args.linear_contract, "r767a_contract_sha256"),
        (args.majority_report, "r776_report_sha256"),
        (args.training_contract, "r764_training_contract_sha256"),
        (args.features, "duesseldorf_feature_report_sha256"),
        (args.review, "duesseldorf_safe_review_sha256"),
        (args.negative_result, "duesseldorf_frozen_r766_result_sha256"),
    )
    for path, key in checks:
        if common.sha256_file(path) != bound[key]:
            raise ValueError(f"bound input mismatch: {path}")
    if common.sha256_file(args.model_dir / "pytorch_model.bin") != bound["dinov2_weights_sha256"]:
        raise ValueError("DINOv2 weights mismatch")
    if lifecycle.verify_json_sidecar(args.majority_report).get("diagnostic_gate_passed") is not True:
        raise ValueError("r7.76 majority target gate did not pass")
    review = lifecycle.verify_json_sidecar(args.review)
    negative = lifecycle.verify_json_sidecar(args.negative_result)
    if review.get("role") != "true_radial_safe_lateral_negative":
        raise ValueError("review is not a safe-lateral negative")
    if negative.get("review_role") != "true_radial_safe_lateral_negative":
        raise ValueError("frozen diagnostic does not bind a safe-lateral negative")
    linear_contract = common.load_json(args.linear_contract)
    models, audits = fit_models(contract, linear_contract)
    training_contract = common.load_json(args.training_contract)
    features = lifecycle.verify_json_sidecar(args.features)
    expansion = float(linear_contract["feature_vector"]["marker_expansion_object_heights"])
    values, timestamps = accepted_event_vectors(
        negative, features, training_contract, args.model_dir, expansion, args.batch_size
    )
    spec = contract["causal_profile"]
    event = profile.event_profile(
        models, values, timestamps, int(spec["baseline_frame_count"]),
        int(spec["positive_consecutive_samples_to_open"])
    )
    passed = event["open_timestamp_ms"] is None
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "independent_external_challenge_after_failed_posthoc_no_promotion_credit",
        "inputs": {"contract_sha256": common.sha256_file(args.contract)},
        "prototype_ensemble": {"model_count": len(models), "bootstrap_audits": audits,
                               "trainable_parameters": 0, "saved_weights": False},
        "source_id": review["source"]["source_id"],
        "role": review["role"],
        "causal_profile": event,
        "checks": {"safe_lateral_never_opens": passed},
        "external_negative_diagnostic_passed": passed,
        "evidence_limit": "External diagnostic only after r7.77 failed; no prospective, training, calibration, Android, blind, or production authority.",
        "authorization": contract["authorization"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--r777-contract", type=Path, required=True)
    parser.add_argument("--linear-contract", type=Path, required=True)
    parser.add_argument("--majority-report", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--negative-result", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({"ok": True, "open_timestamp_ms": value["causal_profile"]["open_timestamp_ms"],
                      "passed": value["external_negative_diagnostic_passed"],
                      "output_sha256": common.sha256_file(parsed.output)}))
