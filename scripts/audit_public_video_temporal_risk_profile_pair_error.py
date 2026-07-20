#!/usr/bin/env python3
"""Audit a same-source safe-lateral/positive pair after an r7.66 failure."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_dense_future_ego_trace_probe as dense


SCHEMA = "blindassist_public_video_temporal_risk_profile_pair_error_audit_v1"


def pair_checks(
    negative_model_score: float,
    positive_model_score: float,
    negative_teacher_score: float,
    positive_teacher_score: float,
    threshold: float,
) -> dict[str, bool]:
    return {
        "offline_teacher_orders_pair": positive_teacher_score > negative_teacher_score,
        "causal_head_orders_pair": positive_model_score > negative_model_score,
        "positive_head_score_passes_threshold": positive_model_score >= threshold,
        "safe_lateral_head_score_fails_threshold": negative_model_score >= threshold,
    }


def _event_window(report: dict[str, Any], sample_interval_ms: int) -> tuple[int, int]:
    event = report["frozen_radial_event"]
    return int(event["event_entry_timestamp_ms"]), int(event["last_active_timestamp_ms"]) + sample_interval_ms


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.dense_contract,
        args.features,
        args.negative_result,
        args.positive_result,
        args.output,
    ):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")

    features = lifecycle.verify_json_sidecar(args.features)
    negative = lifecycle.verify_json_sidecar(args.negative_result)
    positive = lifecycle.verify_json_sidecar(args.positive_result)
    if negative.get("review_role") != "true_radial_safe_lateral_negative":
        raise ValueError("negative result role mismatch")
    if positive.get("review_role") != "prospective_positive_event":
        raise ValueError("positive result role mismatch")
    source_id = str(negative.get("source_id"))
    if source_id != str(positive.get("source_id")):
        raise ValueError("pair must come from one source")
    source_rows = [row for row in features.get("sources", []) if row.get("source_id") == source_id]
    if len(source_rows) != 1:
        raise ValueError("feature report must bind the pair source exactly once")
    source_sha = str(source_rows[0].get("video_sha256", ""))
    if source_sha != negative["inputs"]["source_video_sha256"] \
            or source_sha != positive["inputs"]["source_video_sha256"]:
        raise ValueError("pair video lineage mismatch")

    contract = common.load_json(args.dense_contract)
    teacher_policy = contract.get("teacher")
    if not isinstance(teacher_policy, dict):
        raise ValueError("dense future-route contract lacks teacher policy")
    sample_interval_ms = int(features.get("sampling", {}).get("sample_interval_ms", 0))
    if sample_interval_ms <= 0:
        raise ValueError("feature sampling interval is invalid")

    negative_teacher = dense.evaluate_event(
        features,
        source_id,
        _event_window(negative, sample_interval_ms),
        teacher_policy,
    )
    positive_teacher = dense.evaluate_event(
        features,
        source_id,
        _event_window(positive, sample_interval_ms),
        teacher_policy,
    )
    negative_model_score = float(negative["event_score"])
    positive_model_score = float(positive["event_score"])
    threshold = float(negative["fixed_event_threshold"])
    if threshold != float(positive["fixed_event_threshold"]):
        raise ValueError("pair threshold mismatch")
    negative_teacher_score = float(negative_teacher["mean_trace_intrusion_score"] or 0.0)
    positive_teacher_score = float(positive_teacher["mean_trace_intrusion_score"] or 0.0)
    checks = pair_checks(
        negative_model_score,
        positive_model_score,
        negative_teacher_score,
        positive_teacher_score,
        threshold,
    )

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "post_failure_diagnostic_only",
        "inputs": {
            "dense_future_teacher_contract_sha256": common.sha256_file(args.dense_contract),
            "feature_report_sha256": common.sha256_file(args.features),
            "negative_risk_profile_result_sha256": common.sha256_file(args.negative_result),
            "positive_risk_profile_result_sha256": common.sha256_file(args.positive_result),
            "source_video_sha256": source_sha,
        },
        "source_id": source_id,
        "pair": {
            "negative": {
                "causal_head_score": negative_model_score,
                "offline_teacher_score": negative_teacher_score,
                "teacher_diagnostics": negative_teacher,
            },
            "positive": {
                "causal_head_score": positive_model_score,
                "offline_teacher_score": positive_teacher_score,
                "teacher_diagnostics": positive_teacher,
            },
            "fixed_threshold": threshold,
            "causal_head_margin": positive_model_score - negative_model_score,
            "offline_teacher_margin": positive_teacher_score - negative_teacher_score,
        },
        "checks": checks,
        "diagnosis": {
            "offline_teacher_remains_directionally_valid": checks["offline_teacher_orders_pair"],
            "causal_head_safe_lateral_rejection_failed": checks["safe_lateral_head_score_fails_threshold"],
            "representation_or_readout_failure_confirmed": (
                checks["offline_teacher_orders_pair"]
                and checks["safe_lateral_head_score_fails_threshold"]
            ),
            "threshold_retuning_authorized": False,
        },
        "evidence_limit": "Post-r7.66 failure diagnosis only; same-source pair is not distinct-source prospective closure or human truth.",
        "authorization": {
            "training": False,
            "calibration": False,
            "blind": False,
            "five_seed_short_runs": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-contract", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--negative-result", type=Path, required=True)
    parser.add_argument("--positive-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({
        "ok": True,
        "representation_or_readout_failure_confirmed": value["diagnosis"]["representation_or_readout_failure_confirmed"],
        "output_sha256": common.sha256_file(parsed.output),
    }))
