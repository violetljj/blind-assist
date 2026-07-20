#!/usr/bin/env python3
"""Evaluate visually selected windows against a pre-frozen DINOv2 contract."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import build_public_video_dinov2_prospective_contract as contract_builder
import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import extract_public_video_dinov2_prospective_features as extractor
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_video_dinov2_prospective_pair_result_v1"


def window_mean(samples: Sequence[dict[str, Any]], window: Sequence[int], minimum_samples: int) -> tuple[np.ndarray, list[int]]:
    if len(window) != 2 or int(window[0]) < 0 or int(window[1]) <= int(window[0]):
        raise ValueError("review window is invalid")
    selected = [row for row in samples if int(window[0]) <= int(row["timestamp_ms"]) < int(window[1])]
    if len(selected) < minimum_samples:
        raise ValueError("review window has too few scheduled samples")
    matrix = np.asarray([row["vector"] for row in selected], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("feature samples are invalid")
    return matrix.mean(axis=0), [int(row["timestamp_ms"]) for row in selected]


def evaluate_windows(samples: Sequence[dict[str, Any]], review: dict[str, Any], direction: np.ndarray, *, minimum_samples: int) -> dict[str, Any]:
    means: dict[str, np.ndarray] = {}
    timestamps: dict[str, list[int]] = {}
    for name, field in {"pre_clear": "pre_risk_clear_window_ms", "risk": "risk_present_window_ms", "post_clear": "stable_post_clear_window_ms"}.items():
        means[name], timestamps[name] = window_mean(samples, review[field], minimum_samples)
    if not (int(review["pre_risk_clear_window_ms"][1]) <= int(review["risk_present_window_ms"][0]) and int(review["risk_present_window_ms"][1]) <= int(review["stable_post_clear_window_ms"][0])):
        raise ValueError("review windows are not ordered and non-overlapping")
    open_projection = float((means["risk"] - means["pre_clear"]) @ direction)
    close_projection = float((means["risk"] - means["post_clear"]) @ direction)
    return {"window_sample_timestamps_ms": timestamps, "open_projection": open_projection, "close_projection": close_projection, "open_ordered": open_projection > 0.0, "close_ordered": close_projection > 0.0}


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.contract, args.feature_report, args.review, args.output):
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = lifecycle.verify_json_sidecar(args.contract.resolve())
    features = lifecycle.verify_json_sidecar(args.feature_report.resolve())
    review_document = lifecycle.verify_json_sidecar(args.review.resolve())
    if contract.get("schema") != contract_builder.SCHEMA or features.get("schema") != extractor.SCHEMA:
        raise ValueError("contract or feature report schema mismatch")
    if features.get("contract_sha256") != common.sha256_file(args.contract):
        raise ValueError("feature report contract mismatch")
    if features.get("hazard_or_lifecycle_verdict_emitted") is not False or features.get("sampling", {}).get("review_windows_received") is not False:
        raise ValueError("feature report was contaminated by review or verdict")
    review = review_document.get("review") or review_document
    if review.get("feature_report_sha256") != common.sha256_file(args.feature_report) or review.get("video_sha256") != features["video"]["sha256"] or review.get("source_id") != features["source"]["source_id"]:
        raise ValueError("review lineage mismatch")
    if review.get("full_original_order_reviewed") is not True or review.get("single_continuous_ego_view") is not True or review.get("hard_cut_in_or_between_windows") is not False:
        raise ValueError("visual continuity review did not pass")
    training_sources = set(contract["derivation_inputs"]["training_parent_source_ids"])
    if features["source"]["source_id"] in training_sources or features["video"]["sha256"] == contract["derivation_inputs"]["rice_video_sha256"]:
        raise ValueError("prospective source participated in derivation")
    direction = np.asarray(contract["frozen_prototype"]["direction"], dtype=np.float64)
    if contract_builder.direction_sha256(direction) != contract["frozen_prototype"]["direction_sha256"] or len(direction) != int(contract["frozen_feature_contract"]["feature_dimension"]):
        raise ValueError("frozen direction is invalid")
    minimum = int(contract["prospective_source_requirements"]["minimum_scheduled_samples_per_window"])
    evaluation = evaluate_windows(features["samples"], review, direction, minimum_samples=minimum)
    passed = bool(evaluation["open_ordered"] and evaluation["close_ordered"])
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"contract_sha256": common.sha256_file(args.contract), "feature_report_sha256": common.sha256_file(args.feature_report), "review_sha256": common.sha256_file(args.review), "video_sha256": features["video"]["sha256"]},
        "source": features["source"],
        "review_windows": {key: review[key] for key in ("pre_risk_clear_window_ms", "risk_present_window_ms", "stable_post_clear_window_ms")},
        "evaluation": evaluation,
        "prospective_pair_gate": {"passed": passed, "requirements": {"new_source_lineage": True, "continuous_original_order_review": True, "open_projection_strictly_positive": True, "close_projection_strictly_positive": True}},
        "training_authorized": False, "calibration_authorized": False, "blind_evaluation_authorized": False,
        "android_runtime_change_authorized": False, "production_model_replacement_authorized": False,
        "evidence_limit": "One model-assisted prospective public-video challenge. A pass supports the representation direction but does not establish calibrated event accuracy or production readiness."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--feature-report", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = run(args)
    print(json.dumps({"ok": True, "gate_passed": payload["prospective_pair_gate"]["passed"], "open_projection": payload["evaluation"]["open_projection"], "close_projection": payload["evaluation"]["close_projection"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
