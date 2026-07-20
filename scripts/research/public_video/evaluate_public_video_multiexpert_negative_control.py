#!/usr/bin/env python3
"""Evaluate a frozen r7.23 multi-expert contract on a visual negative control."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import build_public_video_dinov2_prospective_contract as dino_contract_builder
import evaluate_public_video_dinov2_prospective_pair as dino_eval
import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_multiexpert_risk_profile_contract as multiexpert
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common


SCHEMA = "blindassist_public_video_multiexpert_negative_control_result_v1"


def interval_overlaps(interval: dict[str, Any], window: Sequence[int]) -> bool:
    start = int(interval["event_entry_timestamp_ms"])
    end = int(interval.get("confirmed_clear_timestamp_ms", interval["last_active_timestamp_ms"]))
    return start < int(window[1]) and end >= int(window[0])


def evaluate_channels(
    *,
    dino_samples: Sequence[dict[str, Any]],
    dino_direction: np.ndarray,
    chromatic_samples: Sequence[dict[str, Any]],
    chromatic_contract: dict[str, Any],
    windows: dict[str, Sequence[int]],
    minimum_samples: int,
) -> dict[str, Any]:
    dino_review = {
        "pre_risk_clear_window_ms": windows["pre_clear"],
        "risk_present_window_ms": windows["negative_challenge"],
        "stable_post_clear_window_ms": windows["post_clear"],
    }
    dino = dino_eval.evaluate_windows(
        dino_samples, dino_review, dino_direction, minimum_samples=minimum_samples
    )
    policy = chromatic.validate_policy(chromatic_contract)
    filtered = chromatic.apply_policy(chromatic_samples, policy)
    lc = chromatic_contract["lifecycle"]
    chromatic_lifecycle = lifecycle.tristate_exit_intervals(
        filtered,
        lc["selected_groups"],
        entry_window_samples=int(lc["entry_window_samples"]),
        entry_min_active_samples=int(lc["entry_min_active_samples"]),
        clear_absent_samples=int(lc["clear_absent_samples"]),
    )
    overlapping = [
        row for row in chromatic_lifecycle["intervals"]
        if interval_overlaps(row, windows["negative_challenge"])
    ]
    if chromatic_lifecycle["open_event"] and interval_overlaps(
        chromatic_lifecycle["open_event"], windows["negative_challenge"]
    ):
        overlapping.append(chromatic_lifecycle["open_event"])
    positive_channels = []
    if dino["open_ordered"]:
        positive_channels.append("general_static_dinov2")
    if overlapping:
        positive_channels.append("chromatic_construction_marker")
    return {
        "dinov2": dino,
        "chromatic": {
            "lifecycle": chromatic_lifecycle,
            "negative_challenge_overlapping_events": overlapping,
        },
        "fusion": {
            "positive_channels": positive_channels,
            "would_open_event": bool(positive_channels),
            "negative_control_passed": not positive_channels,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract, contract_meta = multiexpert.load_contract(args.multiexpert_contract)
    dino_contract = lifecycle.verify_json_sidecar(args.dinov2_contract)
    chromatic_contract, chromatic_meta = tristate_contract.load_contract(args.chromatic_contract)
    dino_features = lifecycle.verify_json_sidecar(args.dinov2_features)
    chromatic_features = lifecycle.verify_json_sidecar(args.chromatic_features)
    review = lifecycle.verify_json_sidecar(args.review)

    multiexpert.verify_bound_inputs(
        contract,
        dinov2_contract=args.dinov2_contract,
        chromatic_contract=args.chromatic_contract,
        prototype_report=args.prototype_report,
    )
    if review.get("multiexpert_contract_sha256") != contract_meta["sha256"]:
        raise ValueError("review multi-expert contract hash mismatch")
    if review.get("dinov2_feature_report_sha256") != common.sha256_file(args.dinov2_features):
        raise ValueError("review DINO feature hash mismatch")
    if review.get("chromatic_feature_report_sha256") != common.sha256_file(args.chromatic_features):
        raise ValueError("review chromatic feature hash mismatch")
    if review.get("reviewed_after_all_channel_features_frozen") is not True:
        raise ValueError("review chronology is invalid")
    continuity = review.get("continuity", {})
    if continuity != {
        "continuous_ego_pedestrian_capture": True,
        "original_temporal_order": True,
        "hard_cut_or_montage_observed": False,
    }:
        raise ValueError("review continuity gate failed")
    finding = review.get("visual_finding", {})
    if finding.get("pedestrian_corridor_risk_present") is not False or finding.get("should_open_risk_event") is not False:
        raise ValueError("review is not a negative control")
    if dino_features["video"]["sha256"] != review["video_sha256"]:
        raise ValueError("DINO video lineage mismatch")
    chromatic_source = chromatic_features["sources"][0]
    if chromatic_source["video_sha256"] != review["video_sha256"]:
        raise ValueError("chromatic video lineage mismatch")

    direction = np.asarray(dino_contract["frozen_prototype"]["direction"], dtype=np.float64)
    if dino_contract_builder.direction_sha256(direction) != dino_contract["frozen_prototype"]["direction_sha256"]:
        raise ValueError("DINO direction drift")
    result = evaluate_channels(
        dino_samples=dino_features["samples"],
        dino_direction=direction,
        chromatic_samples=chromatic_source["samples"],
        chromatic_contract=chromatic_contract,
        windows=review["windows"],
        minimum_samples=int(dino_contract["prospective_source_requirements"]["minimum_scheduled_samples_per_window"]),
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "multiexpert_contract": contract_meta,
            "dinov2_contract_sha256": common.sha256_file(args.dinov2_contract),
            "chromatic_contract": chromatic_meta,
            "dinov2_features_sha256": common.sha256_file(args.dinov2_features),
            "chromatic_features_sha256": common.sha256_file(args.chromatic_features),
            "review_sha256": common.sha256_file(args.review),
            "video_sha256": review["video_sha256"],
        },
        "source_id": review["source_id"],
        "visual_negative_control": finding,
        "windows": review["windows"],
        **result,
        "authorizations": {
            "training": False,
            "calibration": False,
            "blind": False,
            "android_runtime_change": False,
            "production_model_replacement": False,
        },
        "evidence_limit": "One large-model-reviewed prospective negative control; not human truth or production evidence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiexpert-contract", type=Path, required=True)
    parser.add_argument("--dinov2-contract", type=Path, required=True)
    parser.add_argument("--chromatic-contract", type=Path, required=True)
    parser.add_argument("--prototype-report", type=Path, required=True)
    parser.add_argument("--dinov2-features", type=Path, required=True)
    parser.add_argument("--chromatic-features", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    value = run(args)
    print(json.dumps({"ok": True, **value["fusion"], "output_sha256": common.sha256_file(args.output)}, ensure_ascii=False))
