#!/usr/bin/env python3
"""Strictly offline r7.66 prospective temporal risk-profile diagnostic."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import public_video_chromatic_marker_policy as chromatic
import public_video_tristate_contract as tristate_contract
import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import run_public_video_ego_route_distance_field_probe as spatial
import run_public_video_obstacle_aware_route_width_probe as route_width
import run_public_video_radial_lifecycle_gap_bridge_probe as gap
import train_public_video_temporal_route_head as training
from extract_public_video_temporal_route_features import (
    PatchExtractor,
    causal_flow_grid,
    compose_feature_grid,
)


SCHEMA = "blindassist_public_video_temporal_risk_profile_prospective_diagnostic_v1"
EXPECTED_INPUT_CHANNELS = 43


def relative_peak_readout(predicted: np.ndarray, obstacle: np.ndarray) -> dict[str, Any]:
    """Return the fixed r7.66 per-horizon and mean relative-peak score."""
    predicted = np.asarray(predicted)
    obstacle = np.asarray(obstacle, dtype=bool)
    if predicted.ndim != 3 or predicted.shape[0] != 3:
        raise ValueError("predicted logits must have shape (3, H, W)")
    if obstacle.shape != predicted.shape[1:]:
        raise ValueError("obstacle mask shape differs from route logits")
    if not obstacle.any():
        scores = [0.0] * predicted.shape[0]
    else:
        scores = [
            float(np.exp(float(channel[obstacle].max()) - float(channel.max())))
            for channel in predicted
        ]
    return {"per_horizon_relative_peak": scores, "frame_score": float(np.mean(scores))}


def threshold_check(score: float, expected_positive: bool, threshold: float) -> bool:
    """Apply the frozen asymmetric acceptance boundary without calibration."""
    return score >= threshold if expected_positive else score < threshold


def ensure_prospective_source(source_id: str, forbidden_source_ids: set[str]) -> None:
    if source_id in forbidden_source_ids:
        raise ValueError(f"r7.54-r7.65 derivation source is forbidden: {source_id}")


def forbidden_sources_from_bound_cache(path: Path) -> set[str]:
    cache = np.load(path, allow_pickle=False)
    required = {"train_sources", "eval_sources"}
    if not required.issubset(cache.files):
        raise ValueError("bound derivation cache lacks source lineage")
    return set(cache["train_sources"].astype(str).tolist()) | set(cache["eval_sources"].astype(str).tolist())


def forbidden_lineage_from_derivation_contract(path: Path) -> tuple[set[str], set[str]]:
    """Verify r7.54 feature reports and return every derivation source ID and video hash."""
    contract = common.load_json(path)
    feature_reports = contract.get("feature_reports")
    if not isinstance(feature_reports, dict) or not feature_reports:
        raise ValueError("derivation contract lacks feature reports")
    source_ids: set[str] = set()
    video_hashes: set[str] = set()
    for binding in feature_reports.values():
        report_path = Path(binding["path"])
        if not report_path.is_absolute():
            report_path = (Path.cwd() / report_path).resolve()
        if common.sha256_file(report_path) != binding["sha256"]:
            raise ValueError("r7.54 derivation feature report hash mismatch")
        report = common.load_json(report_path)
        for source in report.get("sources", []):
            source_ids.add(str(source["source_id"]))
            video_hashes.add(str(source["video_sha256"]).lower())
    return source_ids, video_hashes


def _bound_review_hash(review: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = review.get(key)
        if isinstance(value, str):
            return value
    return None


def _select_event(events: list[dict[str, Any]], review: dict[str, Any], role: str) -> dict[str, Any]:
    passed = [event for event in events if event.get("radial_approach_passed") is True]
    section = review[role]
    candidate_id = section.get("candidate_id")
    if candidate_id is not None:
        passed = [event for event in passed if event.get("candidate_id") == candidate_id]
    window = section.get("review_window_ms")
    if window and all(value is not None for value in window):
        start, end = map(int, window)
        passed = [event for event in passed if int(event["event_entry_timestamp_ms"]) < end
                  and int(event["last_active_timestamp_ms"]) >= start]
    if len(passed) != 1:
        raise ValueError("review must bind exactly one frozen radial event")
    return passed[0]


def replay_selected_event_lifecycle(
    samples: list[dict[str, Any]],
    policy: dict[str, Any],
    event: dict[str, Any],
) -> dict[str, Any]:
    """Replay one bound event so the one-reminder gate is event-local.

    The candidate report remains bound in full.  Isolation here prevents other
    valid events in the same source from being counted as duplicate reminders
    for the event selected by the prospective review.
    """
    return gap.radial_entry_lifecycle(
        samples,
        policy,
        [event],
        clear_absent_samples=9,
    )


def _build_features(
    source: dict[str, Any], timestamps: list[int], training_contract: dict[str, Any], model_dir: Path,
    batch_size: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    by_timestamp = {int(sample["timestamp_ms"]): sample for sample in source.get("samples", [])}
    if any(timestamp not in by_timestamp for timestamp in timestamps):
        raise ValueError("frozen event timestamp is absent from chromatic feature report")
    input_spec = training_contract["input"]
    side = int(input_spec["grid_side"])
    past_horizons = list(map(int, input_spec["causal_past_flow"]["horizons_ms"]))
    decode_times = sorted(set(timestamps + [timestamp - horizon for timestamp in timestamps for horizon in past_horizons]))
    if not decode_times or decode_times[0] < 0:
        raise ValueError("causal history precedes source start")
    video_path = Path(source["local_video_path"])
    if not video_path.is_file():
        raise ValueError(f"local video missing: {video_path}")
    if common.sha256_file(video_path) != source["video_sha256"]:
        raise ValueError("local video differs from frozen chromatic feature report")
    decoded = route_width.decode_at(video_path, decode_times)
    frames = dict(zip(decode_times, decoded))
    current = [frames[timestamp] for timestamp in timestamps]
    projection_spec = input_spec["current_dinov2_patch_projection"]
    projection = spatial.fixed_projection(
        int(projection_spec["input_dimension"]), int(projection_spec["output_dimension"]),
        int(projection_spec["seed"]),
    )
    tokens = PatchExtractor(model_dir).extract(current, batch_size) @ projection
    rows = []
    for timestamp, image, token_grid in zip(timestamps, current, tokens):
        flow = causal_flow_grid(image, [frames[timestamp - horizon] for horizon in past_horizons], side)
        rows.append(compose_feature_grid(token_grid, image, flow))
    values = np.stack(rows).astype(np.float32)
    if values.shape[1:] != (EXPECTED_INPUT_CHANNELS, side, side):
        raise ValueError(f"r7.64 input must be exactly 43x{side}x{side}, got {values.shape[1:]}")
    return values, [by_timestamp[timestamp] for timestamp in timestamps]


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = (args.contract, args.training_contract, args.derivation_contract, args.derivation_cache, args.chromatic_contract,
             args.features, args.candidates, args.review, args.model_dir, args.weights, args.output)
    for path in paths:
        mil.reject_independent_direction(path)
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    training_contract = json.loads(args.training_contract.read_text(encoding="utf-8"))
    _, chromatic_meta = tristate_contract.load_contract(args.chromatic_contract)
    policy = chromatic.validate_policy(json.loads(args.chromatic_contract.read_text(encoding="utf-8")))
    features = lifecycle.verify_json_sidecar(args.features)
    candidates = lifecycle.verify_json_sidecar(args.candidates)
    review = lifecycle.verify_json_sidecar(args.review)
    contract_sha = common.sha256_file(args.contract)
    training_sha = common.sha256_file(args.training_contract)
    derivation_contract_sha = common.sha256_file(args.derivation_contract)
    derivation_sha = common.sha256_file(args.derivation_cache)
    feature_sha = common.sha256_file(args.features)
    candidate_sha = common.sha256_file(args.candidates)
    review_sha = common.sha256_file(args.review)
    weights_sha = common.sha256_file(args.weights)
    bound = contract["bound_inputs"]
    if training_sha != bound["training_contract_sha256"]:
        raise ValueError("r7.64 training contract hash mismatch")
    if derivation_sha != bound["feature_cache_sha256"]:
        raise ValueError("r7.54-r7.65 derivation cache hash mismatch")
    if features.get("prospective_contract", {}).get("sha256") != chromatic_meta["sha256"]:
        raise ValueError("chromatic feature report contract mismatch")
    if candidates.get("feature_report_sha256") != feature_sha:
        raise ValueError("frozen radial candidates do not bind the feature report")
    if review.get("contract_sha256") != contract_sha:
        raise ValueError("review does not bind the r7.66 contract")
    if review.get("temporal_route_head_weights_sha256") != weights_sha:
        raise ValueError("review does not bind the frozen temporal-head weights")
    if _bound_review_hash(review, ("feature_report_sha256", "full_feature_report_sha256")) != feature_sha:
        raise ValueError("review does not bind the frozen feature report")
    if _bound_review_hash(review, ("candidate_report_sha256", "frozen_radial_candidate_report_sha256")) != candidate_sha:
        raise ValueError("review does not bind the frozen radial candidates")
    if review.get("evidence_role") != "provisional_large_model_silver_not_human_truth":
        raise ValueError("review evidence role must remain provisional silver")
    if review.get("reviewed_after_features_and_candidates_frozen") is not True:
        raise ValueError("review chronology is not prospective")
    source_review = review.get("source", {})
    if source_review.get("registered_before_download_and_visual_review") is not True:
        raise ValueError("source was not prospectively registered")
    if source_review.get("not_used_to_derive_r754_through_r765") is not True:
        raise ValueError("source derivation isolation is not attested")
    if source_review.get("continuous_ego_pedestrian_capture") is not True:
        raise ValueError("source is not a continuous ego-pedestrian capture")
    if source_review.get("original_temporal_order") is not True:
        raise ValueError("source temporal order is not original")
    role = review.get("role")
    if role not in {"prospective_positive_event", "true_radial_safe_lateral_negative"}:
        raise ValueError("unsupported prospective review role")
    if review.get(role, {}).get("applicable") is not True:
        raise ValueError("review role is not applicable")
    if review.get(role, {}).get("hard_cut_or_montage_present") is not False:
        raise ValueError("reviewed event contains a hard cut or montage")
    source_id = source_review.get("source_id")
    cache_forbidden_ids = forbidden_sources_from_bound_cache(args.derivation_cache)
    lineage_forbidden_ids, lineage_forbidden_hashes = forbidden_lineage_from_derivation_contract(
        args.derivation_contract
    )
    ensure_prospective_source(source_id, cache_forbidden_ids | lineage_forbidden_ids)
    source_rows = [row for row in features.get("sources", []) if row.get("source_id") == source_id]
    candidate_rows = [row for row in candidates.get("sources", []) if row.get("source_id") == source_id]
    if len(source_rows) != 1 or len(candidate_rows) != 1:
        raise ValueError("review source must bind one feature and candidate row")
    source = source_rows[0]
    if str(source.get("video_sha256", "")).lower() in lineage_forbidden_hashes:
        raise ValueError("r7.54-r7.65 derivation video hash is forbidden")
    if source_review.get("source_video_sha256") != source.get("video_sha256"):
        raise ValueError("review video lineage mismatch")
    event = _select_event(candidate_rows[0].get("events", []), review, role)
    entry = int(event["event_entry_timestamp_ms"])
    last_active = int(event["last_active_timestamp_ms"])
    event_samples = [sample for sample in source.get("samples", []) if entry <= int(sample["timestamp_ms"]) <= last_active]
    if not event_samples:
        raise ValueError("frozen radial event contains no feature frames")
    timestamps = [int(sample["timestamp_ms"]) for sample in event_samples]
    model_weights = args.model_dir / "pytorch_model.bin"
    if common.sha256_file(model_weights) != training_contract["bound_inputs"]["dinov2_weights_sha256"]:
        raise ValueError("local DINO weights differ from r7.64")
    values, event_samples = _build_features(source, timestamps, training_contract, args.model_dir, args.batch_size)
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if checkpoint.get("contract_sha256") != contract_sha or int(checkpoint.get("input_channels", -1)) != EXPECTED_INPUT_CHANNELS:
        raise ValueError("frozen temporal-head checkpoint binding mismatch")
    model = training.TemporalRouteHead(EXPECTED_INPUT_CHANNELS)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    predicted = training.predict(model, values, args.batch_size)
    side = int(training_contract["input"]["grid_side"])
    expansion = float(contract["risk_profile"]["marker_expansion_object_heights"])
    frame_rows = []
    for timestamp, logits, sample in zip(timestamps, predicted, event_samples):
        obstacle = spatial.obstacle_grid_mask(sample.get("detections", []), side, expansion)
        frame_rows.append({"timestamp_ms": timestamp, **relative_peak_readout(logits, obstacle),
                           "expanded_marker_present": bool(obstacle.any())})
    event_score = float(np.mean([row["frame_score"] for row in frame_rows]))
    threshold = float(contract["risk_profile"]["fixed_event_threshold"])
    expected_positive = role == "prospective_positive_event"
    lifecycle_state = replay_selected_event_lifecycle(source["samples"], policy, event)
    matching_lifecycle = [row for row in lifecycle_state["intervals"]
                          if int(row["event_entry_timestamp_ms"]) == entry]
    positive_lifecycle_gate = None
    if expected_positive:
        positive_lifecycle_gate = gap.score_sweep_row(lifecycle_state, review[role])
    lifecycle_passed = bool(positive_lifecycle_gate and positive_lifecycle_gate.get("passed")) \
        if expected_positive else True
    threshold_passed = threshold_check(event_score, expected_positive, threshold)
    report = {
        "schema": SCHEMA, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "strictly_offline_diagnostic_only", "network_access_used": False,
        "inputs": {"contract_sha256": contract_sha, "training_contract_sha256": training_sha,
                   "derivation_contract_sha256": derivation_contract_sha,
                   "derivation_cache_sha256": derivation_sha, "chromatic_feature_report_sha256": feature_sha,
                   "radial_candidate_report_sha256": candidate_sha, "candidate_review_sha256": review_sha,
                   "temporal_route_head_weights_sha256": weights_sha,
                   "dinov2_weights_sha256": common.sha256_file(model_weights),
                   "source_video_sha256": source["video_sha256"]},
        "source_id": source_id, "review_role": role, "frozen_radial_event": event,
        "feature_contract": {"channels": EXPECTED_INPUT_CHANNELS, "shape": list(values.shape),
                             "same_as_r764": True, "future_frames_used": False},
        "frame_readouts": frame_rows, "event_score": event_score, "fixed_event_threshold": threshold,
        "threshold_relation": ">= threshold" if expected_positive else "< threshold",
        "diagnostic_role_matches_fixed_threshold": threshold_passed,
        "diagnostic_role_matches_fixed_threshold_and_lifecycle": threshold_passed and lifecycle_passed,
        "lifecycle": {"implementation": "frozen_r730_radial_entry_lifecycle",
                      "clear_absent_samples": 9, "state": lifecycle_state,
                      "matching_event_count": len(matching_lifecycle),
                      "positive_timing_gate": positive_lifecycle_gate},
        "evidence_limit": "Diagnostic-only prospective large-model silver evidence, not human truth; no calibration, training, Android, blind-evaluation, or production authority.",
        "prospective_acceptance_credit_present": False,
        "authorization": {**contract["authorization"], "five_seed_short_runs": False,
                          "risk_event_training_authorized": False, "calibration_authorized": False,
                          "blind_evaluation_authorized": False, "android_runtime_change_authorized": False,
                          "production_model_replacement_authorized": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--derivation-contract", type=Path, required=True)
    parser.add_argument("--derivation-cache", type=Path, required=True)
    parser.add_argument("--chromatic-contract", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    value = run(parsed)
    print(json.dumps({"ok": True, "event_score": value["event_score"],
                      "diagnostic_role_matches_fixed_threshold": value["diagnostic_role_matches_fixed_threshold"],
                      "output_sha256": common.sha256_file(parsed.output)}, ensure_ascii=False))
