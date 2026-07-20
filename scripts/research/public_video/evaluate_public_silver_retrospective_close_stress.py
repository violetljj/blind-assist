"""Evaluate a frozen, model-reviewed retrospective risk-to-clear stress case.

The visual review is hash-bound before detector scoring.  The evaluator then
recomputes the already-published risk-window score with the same frozen detector
and compares it with a later unused clear window from the same source timeline.
The result is diagnostic only and cannot satisfy a prospective source gate.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import evaluate_public_video_tristate_lifecycle_external_challenge as lifecycle
import run_public_silver_frozen_feature_probe as common
import run_public_silver_mechanism_temporal_range_probe as temporal
import run_public_silver_object_trajectory_probe as trajectory
import run_public_silver_risk_lifecycle_mil_head as mil


SCHEMA = "blindassist_public_silver_retrospective_close_stress_v1"
REVIEW_SCHEMA = "blindassist_public_silver_retrospective_close_review_v1"
TRAJECTORY_SCHEMA = "blindassist_public_silver_object_trajectory_probe_v1"
REVIEW_DECISION = "accept_as_retrospective_dynamic_risk_to_clear_stress_case"


def occupancy_sequence(frames: Sequence[Sequence[dict[str, Any]]]) -> list[float]:
    return [
        float(max((row["area"] * row["corridor_overlap"] for row in detections), default=0.0))
        for detections in frames
    ]


def score_change(*, risk_score: float, clear_score: float) -> dict[str, Any]:
    if risk_score < 0 or clear_score < 0:
        raise ValueError("temporal-range scores must be non-negative")
    delta = clear_score - risk_score
    margin = abs(delta) / max(abs(risk_score), abs(clear_score), 1e-12)
    predicted = "close_event" if delta < 0 else ("open_event" if delta > 0 else "abstain")
    return {
        "risk_score": risk_score,
        "clear_score": clear_score,
        "signed_score_delta": delta,
        "normalized_absolute_margin": margin,
        "expected_transition": "close_event",
        "predicted_transition": predicted,
        "correct": predicted == "close_event",
    }


def validate_review(review: dict[str, Any]) -> tuple[list[int], list[int]]:
    if review.get("schema") != REVIEW_SCHEMA or review.get("decision") != REVIEW_DECISION:
        raise ValueError("retrospective close review schema or decision mismatch")
    chronology = review.get("chronology_attestation")
    if not isinstance(chronology, dict) or chronology.get("original_frame_order_used") is not True:
        raise ValueError("review does not attest original frame order")
    if chronology.get("hard_cut_observed") is not False:
        raise ValueError("hard-cut review cannot enter the close stress case")
    if review.get("policy_frozen_before_detector_scoring") is not True:
        raise ValueError("visual review was not frozen before detector scoring")
    limitations = review.get("limitations")
    if not isinstance(limitations, dict):
        raise ValueError("review limitations are missing")
    required_false = (
        "new_source", "prospective", "independent_from_prior_source_review",
        "eligible_for_r712_positive_source_gate", "training_authorized",
        "calibration_authorized", "blind_evaluation_authorized",
        "android_integration_authorized", "production_authorized",
    )
    if any(limitations.get(key) is not False for key in required_false):
        raise ValueError("review overstates retrospective evidence authorization")
    risk = review.get("risk_window")
    clear = review.get("clear_window")
    if not isinstance(risk, dict) or not isinstance(clear, dict):
        raise ValueError("review risk/clear windows are missing")
    risk_indices = risk.get("frame_indices")
    clear_indices = clear.get("frame_indices")
    if (
        not isinstance(risk_indices, list) or len(risk_indices) < 2
        or not isinstance(clear_indices, list) or len(clear_indices) < 2
        or not all(isinstance(value, int) for value in risk_indices + clear_indices)
    ):
        raise ValueError("review windows need at least two integer frame indices")
    if max(risk_indices) >= min(clear_indices):
        raise ValueError("review risk window must precede the clear window")
    return risk_indices, clear_indices


def bound_frames(
    *,
    review_window: dict[str, Any],
    frame_indices: Sequence[int],
    source: dict[str, Any],
) -> list[dict[str, Any]]:
    image_root = Path(source["promotion"]["image_root"]).resolve()
    by_index = {row.get("frame_index"): row for row in source.get("frames", []) if isinstance(row, dict)}
    expected_hashes = review_window.get("frame_sha256")
    expected_source_indices = review_window.get("source_frame_indices")
    if not isinstance(expected_hashes, list) or len(expected_hashes) != len(frame_indices):
        raise ValueError("review frame hashes are missing or misaligned")
    if not isinstance(expected_source_indices, list) or len(expected_source_indices) != len(frame_indices):
        raise ValueError("review source frame indices are missing or misaligned")
    rows = []
    for position, frame_index in enumerate(frame_indices):
        frame = by_index.get(frame_index)
        if not isinstance(frame, dict):
            raise ValueError(f"review frame is absent from source manifest: {frame_index}")
        if frame.get("sha256") != expected_hashes[position]:
            raise ValueError(f"review frame hash differs from source manifest: {frame_index}")
        if frame.get("source_frame_index") != expected_source_indices[position]:
            raise ValueError(f"review source frame index differs from manifest: {frame_index}")
        path = (image_root / str(frame["file_name"])).resolve()
        if not path.is_relative_to(image_root) or not path.is_file():
            raise ValueError(f"review image path is invalid: {path}")
        if common.sha256_file(path) != frame["sha256"]:
            raise ValueError(f"review image bytes differ from source manifest: {path}")
        rows.append({"frame_index": frame_index, "sha256": frame["sha256"], "path": str(path)})
    return rows


def find_frozen_risk_score(report: dict[str, Any], source_id: str) -> dict[str, Any]:
    if report.get("schema") != temporal.SCHEMA:
        raise ValueError("temporal report schema mismatch")
    rows = (report.get("pair_scores") or {}).get(temporal.DYNAMIC)
    matches = [row for row in rows or [] if row.get("source_id") == source_id]
    if len(matches) != 1:
        raise ValueError("temporal report must contain exactly one dynamic pair for the review source")
    return matches[0]


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.review, args.source_manifest, args.temporal_report,
        args.trajectory_report, args.detector_weights, args.cache_dir, args.output,
    ):
        mil.reject_independent_direction(path)
    review = lifecycle.verify_json_sidecar(args.review)
    temporal_report = lifecycle.verify_json_sidecar(args.temporal_report)
    trajectory_report = lifecycle.verify_json_sidecar(args.trajectory_report)
    risk_indices, clear_indices = validate_review(review)

    source_path = args.source_manifest.resolve()
    if not source_path.is_file() or common.sha256_file(source_path) != (review.get("source_manifest") or {}).get("sha256"):
        raise ValueError("source manifest is missing or differs from the frozen review")
    if Path((review.get("source_manifest") or {}).get("path", "")).resolve() != source_path:
        raise ValueError("supplied source manifest path differs from the frozen review")
    source = common.load_json(source_path)
    risk_frames = bound_frames(
        review_window=review["risk_window"], frame_indices=risk_indices, source=source,
    )
    clear_frames = bound_frames(
        review_window=review["clear_window"], frame_indices=clear_indices, source=source,
    )

    source_id = review.get("source_id")
    frozen_score_row = find_frozen_risk_score(temporal_report, str(source_id))
    if trajectory_report.get("schema") != TRAJECTORY_SCHEMA:
        raise ValueError("trajectory provenance report schema mismatch")
    feature_source = trajectory_report.get("feature_source")
    if not isinstance(feature_source, dict):
        raise ValueError("trajectory detector provenance is missing")
    weights = args.detector_weights.resolve()
    if common.sha256_file(weights) != feature_source.get("weights_sha256"):
        raise ValueError("detector weights differ from frozen trajectory provenance")
    if int(feature_source.get("image_size", -1)) != 320 or float(feature_source.get("confidence", -1)) != 0.15:
        raise ValueError("frozen detector inference contract differs from the close stress contract")

    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(cache)
    from ultralytics import YOLO
    detector = YOLO(str(weights))
    episodes = [
        {"episode_id": "risk_window", "frames": risk_frames},
        {"episode_id": "clear_window", "frames": clear_frames},
    ]
    detections = trajectory.extract_frame_detections(
        detector, episodes, image_size=320, confidence=0.15,
    )
    risk_occupancy = occupancy_sequence(detections[0])
    clear_occupancy = occupancy_sequence(detections[1])
    risk_score = float(max(risk_occupancy) - min(risk_occupancy))
    clear_score = float(max(clear_occupancy) - min(clear_occupancy))
    frozen_risk_score = float(frozen_score_row["alert_score"])
    if abs(risk_score - frozen_risk_score) > 1e-12:
        raise ValueError("recomputed risk score differs from the frozen temporal report")
    transition = score_change(risk_score=risk_score, clear_score=clear_score)

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "review": {"path": str(args.review.resolve()), "sha256": common.sha256_file(args.review)},
            "source_manifest": {"path": str(source_path), "sha256": common.sha256_file(source_path)},
            "temporal_report": {"path": str(args.temporal_report.resolve()), "sha256": common.sha256_file(args.temporal_report)},
            "trajectory_report": {"path": str(args.trajectory_report.resolve()), "sha256": common.sha256_file(args.trajectory_report)},
            "detector_weights": {"path": str(weights), "sha256": common.sha256_file(weights)},
        },
        "source_id": source_id,
        "mechanism": temporal.DYNAMIC,
        "risk_window": {
            "frame_indices": risk_indices,
            "occupancy_sequence": risk_occupancy,
            "score": risk_score,
            "matches_frozen_temporal_score": True,
        },
        "clear_window": {
            "frame_indices": clear_indices,
            "occupancy_sequence": clear_occupancy,
            "score": clear_score,
        },
        "transition": transition,
        "acceptance": {
            "visual_review_frozen_before_scoring": True,
            "frozen_risk_score_reproduced": True,
            "retrospective_close_direction_correct": transition["correct"],
            "passed": transition["correct"],
        },
        "evidence_limit": "Retrospective model-reviewed dynamic close stress case from an existing derivation source; not a new or prospective source, not human truth, and not eligible for the r7.12 source gate.",
        "training_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "android_integration_authorized": False,
        "production_authorized": False,
    }
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError(f"refusing to overwrite output or sidecar: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--temporal-report", type=Path, required=True)
    parser.add_argument("--trajectory-report", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
