#!/usr/bin/env python3
"""Train on isolated provisional episodes and transfer to SANPO events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_stage_c_d6_sanpo_real_event_transfer import (
    DEFAULT_BASELINE,
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    ManifestFrames,
    aggregate,
    hold_sampled_values,
    infer_manifest_probabilities,
    load_model,
    reference_comparison,
    sampled_indices,
    score_event,
)
from evaluate_stage_c_d5_tartanground_event_proxy import (
    causal_confirmation,
)
from run_stage_c_d6_sanpo_weak_relation_head import (
    L2_STRENGTH,
    PRIMARY_CONFIRMATION_STEPS,
    event_balanced_weights,
    feature_matrix,
    fit_logistic,
    predict_event,
    weighted_standardize,
)
from run_stage_c_d6_sanpo_spatial_relation_head import (
    SPATIAL_L2_STRENGTH,
    infer_spatial_matrices,
)
from train_stage_c_d5_tartanground_development_student import sha256


DEFAULT_PROVISIONAL_ROOT = Path(
    "artifacts.local/evidence/"
    "public-video-provisional-training-r7-20260717"
)
DEFAULT_NORMAL_NEGATIVE_ROOT = Path(
    "artifacts.local/evidence/candidate-event-mining/"
    "cem-r0-real-20260802-2hz-yolo-depth-proxy"
)
EXPECTED_EPISODE_COUNT = 16
EXPECTED_SOURCE_COUNT = 9
EXPECTED_FRAME_COUNT = 611
PARALLEL_CURB_EPISODE_COUNT = 17
PARALLEL_CURB_SOURCE_COUNT = 10
PARALLEL_CURB_FRAME_COUNT = 692
LABELS = {
    "candidate_no_alert": 0,
    "candidate_alert": 1,
}
FEATURE_FAMILIES = (
    "direction_profiles",
    "spatial_grid_3x6",
)
DEVELOPMENT_THRESHOLDS = tuple(
    round(0.30 + 0.05 * index, 2)
    for index in range(11)
)


def source_session_id(
    source_id: str,
    manifest: dict[str, Any],
) -> str | None:
    source = manifest.get("source", {})
    if isinstance(source, dict) and source.get("session_id"):
        return str(source["session_id"])
    if source_id.startswith("sanpo_real_"):
        return source_id[len("sanpo_real_") :]
    return None


def collect_training_episodes(
    root: Path,
    evaluation_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evaluation_sessions = {
        event["source_session_id"]
        for event in evaluation_manifest["events"]
    }
    episodes = []
    exclusions = []
    for label_path in sorted(root.rglob("silver_labels_v2.json")):
        labels = json.loads(label_path.read_text(encoding="utf-8"))
        if not labels.get("training_execution_authorized", False):
            exclusions.append(
                {
                    "source_id": labels["source"]["source_id"],
                    "reason": "training_not_authorized",
                }
            )
            continue
        manifest_path = label_path.parent / labels["source"][
            "source_manifest_path"
        ]
        source_manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        source_id = str(labels["source"]["source_id"])
        session_id = source_session_id(source_id, source_manifest)
        if session_id in evaluation_sessions:
            exclusions.append(
                {
                    "source_id": source_id,
                    "reason": "overlaps_evaluation_session",
                }
            )
            continue
        frame_by_sha = {
            frame["sha256"]: frame
            for frame in source_manifest["frames"]
        }
        image_root = Path(source_manifest["promotion"]["image_root"])
        source_episodes = []
        source_missing = []
        for episode in labels["episodes"]:
            label_name = episode["silver_should_alert"]
            if label_name not in LABELS:
                continue
            frames = []
            for frame_sha in episode["evidence_frame_sha256"]:
                frame = frame_by_sha.get(frame_sha)
                if frame is None:
                    source_missing.append(
                        f"unmapped_sha256:{frame_sha}"
                    )
                    continue
                image_path = image_root / frame["file_name"]
                if not image_path.is_file():
                    source_missing.append(str(image_path))
                    continue
                frames.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "image_sha256": frame_sha,
                    }
                )
            source_episodes.append(
                {
                    "episode_id": episode["episode_id"],
                    "source_id": source_id,
                    "source_session_id": session_id,
                    "label_name": label_name,
                    "label": LABELS[label_name],
                    "confidence": float(episode["confidence"]),
                    "risk_profile": episode.get("risk_profile", {}),
                    "frames": frames,
                    "label_path": str(label_path.resolve()),
                    "label_sha256": sha256(label_path),
                    "source_manifest_path": str(
                        manifest_path.resolve()
                    ),
                    "source_manifest_sha256": sha256(manifest_path),
                }
            )
        if source_missing:
            exclusions.append(
                {
                    "source_id": source_id,
                    "reason": "missing_evidence_frames",
                    "missing_count": len(source_missing),
                    "examples": source_missing[:3],
                }
            )
            continue
        episodes.extend(source_episodes)
    episode_ids = [episode["episode_id"] for episode in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Duplicate provisional episode ID")
    return episodes, exclusions


def collect_reviewed_negative_sources(
    root: Path,
    include_quarantined_parallel_curb: bool = False,
) -> list[dict[str, Any]]:
    reviews = {
        row["candidate_id"]: row
        for row in (
            json.loads(line)
            for line in (root / "luna_reviews.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    }
    report = json.loads(
        (root / "review_queue_report.json").read_text(encoding="utf-8")
    )
    frames_by_source: dict[str, dict[str, dict[str, Any]]] = {}
    candidate_ids_by_source: dict[str, set[str]] = {}
    observed_types_by_source: dict[str, set[str]] = {}
    for candidate in report["candidates"]:
        review = reviews.get(candidate["candidate_id"])
        if review is None:
            raise ValueError(
                f"Missing candidate review: {candidate['candidate_id']}"
            )
        normal_passage = (
            review["disposition"] == "reject"
            and review["observed_types"]
            == ["normal_passage_negative"]
        )
        provisional_parallel_curb = (
            include_quarantined_parallel_curb
            and review["disposition"] == "quarantine"
            and review["observed_types"] == ["parallel_curb"]
            and review["abstained"] is True
        )
        if not normal_passage and not provisional_parallel_curb:
            continue
        source_id = candidate["source_id"]
        candidate_ids_by_source.setdefault(source_id, set()).add(
            candidate["candidate_id"]
        )
        source_frames = frames_by_source.setdefault(source_id, {})
        observed_types_by_source.setdefault(source_id, set()).update(
            review["observed_types"]
        )
        for frame in candidate["frame_refs"]:
            path = Path(frame["frame_ref"])
            if not path.is_file():
                raise ValueError(
                    f"Missing normal-passage frame: {path}"
                )
            existing = source_frames.get(frame["frame_sha256"])
            row = {
                "image_path": str(path.resolve()),
                "image_sha256": frame["frame_sha256"],
                "timestamp_ms": int(frame["timestamp_ms"]),
            }
            if (
                existing is not None
                and existing["image_path"] != row["image_path"]
            ):
                raise ValueError(
                    "One normal-passage SHA maps to multiple paths"
                )
            source_frames[frame["frame_sha256"]] = row
    episodes = []
    for source_id in sorted(frames_by_source):
        frames = sorted(
            frames_by_source[source_id].values(),
            key=lambda row: (
                row["timestamp_ms"],
                row["image_sha256"],
            ),
        )
        episodes.append(
            {
                "episode_id": (
                    "candidate-mining-normal-passage/"
                    f"{source_id}"
                ),
                "source_id": source_id,
                "source_session_id": None,
                "label_name": "candidate_no_alert",
                "label": 0,
                "confidence": None,
                "risk_profile": {
                    "primary_hazard_type": "none",
                    "corridor_relation": "weak_external_negative",
                    "lifecycle": "no_alert",
                    "observed_types": sorted(
                        observed_types_by_source[source_id]
                    ),
                },
                "frames": frames,
                "candidate_ids": sorted(
                    candidate_ids_by_source[source_id]
                ),
                "review_path": str(
                    (root / "luna_reviews.jsonl").resolve()
                ),
                "review_sha256": sha256(
                    root / "luna_reviews.jsonl"
                ),
                "candidate_report_path": str(
                    (root / "review_queue_report.json").resolve()
                ),
                "candidate_report_sha256": sha256(
                    root / "review_queue_report.json"
                ),
            }
        )
    return episodes


def infer_profile_matrices(
    model: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    batch_size: int,
) -> tuple[list[np.ndarray], list[str]]:
    dataset = ManifestFrames(manifest_path, manifest)
    risks, knowns = infer_manifest_probabilities(
        model,
        dataset,
        manifest,
        batch_size,
    )
    matrices = []
    feature_names = None
    for risk, known in zip(risks, knowns, strict=True):
        matrix, names = feature_matrix(risk, known)
        matrices.append(matrix)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature order drift")
    if feature_names is None:
        raise ValueError("No profile features")
    return matrices, feature_names


def infer_feature_matrices(
    model: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    batch_size: int,
    feature_family: str,
) -> tuple[list[np.ndarray], list[str]]:
    if feature_family == "direction_profiles":
        return infer_profile_matrices(
            model,
            manifest_path,
            manifest,
            batch_size,
        )
    if feature_family == "spatial_grid_3x6":
        return infer_spatial_matrices(
            model,
            ManifestFrames(manifest_path, manifest),
            manifest,
            batch_size,
        )
    raise ValueError(f"Unsupported feature family: {feature_family}")


def active_at_threshold(
    event: dict[str, Any],
    held_probabilities: list[float],
    threshold: float,
) -> list[bool]:
    indices = sampled_indices(event)
    immediate = [
        held_probabilities[index] >= threshold
        for index in indices
    ]
    confirmed = causal_confirmation(
        immediate,
        PRIMARY_CONFIRMATION_STEPS,
    )
    return hold_sampled_values(
        confirmed,
        indices,
        len(event["frames"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--provisional-root",
        type=Path,
        default=DEFAULT_PROVISIONAL_ROOT,
    )
    parser.add_argument(
        "--normal-negative-root",
        type=Path,
        default=DEFAULT_NORMAL_NEGATIVE_ROOT,
    )
    parser.add_argument(
        "--include-quarantined-parallel-curb",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--feature-family",
        choices=FEATURE_FAMILIES,
        default="direction_profiles",
    )
    args = parser.parse_args()

    evaluation_manifest = json.loads(
        args.manifest.read_text(encoding="utf-8")
    )
    events = evaluation_manifest["events"]
    if (
        int(evaluation_manifest["event_count"]) != 30
        or sum(len(event["frames"]) for event in events) != 1920
    ):
        raise ValueError("Expected the 30-event / 1,920-frame SANPO view")
    episodes, exclusions = collect_training_episodes(
        args.provisional_root,
        evaluation_manifest,
    )
    reviewed_negative_episodes = collect_reviewed_negative_sources(
        args.normal_negative_root,
        args.include_quarantined_parallel_curb,
    )
    episodes.extend(reviewed_negative_episodes)
    source_ids = {episode["source_id"] for episode in episodes}
    frame_count = sum(len(episode["frames"]) for episode in episodes)
    label_counts = {
        label_name: sum(
            episode["label_name"] == label_name
            for episode in episodes
        )
        for label_name in LABELS
    }
    expected_inventory = (
        (
            PARALLEL_CURB_EPISODE_COUNT,
            PARALLEL_CURB_SOURCE_COUNT,
            PARALLEL_CURB_FRAME_COUNT,
            {"candidate_no_alert": 9, "candidate_alert": 8},
        )
        if args.include_quarantined_parallel_curb
        else (
            EXPECTED_EPISODE_COUNT,
            EXPECTED_SOURCE_COUNT,
            EXPECTED_FRAME_COUNT,
            {"candidate_no_alert": 8, "candidate_alert": 8},
        )
    )
    if (
        len(episodes) != expected_inventory[0]
        or len(source_ids) != expected_inventory[1]
        or frame_count != expected_inventory[2]
        or label_counts != expected_inventory[3]
    ):
        raise ValueError(
            "Unexpected provisional training inventory: "
            f"episodes={len(episodes)}, sources={len(source_ids)}, "
            f"frames={frame_count}, labels={label_counts}"
        )

    model, checkpoint = load_model(
        args.pretrained,
        args.checkpoint,
    )
    training_manifest = {
        "events": [
            {"frames": episode["frames"]}
            for episode in episodes
        ]
    }
    training_matrices, feature_names = infer_feature_matrices(
        model,
        Path("provisional-training-manifest.json"),
        training_manifest,
        args.batch_size,
        args.feature_family,
    )
    x_train = np.concatenate(training_matrices, axis=0)
    y_train = np.concatenate(
        [
            np.full(
                len(matrix),
                episode["label"],
                dtype=np.int64,
            )
            for episode, matrix in zip(
                episodes,
                training_matrices,
                strict=True,
            )
        ]
    )
    training_episode_ids = [
        episode["episode_id"]
        for episode, matrix in zip(
            episodes,
            training_matrices,
            strict=True,
        )
        for _ in range(len(matrix))
    ]
    weights = event_balanced_weights(
        training_episode_ids,
        y_train,
    )
    mean, scale = weighted_standardize(x_train, weights)
    l2_strength = (
        SPATIAL_L2_STRENGTH
        if args.feature_family == "spatial_grid_3x6"
        else L2_STRENGTH
    )
    coefficient, intercept, loss = fit_logistic(
        (x_train - mean) / scale,
        y_train,
        weights,
        l2_strength=l2_strength,
    )

    evaluation_matrices, evaluation_names = infer_feature_matrices(
        model,
        args.manifest,
        evaluation_manifest,
        args.batch_size,
        args.feature_family,
    )
    if feature_names != evaluation_names:
        raise ValueError("Training/evaluation feature order drift")
    scored_events = []
    threshold_event_scores = {
        threshold: [] for threshold in DEVELOPMENT_THRESHOLDS
    }
    for event, matrix in zip(
        events,
        evaluation_matrices,
        strict=True,
    ):
        active, probabilities = predict_event(
            event,
            matrix,
            coefficient,
            intercept,
            mean,
            scale,
        )
        score = score_event(event, active)
        score["probability_median"] = float(
            np.median(probabilities)
        )
        score["probability_max"] = float(
            np.max(probabilities)
        )
        scored_events.append(score)
        for threshold in DEVELOPMENT_THRESHOLDS:
            threshold_event_scores[threshold].append(
                score_event(
                    event,
                    active_at_threshold(
                        event,
                        probabilities,
                        threshold,
                    ),
                )
            )
    metrics = aggregate(scored_events)
    baseline_result = json.loads(
        args.baseline.read_text(encoding="utf-8")
    )
    baseline = baseline_result["event_evaluation"][
        "current_yolo_reference"
    ]
    development_threshold_sweep = [
        {
            "threshold": threshold,
            "metrics": threshold_metrics,
            "comparison_to_current_yolo": reference_comparison(
                threshold_metrics,
                baseline,
            ),
        }
        for threshold in DEVELOPMENT_THRESHOLDS
        for threshold_metrics in [
            aggregate(threshold_event_scores[threshold])
        ]
    ]
    spatial = args.feature_family == "spatial_grid_3x6"
    weak_parallel_curb = (
        args.include_quarantined_parallel_curb
    )
    result = {
        "schema": (
            "blindassist_hftf_stage_c_d6_provisional_relation_"
            + (
                "cross_source_spatial_weak_parallel_curb_v0"
                if spatial and weak_parallel_curb
                else "cross_source_spatial_normal_negative_v0"
                if spatial
                else "cross_source_transfer_normal_negative_v1"
            )
        ),
        "status": (
            "PROVISIONAL_RELATION_CROSS_SOURCE_"
            + (
                "SPATIAL_WEAK_PARALLEL_CURB_COMPLETE"
                if spatial and weak_parallel_curb
                else "SPATIAL_NORMAL_NEGATIVE_COMPLETE"
                if spatial
                else "TRANSFER_NORMAL_NEGATIVE_COMPLETE"
            )
        ),
        "policy": {
            "training_data_role": "provisional_model_supervision",
            "evaluation_data_role": "consumed_development",
            "training_evaluation_source_isolated": True,
            "evaluation_sessions_used_for_fit": False,
            "fixed_hftf_backbone": True,
            "feature_family": args.feature_family,
            "feature_count": len(feature_names),
            "episode_balanced_training": True,
            "class_balanced_training": True,
            "l2_strength": l2_strength,
            "probability_threshold": 0.5,
            "causal_confirmation_steps_at_5hz": (
                PRIMARY_CONFIRMATION_STEPS
            ),
            "human_event_truth_training": False,
            "human_safety_or_app_claim": False,
            "normal_passage_negatives_reviewed": True,
            "quarantine_candidates_used": weak_parallel_curb,
            "quarantined_parallel_curb_canary": (
                weak_parallel_curb
            ),
        },
        "model": {
            "name": args.name,
            "architecture": checkpoint.get("architecture", "pooled"),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256(args.checkpoint),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "training": {
            "root": str(args.provisional_root.resolve()),
            "normal_negative_root": str(
                args.normal_negative_root.resolve()
            ),
            "include_quarantined_parallel_curb": (
                weak_parallel_curb
            ),
            "episode_count": len(episodes),
            "source_count": len(source_ids),
            "frame_count": frame_count,
            "label_counts": label_counts,
            "weighted_regularized_train_loss": loss,
            "intercept": intercept,
            "coefficient_l2_norm": float(
                np.linalg.norm(coefficient)
            ),
            "coefficients": {
                name: float(value)
                for name, value in zip(
                    feature_names,
                    coefficient,
                    strict=True,
                )
            },
            "episodes": episodes,
            "excluded_sources": exclusions,
        },
        "evaluation": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "event_count": evaluation_manifest["event_count"],
            "frame_count": sum(
                len(event["frames"]) for event in events
            ),
            "bucket_counts": evaluation_manifest["bucket_counts"],
        },
        "current_yolo_reference": baseline,
        "metrics": metrics,
        "comparison_to_current_yolo": reference_comparison(
            metrics,
            baseline,
        ),
        "development_threshold_sweep": {
            "data_role": "consumed_development",
            "selection_authority": False,
            "purpose": (
                "diagnose a fixed operating point for future "
                "outcome-unseen evaluation"
            ),
            "rows": development_threshold_sweep,
        },
        "events": scored_events,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
