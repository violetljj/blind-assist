#!/usr/bin/env python3
"""Train on isolated provisional episodes and transfer to SANPO events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

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
from train_stage_c_d5_tartanground_development_student import (
    MEAN,
    STD,
)


DEFAULT_PROVISIONAL_ROOT = Path(
    "artifacts.local/evidence/"
    "public-video-provisional-training-r7-20260717"
)
DEFAULT_NORMAL_NEGATIVE_ROOT = Path(
    "artifacts.local/evidence/candidate-event-mining/"
    "cem-r0-real-20260802-2hz-yolo-depth-proxy"
)
DEFAULT_MERGED_CANDIDATE_POOL = (
    DEFAULT_NORMAL_NEGATIVE_ROOT / "candidate_pool_merged.json"
)
EXPECTED_EPISODE_COUNT = 16
EXPECTED_SOURCE_COUNT = 9
EXPECTED_FRAME_COUNT = 611
PARALLEL_CURB_EPISODE_COUNT = 17
PARALLEL_CURB_SOURCE_COUNT = 10
PARALLEL_CURB_FRAME_COUNT = 692
MERGED_RELATION_EPISODE_COUNT = 21
MERGED_RELATION_SOURCE_COUNT = 11
MERGED_RELATION_FRAME_COUNT = 771
PUBLIC_VIDEO_EPISODE_COUNT = 42
PUBLIC_VIDEO_SOURCE_COUNT = 18
PUBLIC_VIDEO_FRAME_COUNT = 485
PUBLIC_VIDEO_STEP_MS = 500
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
MERGED_POSITIVE_TYPES = {
    ("front_obstacle_approach",),
    ("static_obstacle_approach",),
    (
        "front_obstacle_approach",
        "static_obstacle_approach",
    ),
}
MERGED_NEGATIVE_TYPES = {
    ("normal_passage_negative",),
    ("parallel_curb",),
}


class MixedRelationFrames(
    Dataset[tuple[torch.Tensor, int, int]]
):
    """Read image evidence or decode deterministic video timestamps."""

    def __init__(self, manifest: dict[str, Any]) -> None:
        self.rows = [
            (event_index, frame_index, frame)
            for event_index, event in enumerate(manifest["events"])
            for frame_index, frame in enumerate(event["frames"])
        ]
        self.captures: dict[str, cv2.VideoCapture] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, int, int]:
        event_index, frame_index, frame = self.rows[index]
        if frame.get("image_path"):
            with Image.open(frame["image_path"]) as image:
                source = image.convert("RGB")
        else:
            video_path = str(frame["video_path"])
            capture = self.captures.get(video_path)
            if capture is None:
                capture = cv2.VideoCapture(video_path)
                if not capture.isOpened():
                    raise OSError(
                        f"Could not open source video: {video_path}"
                    )
                self.captures[video_path] = capture
            timestamp_ms = int(frame["timestamp_ms"])
            if not capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms):
                raise OSError(
                    f"Could not seek source video: {video_path} "
                    f"at {timestamp_ms} ms"
                )
            ok, bgr = capture.read()
            if not ok or bgr is None:
                raise OSError(
                    f"Could not decode source video: {video_path} "
                    f"at {timestamp_ms} ms"
                )
            source = Image.fromarray(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            )
        value = tvf.resize(
            source,
            [128, 224],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = tvf.pil_to_tensor(value).float().div_(255.0)
        tensor = tvf.normalize(tensor, MEAN, STD)
        return tensor, event_index, frame_index

    def close(self) -> None:
        for capture in getattr(self, "captures", {}).values():
            capture.release()
        self.captures.clear()

    def __del__(self) -> None:
        self.close()


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


def collect_merged_relation_sources(
    pool_path: Path,
) -> list[dict[str, Any]]:
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    frames_by_key: dict[
        tuple[str, int],
        dict[str, dict[str, Any]],
    ] = {}
    candidate_ids_by_key: dict[
        tuple[str, int],
        set[str],
    ] = {}
    observed_types_by_key: dict[
        tuple[str, int],
        set[str],
    ] = {}
    confidences_by_key: dict[
        tuple[str, int],
        list[float],
    ] = {}
    review_hashes_by_key: dict[
        tuple[str, int],
        set[str],
    ] = {}
    for candidate in pool["pool"]:
        observed_types = tuple(candidate["luna_reviewed_types"])
        if observed_types in MERGED_POSITIVE_TYPES:
            label = 1
        elif observed_types in MERGED_NEGATIVE_TYPES:
            label = 0
        else:
            continue
        if (
            candidate["candidate_status"] != "luna_reviewed_keep"
            or candidate["pool_authority"]
            != "DISCOVERY_CANDIDATE_ONLY"
            or candidate["truth_status"] != "not_evaluated"
        ):
            raise ValueError(
                "Unexpected merged-candidate authority state"
            )
        key = (str(candidate["source_id"]), label)
        candidate_ids_by_key.setdefault(key, set()).add(
            candidate["candidate_id"]
        )
        observed_types_by_key.setdefault(key, set()).update(
            observed_types
        )
        confidences_by_key.setdefault(key, []).append(
            float(candidate["luna_review_confidence"])
        )
        review_hashes_by_key.setdefault(key, set()).add(
            candidate["luna_review_sha256"]
        )
        frames = frames_by_key.setdefault(key, {})
        candidate_frames = candidate["frame_refs"]
        if label == 1:
            pretrigger_key = (key[0], 0)
            candidate_ids_by_key.setdefault(
                pretrigger_key,
                set(),
            ).add(candidate["candidate_id"])
            observed_types_by_key.setdefault(
                pretrigger_key,
                set(),
            ).add("positive_candidate_pretrigger_context")
            confidences_by_key.setdefault(
                pretrigger_key,
                [],
            ).append(float(candidate["luna_review_confidence"]))
            review_hashes_by_key.setdefault(
                pretrigger_key,
                set(),
            ).add(candidate["luna_review_sha256"])
            pretrigger_frames = frames_by_key.setdefault(
                pretrigger_key,
                {},
            )
            for frame in candidate["frame_refs"]:
                if int(frame["timestamp_ms"]) >= int(
                    candidate["start_timestamp_ms"]
                ):
                    continue
                path = Path(frame["frame_ref"])
                if not path.is_file():
                    raise ValueError(
                        "Missing merged-candidate pretrigger "
                        f"frame: {path}"
                    )
                pretrigger_frames[frame["frame_sha256"]] = {
                    "image_path": str(path.resolve()),
                    "image_sha256": frame["frame_sha256"],
                    "timestamp_ms": int(frame["timestamp_ms"]),
                }
            candidate_frames = [
                frame
                for frame in candidate_frames
                if int(candidate["start_timestamp_ms"])
                <= int(frame["timestamp_ms"])
                <= int(candidate["end_timestamp_ms"])
            ]
            if not candidate_frames:
                candidate_frames = [
                    min(
                        candidate["frame_refs"],
                        key=lambda frame: abs(
                            int(frame["frame_index"])
                            - int(candidate["peak_frame_index"])
                        ),
                    )
                ]
        for frame in candidate_frames:
            path = Path(frame["frame_ref"])
            if not path.is_file():
                raise ValueError(
                    f"Missing merged-candidate frame: {path}"
                )
            row = {
                "image_path": str(path.resolve()),
                "image_sha256": frame["frame_sha256"],
                "timestamp_ms": int(frame["timestamp_ms"]),
            }
            existing = frames.get(frame["frame_sha256"])
            if (
                existing is not None
                and existing["image_path"] != row["image_path"]
            ):
                raise ValueError(
                    "One merged-candidate SHA maps to multiple paths"
                )
            frames[frame["frame_sha256"]] = row

    source_ids = {source_id for source_id, _ in frames_by_key}
    ambiguous_by_source: dict[str, set[str]] = {}
    for source_id in source_ids:
        ambiguous_by_source[source_id] = (
            set(frames_by_key.get((source_id, 0), {}))
            & set(frames_by_key.get((source_id, 1), {}))
        )

    episodes = []
    for key in sorted(frames_by_key):
        source_id, label = key
        ambiguous = ambiguous_by_source[source_id]
        frames = sorted(
            (
                row
                for frame_sha, row in frames_by_key[key].items()
                if frame_sha not in ambiguous
            ),
            key=lambda row: (
                row["timestamp_ms"],
                row["image_sha256"],
            ),
        )
        if not frames:
            raise ValueError(
                f"No unambiguous merged frames for {key}"
            )
        label_name = (
            "candidate_alert"
            if label == 1
            else "candidate_no_alert"
        )
        confidences = confidences_by_key[key]
        episodes.append(
            {
                "episode_id": (
                    "candidate-pool-merged-relation/"
                    f"{source_id}/{label_name}"
                ),
                "source_id": source_id,
                "source_session_id": None,
                "label_name": label_name,
                "label": label,
                "confidence": {
                    "minimum": min(confidences),
                    "mean": float(np.mean(confidences)),
                    "maximum": max(confidences),
                },
                "risk_profile": {
                    "primary_hazard_type": (
                        "obstacle_approach"
                        if label == 1
                        else "none"
                    ),
                    "corridor_relation": (
                        "front_or_static_obstacle_approach"
                        if label == 1
                        else "normal_or_parallel_passage"
                    ),
                    "lifecycle": (
                        "alert" if label == 1 else "no_alert"
                    ),
                    "observed_types": sorted(
                        observed_types_by_key[key]
                    ),
                },
                "frames": frames,
                "candidate_ids": sorted(
                    candidate_ids_by_key[key]
                ),
                "frame_selection": (
                    "trigger_active_interval_or_nearest_peak"
                    if label == 1
                    else (
                        "reviewed_negative_context_and_"
                        "positive_pretrigger_context"
                        if (
                            "positive_candidate_pretrigger_context"
                            in observed_types_by_key[key]
                        )
                        else "reviewed_negative_context"
                    )
                ),
                "excluded_cross_label_frame_count": len(
                    ambiguous
                ),
                "luna_review_sha256": sorted(
                    review_hashes_by_key[key]
                ),
                "candidate_pool_path": str(
                    pool_path.resolve()
                ),
                "candidate_pool_sha256": sha256(pool_path),
            }
        )
    return episodes


def actionability_segments(
    item: dict[str, Any],
) -> list[tuple[int, int, int]]:
    start_ms, end_ms = map(int, item["window_ms"])
    if start_ms >= end_ms:
        raise ValueError(f"Invalid actionability window: {item['item_id']}")
    transitions = sorted(
        item.get("transitions", []),
        key=lambda row: int(row["timestamp_ms"]),
    )
    if not item["intervention_required"]:
        if transitions:
            raise ValueError(
                "Context-only actionability item has transitions: "
                f"{item['item_id']}"
            )
        return [(start_ms, end_ms, 0)]
    if not transitions or not any(
        row["state"] == "intervention_needed"
        for row in transitions
    ):
        raise ValueError(
            f"Positive actionability item lacks intervention: "
            f"{item['item_id']}"
        )
    segments = []
    cursor = start_ms
    label = 0
    for transition in transitions:
        timestamp_ms = int(transition["timestamp_ms"])
        if not start_ms <= timestamp_ms < end_ms:
            raise ValueError(
                f"Actionability transition outside window: "
                f"{item['item_id']}"
            )
        if timestamp_ms > cursor:
            segments.append((cursor, timestamp_ms, label))
        state = transition["state"]
        if state == "intervention_needed":
            label = 1
        elif state == "route_clear":
            label = 0
        else:
            raise ValueError(
                f"Unknown actionability state: {state}"
            )
        cursor = timestamp_ms
    if cursor < end_ms:
        segments.append((cursor, end_ms, label))
    return segments


def collect_public_video_actionability_episodes(
    manifest_path: Path,
    feature_contract_path: Path,
    step_ms: int = PUBLIC_VIDEO_STEP_MS,
) -> list[dict[str, Any]]:
    if step_ms <= 0:
        raise ValueError("Public-video sampling step must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(
        feature_contract_path.read_text(encoding="utf-8")
    )
    sources: dict[str, dict[str, Any]] = {}
    for binding in contract["feature_reports"].values():
        report_path = Path(binding["path"])
        if sha256(report_path) != binding["sha256"]:
            raise ValueError(
                f"Public-video feature report hash mismatch: "
                f"{report_path}"
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for source in report["sources"]:
            source_id = str(source["source_id"])
            incoming = {
                "video_path": str(
                    Path(source["local_video_path"]).resolve()
                ),
                "video_sha256": source["video_sha256"],
            }
            existing = sources.get(source_id)
            report_binding = {
                "path": str(report_path.resolve()),
                "sha256": binding["sha256"],
            }
            if existing is not None:
                if (
                    existing["video_path"] != incoming["video_path"]
                    or existing["video_sha256"]
                    != incoming["video_sha256"]
                ):
                    raise ValueError(
                        "Public-video source metadata mismatch: "
                        f"{source_id}"
                    )
                existing["feature_report_bindings"].append(
                    report_binding
                )
            else:
                sources[source_id] = {
                    **incoming,
                    "feature_report_bindings": [report_binding],
                }

    episodes = []
    for item in manifest["items"]:
        source_id = str(item["parent_source_id"])
        source = sources.get(source_id)
        if source is None:
            raise ValueError(
                f"Missing public-video source binding: {source_id}"
            )
        if not Path(source["video_path"]).is_file():
            raise OSError(
                f"Missing public-video source: {source['video_path']}"
            )
        segments = actionability_segments(item)
        for segment_index, (start_ms, end_ms, label) in enumerate(
            segments
        ):
            timestamps = list(range(start_ms, end_ms, step_ms))
            if not timestamps:
                timestamps = [start_ms]
            label_name = (
                "candidate_alert"
                if label == 1
                else "candidate_no_alert"
            )
            episodes.append(
                {
                    "episode_id": (
                        "public-video-actionability/"
                        f"{item['item_id']}/segment-{segment_index:02d}"
                    ),
                    "source_id": source_id,
                    "source_session_id": None,
                    "label_name": label_name,
                    "label": label,
                    "confidence": None,
                    "risk_profile": {
                        "actionability_class": item[
                            "actionability_class"
                        ],
                        "segment_state": (
                            "intervention_needed"
                            if label == 1
                            else "route_clear_or_context"
                        ),
                    },
                    "frames": [
                        {
                            "video_path": source["video_path"],
                            "video_sha256": source["video_sha256"],
                            "timestamp_ms": timestamp_ms,
                        }
                        for timestamp_ms in timestamps
                    ],
                    "window_ms": [start_ms, end_ms],
                    "frame_selection": (
                        "frozen_actionability_state_segment_at_2hz"
                    ),
                    "label_basis": item["label_basis"],
                    "origin_report": item["origin_report"],
                    "actionability_manifest_path": str(
                        manifest_path.resolve()
                    ),
                    "actionability_manifest_sha256": sha256(
                        manifest_path
                    ),
                    **source,
                }
            )
    return episodes


def infer_profile_matrices(
    model: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    batch_size: int,
    dataset: Dataset | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    if dataset is None:
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
    dataset: Dataset | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    if feature_family == "direction_profiles":
        return infer_profile_matrices(
            model,
            manifest_path,
            manifest,
            batch_size,
            dataset,
        )
    if feature_family == "spatial_grid_3x6":
        if dataset is None:
            dataset = ManifestFrames(manifest_path, manifest)
        return infer_spatial_matrices(
            model,
            dataset,
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


def source_heldout_training_diagnostic(
    x: np.ndarray,
    y: np.ndarray,
    sources: np.ndarray,
    episode_ids: np.ndarray,
    l2_strength: float,
) -> dict[str, Any]:
    probabilities = np.full(len(y), np.nan, dtype=np.float64)
    folds = []
    for held_out_source in sorted(set(sources.tolist())):
        train = np.flatnonzero(sources != held_out_source)
        test = np.flatnonzero(sources == held_out_source)
        train_classes = sorted(set(y[train].tolist()))
        if train_classes != [0, 1]:
            folds.append(
                {
                    "held_out_source_id": held_out_source,
                    "train_class_count": len(train_classes),
                    "test_frame_count": len(test),
                    "evaluated": False,
                }
            )
            continue
        weights = event_balanced_weights(
            episode_ids[train].tolist(),
            y[train],
        )
        mean, scale = weighted_standardize(x[train], weights)
        coefficient, intercept, _ = fit_logistic(
            (x[train] - mean) / scale,
            y[train],
            weights,
            l2_strength=l2_strength,
        )
        logits = (
            ((x[test] - mean) / scale) @ coefficient
            + intercept
        )
        probabilities[test] = 1.0 / (
            1.0 + np.exp(-np.clip(logits, -50.0, 50.0))
        )
        folds.append(
            {
                "held_out_source_id": held_out_source,
                "train_class_count": len(train_classes),
                "test_frame_count": len(test),
                "evaluated": True,
            }
        )
    evaluated = np.isfinite(probabilities)
    if not evaluated.all():
        return {
            "complete": False,
            "folds": folds,
            "unevaluated_frame_count": int((~evaluated).sum()),
        }
    frame_predictions = (probabilities >= 0.5).astype(np.int64)
    frame_alert_recall = float(
        (frame_predictions[y == 1] == 1).mean()
    )
    frame_no_alert_recall = float(
        (frame_predictions[y == 0] == 0).mean()
    )
    unique_episode_ids = sorted(set(episode_ids.tolist()))
    episode_labels = []
    episode_predictions = []
    episode_rows = []
    for episode_id in unique_episode_ids:
        indices = np.flatnonzero(episode_ids == episode_id)
        labels = sorted(set(y[indices].tolist()))
        if len(labels) != 1:
            raise ValueError(
                f"Training episode has mixed labels: {episode_id}"
            )
        score = float(np.mean(probabilities[indices]))
        label = labels[0]
        prediction = int(score >= 0.5)
        episode_labels.append(label)
        episode_predictions.append(prediction)
        episode_rows.append(
            {
                "episode_id": episode_id,
                "source_id": str(sources[indices[0]]),
                "label": label,
                "score": score,
                "prediction": prediction,
            }
        )
    episode_labels_array = np.asarray(
        episode_labels,
        dtype=np.int64,
    )
    episode_predictions_array = np.asarray(
        episode_predictions,
        dtype=np.int64,
    )
    episode_alert_recall = float(
        (
            episode_predictions_array[episode_labels_array == 1]
            == 1
        ).mean()
    )
    episode_no_alert_recall = float(
        (
            episode_predictions_array[episode_labels_array == 0]
            == 0
        ).mean()
    )
    return {
        "complete": True,
        "split_unit": "source_id",
        "threshold": 0.5,
        "frame_count": len(y),
        "episode_count": len(unique_episode_ids),
        "source_count": len(set(sources.tolist())),
        "frame_alert_recall": frame_alert_recall,
        "frame_no_alert_recall": frame_no_alert_recall,
        "frame_balanced_accuracy": (
            frame_alert_recall + frame_no_alert_recall
        )
        / 2.0,
        "episode_alert_recall": episode_alert_recall,
        "episode_no_alert_recall": episode_no_alert_recall,
        "episode_balanced_accuracy": (
            episode_alert_recall + episode_no_alert_recall
        )
        / 2.0,
        "folds": folds,
        "episodes": episode_rows,
    }


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
        "--merged-candidate-pool",
        type=Path,
    )
    parser.add_argument(
        "--public-video-actionability-manifest",
        type=Path,
    )
    parser.add_argument(
        "--public-video-feature-contract",
        type=Path,
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
    public_video_relation = (
        args.public_video_actionability_manifest is not None
    )
    if public_video_relation != (
        args.public_video_feature_contract is not None
    ):
        raise ValueError(
            "Public-video manifest and feature contract are required "
            "together"
        )
    if (
        sum(
            (
                args.merged_candidate_pool is not None,
                args.include_quarantined_parallel_curb,
                public_video_relation,
            )
        )
        > 1
    ):
        raise ValueError(
            "Merged pool, quarantine canary, and public-video relation "
            "modes are exclusive"
        )
    if public_video_relation:
        episodes.extend(
            collect_public_video_actionability_episodes(
                args.public_video_actionability_manifest,
                args.public_video_feature_contract,
            )
        )
    elif args.merged_candidate_pool is not None:
        episodes.extend(
            collect_merged_relation_sources(
                args.merged_candidate_pool
            )
        )
    else:
        episodes.extend(
            collect_reviewed_negative_sources(
                args.normal_negative_root,
                args.include_quarantined_parallel_curb,
            )
        )
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
            PUBLIC_VIDEO_EPISODE_COUNT,
            PUBLIC_VIDEO_SOURCE_COUNT,
            PUBLIC_VIDEO_FRAME_COUNT,
            {"candidate_no_alert": 27, "candidate_alert": 15},
        )
        if public_video_relation
        else
        (
            MERGED_RELATION_EPISODE_COUNT,
            MERGED_RELATION_SOURCE_COUNT,
            MERGED_RELATION_FRAME_COUNT,
            {"candidate_no_alert": 10, "candidate_alert": 11},
        )
        if args.merged_candidate_pool is not None
        else
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
    training_dataset = (
        MixedRelationFrames(training_manifest)
        if public_video_relation
        else None
    )
    training_matrices, feature_names = infer_feature_matrices(
        model,
        Path("provisional-training-manifest.json"),
        training_manifest,
        args.batch_size,
        args.feature_family,
        training_dataset,
    )
    if training_dataset is not None:
        training_dataset.close()
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
    training_source_ids = np.asarray(
        [
            episode["source_id"]
            for episode, matrix in zip(
                episodes,
                training_matrices,
                strict=True,
            )
            for _ in range(len(matrix))
        ],
        dtype=str,
    )
    training_episode_ids_array = np.asarray(
        training_episode_ids,
        dtype=str,
    )
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
    if public_video_relation:
        public_video_rows = np.char.startswith(
            training_episode_ids_array,
            "public-video-actionability/",
        )
        training_source_heldout = (
            source_heldout_training_diagnostic(
                x_train[public_video_rows],
                y_train[public_video_rows],
                training_source_ids[public_video_rows],
                training_episode_ids_array[public_video_rows],
                l2_strength,
            )
        )
    else:
        training_source_heldout = None

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
    merged_relation = args.merged_candidate_pool is not None
    result = {
        "schema": (
            "blindassist_hftf_stage_c_d6_provisional_relation_"
            + (
                "cross_source_spatial_merged_relation_v0"
                if spatial and merged_relation
                else
                "cross_source_spatial_public_video_actionability_v1"
                if spatial and public_video_relation
                else
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
                "SPATIAL_MERGED_RELATION_COMPLETE"
                if spatial and merged_relation
                else
                "SPATIAL_PUBLIC_VIDEO_ACTIONABILITY_COMPLETE"
                if spatial and public_video_relation
                else
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
            "normal_passage_negatives_reviewed": (
                not public_video_relation
            ),
            "quarantine_candidates_used": weak_parallel_curb,
            "quarantined_parallel_curb_canary": (
                weak_parallel_curb
            ),
            "merged_relation_candidate_pool_used": (
                merged_relation
            ),
            "public_video_actionability_relation_used": (
                public_video_relation
            ),
            "public_video_state_transition_labels": (
                public_video_relation
            ),
            "public_video_sampling_step_ms": (
                PUBLIC_VIDEO_STEP_MS
                if public_video_relation
                else None
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
            "merged_candidate_pool": (
                str(args.merged_candidate_pool.resolve())
                if args.merged_candidate_pool is not None
                else None
            ),
            "public_video_actionability_manifest": (
                str(
                    args.public_video_actionability_manifest.resolve()
                )
                if public_video_relation
                else None
            ),
            "public_video_actionability_manifest_sha256": (
                sha256(args.public_video_actionability_manifest)
                if public_video_relation
                else None
            ),
            "public_video_feature_contract": (
                str(args.public_video_feature_contract.resolve())
                if public_video_relation
                else None
            ),
            "public_video_feature_contract_sha256": (
                sha256(args.public_video_feature_contract)
                if public_video_relation
                else None
            ),
            "episode_count": len(episodes),
            "source_count": len(source_ids),
            "frame_count": frame_count,
            "label_counts": label_counts,
            "weighted_regularized_train_loss": loss,
            "source_heldout_diagnostic": (
                training_source_heldout
            ),
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
