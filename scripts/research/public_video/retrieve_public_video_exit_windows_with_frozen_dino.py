#!/usr/bin/env python3
"""Retrieve licensed public-video exit windows with a frozen DINO direction.

The discovery direction is built from present/clear timestamps in one external
challenge video. It is projected onto fixed-interval frames from registered
videos, then used only to rank sustained high-to-low transitions. No label,
head training, source mask, or production decision is produced.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import run_public_silver_frozen_feature_probe as common
import run_public_silver_risk_lifecycle_mil_head as mil
import sanpo_depth_anything_linear_probe as depth_probe
import scan_public_video_prompt_free_exit_candidates as discovery


SCHEMA = "blindassist_public_video_frozen_dino_exit_retrieval_v1"


def robust_zscores(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("robust z-score needs finite one-dimensional values")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        standard = float(np.std(array))
        scale = standard if standard > 1e-12 else 1.0
    return (array - median) / scale


def prototype_direction(
    positive_vectors: np.ndarray, negative_vectors: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positive = np.asarray(positive_vectors, dtype=np.float64)
    negative = np.asarray(negative_vectors, dtype=np.float64)
    if (
        positive.ndim != 2
        or negative.ndim != 2
        or not len(positive)
        or not len(negative)
        or positive.shape[1] != negative.shape[1]
    ):
        raise ValueError("prototype vectors must be non-empty aligned matrices")
    positive_center = positive.mean(axis=0)
    negative_center = negative.mean(axis=0)
    direction = positive_center - negative_center
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("present and clear prototypes produce a zero direction")
    return direction / norm, positive_center, negative_center


def rank_exit_transitions(
    timestamps_ms: Sequence[int],
    scores: Sequence[float],
    *,
    sample_interval_ms: int,
    prior_samples: int,
    future_samples: int,
    top_k: int,
    minimum_separation_ms: int,
) -> list[dict[str, Any]]:
    timestamps = np.asarray(timestamps_ms, dtype=np.int64)
    values = np.asarray(scores, dtype=np.float64)
    if (
        timestamps.ndim != 1
        or values.ndim != 1
        or len(timestamps) != len(values)
        or prior_samples <= 0
        or future_samples <= 0
        or top_k <= 0
    ):
        raise ValueError("transition ranking arguments are invalid")
    if len(values) < prior_samples + future_samples:
        return []
    zscores = robust_zscores(values)
    proposals: list[dict[str, Any]] = []
    for split in range(prior_samples, len(values) - future_samples + 1):
        prior_timestamps = timestamps[split - prior_samples:split]
        future_timestamps = timestamps[split:split + future_samples]
        combined = np.concatenate([prior_timestamps, future_timestamps])
        if np.any(np.diff(combined) <= 0) or np.any(np.diff(combined) > sample_interval_ms):
            continue
        prior = zscores[split - prior_samples:split]
        future = zscores[split:split + future_samples]
        proposals.append({
            "present_timestamp_ms": int(timestamps[split - 1]),
            "clear_timestamp_ms": int(timestamps[split]),
            "prior_mean_z": float(prior.mean()),
            "future_mean_z": float(future.mean()),
            "sustained_drop_z": float(prior.mean() - future.mean()),
            "future_max_z": float(future.max()),
            "prior_samples": prior_samples,
            "future_samples": future_samples,
        })
    proposals.sort(key=lambda row: (
        -float(row["sustained_drop_z"]),
        int(row["clear_timestamp_ms"]),
    ))
    selected: list[dict[str, Any]] = []
    for proposal in proposals:
        timestamp = int(proposal["clear_timestamp_ms"])
        if all(
            abs(timestamp - int(existing["clear_timestamp_ms"])) >= minimum_separation_ms
            for existing in selected
        ):
            selected.append(proposal)
            if len(selected) >= top_k:
                break
    return selected


def merge_registries(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    sources: list[dict[str, Any]] = []
    attestations: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        mil.reject_independent_direction(path)
        registry = common.load_json(path)
        rows = discovery.validate_registry(registry, path.resolve())
        for source in rows:
            source_id = str(source["source_id"])
            if source_id in seen:
                raise ValueError(f"duplicate source across registries: {source_id}")
            seen.add(source_id)
            sources.append(source)
        attestations.append({
            "path": str(path.resolve()),
            "sha256": common.sha256_file(path),
        })
    if not sources:
        raise ValueError("at least one registered source is required")
    return sources, attestations


def decode_frames(video_path: Path, timestamps_ms: Sequence[int]) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames: list[np.ndarray] = []
    try:
        for timestamp_ms in timestamps_ms:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot decode {video_path} at {timestamp_ms} ms")
            frames.append(frame)
    finally:
        capture.release()
    return frames


def video_timestamps(video_path: Path, sample_interval_ms: int) -> tuple[list[int], dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"video timing metadata is invalid: {video_path}")
    duration_ms = int(round(frame_count / fps * 1000.0))
    return list(range(0, duration_ms, sample_interval_ms)), {
        "fps": fps,
        "frame_count": frame_count,
        "duration_ms": duration_ms,
    }


def pooled_token_vectors(tokens: Any, patch_height: int, patch_width: int) -> np.ndarray:
    array = np.asarray(tokens.detach().cpu().numpy(), dtype=np.float32)
    if array.ndim != 3 or array.shape[1] != patch_height * patch_width:
        raise ValueError(f"unexpected DINO token shape: {tuple(array.shape)}")
    maps = array.reshape(array.shape[0], patch_height, patch_width, array.shape[-1])
    center = maps[:, :, patch_width // 4:max(patch_width // 4 + 1, 3 * patch_width // 4)]
    lower_center = maps[
        :, patch_height // 2:, patch_width // 4:max(patch_width // 4 + 1, 3 * patch_width // 4)
    ]
    return np.concatenate([
        maps.mean(axis=(1, 2)),
        maps.max(axis=(1, 2)),
        center.mean(axis=(1, 2)),
        lower_center.mean(axis=(1, 2)),
    ], axis=1).astype(np.float64)


def extract_vectors(
    model: Any,
    frames: Sequence[np.ndarray],
    *,
    input_size: int,
    layer_index: int,
    batch_size: int,
) -> np.ndarray:
    import torch

    vectors: list[np.ndarray] = []
    tensors: list[Any] = []
    patch_shape: tuple[int, int] | None = None

    def flush() -> None:
        nonlocal tensors
        if not tensors:
            return
        batch = torch.cat(tensors, dim=0)
        outputs = model.pretrained.get_intermediate_layers(
            batch, [layer_index], return_class_token=True
        )
        tokens = outputs[0][0]
        assert patch_shape is not None
        vectors.append(pooled_token_vectors(tokens, *patch_shape))
        tensors = []

    with torch.no_grad():
        for frame in frames:
            tensor, _original_size = model.image2tensor(frame, input_size=input_size)
            shape = (int(tensor.shape[-2] // 14), int(tensor.shape[-1] // 14))
            if patch_shape is None:
                patch_shape = shape
            if shape != patch_shape:
                flush()
                patch_shape = shape
            tensors.append(tensor)
            if len(tensors) >= batch_size:
                flush()
        flush()
    if not vectors:
        raise RuntimeError("no frozen DINO vectors were extracted")
    return np.concatenate(vectors, axis=0)


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.prototype_video, args.src_root, args.checkpoint, args.output):
        mil.reject_independent_direction(path)
    if not args.prototype_video.is_file() or not args.src_root.is_dir() or not args.checkpoint.is_file():
        raise FileNotFoundError("prototype video, DINO source, or checkpoint is missing")
    sources, registry_attestations = merge_registries(args.source_registry)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = depth_probe.depth_anything.load_model(args.src_root, args.checkpoint, "vits")
    model.eval()

    prototype_timestamps = args.positive_ms + args.negative_ms
    prototype_frames = decode_frames(args.prototype_video, prototype_timestamps)
    prototype_vectors = extract_vectors(
        model,
        prototype_frames,
        input_size=args.input_size,
        layer_index=args.layer_index,
        batch_size=args.batch_size,
    )
    direction, positive_center, negative_center = prototype_direction(
        prototype_vectors[:len(args.positive_ms)], prototype_vectors[len(args.positive_ms):]
    )

    source_reports: list[dict[str, Any]] = []
    for source in sources:
        video_path = Path(source["local_video_path"])
        timestamps, metadata = video_timestamps(video_path, args.sample_interval_ms)
        frames = decode_frames(video_path, timestamps)
        vectors = extract_vectors(
            model,
            frames,
            input_size=args.input_size,
            layer_index=args.layer_index,
            batch_size=args.batch_size,
        )
        if len(vectors) != len(timestamps):
            raise RuntimeError(f"feature/timestamp mismatch for {source['source_id']}")
        raw_scores = vectors @ direction
        zscores = robust_zscores(raw_scores)
        top_sample_indices = sorted(
            range(len(timestamps)), key=lambda index: (-float(zscores[index]), timestamps[index])
        )[:args.top_samples_per_source]
        transitions = rank_exit_transitions(
            timestamps,
            raw_scores,
            sample_interval_ms=args.sample_interval_ms,
            prior_samples=args.prior_samples,
            future_samples=args.future_samples,
            top_k=args.top_transitions_per_source,
            minimum_separation_ms=args.minimum_separation_ms,
        )
        source_reports.append({
            **source,
            "video_sha256": common.sha256_file(video_path),
            "video": metadata,
            "sample_count": len(timestamps),
            "score_summary": {
                "minimum": float(raw_scores.min()),
                "median": float(np.median(raw_scores)),
                "maximum": float(raw_scores.max()),
            },
            "top_similarity_samples": [{
                "timestamp_ms": int(timestamps[index]),
                "raw_direction_score": float(raw_scores[index]),
                "within_source_robust_z": float(zscores[index]),
            } for index in top_sample_indices],
            "top_sustained_exit_transitions": transitions,
            "samples": [{
                "timestamp_ms": int(timestamp),
                "raw_direction_score": float(score),
                "within_source_robust_z": float(zscore),
            } for timestamp, score, zscore in zip(timestamps, raw_scores, zscores)],
        })

    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_registries": registry_attestations,
        "prototype": {
            "video_path": str(args.prototype_video),
            "video_sha256": common.sha256_file(args.prototype_video),
            "present_timestamps_ms": args.positive_ms,
            "clear_timestamps_ms": args.negative_ms,
            "positive_center_norm": float(np.linalg.norm(positive_center)),
            "negative_center_norm": float(np.linalg.norm(negative_center)),
            "unit_direction_norm": float(np.linalg.norm(direction)),
        },
        "feature_contract": {
            "model": "Depth Anything V2 frozen DINO-S encoder",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": common.sha256_file(args.checkpoint),
            "layer_index": args.layer_index,
            "input_size": args.input_size,
            "pooling": "global_mean+global_max+center_mean+lower_center_mean",
            "trainable_parameters": 0,
            "sample_interval_ms": args.sample_interval_ms,
            "seed": args.seed,
        },
        "transition_contract": {
            "prior_samples": args.prior_samples,
            "future_samples": args.future_samples,
            "minimum_separation_ms": args.minimum_separation_ms,
            "ranking": "within-source robust-z prior mean minus persistent future mean",
        },
        "sources": source_reports,
        "output": str(args.output),
        "evidence_limit": "Frozen feature retrieval proposes review windows only. Scores and transitions are not event labels, human truth, calibration evidence, blind evidence, production evidence, or permission to train.",
        "training_authorized": False,
        "calibration_authorized": False,
        "blind_evaluation_authorized": False,
        "production_authorized": False,
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(str(args.output) + ".sha256").write_text(common.sha256_file(args.output) + "\n", encoding="ascii")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-registry", type=Path, action="append", required=True)
    parser.add_argument("--prototype-video", type=Path, required=True)
    parser.add_argument("--positive-ms", type=int, action="append", required=True)
    parser.add_argument("--negative-ms", type=int, action="append", required=True)
    parser.add_argument("--src-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-interval-ms", type=int, default=5000)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--layer-index", type=int, choices=range(12), default=11)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prior-samples", type=int, default=2)
    parser.add_argument("--future-samples", type=int, default=3)
    parser.add_argument("--top-samples-per-source", type=int, default=8)
    parser.add_argument("--top-transitions-per-source", type=int, default=6)
    parser.add_argument("--minimum-separation-ms", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()
    for field in ("source_registry",):
        setattr(args, field, [path.resolve() for path in getattr(args, field)])
    for field in ("prototype_video", "src_root", "checkpoint", "output"):
        setattr(args, field, getattr(args, field).resolve())
    positive_fields = (
        args.sample_interval_ms,
        args.input_size,
        args.batch_size,
        args.prior_samples,
        args.future_samples,
        args.top_samples_per_source,
        args.top_transitions_per_source,
        args.minimum_separation_ms,
    )
    if any(value <= 0 for value in positive_fields) or args.input_size % 14:
        parser.error("intervals/counts must be positive and input size must be a multiple of 14")
    if min(args.positive_ms + args.negative_ms) < 0:
        parser.error("prototype timestamps must be non-negative")
    return args


def main() -> int:
    try:
        report = run(parse_args())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({
        "ok": True,
        "source_count": len(report["sources"]),
        "sample_count": sum(source["sample_count"] for source in report["sources"]),
        "output": report["output"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
