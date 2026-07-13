#!/usr/bin/env python3
"""Train a gated SANPO candidate with a step-budgeted, multi-seed protocol.

The trainer consumes only the attested train/dev manifest.  It never opens the
benchmark-only blind holdout.  Optimizer steps, not epochs, define the compute
budget so batch-size or dataset-size changes remain comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

import sanpo_training_gate as training_gate
import train_export_sanpo_segmentation as shared


DEFAULT_WEIGHTS = "test-artifacts.local/segmentation-candidate/torch/mobilenetv3_lraspp.weights.h5"
DEFAULT_REPORT = "test-artifacts.local/segmentation-candidate/torch/training_report.json"
DEFAULT_SEEDS = (20260711, 20260712, 20260713)
ALLOWED_INPUT_SIZES = (256, 384, 512)
BOUNDARY_CLASS_ID = shared.CLASS_IDS["boundary_step_curb"]
OBSTACLE_CLASS_ID = shared.CLASS_IDS["obstacle"]
HEAD_LAYER_PREFIXES = ("lraspp_", "semantic_logits")


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_seed_list(value: str) -> tuple[int, ...]:
    seeds: list[int] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        seed = int(stripped)
        if seed < 0:
            raise ValueError("seeds must be non-negative integers")
        if seed not in seeds:
            seeds.append(seed)
    if not seeds:
        raise ValueError("at least one seed is required")
    return tuple(seeds)


def parse_seed_pairs(value: str) -> tuple[tuple[int, int], ...]:
    """Parse model:sampler seed pairs used by the P0 variance audit."""
    pairs: list[tuple[int, int]] = []
    for item in value.split(","):
        stripped = item.strip()
        if not stripped:
            continue
        parts = stripped.split(":")
        if len(parts) != 2:
            raise ValueError("seed pairs must use model_seed:sampler_seed")
        pair = (int(parts[0]), int(parts[1]))
        if min(pair) < 0:
            raise ValueError("seed pairs must be non-negative integers")
        if pair not in pairs:
            pairs.append(pair)
    if not pairs:
        raise ValueError("at least one seed pair is required")
    return tuple(pairs)


def class_loss_weights(pixel_counts: dict[str, int], maximum: float = 4.0) -> np.ndarray:
    counts = np.asarray([pixel_counts[name] for name in shared.CLASS_NAMES], dtype=np.float64)
    frequencies = counts / max(1.0, counts.sum())
    raw = 1.0 / np.sqrt(np.maximum(frequencies, 1e-12))
    normalized = raw / max(1e-12, raw.mean())
    return np.clip(normalized, 0.35, maximum).astype(np.float32)


def selection_score(metrics: dict[str, Any]) -> float:
    """Harmonic mIoU/boundary score: neither metric can hide collapse in the other."""
    mean_iou = float(metrics["mean_iou"])
    boundary_iou = float(metrics["per_class"]["boundary_step_curb"]["iou"])
    if mean_iou <= 0.0 or boundary_iou <= 0.0:
        return 0.0
    return float(2.0 * mean_iou * boundary_iou / (mean_iou + boundary_iou))


def checkpoint_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        selection_score(metrics),
        float(metrics["per_class"]["boundary_step_curb"]["iou"]),
        float(metrics["mean_iou"]),
    )


def aggregate_seed_metrics(seed_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not seed_runs:
        raise ValueError("seed_runs must not be empty")
    fields = {
        "selection_score": [float(run["selection_score"]) for run in seed_runs],
        "mean_iou": [float(run["dev_mask_metrics"]["mean_iou"]) for run in seed_runs],
        "boundary_iou": [
            float(run["dev_mask_metrics"]["per_class"]["boundary_step_curb"]["iou"])
            for run in seed_runs
        ],
        "pixel_accuracy": [float(run["dev_mask_metrics"]["pixel_accuracy"]) for run in seed_runs],
        "unknown_iou": [
            float(run["dev_mask_metrics"]["per_class"]["unknown_nonwalkable"]["iou"])
            for run in seed_runs
        ],
    }
    summary: dict[str, Any] = {"seed_count": len(seed_runs)}
    for name, values in fields.items():
        array = np.asarray(values, dtype=np.float64)
        summary[name] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "minimum": float(array.min()),
            "maximum": float(array.max()),
            "values": values,
        }
    return summary


def factor_variation_summary(seed_runs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Summarize one-factor-at-a-time P0 seed variation without claiming causality."""
    metric_getters = {
        "selection_score": lambda run: float(run["selection_score"]),
        "mean_iou": lambda run: float(run["dev_mask_metrics"]["mean_iou"]),
        "boundary_iou": lambda run: float(
            run["dev_mask_metrics"]["per_class"]["boundary_step_curb"]["iou"]
        ),
        "unknown_iou": lambda run: float(
            run["dev_mask_metrics"]["per_class"]["unknown_nonwalkable"]["iou"]
        ),
    }

    def grouped(fixed_field: str, varying_field: str) -> list[dict[str, Any]]:
        buckets: dict[int, list[dict[str, Any]]] = {}
        for run in seed_runs:
            buckets.setdefault(int(run[fixed_field]), []).append(run)
        result: list[dict[str, Any]] = []
        for fixed_seed, runs in sorted(buckets.items()):
            varying_seeds = {int(run[varying_field]) for run in runs}
            if len(varying_seeds) < 2:
                continue
            metrics: dict[str, Any] = {}
            for name, getter in metric_getters.items():
                values = np.asarray([getter(run) for run in runs], dtype=np.float64)
                metrics[name] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "range": float(values.max() - values.min()),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            result.append({
                fixed_field: fixed_seed,
                f"{varying_field}s": sorted(varying_seeds),
                "run_count": len(runs),
                "metrics": metrics,
            })
        return result

    return {
        "model_initialization_effect_with_sampler_fixed": grouped("sampler_seed", "model_seed"),
        "sampler_effect_with_model_initialization_fixed": grouped("model_seed", "sampler_seed"),
        "interpretation_boundary": "descriptive_one_factor_ranges_not_causal_significance",
    }


def seeded_weight_path(base: Path, seed: int) -> Path:
    suffix = ".weights.h5"
    if not base.name.endswith(suffix):
        raise ValueError("Keras weight output must end with .weights.h5")
    return base.with_name(f"{base.name[:-len(suffix)]}.seed-{seed}{suffix}")


def stage_weight_path(base: Path, seed: int, stage: str) -> Path:
    suffix = ".weights.h5"
    seeded = seeded_weight_path(base, seed)
    safe_stage = stage.replace("_", "-")
    return seeded.with_name(f"{seeded.name[:-len(suffix)]}.stage-{safe_stage}{suffix}")


def seed_pair_weight_path(base: Path, model_seed: int, sampler_seed: int) -> Path:
    if model_seed == sampler_seed:
        return seeded_weight_path(base, model_seed)
    suffix = ".weights.h5"
    if not base.name.endswith(suffix):
        raise ValueError("Keras weight output must end with .weights.h5")
    stem = base.name[:-len(suffix)]
    return base.with_name(
        f"{stem}.model-seed-{model_seed}.sampler-seed-{sampler_seed}{suffix}"
    )


def seed_pair_stage_weight_path(
    base: Path, model_seed: int, sampler_seed: int, stage: str,
) -> Path:
    suffix = ".weights.h5"
    paired = seed_pair_weight_path(base, model_seed, sampler_seed)
    safe_stage = stage.replace("_", "-")
    return paired.with_name(f"{paired.name[:-len(suffix)]}.stage-{safe_stage}{suffix}")


def cosine_decay_value(initial: float, final_ratio: float, step: int, decay_steps: int) -> float:
    progress = min(max(step, 0), max(1, decay_steps)) / max(1, decay_steps)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(initial * (final_ratio + (1.0 - final_ratio) * cosine))


def configure_trainable_layers(
    model: Any,
    keras: Any,
    *,
    backbone_trainable: bool,
    keep_batchnorm_frozen: bool,
) -> dict[str, Any]:
    """Freeze/unfreeze the backbone while always leaving LR-ASPP trainable."""
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    frozen_batchnorm: list[str] = []
    for layer in model.layers:
        is_head = layer.name.startswith(HEAD_LAYER_PREFIXES)
        is_batchnorm = isinstance(layer, keras.layers.BatchNormalization)
        if is_head:
            layer.trainable = True
        elif backbone_trainable and not (keep_batchnorm_frozen and is_batchnorm):
            layer.trainable = True
        else:
            layer.trainable = False
            if is_batchnorm:
                frozen_batchnorm.append(layer.name)
        (trainable_names if layer.trainable else frozen_names).append(layer.name)
    return {
        "backbone_trainable": backbone_trainable,
        "keep_batchnorm_frozen": keep_batchnorm_frozen,
        "trainable_layer_count": len(trainable_names),
        "frozen_layer_count": len(frozen_names),
        "frozen_batchnorm_count": len(frozen_batchnorm),
        "trainable_head_layers": [name for name in trainable_names if name.startswith(HEAD_LAYER_PREFIXES)],
    }


class SessionBalancedCropSampler:
    """Uniformly draw sessions and mix full frames with rare-class guided crops."""

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        records: Sequence[shared.Record],
        *,
        batch_size: int,
        seed: int,
        guided_crop_fraction: float,
        crop_min_fraction: float,
        crop_max_fraction: float,
        horizontal_flip_probability: float,
        boundary_guided_probability: float = 0.65,
    ) -> None:
        if len(images) != len(masks) or len(images) != len(records) or not len(records):
            raise ValueError("images, masks and records must have the same non-zero length")
        if not 0.0 <= guided_crop_fraction <= 1.0:
            raise ValueError("guided_crop_fraction must be in 0..1")
        if not 0.0 < crop_min_fraction <= crop_max_fraction <= 1.0:
            raise ValueError("crop fractions must satisfy 0 < min <= max <= 1")
        if not 0.0 <= horizontal_flip_probability <= 1.0:
            raise ValueError("horizontal_flip_probability must be in 0..1")
        if not 0.0 <= boundary_guided_probability <= 1.0:
            raise ValueError("boundary_guided_probability must be in 0..1")
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.guided_crop_fraction = guided_crop_fraction
        self.crop_min_fraction = crop_min_fraction
        self.crop_max_fraction = crop_max_fraction
        self.horizontal_flip_probability = horizontal_flip_probability
        self.boundary_guided_probability = boundary_guided_probability
        self.rng = np.random.default_rng(seed)
        self.sessions: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            self.sessions.setdefault(record.session_id, []).append(index)
        self.session_ids = tuple(sorted(self.sessions))
        self.candidates: dict[str, dict[int, list[int]]] = {}
        for session_id, indices in self.sessions.items():
            self.candidates[session_id] = {
                class_id: [index for index in indices if np.any(masks[index] == class_id)]
                for class_id in (BOUNDARY_CLASS_ID, OBSTACLE_CLASS_ID)
            }
        self.session_draws = {session_id: 0 for session_id in self.session_ids}
        self.guided_crop_attempts = 0
        self.guided_crop_hits = {"boundary_step_curb": 0, "obstacle": 0}

    def _crop(self, image: np.ndarray, mask: np.ndarray, class_id: int) -> tuple[np.ndarray, np.ndarray]:
        points = np.argwhere(mask == class_id)
        if not len(points):
            return image, mask
        target_y, target_x = points[int(self.rng.integers(len(points)))]
        height, width = mask.shape
        fraction = float(self.rng.uniform(self.crop_min_fraction, self.crop_max_fraction))
        crop_h = max(2, min(height, int(round(height * fraction))))
        crop_w = max(2, min(width, int(round(width * fraction))))
        left = int(np.clip(target_x - self.rng.integers(max(1, crop_w)), 0, width - crop_w))
        top = int(np.clip(target_y - self.rng.integers(max(1, crop_h)), 0, height - crop_h))
        box = (left, top, left + crop_w, top + crop_h)
        resized_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").crop(box).resize(
            (width, height), Image.Resampling.BILINEAR,
        )
        resized_mask = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").crop(box).resize(
            (width, height), Image.Resampling.NEAREST,
        )
        return np.asarray(resized_image, dtype=np.float32), np.asarray(resized_mask, dtype=np.int64)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        batch_images: list[np.ndarray] = []
        batch_masks: list[np.ndarray] = []
        for _ in range(self.batch_size):
            session_id = self.session_ids[int(self.rng.integers(len(self.session_ids)))]
            self.session_draws[session_id] += 1
            indices = self.sessions[session_id]
            class_id: int | None = None
            if self.rng.random() < self.guided_crop_fraction:
                self.guided_crop_attempts += 1
                class_id = (
                    BOUNDARY_CLASS_ID
                    if self.rng.random() < self.boundary_guided_probability
                    else OBSTACLE_CLASS_ID
                )
                class_candidates = self.candidates[session_id][class_id]
                if class_candidates:
                    indices = class_candidates
            index = indices[int(self.rng.integers(len(indices)))]
            image = self.images[index]
            mask = self.masks[index]
            if class_id is not None and np.any(mask == class_id):
                image, mask = self._crop(image, mask, class_id)
                self.guided_crop_hits[shared.CLASS_NAMES[class_id]] += 1
            else:
                image = image.copy()
                mask = mask.copy()
            if self.rng.random() < self.horizontal_flip_probability:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            batch_images.append(image)
            batch_masks.append(mask)
        return np.stack(batch_images).astype(np.float32), np.stack(batch_masks).astype(np.int64)

    def report(self) -> dict[str, Any]:
        return {
            "session_strategy": "uniform_session_then_uniform_frame",
            "session_draws": dict(self.session_draws),
            "guided_crop_fraction": self.guided_crop_fraction,
            "guided_target_probabilities": {
                "boundary_step_curb": self.boundary_guided_probability,
                "obstacle": 1.0 - self.boundary_guided_probability,
            },
            "guided_crop_attempts": self.guided_crop_attempts,
            "guided_crop_hits": dict(self.guided_crop_hits),
            "crop_fraction_range": [self.crop_min_fraction, self.crop_max_fraction],
            "horizontal_flip_probability": self.horizontal_flip_probability,
        }


QUOTA_NAMES = ("boundary", "obstacle", "hard_negative", "unknown_full_frame")


class DeterministicQuotaSampler(SessionBalancedCropSampler):
    """Cycle four sampling intents exactly, independent of optimization batch size."""

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        records: Sequence[shared.Record],
        *,
        batch_size: int,
        seed: int,
        crop_min_fraction: float,
        crop_max_fraction: float,
        horizontal_flip_probability: float,
        unknown_rich_quantile: float = 0.75,
        source_mask_stats: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        if not 0.0 <= unknown_rich_quantile < 1.0:
            raise ValueError("unknown_rich_quantile must be in [0, 1)")
        super().__init__(
            images,
            masks,
            records,
            batch_size=batch_size,
            seed=seed,
            guided_crop_fraction=0.0,
            crop_min_fraction=crop_min_fraction,
            crop_max_fraction=crop_max_fraction,
            horizontal_flip_probability=horizontal_flip_probability,
            boundary_guided_probability=0.5,
        )
        self.seed = seed
        self.unknown_rich_quantile = unknown_rich_quantile
        resized_presence = np.asarray([
            [bool(np.any(mask == class_id)) for class_id in range(len(shared.CLASS_NAMES))]
            for mask in masks
        ], dtype=bool)
        if source_mask_stats is None:
            source_presence = resized_presence
            unknown_shares = np.mean(
                masks == shared.CLASS_IDS["unknown_nonwalkable"], axis=(1, 2)
            )
            stats_source = "resized_training_masks"
        else:
            if len(source_mask_stats) != len(records):
                raise ValueError("source_mask_stats must align with records")
            source_presence = np.asarray(
                [stats["class_presence"] for stats in source_mask_stats], dtype=bool,
            )
            unknown_shares = np.asarray(
                [float(stats["unknown_share"]) for stats in source_mask_stats], dtype=np.float64,
            )
            stats_source = "canonical_source_resolution_masks"
        if not np.any(unknown_shares > 0.0):
            raise ValueError("deterministic quota has no candidates: unknown_full_frame")
        self.unknown_rich_threshold_by_session: dict[str, float] = {}
        unknown_rich_indices: list[int] = []
        for session_id, indices in self.sessions.items():
            values = unknown_shares[indices]
            threshold = float(np.quantile(values, unknown_rich_quantile))
            self.unknown_rich_threshold_by_session[session_id] = threshold
            unknown_rich_indices.extend(
                index for index in indices
                if float(unknown_shares[index]) > 0.0 and float(unknown_shares[index]) >= threshold
            )
        self.stats_source = stats_source
        self.resized_boundary_loss_count = int(np.sum(
            source_presence[:, BOUNDARY_CLASS_ID] & ~resized_presence[:, BOUNDARY_CLASS_ID]
        ))
        quota_indices = {
            "boundary": [
                index for index in range(len(masks))
                if source_presence[index, BOUNDARY_CLASS_ID]
                and resized_presence[index, BOUNDARY_CLASS_ID]
            ],
            "obstacle": [
                index for index in range(len(masks))
                if source_presence[index, OBSTACLE_CLASS_ID]
                and resized_presence[index, OBSTACLE_CLASS_ID]
            ],
            "hard_negative": [
                index for index in range(len(masks))
                if not source_presence[index, BOUNDARY_CLASS_ID]
            ],
            "unknown_full_frame": unknown_rich_indices,
        }
        self.quota_by_session: dict[str, dict[str, tuple[int, ...]]] = {}
        self.quota_sessions: dict[str, tuple[str, ...]] = {}
        self.frame_cursors: dict[tuple[str, str], int] = {}
        for quota, indices in quota_indices.items():
            allowed = set(indices)
            by_session: dict[str, tuple[int, ...]] = {}
            for session_id in sorted(self.sessions):
                values = [index for index in self.sessions[session_id] if index in allowed]
                if values:
                    by_session[session_id] = tuple(int(index) for index in self.rng.permutation(values))
            if not by_session:
                raise ValueError(f"deterministic quota has no candidates: {quota}")
            ordered_sessions = tuple(str(value) for value in self.rng.permutation(sorted(by_session)))
            if quota == "hard_negative":
                weighted = [
                    session_id for session_id, values in by_session.items() for _ in range(len(values))
                ]
                ordered_sessions = tuple(str(value) for value in self.rng.permutation(weighted))
            self.quota_by_session[quota] = by_session
            self.quota_sessions[quota] = ordered_sessions
            for session_id in ordered_sessions:
                self.frame_cursors[(quota, session_id)] = 0
        self.quota_candidate_counts = {name: len(values) for name, values in quota_indices.items()}
        self.quota_session_strategy = {
            "boundary": "uniform_eligible_session_round_robin",
            "obstacle": "uniform_eligible_session_round_robin",
            "hard_negative": "candidate_count_weighted_session_round_robin",
            "unknown_full_frame": "uniform_eligible_session_round_robin",
        }
        self.quota_draws = {name: 0 for name in QUOTA_NAMES}
        self.quota_session_draws = {
            name: {session_id: 0 for session_id in self.quota_sessions[name]}
            for name in QUOTA_NAMES
        }
        self.full_frame_draws = {"hard_negative": 0, "unknown_full_frame": 0}
        self.sample_cursor = 0
        self.sample_trace_hasher = hashlib.sha256()
        self.quota_class_presence = {
            quota: {name: 0 for name in shared.CLASS_NAMES} for quota in QUOTA_NAMES
        }

    def _next_index(self, quota: str) -> tuple[int, str]:
        draw = self.quota_draws[quota]
        sessions = self.quota_sessions[quota]
        session_id = sessions[draw % len(sessions)]
        values = self.quota_by_session[quota][session_id]
        cursor_key = (quota, session_id)
        cursor = self.frame_cursors[cursor_key]
        index = values[cursor % len(values)]
        self.frame_cursors[cursor_key] = cursor + 1
        return index, session_id

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        batch_images: list[np.ndarray] = []
        batch_masks: list[np.ndarray] = []
        for _ in range(self.batch_size):
            quota = QUOTA_NAMES[self.sample_cursor % len(QUOTA_NAMES)]
            self.sample_cursor += 1
            index, session_id = self._next_index(quota)
            self.quota_draws[quota] += 1
            self.quota_session_draws[quota][session_id] += 1
            self.session_draws[session_id] += 1
            self.sample_trace_hasher.update(
                f"{self.sample_cursor}:{quota}:{session_id}:{index};".encode("utf-8")
            )
            image = self.images[index]
            mask = self.masks[index]
            for class_id, class_name in enumerate(shared.CLASS_NAMES):
                if np.any(mask == class_id):
                    self.quota_class_presence[quota][class_name] += 1
            if quota == "boundary":
                image, mask = self._crop(image, mask, BOUNDARY_CLASS_ID)
                if not np.any(mask == BOUNDARY_CLASS_ID):
                    raise RuntimeError("boundary quota crop lost its target class")
                self.guided_crop_hits[shared.CLASS_NAMES[BOUNDARY_CLASS_ID]] += 1
            elif quota == "obstacle":
                image, mask = self._crop(image, mask, OBSTACLE_CLASS_ID)
                if not np.any(mask == OBSTACLE_CLASS_ID):
                    raise RuntimeError("obstacle quota crop lost its target class")
                self.guided_crop_hits[shared.CLASS_NAMES[OBSTACLE_CLASS_ID]] += 1
            else:
                image = image.copy()
                mask = mask.copy()
                if quota == "hard_negative" and np.any(mask == BOUNDARY_CLASS_ID):
                    raise RuntimeError("hard-negative quota selected a boundary-positive frame")
                self.full_frame_draws[quota] += 1
            if self.rng.random() < self.horizontal_flip_probability:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            batch_images.append(image)
            batch_masks.append(mask)
        return np.stack(batch_images).astype(np.float32), np.stack(batch_masks).astype(np.int64)

    def report(self) -> dict[str, Any]:
        total = max(1, sum(self.quota_draws.values()))
        return {
            "sampler_strategy": "deterministic_quota_round_robin",
            "seed": self.seed,
            "schedule": list(QUOTA_NAMES),
            "schedule_period_samples": len(QUOTA_NAMES),
            "quota_targets": {name: 0.25 for name in QUOTA_NAMES},
            "quota_draws": dict(self.quota_draws),
            "quota_realized_share": {
                name: float(count / total) for name, count in self.quota_draws.items()
            },
            "quota_candidate_counts": dict(self.quota_candidate_counts),
            "quota_candidate_session_counts": {
                name: len(self.quota_by_session[name]) for name in QUOTA_NAMES
            },
            "quota_session_draws": {
                name: dict(values) for name, values in self.quota_session_draws.items()
            },
            "unknown_rich_quantile": self.unknown_rich_quantile,
            "unknown_rich_threshold_by_session": dict(self.unknown_rich_threshold_by_session),
            "sampling_stats_source": self.stats_source,
            "resized_boundary_loss_count": self.resized_boundary_loss_count,
            "quota_session_strategy": dict(self.quota_session_strategy),
            "quota_repeat_factor": {
                name: float(self.quota_draws[name] / max(1, self.quota_candidate_counts[name]))
                for name in QUOTA_NAMES
            },
            "session_draws": dict(self.session_draws),
            "guided_crop_hits": dict(self.guided_crop_hits),
            "full_frame_draws": dict(self.full_frame_draws),
            "crop_fraction_range": [self.crop_min_fraction, self.crop_max_fraction],
            "horizontal_flip_probability": self.horizontal_flip_probability,
            "completed_schedule_cycles": self.sample_cursor // len(QUOTA_NAMES),
            "schedule_cycle_complete": self.sample_cursor % len(QUOTA_NAMES) == 0,
            "maximum_quota_draw_imbalance": max(self.quota_draws.values()) - min(self.quota_draws.values()),
            "sample_trace_sha256": self.sample_trace_hasher.hexdigest(),
            "quota_class_presence": {
                name: dict(values) for name, values in self.quota_class_presence.items()
            },
        }


def build_composite_loss(keras: Any, class_weights: np.ndarray, args: argparse.Namespace) -> Any:
    weights_tensor = keras.ops.convert_to_tensor(class_weights)

    class CompositeSegmentationLoss(keras.losses.Loss):
        def call(self, y_true: Any, y_pred: Any) -> Any:
            labels = keras.ops.cast(y_true, "int32")
            logits = keras.ops.cast(y_pred, "float32")
            one_hot = keras.ops.one_hot(labels, len(shared.CLASS_NAMES))
            probabilities = keras.ops.softmax(logits, axis=-1)
            pixel_weights = keras.ops.take(weights_tensor, labels)
            crossentropy = keras.ops.sparse_categorical_crossentropy(labels, logits, from_logits=True)
            true_probability = keras.ops.sum(probabilities * one_hot, axis=-1)
            focal = -keras.ops.power(1.0 - true_probability, args.focal_gamma) * keras.ops.log(
                keras.ops.maximum(true_probability, 1e-7)
            )
            axes = (0, 1, 2)
            intersection = keras.ops.sum(probabilities * one_hot, axis=axes)
            denominator = keras.ops.sum(probabilities + one_hot, axis=axes)
            dice_per_class = (2.0 * intersection + 1.0) / (denominator + 1.0)
            normalized_weights = weights_tensor / keras.ops.sum(weights_tensor)
            dice_loss = 1.0 - keras.ops.sum(dice_per_class * normalized_weights)
            return (
                args.ce_weight * crossentropy * pixel_weights
                + args.focal_weight * focal * pixel_weights
                + args.dice_weight * dice_loss
            )

    return CompositeSegmentationLoss(name="weighted_ce_dice_focal")


def set_seed(keras: Any, torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    keras.utils.set_random_seed(seed)


def evaluate_arrays(model: Any, images: np.ndarray, masks: np.ndarray, batch_size: int) -> dict[str, Any]:
    logits = model.predict(images, batch_size=batch_size, verbose=0)
    return shared.confusion_and_metrics(np.argmax(logits, axis=-1), masks)


def metric_snapshot(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "selection_score": selection_score(metrics),
        "mean_iou": float(metrics["mean_iou"]),
        "boundary_iou": float(metrics["per_class"]["boundary_step_curb"]["iou"]),
        "unknown_iou": float(metrics["per_class"]["unknown_nonwalkable"]["iou"]),
        "pixel_accuracy": float(metrics["pixel_accuracy"]),
    }


def grouped_dev_metrics(
    predictions: np.ndarray,
    masks: np.ndarray,
    records: Sequence[shared.Record],
    key_for: Any,
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        key = key_for(record)
        if key:
            groups.setdefault(str(key), []).append(index)
    return {
        name: shared.confusion_and_metrics(predictions[indices], masks[indices])
        for name, indices in sorted(groups.items())
    }


def summarize_group_metrics(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not groups:
        return {"group_count": 0}
    snapshots = {name: metric_snapshot(metrics) for name, metrics in groups.items()}
    fields = tuple(next(iter(snapshots.values())))
    return {
        "group_count": len(groups),
        "macro": {
            field: float(np.mean([values[field] for values in snapshots.values()]))
            for field in fields
        },
        "worst": {
            field: {
                "value": float(min(values[field] for values in snapshots.values())),
                "group": min(snapshots, key=lambda name: snapshots[name][field]),
            }
            for field in fields
        },
        "groups": snapshots,
    }


def evaluate_dev_breakdown(
    model: Any,
    images: np.ndarray,
    masks: np.ndarray,
    records: Sequence[shared.Record],
    batch_size: int,
) -> dict[str, Any]:
    logits = model.predict(images, batch_size=batch_size, verbose=0)
    predictions = np.argmax(logits, axis=-1)
    global_metrics = shared.confusion_and_metrics(predictions, masks)
    sessions = grouped_dev_metrics(predictions, masks, records, lambda record: record.session_id)
    scenes = grouped_dev_metrics(
        predictions, masks, records, lambda record: record.scene_bucket or "unassigned",
    )
    return {
        "global": global_metrics,
        "global_snapshot": metric_snapshot(global_metrics),
        "macro_session": summarize_group_metrics(sessions),
        "scene": summarize_group_metrics(scenes),
    }


def train_seed(
    args: argparse.Namespace,
    *,
    keras: Any,
    torch: Any,
    train_images: np.ndarray,
    train_masks: np.ndarray,
    train_records: Sequence[shared.Record],
    train_sampling_stats: Sequence[dict[str, Any]],
    dev_images: np.ndarray,
    dev_masks: np.ndarray,
    dev_records: Sequence[shared.Record],
    loss_weights: np.ndarray,
    base_weights: Path,
    model_seed: int,
    sampler_seed: int,
) -> dict[str, Any]:
    keras.backend.clear_session()
    set_seed(keras, torch, model_seed)
    model = shared.sanpo_segmentation_model.build_mobilenetv3_lraspp(
        keras,
        args.input_size,
        len(shared.CLASS_NAMES),
        backbone_weights="imagenet",
        backbone_alpha=args.backbone_alpha,
        decoder_channels=args.decoder_channels,
        detail_output_stride=args.detail_output_stride,
        semantic_output_stride=args.semantic_output_stride,
    )
    if args.sampler_strategy == "deterministic_quota":
        sampler: Any = DeterministicQuotaSampler(
            train_images,
            train_masks,
            train_records,
            batch_size=args.batch_size,
            seed=sampler_seed,
            crop_min_fraction=args.crop_min_fraction,
            crop_max_fraction=args.crop_max_fraction,
            horizontal_flip_probability=args.horizontal_flip_probability,
            unknown_rich_quantile=args.unknown_rich_quantile,
            source_mask_stats=train_sampling_stats,
        )
    else:
        sampler = SessionBalancedCropSampler(
            train_images,
            train_masks,
            train_records,
            batch_size=args.batch_size,
            seed=sampler_seed,
            guided_crop_fraction=args.guided_crop_fraction,
            crop_min_fraction=args.crop_min_fraction,
            crop_max_fraction=args.crop_max_fraction,
            horizontal_flip_probability=args.horizontal_flip_probability,
            boundary_guided_probability=args.boundary_guided_probability,
        )
    seed_weights = seed_pair_weight_path(base_weights, model_seed, sampler_seed)
    seed_weights.parent.mkdir(parents=True, exist_ok=True)
    evaluations: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []
    global_best_key: tuple[float, float, float] | None = None
    completed_steps = 0
    next_evaluation_step = args.eval_every_steps
    fit_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    if args.head_only:
        stages = [{
            "name": "head_only",
            "steps": args.optimizer_steps,
            "backbone_trainable": False,
            "initial_learning_rate": args.learning_rate,
            "final_learning_rate_ratio": 1.0,
            "early_stopping": False,
        }]
    elif args.two_stage:
        stages = [
            {
                "name": "head_warmup",
                "steps": args.head_warmup_steps,
                "backbone_trainable": False,
                "initial_learning_rate": args.learning_rate,
                "final_learning_rate_ratio": 1.0,
                "early_stopping": False,
            },
            {
                "name": "backbone_finetune",
                "steps": args.optimizer_steps - args.head_warmup_steps,
                "backbone_trainable": True,
                "initial_learning_rate": args.finetune_learning_rate,
                "final_learning_rate_ratio": args.finetune_final_lr_ratio,
                "early_stopping": True,
            },
        ]
    else:
        stages = [{
            "name": "joint_training",
            "steps": args.optimizer_steps,
            "backbone_trainable": True,
            "initial_learning_rate": args.learning_rate,
            "final_learning_rate_ratio": 1.0,
            "early_stopping": True,
        }]

    for stage_index, stage in enumerate(stages):
        if stage_index > 0:
            previous_checkpoint = Path(stage_reports[-1]["checkpoint"])
            model.load_weights(previous_checkpoint)
        trainability = configure_trainable_layers(
            model,
            keras,
            backbone_trainable=bool(stage["backbone_trainable"]),
            keep_batchnorm_frozen=args.freeze_backbone_batchnorm,
        )
        if stage["final_learning_rate_ratio"] < 1.0:
            learning_rate: Any = keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=stage["initial_learning_rate"],
                decay_steps=max(1, int(stage["steps"])),
                alpha=stage["final_learning_rate_ratio"],
            )
            learning_rate_schedule = "cosine_decay"
        else:
            learning_rate = stage["initial_learning_rate"]
            learning_rate_schedule = "constant"
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=build_composite_loss(keras, loss_weights, args),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="pixel_accuracy")],
            jit_compile=args.jit_compile,
        )
        recent_losses: list[float] = []
        stage_best_key: tuple[float, float, float] | None = None
        stage_no_improvement = 0
        stage_completed_steps = 0
        checkpoint = seed_pair_stage_weight_path(
            base_weights, model_seed, sampler_seed, str(stage["name"]),
        )
        for stage_step in range(1, int(stage["steps"]) + 1):
            batch_images, batch_masks = sampler.next_batch()
            train_logs = model.train_on_batch(batch_images, batch_masks, return_dict=True)
            recent_losses.append(float(train_logs["loss"]))
            model.reset_metrics()
            stage_completed_steps = stage_step
            completed_steps += 1
            quota_cycle_complete = (
                not isinstance(sampler, DeterministicQuotaSampler)
                or sampler.sample_cursor % len(QUOTA_NAMES) == 0
            )
            scheduled_evaluation = completed_steps >= next_evaluation_step and quota_cycle_complete
            final_stage_step = stage_step == stage["steps"]
            if not scheduled_evaluation and not final_stage_step:
                continue
            requested_evaluation_step = next_evaluation_step if scheduled_evaluation else completed_steps
            if scheduled_evaluation:
                while next_evaluation_step <= completed_steps:
                    next_evaluation_step += args.eval_every_steps
            metrics = evaluate_arrays(model, dev_images, dev_masks, args.batch_size)
            key = checkpoint_key(metrics)
            stage_improved = (
                stage_best_key is None
                or key[0] > stage_best_key[0] + args.early_stopping_min_delta
                or (abs(key[0] - stage_best_key[0]) <= 1e-12 and key[1:] > stage_best_key[1:])
            )
            global_improved = (
                global_best_key is None
                or key[0] > global_best_key[0] + args.early_stopping_min_delta
                or (abs(key[0] - global_best_key[0]) <= 1e-12 and key[1:] > global_best_key[1:])
            )
            if stage_improved:
                stage_best_key = key
                stage_no_improvement = 0
                model.save_weights(checkpoint)
            else:
                stage_no_improvement += 1
            if global_improved:
                global_best_key = key
                model.save_weights(seed_weights)
            if learning_rate_schedule == "cosine_decay":
                current_learning_rate = cosine_decay_value(
                    float(stage["initial_learning_rate"]),
                    float(stage["final_learning_rate_ratio"]),
                    stage_step - 1,
                    int(stage["steps"]),
                )
            else:
                current_learning_rate = float(stage["initial_learning_rate"])
            evaluations.append({
                "stage": stage["name"],
                "stage_optimizer_step": stage_step,
                "optimizer_step": completed_steps,
                "requested_evaluation_step": requested_evaluation_step,
                "quota_cycle_complete": quota_cycle_complete,
                "learning_rate": current_learning_rate,
                "mean_train_loss_since_last_eval": float(np.mean(recent_losses)),
                "selection_score": key[0],
                "dev_mean_iou": float(metrics["mean_iou"]),
                "dev_boundary_iou": float(metrics["per_class"]["boundary_step_curb"]["iou"]),
                "dev_pixel_accuracy": float(metrics["pixel_accuracy"]),
                "stage_checkpoint_saved": stage_improved,
                "global_checkpoint_saved": global_improved,
                "checkpoint_saved": global_improved,
            })
            recent_losses.clear()
            if (
                stage["early_stopping"]
                and completed_steps >= args.minimum_optimizer_steps
                and stage_no_improvement >= args.patience_evaluations
            ):
                break
        if stage_best_key is None or not checkpoint.is_file():
            raise RuntimeError(f"stage {stage['name']} completed without a dev checkpoint")
        stage_reports.append({
            "name": stage["name"],
            "requested_optimizer_steps": stage["steps"],
            "completed_optimizer_steps": stage_completed_steps,
            "early_stopped": stage_completed_steps < stage["steps"],
            "trainability": trainability,
            "learning_rate": {
                "schedule": learning_rate_schedule,
                "initial": stage["initial_learning_rate"],
                "final_ratio": stage["final_learning_rate_ratio"],
            },
            "best_selection_score": stage_best_key[0],
            "best_boundary_iou": stage_best_key[1],
            "best_mean_iou": stage_best_key[2],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": shared.sha256_file(checkpoint),
            "restored_before_next_stage": stage_index < len(stages) - 1,
        })
    torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - fit_started
    if global_best_key is None or not seed_weights.is_file():
        raise RuntimeError("training completed without a dev checkpoint")
    model.load_weights(seed_weights)
    dev_breakdown = evaluate_dev_breakdown(
        model, dev_images, dev_masks, dev_records, args.batch_size,
    )
    final_metrics = dev_breakdown["global"]
    return {
        "seed": model_seed,
        "model_seed": model_seed,
        "sampler_seed": sampler_seed,
        "requested_optimizer_steps": args.optimizer_steps,
        "completed_optimizer_steps": completed_steps,
        "early_stopped": completed_steps < args.optimizer_steps,
        "stages": stage_reports,
        "selection_score": selection_score(final_metrics),
        "dev_mask_metrics": final_metrics,
        "dev_breakdown": dev_breakdown,
        "evaluations": evaluations,
        "sampler": sampler.report(),
        "fit_seconds": fit_seconds,
        "training_images_per_second": completed_steps * args.batch_size / max(fit_seconds, 1e-9),
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "weights": str(seed_weights),
        "weights_sha256": shared.sha256_file(seed_weights),
        "parameter_count": int(model.count_params()),
        "feature_contract": dict(model.sanpo_feature_contract),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = shared.project_root()
    dataset_root = resolve(root, args.dataset_root).resolve()
    manifest = dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST
    report_path = resolve(root, args.report).resolve()
    gate_path = resolve(dataset_root, args.training_gate_report).resolve()
    gate_report = training_gate.consume_training_authorization(dataset_root, gate_path)

    records = shared.load_records(manifest)
    train_records = shared.records_by_split(records, "train")
    dev_records = shared.records_by_split(records, "dev")
    source_mask_stats_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        source_mask = shared.validate_binary_masks(record)
        source_mask_stats_by_id[record.sample_id] = {
            "class_presence": [
                bool(np.any(source_mask == class_id)) for class_id in range(len(shared.CLASS_NAMES))
            ],
            "unknown_share": float(np.mean(source_mask == shared.CLASS_IDS["unknown_nonwalkable"])),
        }
    train_sampling_stats = [source_mask_stats_by_id[record.sample_id] for record in train_records]

    # Keras selects its backend at import time. Fail closed if the parent process
    # imported another backend or configured a conflicting value.
    os.environ["KERAS_BACKEND"] = "torch"
    import keras
    import torch

    if keras.backend.backend() != "torch":
        raise RuntimeError(f"Expected KERAS_BACKEND=torch, got {keras.backend.backend()!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is unavailable; refusing to silently train on CPU")

    def preload(values: Sequence[shared.Record]) -> tuple[np.ndarray, np.ndarray]:
        examples = [shared.load_example(record, args.input_size) for record in values]
        images, masks = zip(*examples)
        # Keep the preloaded pool compact at 384/512; sampler batches are cast
        # to float32/int64 immediately before optimization.
        return np.stack(images).astype(np.uint8), np.stack(masks).astype(np.uint8)

    train_images, train_masks = preload(train_records)
    dev_images, dev_masks = preload(dev_records)
    keras.mixed_precision.set_global_policy("mixed_float16")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    pixel_counts = shared.class_pixel_counts(train_records)
    loss_weights = class_loss_weights(pixel_counts, args.maximum_class_weight)
    weights = resolve(root, args.weights).resolve()
    seed_runs = [
        train_seed(
            args,
            keras=keras,
            torch=torch,
            train_images=train_images,
            train_masks=train_masks,
            train_records=train_records,
            train_sampling_stats=train_sampling_stats,
            dev_images=dev_images,
            dev_masks=dev_masks,
            dev_records=dev_records,
            loss_weights=loss_weights,
            base_weights=weights,
            model_seed=model_seed,
            sampler_seed=sampler_seed,
        )
        for model_seed, sampler_seed in args.seed_pairs
    ]
    selected = max(seed_runs, key=lambda run: checkpoint_key(run["dev_mask_metrics"]))
    weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected["weights"], weights)
    stability = aggregate_seed_metrics(seed_runs)
    report = {
        "schema_version": 4,
        "candidate": "MobileNetV3Small+LR-ASPP",
        "architecture_revision": shared.sanpo_segmentation_model.ARCHITECTURE_REVISION,
        "model_definition_sha256": shared.sha256_file(Path(shared.sanpo_segmentation_model.__file__)),
        "benchmark_only": True,
        "promotion": "do_not_replace_default_model",
        "backend": "keras3_torch",
        "device": str(torch.cuda.get_device_name(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mixed_precision_policy": keras.mixed_precision.global_policy().name,
        "numeric_contract": {"tf32": False, "float32_matmul_precision": "highest"},
        "model_hyperparameters": {
            "backbone": "MobileNetV3Small",
            "input_size": args.input_size,
            "backbone_alpha": args.backbone_alpha,
            "decoder_channels": args.decoder_channels,
            "detail_output_stride": args.detail_output_stride,
            "semantic_output_stride": args.semantic_output_stride,
            "feature_contract": selected["feature_contract"],
            "parameter_count": selected["parameter_count"],
        },
        "data_pipeline": f"preloaded_uint8_{args.sampler_strategy}",
        "manifest": str(manifest),
        "manifest_sha256": shared.sha256_file(manifest),
        "training_gate_report": str(gate_path),
        "training_gate_report_sha256": gate_report["report_sha256"],
        "blind_holdout_access": "not_accessed_by_trainer",
        "record_counts": {"train": len(train_records), "dev": len(dev_records)},
        "session_counts": {
            "train": len({record.session_id for record in train_records}),
            "dev": len({record.session_id for record in dev_records}),
        },
        "class_pixel_counts": pixel_counts,
        "class_loss_weights": {
            name: float(loss_weights[index]) for index, name in enumerate(shared.CLASS_NAMES)
        },
        "loss": {
            "name": "weighted_ce_dice_focal",
            "ce_weight": args.ce_weight,
            "dice_weight": args.dice_weight,
            "focal_weight": args.focal_weight,
            "focal_gamma": args.focal_gamma,
            "maximum_class_weight": args.maximum_class_weight,
        },
        "training_protocol": {
            "budget_unit": "optimizer_step",
            "optimizer_steps_per_seed": args.optimizer_steps,
            "minimum_optimizer_steps": args.minimum_optimizer_steps,
            "eval_every_steps": args.eval_every_steps,
            "batch_size": args.batch_size,
            "head_only": args.head_only,
            "two_stage": args.two_stage and not args.head_only,
            "head_warmup_steps": args.head_warmup_steps if args.two_stage and not args.head_only else 0,
            "head_learning_rate": args.learning_rate,
            "finetune_learning_rate": args.finetune_learning_rate if args.two_stage else None,
            "finetune_final_learning_rate_ratio": args.finetune_final_lr_ratio if args.two_stage else None,
            "freeze_backbone_batchnorm": args.freeze_backbone_batchnorm,
            "seeds": list(args.seeds),
            "seed_pairs": [
                {"model_seed": model_seed, "sampler_seed": sampler_seed}
                for model_seed, sampler_seed in args.seed_pairs
            ],
            "sampler_strategy": args.sampler_strategy,
            "unknown_rich_quantile": (
                args.unknown_rich_quantile if args.sampler_strategy == "deterministic_quota" else None
            ),
            "checkpoint_monitor": "harmonic_mean(dev_mean_iou, dev_boundary_step_curb_iou)",
            "checkpoint_tiebreakers": ["dev_boundary_step_curb_iou", "dev_mean_iou"],
            "early_stopping_patience_evaluations": args.patience_evaluations,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "jit_compile": args.jit_compile,
        },
        "seed_runs": seed_runs,
        "seed_stability": stability,
        "p0_factor_variation": factor_variation_summary(seed_runs) if args.head_only else None,
        "selected_seed": selected["seed"],
        "selected_model_seed": selected["model_seed"],
        "selected_sampler_seed": selected["sampler_seed"],
        "dev_mask_metrics": selected["dev_mask_metrics"],
        "selection_score": selected["selection_score"],
        "weights": str(weights),
        "weights_sha256": shared.sha256_file(weights),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    shared.write_json(report_path, report)
    Path(str(report_path) + ".sha256").write_text(shared.sha256_file(report_path) + "\n", encoding="ascii")
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the gated SANPO candidate with Keras 3 + PyTorch CUDA.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--training-gate-report", default="qa/training_gate_report.json")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--input-size", type=int, default=shared.INPUT_SIZE, choices=ALLOWED_INPUT_SIZES)
    parser.add_argument("--optimizer-steps", type=int, default=1200)
    parser.add_argument("--minimum-optimizer-steps", type=int, default=300)
    parser.add_argument("--eval-every-steps", type=int, default=50)
    parser.add_argument("--patience-evaluations", type=int, default=6)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--batch-size", type=int, default=12,
        help="Optimization batch size. Throughput benchmarking is separate from the learning protocol.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--two-stage", action=argparse.BooleanOptionalAction, default=True,
        help="Warm up LR-ASPP with a frozen backbone, then fine-tune the backbone at a lower decayed LR.",
    )
    parser.add_argument(
        "--head-only", action=argparse.BooleanOptionalAction, default=False,
        help="Freeze the backbone for the full run; intended for short P0 seed-factor audits.",
    )
    parser.add_argument("--head-warmup-steps", type=int, default=100)
    parser.add_argument("--finetune-learning-rate", type=float, default=5e-5)
    parser.add_argument("--finetune-final-lr-ratio", type=float, default=0.10)
    parser.add_argument(
        "--freeze-backbone-batchnorm", action=argparse.BooleanOptionalAction, default=True,
        help="Keep pretrained backbone BatchNorm statistics frozen during small-data fine-tuning.",
    )
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--seed", type=int, help="Compatibility shortcut for a deliberate single-seed audit run.")
    parser.add_argument(
        "--seed-pairs",
        help="Comma-separated model_seed:sampler_seed pairs; separates initialization and sampling variance.",
    )
    parser.add_argument("--guided-crop-fraction", type=float, default=0.70)
    parser.add_argument(
        "--sampler-strategy",
        choices=("session_balanced_guided", "deterministic_quota"),
        default="session_balanced_guided",
    )
    parser.add_argument("--unknown-rich-quantile", type=float, default=0.75)
    parser.add_argument("--boundary-guided-probability", type=float, default=0.65)
    parser.add_argument("--crop-min-fraction", type=float, default=0.55)
    parser.add_argument("--crop-max-fraction", type=float, default=0.85)
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.50)
    parser.add_argument("--ce-weight", type=float, default=0.50)
    parser.add_argument("--dice-weight", type=float, default=0.40)
    parser.add_argument("--focal-weight", type=float, default=0.10)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--maximum-class-weight", type=float, default=4.0)
    parser.add_argument("--backbone-alpha", type=float, choices=[0.75, 1.0], default=0.75)
    parser.add_argument("--decoder-channels", type=int, default=96)
    parser.add_argument("--detail-output-stride", type=int, choices=[4, 8], default=8)
    parser.add_argument("--semantic-output-stride", type=int, choices=[16, 32], default=32)
    parser.add_argument(
        "--jit-compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use torch.compile through Keras; keep identical across ablations.",
    )
    args = parser.parse_args(argv)
    try:
        if args.seed_pairs and args.seed is not None:
            parser.error("--seed-pairs and --seed are mutually exclusive")
        if args.seed_pairs:
            args.seed_pairs = parse_seed_pairs(args.seed_pairs)
            args.seeds = tuple(dict.fromkeys(model_seed for model_seed, _ in args.seed_pairs))
        else:
            args.seeds = (args.seed,) if args.seed is not None else parse_seed_list(args.seeds)
            args.seed_pairs = tuple((seed, seed) for seed in args.seeds)
    except ValueError as error:
        parser.error(str(error))
    positive = (
        args.optimizer_steps,
        args.minimum_optimizer_steps,
        args.eval_every_steps,
        args.patience_evaluations,
        args.batch_size,
        args.learning_rate,
        args.head_warmup_steps,
        args.finetune_learning_rate,
        args.focal_gamma,
        args.maximum_class_weight,
        args.decoder_channels,
    )
    if any(value <= 0 for value in positive):
        parser.error("step counts, batch-size, learning-rate, focal-gamma, class-weight cap and decoder channels must be positive")
    if args.minimum_optimizer_steps > args.optimizer_steps:
        parser.error("minimum-optimizer-steps must not exceed optimizer-steps")
    if args.two_stage and not args.head_only and args.head_warmup_steps >= args.optimizer_steps:
        parser.error("two-stage training requires head-warmup-steps < optimizer-steps")
    if not 0 < args.finetune_final_lr_ratio <= 1:
        parser.error("finetune-final-lr-ratio must be in (0, 1]")
    if args.early_stopping_min_delta < 0:
        parser.error("early-stopping-min-delta must be non-negative")
    if (
        not 0 <= args.guided_crop_fraction <= 1
        or not 0 <= args.horizontal_flip_probability <= 1
        or not 0 <= args.boundary_guided_probability <= 1
    ):
        parser.error("crop/flip probabilities must be in 0..1")
    if not 0 < args.crop_min_fraction <= args.crop_max_fraction <= 1:
        parser.error("crop fractions must satisfy 0 < min <= max <= 1")
    if not 0 <= args.unknown_rich_quantile < 1:
        parser.error("unknown-rich-quantile must be in [0, 1)")
    if min(args.ce_weight, args.dice_weight, args.focal_weight) < 0 or abs(
        args.ce_weight + args.dice_weight + args.focal_weight - 1.0
    ) > 1e-6:
        parser.error("ce-weight, dice-weight and focal-weight must be non-negative and sum to 1")
    return args


def main() -> None:
    report = run(parse_args())
    print(f"weights={report['weights']}")
    print(f"weights_sha256={report['weights_sha256']}")
    print(f"selected_seed={report['selected_seed']}")
    print(f"dev_mean_iou={report['dev_mask_metrics']['mean_iou']:.6f}")
    print(f"dev_boundary_iou={report['dev_mask_metrics']['per_class']['boundary_step_curb']['iou']:.6f}")
    print("promotion=do_not_replace_default_model")


if __name__ == "__main__":
    main()
