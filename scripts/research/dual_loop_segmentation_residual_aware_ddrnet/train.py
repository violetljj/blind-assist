#!/usr/bin/env python3
"""Train the one frozen FP-aware DDRNet successor.

The only scientific variable relative to the archived R1 DDRNet trainer is the
frame-selection probability inside the 30% unguided full-frame sampling branch.
Within the already selected session, frames are weighted by the number of
train-only pixels where the same-seed frozen R1 baseline predicts a hazard
class while canonical truth is walkable or unknown. The input remains the full
frame. Backbone, source initialization, four classes, loss, positive-guided
70% branch, optimizer, schedule, seeds and checkpoint rule remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from . import CANDIDATE_ID, PROTOCOL_ID
from .contract import validate_config_contract, validate_config_sha256
from .models import DDRNet23SlimSegmenter, load_exact_checkpoint, sha256_file


CLASS_NAMES = ("walkable", "boundary_step_curb", "obstacle", "unknown_nonwalkable")
BOUNDARY_ID = 1
OBSTACLE_ID = 2
HAZARD_IDS = (BOUNDARY_ID, OBSTACLE_ID)
NON_HAZARD_IDS = (0, 3)


def resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_shared(repo_root: Path) -> Any:
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import train_export_sanpo_segmentation as shared

    return shared


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_arrays(shared: Any, records: Sequence[Any], input_size: int) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for record in records:
        image, mask = shared.load_example(record, input_size)
        images.append(image)
        masks.append(mask)
    return np.stack(images).astype(np.float32), np.stack(masks).astype(np.int64)


def class_weights(shared: Any, train_records: Sequence[Any]) -> np.ndarray:
    counts = shared.class_pixel_counts(train_records)
    values = np.asarray([counts[name] for name in CLASS_NAMES], dtype=np.float64)
    frequencies = values / max(1.0, values.sum())
    raw = 1.0 / np.sqrt(np.maximum(frequencies, 1e-12))
    return np.clip(raw / max(1e-12, raw.mean()), 0.35, 4.0).astype(np.float32)


def batch_tensor(images: np.ndarray, masks: np.ndarray, device: torch.device) -> tuple[Tensor, Tensor]:
    image_tensor = torch.from_numpy(images).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    mask_tensor = torch.from_numpy(masks).to(device=device, dtype=torch.long)
    return image_tensor, mask_tensor


def train_mode(model: DDRNet23SlimSegmenter) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def predict_classes(
    model: DDRNet23SlimSegmenter,
    images: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    values: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            tensor = torch.from_numpy(images[start : start + batch_size]).to(
                device=device, dtype=torch.float32
            ).permute(0, 3, 1, 2)
            values.append(model(tensor).argmax(dim=1).cpu().numpy().astype(np.uint8))
    return np.concatenate(values, axis=0)


def build_hard_negative_masks(baseline_predictions: np.ndarray, truth_masks: np.ndarray) -> np.ndarray:
    if baseline_predictions.shape != truth_masks.shape:
        raise ValueError("baseline predictions and truth masks must have the same shape")
    return np.isin(baseline_predictions, HAZARD_IDS) & np.isin(truth_masks, NON_HAZARD_IDS)


class FpAwareSampler:
    """R1 sampler with only its 30% full-frame selection weights replaced."""

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        records: Sequence[Any],
        hard_negative_masks: np.ndarray,
        *,
        batch_size: int,
        seed: int,
        guided_crop_fraction: float,
        boundary_guided_probability: float,
        crop_min_fraction: float,
        crop_max_fraction: float,
        horizontal_flip_probability: float,
    ) -> None:
        if not (len(images) == len(masks) == len(records) == len(hard_negative_masks) and records):
            raise ValueError("sampler inputs must have equal non-zero length")
        self.images = images
        self.masks = masks
        self.hard_negative_masks = hard_negative_masks
        self.batch_size = batch_size
        self.guided_crop_fraction = guided_crop_fraction
        self.boundary_guided_probability = boundary_guided_probability
        self.crop_min_fraction = crop_min_fraction
        self.crop_max_fraction = crop_max_fraction
        self.horizontal_flip_probability = horizontal_flip_probability
        self.rng = np.random.default_rng(seed)
        self.sessions: dict[str, list[int]] = {}
        for index, record in enumerate(records):
            self.sessions.setdefault(str(record.session_id), []).append(index)
        self.session_ids = tuple(sorted(self.sessions))
        self.positive_candidates = {
            session: {
                class_id: [index for index in indices if np.any(masks[index] == class_id)]
                for class_id in HAZARD_IDS
            }
            for session, indices in self.sessions.items()
        }
        self.hard_negative_weights = {
            session: np.asarray(
                [int(hard_negative_masks[index].sum()) for index in indices],
                dtype=np.float64,
            )
            for session, indices in self.sessions.items()
        }
        self.session_draws = Counter()
        self.branch_draws = Counter()
        self.positive_guided_hits = Counter()
        self.hard_negative_fallbacks = 0

    def validate_pool_coverage(self) -> None:
        missing = [
            session
            for session in self.session_ids
            if float(self.hard_negative_weights[session].sum()) <= 0
        ]
        if missing:
            raise RuntimeError(f"train-only hard-negative pool has zero candidates for sessions: {missing}")

    def pool_receipt(self) -> dict[str, Any]:
        return {
            "definition": "baseline_argmax_in_1_2 AND canonical_truth_in_0_3",
            "session_frame_counts": {
                session: int(np.count_nonzero(self.hard_negative_weights[session]))
                for session in self.session_ids
            },
            "session_pixel_counts": {
                session: int(
                    sum(self.hard_negative_masks[index].sum() for index in self.sessions[session])
                )
                for session in self.session_ids
            },
            "total_frame_count": int(sum(bool(np.any(mask)) for mask in self.hard_negative_masks)),
            "total_pixel_count": int(self.hard_negative_masks.sum()),
            "all_train_sessions_covered": all(
                float(self.hard_negative_weights[session].sum()) > 0
                for session in self.session_ids
            ),
            "sampling_unit": "full_frame",
            "within_session_probability": "frame_fp_pixel_count / session_fp_pixel_count",
        }

    def _crop_around(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not len(points):
            return image.copy(), mask.copy()
        target_y, target_x = points[int(self.rng.integers(len(points)))]
        height, width = mask.shape
        fraction = float(self.rng.uniform(self.crop_min_fraction, self.crop_max_fraction))
        crop_h = max(2, min(height, int(round(height * fraction))))
        crop_w = max(2, min(width, int(round(width * fraction))))
        left = int(np.clip(target_x - self.rng.integers(max(1, crop_w)), 0, width - crop_w))
        top = int(np.clip(target_y - self.rng.integers(max(1, crop_h)), 0, height - crop_h))
        box = (left, top, left + crop_w, top + crop_h)
        crop_image = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").crop(box).resize(
            (width, height), Image.Resampling.BILINEAR
        )
        crop_mask = Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").crop(box).resize(
            (width, height), Image.Resampling.NEAREST
        )
        return np.asarray(crop_image, dtype=np.float32), np.asarray(crop_mask, dtype=np.int64)

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for _ in range(self.batch_size):
            session = self.session_ids[int(self.rng.integers(len(self.session_ids)))]
            self.session_draws[session] += 1
            if self.rng.random() < self.guided_crop_fraction:
                self.branch_draws["positive_guided"] += 1
                class_id = (
                    BOUNDARY_ID
                    if self.rng.random() < self.boundary_guided_probability
                    else OBSTACLE_ID
                )
                indices = self.positive_candidates[session][class_id] or self.sessions[session]
                index = indices[int(self.rng.integers(len(indices)))]
                points = np.argwhere(self.masks[index] == class_id)
                image, mask = self._crop_around(self.images[index], self.masks[index], points)
                if len(points):
                    self.positive_guided_hits[CLASS_NAMES[class_id]] += 1
            else:
                self.branch_draws["fp_weighted_full_frame"] += 1
                indices = self.sessions[session]
                weights = self.hard_negative_weights[session]
                total_weight = float(weights.sum())
                if total_weight <= 0:
                    self.hard_negative_fallbacks += 1
                    index = indices[int(self.rng.integers(len(indices)))]
                else:
                    local_index = int(self.rng.choice(len(indices), p=weights / total_weight))
                    index = indices[local_index]
                image, mask = self.images[index].copy(), self.masks[index].copy()
            if self.rng.random() < self.horizontal_flip_probability:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            images.append(image)
            masks.append(mask)
        return np.stack(images).astype(np.float32), np.stack(masks).astype(np.int64)

    def receipt(self) -> dict[str, Any]:
        return {
            "strategy": CANDIDATE_ID,
            "session_draws": dict(sorted(self.session_draws.items())),
            "branch_draws": dict(sorted(self.branch_draws.items())),
            "positive_guided_hits": dict(sorted(self.positive_guided_hits.items())),
            "hard_negative_fallbacks": self.hard_negative_fallbacks,
            "guided_crop_fraction": self.guided_crop_fraction,
            "fp_weighted_full_frame_fraction": 1.0 - self.guided_crop_fraction,
            "boundary_guided_probability_within_positive_branch": self.boundary_guided_probability,
            "crop_fraction_range": [self.crop_min_fraction, self.crop_max_fraction],
            "horizontal_flip_probability": self.horizontal_flip_probability,
            "unguided_transform": "unchanged_full_frame",
        }


def composite_loss(logits: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=len(CLASS_NAMES)).permute(0, 3, 1, 2).float()
    per_pixel_weights = weights[target]
    cross_entropy = F.cross_entropy(logits, target, reduction="none")
    true_probability = (probabilities * one_hot).sum(dim=1).clamp_min(1e-7)
    focal = -((1.0 - true_probability) ** 2.0) * torch.log(true_probability)
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = (probabilities + one_hot).sum(dim=(0, 2, 3))
    dice_per_class = (2.0 * intersection + 1.0) / (denominator + 1.0)
    normalized_weights = weights / weights.sum()
    dice_loss = 1.0 - (dice_per_class * normalized_weights).sum()
    return (
        0.5 * (cross_entropy * per_pixel_weights).mean()
        + 0.1 * (focal * per_pixel_weights).mean()
        + 0.4 * dice_loss
    )


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    mean_iou = float(metrics["mean_iou"])
    boundary_iou = float(metrics["per_class"]["boundary_step_curb"]["iou"])
    harmonic = (
        0.0
        if min(mean_iou, boundary_iou) <= 0
        else 2.0 * mean_iou * boundary_iou / (mean_iou + boundary_iou)
    )
    return harmonic, boundary_iou, mean_iou


def evaluate_dev(
    shared: Any,
    model: DDRNet23SlimSegmenter,
    images: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    predictions = predict_classes(model, images, device, batch_size)
    return shared.confusion_and_metrics(predictions, masks)


def state_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_seed(
    *,
    shared: Any,
    model: DDRNet23SlimSegmenter,
    sampler: FpAwareSampler,
    dev_images: np.ndarray,
    dev_masks: np.ndarray,
    seed: int,
    device: torch.device,
    output_dir: Path,
    weights: Tensor,
    progress_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    set_seed(seed)
    training = config["training"]
    total_steps = int(training["optimizer_steps"])
    warmup_steps = int(training["head_warmup_steps"])
    eval_every = int(training["eval_every_steps"])
    batch_size = int(training["batch_size"])
    best_key: tuple[float, float, float] | None = None
    best_metrics: dict[str, Any] | None = None
    best_state: dict[str, Tensor] | None = None
    best_step = 0
    completed = 0
    history: list[dict[str, Any]] = []
    stages = (
        ("head_warmup", warmup_steps, True, float(training["head_learning_rate"]), 1.0),
        (
            "backbone_finetune",
            total_steps - warmup_steps,
            False,
            float(training["finetune_learning_rate"]),
            float(training["finetune_final_lr_ratio"]),
        ),
    )
    stage_reports: list[dict[str, Any]] = []
    for stage_name, stage_steps, head_only, initial_lr, final_ratio in stages:
        trainability = model.set_stage_trainability(head_only=head_only)
        train_mode(model)
        optimizer = torch.optim.Adam(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=initial_lr,
        )
        scheduler = (
            None
            if final_ratio >= 1.0
            else torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=stage_steps,
                eta_min=initial_lr * final_ratio,
            )
        )
        stage_started = time.perf_counter()
        for _ in range(stage_steps):
            batch_images, batch_masks = sampler.next_batch()
            tensor, target = batch_tensor(batch_images, batch_masks, device)
            optimizer.zero_grad(set_to_none=True)
            loss = composite_loss(model(tensor), target, weights)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            completed += 1
            if completed % eval_every and completed != total_steps:
                continue
            metrics = evaluate_dev(shared, model, dev_images, dev_masks, device, batch_size)
            key = selection_key(metrics)
            improved = best_key is None or key > best_key
            if improved:
                best_key = key
                best_metrics = metrics
                best_state = state_cpu(model)
                best_step = completed
            history.append(
                {
                    "stage": stage_name,
                    "optimizer_step": completed,
                    "loss": float(loss.detach().cpu()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "selection_key": list(key),
                    "dev_mean_iou": float(metrics["mean_iou"]),
                    "dev_boundary_iou": float(metrics["per_class"]["boundary_step_curb"]["iou"]),
                    "checkpoint_saved": improved,
                }
            )
            write_json(
                progress_path,
                {
                    "schema_version": "blindassist.dual_loop_segmentation_residual_aware_ddrnet_r0.progress.v1",
                    "protocol_id": PROTOCOL_ID,
                    "status": "RUNNING",
                    "seed": seed,
                    "completed_steps_current_seed": completed,
                    "steps_per_seed": total_steps,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            print(
                f"[{CANDIDATE_ID}] seed={seed} step={completed}/{total_steps} "
                f"selection={key[0]:.6f} boundary_iou={key[1]:.6f}",
                flush=True,
            )
            train_mode(model)
        stage_reports.append(
            {
                "name": stage_name,
                "completed_steps": stage_steps,
                "head_only": head_only,
                "initial_learning_rate": initial_lr,
                "final_learning_rate_ratio": final_ratio,
                "trainability": trainability,
                "elapsed_ms": (time.perf_counter() - stage_started) * 1000.0,
            }
        )
    if best_key is None or best_metrics is None or best_state is None:
        raise RuntimeError(f"seed {seed} completed without a dev checkpoint")
    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = seed_dir / "best_fp32.pt"
    torch.save(
        {
            "state_dict": best_state,
            "model_id": "DDRNet-23-Slim",
            "candidate_id": CANDIDATE_ID,
            "protocol_id": PROTOCOL_ID,
            "seed": seed,
            "optimizer_step": best_step,
            "selection_key": list(best_key),
            "dev_mask_metrics": best_metrics,
            "input_contract": {"shape": [1, 256, 256, 3], "dtype": "float32", "range": "0..255"},
        },
        checkpoint,
    )
    report = {
        "seed": seed,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_optimizer_step": best_step,
        "selection_key": list(best_key),
        "dev_mask_metrics": best_metrics,
        "history": history,
        "stages": stage_reports,
        "sampler": sampler.receipt(),
    }
    write_json(seed_dir / "seed_report.json", report)
    return report


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    config_path = resolve(repo_root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config_sha256(sha256_file(config_path))
    validate_config_contract(config)
    inputs = config["inputs"]
    bound_paths = {
        name: resolve(repo_root, item["path"])
        for name, item in inputs.items()
        if isinstance(item, dict) and "path" in item
    }
    for name, path in bound_paths.items():
        expected = str(inputs[name]["sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{name} SHA256 mismatch: {actual} != {expected}")
    dataset_root = resolve(repo_root, config["data"]["dataset_root"])
    manifest = dataset_root / config["data"]["training_manifest"]
    if sha256_file(manifest) != config["data"]["training_manifest_sha256"]:
        raise ValueError("training manifest SHA256 mismatch")
    shared = load_shared(repo_root)
    records = shared.load_records(manifest)
    train_records = [record for record in records if record.split == "train"]
    dev_records = [record for record in records if record.split == "dev"]
    if len(train_records) != 400 or len(dev_records) != 200:
        raise ValueError("successor requires the frozen 400 train + 200 dev view")
    if len({str(record.session_id) for record in train_records}) != 8:
        raise ValueError("successor requires all 8 train sessions")
    if len({str(record.session_id) for record in dev_records}) != 4:
        raise ValueError("successor requires all 4 dev sessions")
    train_images, train_masks = load_arrays(shared, train_records, 256)
    dev_images, dev_masks = load_arrays(shared, dev_records, 256)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    architecture = bound_paths["ddrnet_architecture"]
    source_checkpoint = bound_paths["ddrnet_source_checkpoint"]
    hard_negative_masks_by_seed: dict[int, np.ndarray] = {}
    pool_receipts: dict[str, dict[str, Any]] = {}
    sampling = config["training"]["sampling"]
    for seed_value in config["training"]["seeds"]:
        seed = int(seed_value)
        baseline_checkpoint = bound_paths[f"r1_baseline_seed_{seed}"]
        baseline = DDRNet23SlimSegmenter(architecture, source_checkpoint).to(device)
        load_exact_checkpoint(baseline, baseline_checkpoint)
        baseline_predictions = predict_classes(
            baseline,
            train_images,
            device,
            int(config["training"]["batch_size"]),
        )
        del baseline
        if device.type == "cuda":
            torch.cuda.empty_cache()
        masks_for_seed = build_hard_negative_masks(baseline_predictions, train_masks)
        hard_negative_masks_by_seed[seed] = masks_for_seed
        sampler = FpAwareSampler(
            train_images,
            train_masks,
            train_records,
            masks_for_seed,
            batch_size=int(config["training"]["batch_size"]),
            seed=seed,
            guided_crop_fraction=float(sampling["positive_guided_fraction"]),
            boundary_guided_probability=float(
                sampling["boundary_probability_within_positive_branch"]
            ),
            crop_min_fraction=float(sampling["crop_min_fraction"]),
            crop_max_fraction=float(sampling["crop_max_fraction"]),
            horizontal_flip_probability=float(sampling["horizontal_flip_probability"]),
        )
        sampler.validate_pool_coverage()
        pool_receipts[str(seed)] = sampler.pool_receipt()
    return {
        "repo_root": repo_root,
        "config_path": config_path,
        "config": config,
        "shared": shared,
        "train_records": train_records,
        "dev_records": dev_records,
        "train_images": train_images,
        "train_masks": train_masks,
        "dev_images": dev_images,
        "dev_masks": dev_masks,
        "hard_negative_masks_by_seed": hard_negative_masks_by_seed,
        "pool_receipts": pool_receipts,
        "device": device,
        "architecture": architecture,
        "source_checkpoint": source_checkpoint,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    early_output_dir: Path | None = None
    if not args.preflight_only:
        early_repo_root = Path(args.repo_root).resolve()
        early_config_path = resolve(early_repo_root, args.config)
        early_config = json.loads(early_config_path.read_text(encoding="utf-8"))
        validate_config_sha256(sha256_file(early_config_path))
        validate_config_contract(early_config)
        early_output_dir = resolve(
            early_repo_root,
            args.output_dir or early_config["output"]["training_root"],
        )
        if early_output_dir.exists() and any(early_output_dir.iterdir()):
            raise FileExistsError(
                f"refusing to overwrite non-empty output: {early_output_dir}"
            )
    prepared = prepare(args)
    config = prepared["config"]
    if args.preflight_only:
        return {
            "status": "PREFLIGHT_VALID",
            "protocol_id": PROTOCOL_ID,
            "candidate_id": CANDIDATE_ID,
            "records": {"train": 400, "dev": 200},
            "hard_negative_pools_by_seed": prepared["pool_receipts"],
            "candidate_outcomes_accessed": False,
        }
    output_dir = early_output_dir or resolve(
        prepared["repo_root"],
        args.output_dir or config["output"]["training_root"],
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "hard_negative_pools_by_seed.json", prepared["pool_receipts"])
    template = DDRNet23SlimSegmenter(prepared["architecture"], prepared["source_checkpoint"])
    write_json(output_dir / "initialization_receipt.json", template.build_receipt.as_dict())
    del template
    weights = torch.from_numpy(
        class_weights(prepared["shared"], prepared["train_records"])
    ).to(device=prepared["device"], dtype=torch.float32)
    seed_reports: list[dict[str, Any]] = []
    progress_path = output_dir / "training_progress.json"
    started = time.perf_counter()
    sampling = config["training"]["sampling"]
    for seed in config["training"]["seeds"]:
        seed = int(seed)
        set_seed(seed)
        model = DDRNet23SlimSegmenter(
            prepared["architecture"], prepared["source_checkpoint"]
        ).to(prepared["device"])
        sampler = FpAwareSampler(
            prepared["train_images"],
            prepared["train_masks"],
            prepared["train_records"],
            prepared["hard_negative_masks_by_seed"][seed],
            batch_size=int(config["training"]["batch_size"]),
            seed=seed,
            guided_crop_fraction=float(sampling["positive_guided_fraction"]),
            boundary_guided_probability=float(
                sampling["boundary_probability_within_positive_branch"]
            ),
            crop_min_fraction=float(sampling["crop_min_fraction"]),
            crop_max_fraction=float(sampling["crop_max_fraction"]),
            horizontal_flip_probability=float(sampling["horizontal_flip_probability"]),
        )
        sampler.validate_pool_coverage()
        seed_reports.append(
            train_one_seed(
                shared=prepared["shared"],
                model=model,
                sampler=sampler,
                dev_images=prepared["dev_images"],
                dev_masks=prepared["dev_masks"],
                seed=seed,
                device=prepared["device"],
                output_dir=output_dir,
                weights=weights,
                progress_path=progress_path,
                config=config,
            )
        )
        del model
        if prepared["device"].type == "cuda":
            torch.cuda.empty_cache()
    report = {
        "schema_version": "blindassist.dual_loop_segmentation_residual_aware_ddrnet_r0.training_report.v1",
        "protocol_id": PROTOCOL_ID,
        "candidate_id": CANDIDATE_ID,
        "status": "TRAINING_COMPLETE",
        "config_path": str(prepared["config_path"]),
        "config_sha256": sha256_file(prepared["config_path"]),
        "data": {
            "training_manifest": config["data"]["training_manifest"],
            "training_manifest_sha256": config["data"]["training_manifest_sha256"],
            "records": {"train": 400, "dev": 200},
            "training_only_hard_negative_mining": True,
        },
        "single_variable": config["single_variable"],
        "hard_negative_pools_by_seed": prepared["pool_receipts"],
        "class_pixel_weights": weights.detach().cpu().tolist(),
        "training_contract": config["training"],
        "seed_reports": seed_reports,
        "cross_seed_selection": "FORBIDDEN_NOT_PERFORMED",
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "fresh_or_confirmation_outcome_accessed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / "training_report.json", report)
    write_json(
        progress_path,
        {
            "schema_version": "blindassist.dual_loop_segmentation_residual_aware_ddrnet_r0.progress.v1",
            "protocol_id": PROTOCOL_ID,
            "status": "COMPLETE",
            "completed_steps": int(config["training"]["optimizer_steps"])
            * len(config["training"]["seeds"]),
            "cross_seed_selection": "FORBIDDEN_NOT_PERFORMED",
            "seed_checkpoint_sha256": {
                str(item["seed"]): item["checkpoint_sha256"] for item in seed_reports
            },
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=Path.cwd())
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    report = run(parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "protocol_id": report["protocol_id"],
                "candidate_id": report["candidate_id"],
                "cross_seed_selection": report.get("cross_seed_selection"),
                "hard_negative_pools_by_seed": report.get("hard_negative_pools_by_seed"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
