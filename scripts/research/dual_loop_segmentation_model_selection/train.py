#!/usr/bin/env python3
"""Train one R1 segmentation candidate under the frozen shared budget.

This entrypoint accepts only the canonical train/dev manifest.  It has no
argument for a blind/fresh manifest and never imports the fresh holdout.  The
two model IDs use the same sampler, loss, step schedule, input contract and
checkpoint rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from .models import RawRgbSegmenter, build_model, sha256_file, write_build_receipt
except ImportError:  # pragma: no cover - direct script execution
    from models import RawRgbSegmenter, build_model, sha256_file, write_build_receipt


CLASS_NAMES = ("walkable", "boundary_step_curb", "obstacle", "unknown_nonwalkable")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}
BOUNDARY_ID = CLASS_IDS["boundary_step_curb"]
OBSTACLE_ID = CLASS_IDS["obstacle"]
DEFAULT_SEEDS = (20260711, 20260712, 20260713)
HEAD_WARMUP_STEPS = 100
TOTAL_STEPS = 1200
EVAL_EVERY = 50
BATCH_SIZE = 12


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (repo_root / path).resolve()


def load_shared() -> Any:
    repo_root = Path(__file__).resolve().parents[3]
    scripts = repo_root / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import train_export_sanpo_segmentation as shared

    return shared


def class_weights(shared: Any, train_records: Sequence[Any]) -> np.ndarray:
    counts = shared.class_pixel_counts(train_records)
    values = np.asarray([counts[name] for name in CLASS_NAMES], dtype=np.float64)
    frequencies = values / max(1.0, values.sum())
    raw = 1.0 / np.sqrt(np.maximum(frequencies, 1e-12))
    normalized = raw / max(1e-12, raw.mean())
    weights = np.clip(normalized, 0.35, 4.0).astype(np.float32)
    return weights


class SessionBalancedSampler:
    """The frozen R1 session-balanced guided-crop sampler."""

    def __init__(
        self,
        images: np.ndarray,
        masks: np.ndarray,
        records: Sequence[Any],
        *,
        batch_size: int,
        seed: int,
        guided_crop_fraction: float = 0.70,
        boundary_guided_probability: float = 0.65,
        crop_min_fraction: float = 0.55,
        crop_max_fraction: float = 0.85,
        horizontal_flip_probability: float = 0.50,
    ) -> None:
        if not (len(images) == len(masks) == len(records) and records):
            raise ValueError("sampler inputs must have equal non-zero length")
        self.images = images
        self.masks = masks
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
        self.candidates = {
            session: {
                class_id: [index for index in indices if np.any(masks[index] == class_id)]
                for class_id in (BOUNDARY_ID, OBSTACLE_ID)
            }
            for session, indices in self.sessions.items()
        }
        self.draws = {session: 0 for session in self.session_ids}
        self.guided_attempts = 0
        self.guided_hits = {CLASS_NAMES[BOUNDARY_ID]: 0, CLASS_NAMES[OBSTACLE_ID]: 0}

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
        images: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for _ in range(self.batch_size):
            session = self.session_ids[int(self.rng.integers(len(self.session_ids)))]
            self.draws[session] += 1
            indices = self.sessions[session]
            class_id: int | None = None
            if self.rng.random() < self.guided_crop_fraction:
                self.guided_attempts += 1
                class_id = BOUNDARY_ID if self.rng.random() < self.boundary_guided_probability else OBSTACLE_ID
                if self.candidates[session][class_id]:
                    indices = self.candidates[session][class_id]
            index = indices[int(self.rng.integers(len(indices)))]
            image = self.images[index]
            mask = self.masks[index]
            if class_id is not None and np.any(mask == class_id):
                image, mask = self._crop(image, mask, class_id)
                self.guided_hits[CLASS_NAMES[class_id]] += 1
            else:
                image, mask = image.copy(), mask.copy()
            if self.rng.random() < self.horizontal_flip_probability:
                image = np.flip(image, axis=1).copy()
                mask = np.flip(mask, axis=1).copy()
            images.append(image)
            masks.append(mask)
        return np.stack(images).astype(np.float32), np.stack(masks).astype(np.int64)

    def receipt(self) -> dict[str, Any]:
        return {
            "strategy": "uniform_session_then_uniform_frame_with_guided_crop",
            "session_draws": self.draws,
            "guided_crop_fraction": self.guided_crop_fraction,
            "boundary_guided_probability": self.boundary_guided_probability,
            "crop_fraction_range": [self.crop_min_fraction, self.crop_max_fraction],
            "horizontal_flip_probability": self.horizontal_flip_probability,
            "guided_crop_attempts": self.guided_attempts,
            "guided_crop_hits": self.guided_hits,
        }


def load_arrays(shared: Any, records: Sequence[Any], input_size: int) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for record in records:
        image, mask = shared.load_example(record, input_size)
        images.append(image)
        masks.append(mask)
    return np.stack(images).astype(np.float32), np.stack(masks).astype(np.int64)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_mode(model: RawRgbSegmenter) -> None:
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def batch_tensor(images: np.ndarray, masks: np.ndarray, device: torch.device) -> tuple[Tensor, Tensor]:
    image_tensor = torch.from_numpy(images).to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    mask_tensor = torch.from_numpy(masks).to(device=device, dtype=torch.long)
    return image_tensor, mask_tensor


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


def evaluate(shared: Any, model: RawRgbSegmenter, images: np.ndarray, masks: np.ndarray, device: torch.device) -> dict[str, Any]:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(images), BATCH_SIZE):
            tensor, _ = batch_tensor(images[start:start + BATCH_SIZE], masks[start:start + BATCH_SIZE], device)
            predictions.append(model(tensor).argmax(dim=1).cpu().numpy().astype(np.uint8))
    return shared.confusion_and_metrics(np.concatenate(predictions, axis=0), masks)


def selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    mean_iou = float(metrics["mean_iou"])
    boundary_iou = float(metrics["per_class"]["boundary_step_curb"]["iou"])
    harmonic = 0.0 if min(mean_iou, boundary_iou) <= 0 else 2.0 * mean_iou * boundary_iou / (mean_iou + boundary_iou)
    return harmonic, boundary_iou, mean_iou


def state_cpu(model: nn.Module) -> dict[str, Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def train_one_seed(
    shared: Any,
    model: RawRgbSegmenter,
    train_images: np.ndarray,
    train_masks: np.ndarray,
    train_records: Sequence[Any],
    dev_images: np.ndarray,
    dev_masks: np.ndarray,
    *,
    seed: int,
    device: torch.device,
    output_dir: Path,
    weights: Tensor,
    progress_path: Path,
) -> dict[str, Any]:
    set_seed(seed)
    sampler = SessionBalancedSampler(train_images, train_masks, train_records, batch_size=BATCH_SIZE, seed=seed)
    best_key: tuple[float, float, float] | None = None
    best_metrics: dict[str, Any] | None = None
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    stage_reports: list[dict[str, Any]] = []
    completed_steps = 0
    stages = (
        ("head_warmup", HEAD_WARMUP_STEPS, True, 3e-4, 1.0),
        ("backbone_finetune", TOTAL_STEPS - HEAD_WARMUP_STEPS, False, 5e-5, 0.1),
    )
    for stage_name, stage_steps, head_only, initial_lr, final_ratio in stages:
        trainability = model.set_stage_trainability(head_only=head_only)
        train_mode(model)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=initial_lr)
        scheduler = None
        if final_ratio < 1.0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=stage_steps,
                eta_min=initial_lr * final_ratio,
            )
        stage_started = time.perf_counter()
        stage_best_step = 0
        for stage_step in range(1, stage_steps + 1):
            batch_images, batch_masks = sampler.next_batch()
            tensor, target = batch_tensor(batch_images, batch_masks, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensor)
            loss = composite_loss(logits, target, weights)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            completed_steps += 1
            is_eval = completed_steps % EVAL_EVERY == 0 or completed_steps == TOTAL_STEPS
            if not is_eval:
                continue
            metrics = evaluate(shared, model, dev_images, dev_masks, device)
            key = selection_key(metrics)
            improved = best_key is None or key > best_key
            if improved:
                best_key = key
                best_metrics = metrics
                best_step = completed_steps
                stage_best_step = stage_step
                best_state = state_cpu(model)
            history.append({
                "stage": stage_name,
                "stage_step": stage_step,
                "optimizer_step": completed_steps,
                "loss": float(loss.detach().cpu()),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "selection_score": key[0],
                "dev_mean_iou": float(metrics["mean_iou"]),
                "dev_boundary_iou": float(metrics["per_class"]["boundary_step_curb"]["iou"]),
                "checkpoint_saved": improved,
            })
            write_json(progress_path, {
                "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.training_progress.v1",
                "status": "RUNNING",
                "model_id": model.build_receipt.model_id,
                "seed": seed,
                "completed_steps": completed_steps,
                "total_steps": TOTAL_STEPS,
                "last_selection_score": key[0],
                "last_dev_mean_iou": float(metrics["mean_iou"]),
                "last_dev_boundary_iou": float(metrics["per_class"]["boundary_step_curb"]["iou"]),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            print(
                f"[{model.build_receipt.model_id}] seed={seed} step={completed_steps}/{TOTAL_STEPS} "
                f"selection={key[0]:.6f} boundary_iou={key[1]:.6f}",
                flush=True,
            )
            train_mode(model)
        stage_reports.append({
            "name": stage_name,
            "requested_steps": stage_steps,
            "completed_steps": stage_steps,
            "best_stage_step": stage_best_step,
            "head_only": head_only,
            "initial_learning_rate": initial_lr,
            "final_learning_rate_ratio": final_ratio,
            "trainability": trainability,
            "elapsed_ms": (time.perf_counter() - stage_started) * 1000.0,
        })
    if best_state is None or best_metrics is None or best_key is None:
        raise RuntimeError(f"seed {seed} completed without a dev checkpoint")
    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = seed_dir / "best_fp32.pt"
    torch.save({
        "state_dict": best_state,
        "model_id": model.build_receipt.model_id,
        "implementation_identity": initialization_receipt["implementation_identity"],
        "seed": seed,
        "optimizer_step": best_step,
        "selection_key": list(best_key),
        "dev_mask_metrics": best_metrics,
        "input_contract": {"shape": [1, 256, 256, 3], "dtype": "float32", "range": "0..255"},
    }, checkpoint_path)
    report = {
        "seed": seed,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "selected_optimizer_step": best_step,
        "selection_key": list(best_key),
        "selection_score": best_key[0],
        "dev_mask_metrics": best_metrics,
        "history": history,
        "stages": stage_reports,
        "sampler": sampler.receipt(),
    }
    write_json(seed_dir / "seed_report.json", report)
    write_json(seed_dir / "seed_report.json.sha256.json", {"sha256": sha256_file(seed_dir / "seed_report.json")})
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = resolve(repo_root, args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1":
        raise ValueError("training config is not bound to R1")
    if int(config.get("optimizer_steps", 0)) != TOTAL_STEPS or int(config.get("batch_size", 0)) != BATCH_SIZE:
        raise ValueError("training config does not match the frozen R1 budget")
    if list(config.get("seeds", [])) != list(DEFAULT_SEEDS):
        raise ValueError("training config seed list does not match R1")
    dataset_root = resolve(repo_root, args.dataset_root)
    manifest = dataset_root / str(config["training_manifest"])
    if sha256_file(manifest) != str(config["dataset_manifest_sha256"]):
        raise ValueError("training manifest SHA256 differs from the frozen config")
    shared = load_shared()
    records = shared.load_records(manifest)
    train_records = [record for record in records if record.split == "train"]
    dev_records = [record for record in records if record.split == "dev"]
    if len(train_records) != 400 or len(dev_records) != 200:
        raise ValueError("R1 requires the canonical 400 train + 200 dev records")
    train_images, train_masks = load_arrays(shared, train_records, 256)
    dev_images, dev_masks = load_arrays(shared, dev_records, 256)
    numpy_weights = class_weights(shared, train_records)
    output_dir = resolve(repo_root, args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("R1 GPU training requested but CUDA is unavailable")
    model_kwargs = {
        "ddrnet_architecture_source": resolve(repo_root, args.ddrnet_architecture_source) if args.ddrnet_architecture_source else None,
        "ddrnet_checkpoint": resolve(repo_root, args.ddrnet_checkpoint) if args.ddrnet_checkpoint else None,
        "segformer_checkpoint_dir": resolve(repo_root, args.segformer_checkpoint_dir) if args.segformer_checkpoint_dir else None,
    }
    model = build_model(args.model_id, **model_kwargs)
    initialization_receipt = model.build_receipt.as_dict()
    build_receipt_path = output_dir / "initialization_receipt.json"
    write_build_receipt(build_receipt_path, model)
    del model
    weights = torch.from_numpy(numpy_weights).to(device=device, dtype=torch.float32)
    seed_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    progress_path = output_dir / "training_progress.json"
    write_json(progress_path, {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.training_progress.v1",
        "status": "RUNNING",
        "model_id": args.model_id,
        "completed_steps": 0,
        "total_steps": TOTAL_STEPS * len(DEFAULT_SEEDS),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    for seed in DEFAULT_SEEDS:
        set_seed(seed)
        seed_model = build_model(args.model_id, **model_kwargs).to(device)
        seed_reports.append(
            train_one_seed(
                shared,
                seed_model,
                train_images,
                train_masks,
                train_records,
                dev_images,
                dev_masks,
                seed=seed,
                device=device,
                output_dir=output_dir,
                weights=weights,
                progress_path=progress_path,
            )
        )
        del seed_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selected = max(seed_reports, key=lambda report: tuple(report["selection_key"]))
    selected_seed_checkpoint = Path(selected["checkpoint"])
    selected_payload = torch.load(selected_seed_checkpoint, map_location="cpu", weights_only=False)
    final_checkpoint = output_dir / "fp32_checkpoint.pt"
    torch.save(selected_payload, final_checkpoint)
    final_report = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1_training_report.v1",
        "protocol_id": "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1",
        "model_id": args.model_id,
        "implementation_identity": model.build_receipt.implementation_identity,
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "dataset_root": str(dataset_root),
        "training_manifest": str(manifest.resolve()),
        "training_manifest_sha256": sha256_file(manifest),
        "records": {"train": len(train_records), "dev": len(dev_records)},
        "sessions": {
            "train": sorted({record.session_id for record in train_records}),
            "dev": sorted({record.session_id for record in dev_records}),
        },
        "class_pixel_weights": numpy_weights.tolist(),
        "training_contract": {
            "optimizer": "Adam",
            "optimizer_steps_per_seed": TOTAL_STEPS,
            "head_warmup_steps": HEAD_WARMUP_STEPS,
            "evaluation_every_steps": EVAL_EVERY,
            "batch_size": BATCH_SIZE,
            "seeds": list(DEFAULT_SEEDS),
            "augmentation": {
                "session_balanced_sampling": True,
                "guided_crop_fraction": 0.70,
                "boundary_guided_probability": 0.65,
                "crop_min_fraction": 0.55,
                "crop_max_fraction": 0.85,
                "horizontal_flip_probability": 0.50,
            },
            "loss": {"weighted_cross_entropy": 0.5, "weighted_soft_dice": 0.4, "weighted_focal": 0.1, "focal_gamma": 2.0},
        },
        "initialization": initialization_receipt,
        "seed_reports": seed_reports,
        "selected_seed": selected["seed"],
        "selected_checkpoint": str(final_checkpoint.resolve()),
        "selected_checkpoint_sha256": sha256_file(final_checkpoint),
        "selected_dev_mask_metrics": selected["dev_mask_metrics"],
        "worst_seed": min(seed_reports, key=lambda report: tuple(report["selection_key"]))["seed"],
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "fresh_holdout_consumed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    report_path = output_dir / "training_report.json"
    write_json(report_path, final_report)
    write_json(report_path.with_suffix(".sha256.json"), {"sha256": sha256_file(report_path)})
    write_json(progress_path, {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.training_progress.v1",
        "status": "COMPLETE",
        "model_id": args.model_id,
        "completed_steps": TOTAL_STEPS * len(DEFAULT_SEEDS),
        "total_steps": TOTAL_STEPS * len(DEFAULT_SEEDS),
        "selected_seed": selected["seed"],
        "selected_checkpoint_sha256": sha256_file(final_checkpoint),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    return final_report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=("DDRNet-23-Slim", "SegFormer-B0"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddrnet-architecture-source")
    parser.add_argument("--ddrnet-checkpoint")
    parser.add_argument("--segformer-checkpoint-dir")
    args = parser.parse_args(argv)
    if args.model_id == "DDRNet-23-Slim" and not (args.ddrnet_architecture_source and args.ddrnet_checkpoint):
        parser.error("DDRNet requires --ddrnet-architecture-source and --ddrnet-checkpoint")
    if args.model_id == "SegFormer-B0" and not args.segformer_checkpoint_dir:
        parser.error("SegFormer requires --segformer-checkpoint-dir")
    return args


def main() -> None:
    report = run(parse_args())
    print(json.dumps({
        "status": "COMPLETE",
        "model_id": report["model_id"],
        "selected_seed": report["selected_seed"],
        "selected_checkpoint": report["selected_checkpoint"],
        "selected_checkpoint_sha256": report["selected_checkpoint_sha256"],
        "dev_selection_score": report["selected_dev_mask_metrics"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
