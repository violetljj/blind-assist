#!/usr/bin/env python3
"""Train the single frozen A2 392px DA V2 student from teacher-only targets."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.nn import functional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_dav2_model_variant_gate_r0 import sha256_file
from produce_external_rgb_metric_depth_observations import (
    DepthAnythingV2MetricSource,
)


def distillation_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    beta: float,
    gradient_weight: float,
    scale_weight: float,
    minimum_depth: float,
    maximum_depth: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = torch.clamp(prediction.float(), minimum_depth, maximum_depth)
    teacher = torch.clamp(teacher.float(), minimum_depth, maximum_depth)
    log_prediction = torch.log(prediction)
    log_teacher = torch.log(teacher)
    depth_loss = functional.smooth_l1_loss(
        log_prediction, log_teacher, beta=beta
    )
    prediction_dx = log_prediction[:, :, 1:] - log_prediction[:, :, :-1]
    teacher_dx = log_teacher[:, :, 1:] - log_teacher[:, :, :-1]
    prediction_dy = log_prediction[:, 1:, :] - log_prediction[:, :-1, :]
    teacher_dy = log_teacher[:, 1:, :] - log_teacher[:, :-1, :]
    gradient_loss = 0.5 * (
        functional.l1_loss(prediction_dx, teacher_dx)
        + functional.l1_loss(prediction_dy, teacher_dy)
    )
    log_ratio = (log_prediction - log_teacher).flatten(1)
    scale_loss = torch.median(log_ratio, dim=1).values.abs().mean()
    total = depth_loss + gradient_weight * gradient_loss + scale_weight * scale_loss
    return total, {
        "depth": float(depth_loss.detach().cpu()),
        "gradient": float(gradient_loss.detach().cpu()),
        "scale": float(scale_loss.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def load_batch(
    records: list[dict[str, Any]],
    indices: list[int],
    model: torch.nn.Module,
    teacher_depth: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    targets = []
    for index in indices:
        record = records[index]
        rgb_path = Path(str(record["rgb_path"]))
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.shape[:2] != (192, 256):
            raise OSError(f"cannot decode distillation RGB: {rgb_path}")
        image, _shape = model.image2tensor(bgr, 392)
        images.append(image)
        targets.append(np.asarray(teacher_depth[index], dtype=np.float32))
    return (
        torch.cat(images, dim=0).to(device),
        torch.from_numpy(np.stack(targets)).to(device),
    )


def run_validation(
    model: torch.nn.Module,
    records: list[dict[str, Any]],
    indices: list[int],
    teacher_depth: np.ndarray,
    device: torch.device,
    loss_config: dict[str, Any],
    batch_size: int = 4,
) -> dict[str, float]:
    model.eval()
    values: dict[str, list[float]] = {
        "depth": [],
        "gradient": [],
        "scale": [],
        "total": [],
    }
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            images, targets = load_batch(
                records, batch_indices, model, teacher_depth, device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                prediction = model(images)
                prediction = functional.interpolate(
                    prediction[:, None],
                    size=(192, 256),
                    mode="bilinear",
                    align_corners=True,
                )[:, 0]
                _loss, components = distillation_loss(
                    prediction,
                    targets,
                    beta=float(loss_config["log_depth_smooth_l1_beta"]),
                    gradient_weight=float(
                        loss_config["log_depth_gradient_l1_weight"]
                    ),
                    scale_weight=float(loss_config["median_log_scale_weight"]),
                    minimum_depth=float(loss_config["depth_clamp_m"][0]),
                    maximum_depth=float(loss_config["depth_clamp_m"][1]),
                )
            for key, value in components.items():
                values[key].append(value)
    return {key: statistics.fmean(component) for key, component in values.items()}


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save(state, partial)
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--dav2-repo", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    teacher_manifest = json.loads(
        args.teacher_manifest.read_text(encoding="utf-8")
    )
    if protocol.get("status") != "FROZEN_BEFORE_TEACHER_CACHE_OR_STUDENT_TRAINING":
        raise ValueError("A2 protocol is not frozen")
    if teacher_manifest.get("protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("teacher cache is not bound to this A2 protocol")
    if teacher_manifest.get("truth_inputs_opened") is not False:
        raise ValueError("teacher cache truth firewall failed")
    teacher_path = Path(teacher_manifest["teacher_depth"]["path"])
    if sha256_file(teacher_path) != teacher_manifest["teacher_depth"]["sha256"]:
        raise ValueError("teacher depth hash mismatch")
    if sha256_file(args.initial_checkpoint) != protocol["teacher"][
        "checkpoint_sha256"
    ]:
        raise ValueError("initial checkpoint hash mismatch")
    records = teacher_manifest["records"]
    if len(records) != 3000:
        raise ValueError("teacher roster must contain exactly 3000 records")
    for record in records:
        rgb_path = Path(str(record["rgb_path"]))
        if sha256_file(rgb_path) != record["rgb_sha256"]:
            raise ValueError(f"training RGB hash mismatch: {record['frame_id']}")
    train_indices = [
        index for index, row in enumerate(records) if row["role"] == "train"
    ]
    validation_indices = [
        index for index, row in enumerate(records) if row["role"] == "validation"
    ]
    if len(train_indices) != 2400 or len(validation_indices) != 600:
        raise ValueError("teacher roster role counts drifted")
    teacher_depth = np.load(teacher_path, mmap_mode="r")
    if teacher_depth.shape != (3000, 192, 256):
        raise ValueError("teacher cache shape mismatch")

    training = protocol["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    source = DepthAnythingV2MetricSource(
        args.dav2_repo.resolve(),
        args.initial_checkpoint.resolve(),
        "cuda",
        392,
        "fp16",
    )
    model = source.model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    batch_size = int(training["batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    loss_config = training["loss"]
    args.output_root.mkdir(parents=True)
    checkpoint_path = args.output_root / "dav2_392_distilled_best.pth"
    history = []
    best_validation = math.inf
    best_epoch = None
    started_training = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = [train_indices[index] for index in rng.permutation(len(train_indices))]
        optimizer.zero_grad(set_to_none=True)
        train_totals = []
        for batch_number, start in enumerate(range(0, len(order), batch_size)):
            batch_indices = order[start : start + batch_size]
            images, targets = load_batch(
                records, batch_indices, model, teacher_depth, device
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(images)
                prediction = functional.interpolate(
                    prediction[:, None],
                    size=(192, 256),
                    mode="bilinear",
                    align_corners=True,
                )[:, 0]
                loss, components = distillation_loss(
                    prediction,
                    targets,
                    beta=float(loss_config["log_depth_smooth_l1_beta"]),
                    gradient_weight=float(
                        loss_config["log_depth_gradient_l1_weight"]
                    ),
                    scale_weight=float(loss_config["median_log_scale_weight"]),
                    minimum_depth=float(loss_config["depth_clamp_m"][0]),
                    maximum_depth=float(loss_config["depth_clamp_m"][1]),
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            train_totals.append(components["total"])
            final_batch = start + batch_size >= len(order)
            if (batch_number + 1) % accumulation == 0 or final_batch:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(training["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        validation = run_validation(
            model,
            records,
            validation_indices,
            teacher_depth,
            device,
            loss_config,
        )
        epoch_result = {
            "epoch": epoch + 1,
            "train_total_mean": statistics.fmean(train_totals),
            "validation": validation,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        if validation["total"] < best_validation:
            best_validation = validation["total"]
            best_epoch = epoch + 1
            save_checkpoint(model, checkpoint_path)
    duration_s = time.perf_counter() - started_training
    if best_epoch is None or not checkpoint_path.exists():
        raise RuntimeError("no A2 checkpoint was selected")
    result = {
        "schema": "blindassist_dav2_392_distillation_a2_r0_training_result",
        "protocol_sha256": sha256_file(args.protocol),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "teacher_depth_sha256": teacher_manifest["teacher_depth"]["sha256"],
        "initial_checkpoint_sha256": sha256_file(args.initial_checkpoint),
        "truth_inputs_opened": False,
        "seed": seed,
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "history": history,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "training_duration_s": duration_s,
        "terminal": "A2_DISTILLATION_TRAINING_COMPLETE_P1_UNOPENED",
    }
    result_path = args.output_root / "training_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
