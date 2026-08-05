#!/usr/bin/env python3
"""Train the single frozen A4 RGB-D-supervised MobileNet depth student."""

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

from dav2_temporal_mobile_student_r0 import (
    TemporalMobileDepthStudent,
    load_torchvision_encoder_weights,
    normalize_bgr_batch,
    parameter_count,
)
from evaluate_dav2_model_variant_gate_r0 import sha256_file


def truth_paths(record: dict[str, Any]) -> tuple[Path, Path]:
    rgb = Path(str(record["rgb_path"]))
    video_root = rgb.parent.parent
    stem = rgb.stem
    return (
        video_root / "lowres_depth" / f"{stem}.png",
        video_root / "confidence" / f"{stem}.png",
    )


def load_batch(
    records: list[dict[str, Any]],
    indices: list[int],
    teacher_depth: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    truth_values = []
    masks = []
    teachers = []
    for index in indices:
        record = records[index]
        rgb_path = Path(str(record["rgb_path"]))
        depth_path, confidence_path = truth_paths(record)
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or depth_raw is None or confidence is None:
            raise OSError(f"cannot decode A4 frame: {record['frame_id']}")
        if bgr.shape[:2] != (192, 256) or depth_raw.shape != (192, 256):
            raise ValueError(f"A4 frame shape mismatch: {record['frame_id']}")
        truth = depth_raw.astype(np.float32) / 1000.0
        mask = (
            (confidence == 2)
            & np.isfinite(truth)
            & (truth >= 0.25)
            & (truth <= 6.0)
        )
        if not np.any(mask):
            raise ValueError(f"A4 frame has no confidence-2 truth: {record['frame_id']}")
        images.append(torch.from_numpy(bgr.transpose(2, 0, 1).copy()))
        truth_values.append(truth)
        masks.append(mask)
        teachers.append(np.asarray(teacher_depth[index], dtype=np.float32))
    return (
        normalize_bgr_batch(images).to(device),
        torch.from_numpy(np.stack(truth_values)).to(device),
        torch.from_numpy(np.stack(masks)).to(device),
        torch.from_numpy(np.stack(teachers)).to(device),
    )


def masked_gradient_loss(
    prediction_log: torch.Tensor,
    truth_log: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for dimension in (-1, -2):
        pred_delta = torch.diff(prediction_log, dim=dimension)
        truth_delta = torch.diff(truth_log, dim=dimension)
        if dimension == -1:
            pair_mask = mask[..., 1:] & mask[..., :-1]
        else:
            pair_mask = mask[..., 1:, :] & mask[..., :-1, :]
        if torch.any(pair_mask):
            losses.append(functional.l1_loss(pred_delta[pair_mask], truth_delta[pair_mask]))
    if not losses:
        return prediction_log.sum() * 0.0
    return torch.stack(losses).mean()


def rgbd_teacher_loss(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    truth_mask: torch.Tensor,
    teacher: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    minimum, maximum = (float(value) for value in config["depth_clamp_m"])
    prediction_log = torch.log(prediction.float().clamp(minimum, maximum))
    truth_log = torch.log(truth.float().clamp(minimum, maximum))
    teacher_log = torch.log(teacher.float().clamp(minimum, maximum))
    sensor = functional.smooth_l1_loss(
        prediction_log[truth_mask],
        truth_log[truth_mask],
        beta=float(config["sensor_log_depth_smooth_l1_beta"]),
    )
    gradient = masked_gradient_loss(prediction_log, truth_log, truth_mask)
    pred_center = prediction_log - prediction_log.flatten(1).median(dim=1).values[:, None, None]
    teacher_center = teacher_log - teacher_log.flatten(1).median(dim=1).values[:, None, None]
    teacher_structure = functional.smooth_l1_loss(
        pred_center,
        teacher_center,
        beta=float(config["teacher_centered_log_smooth_l1_beta"]),
    )
    total = (
        sensor
        + float(config["sensor_log_gradient_l1_weight"]) * gradient
        + float(config["teacher_centered_log_weight"]) * teacher_structure
    )
    return total, {
        "sensor_log_depth": float(sensor.detach().cpu()),
        "sensor_log_gradient": float(gradient.detach().cpu()),
        "teacher_centered_log": float(teacher_structure.detach().cpu()),
        "total": float(total.detach().cpu()),
    }


def run_validation(
    model: TemporalMobileDepthStudent,
    records: list[dict[str, Any]],
    indices: list[int],
    teacher_depth: np.ndarray,
    device: torch.device,
    config: dict[str, Any],
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    values: dict[str, list[float]] = {}
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            images, truth, mask, teacher = load_batch(records, batch, teacher_depth, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(images, (192, 256))
                _loss, components = rgbd_teacher_loss(prediction, truth, mask, teacher, config)
            for key, value in components.items():
                values.setdefault(key, []).append(value)
    return {key: statistics.fmean(items) for key, items in values.items()}


def save_checkpoint(model: TemporalMobileDepthStudent, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, partial)
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--encoder-weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    teacher_manifest = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_A4_MODEL_INITIALIZATION_OR_TRAINING":
        raise ValueError("A4 protocol is not frozen")
    trainer_source = Path(__file__).resolve()
    model_source = SCRIPT_DIR / "dav2_temporal_mobile_student_r0.py"
    if sha256_file(trainer_source) != protocol["implementation"]["trainer_source_sha256"]:
        raise ValueError("A4 trainer source hash mismatch")
    if sha256_file(model_source) != protocol["implementation"]["model_source_sha256"]:
        raise ValueError("A4 model source hash mismatch")
    if sha256_file(args.source_manifest) != protocol["data"]["source_manifest_sha256"]:
        raise ValueError("A4 source manifest hash mismatch")
    if sha256_file(args.teacher_manifest) != protocol["teacher_cache"]["manifest_sha256"]:
        raise ValueError("A4 teacher manifest hash mismatch")
    teacher_path = Path(str(teacher_manifest["teacher_depth"]["path"]))
    if sha256_file(teacher_path) != protocol["teacher_cache"]["depth_sha256"]:
        raise ValueError("A4 teacher depth hash mismatch")
    if sha256_file(args.encoder_weights) != protocol["model"]["encoder_weights_sha256"]:
        raise ValueError("A4 encoder weights hash mismatch")
    records = teacher_manifest["records"]
    train_indices = [index for index, row in enumerate(records) if row["role"] == "train"]
    validation_indices = [index for index, row in enumerate(records) if row["role"] == "validation"]
    if [len(train_indices), len(validation_indices)] != protocol["data"]["role_counts"]:
        raise ValueError("A4 role counts mismatch")
    for record in records:
        rgb_path = Path(str(record["rgb_path"]))
        if sha256_file(rgb_path) != record["rgb_sha256"]:
            raise ValueError(f"A4 RGB hash mismatch: {record['frame_id']}")
        depth_path, confidence_path = truth_paths(record)
        if not depth_path.is_file() or not confidence_path.is_file():
            raise FileNotFoundError(f"A4 truth assets missing: {record['frame_id']}")
    teacher_depth = np.load(teacher_path, mmap_mode="r")
    if teacher_depth.shape != (3000, 192, 256):
        raise ValueError("A4 teacher cache shape mismatch")

    training = protocol["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda")
    model = TemporalMobileDepthStudent(pretrained=False).to(device)
    load_torchvision_encoder_weights(model, args.encoder_weights)
    parameters = parameter_count(model)
    if parameters > int(protocol["model"]["maximum_parameters"]):
        raise ValueError(f"A4 parameter cap exceeded: {parameters}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])
    loss_config = training["loss"]
    args.output_root.mkdir(parents=True)
    checkpoint_path = args.output_root / "a4_rgbd_mobile_student_best.pth"
    best_validation = math.inf
    best_epoch = None
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = [train_indices[index] for index in np.random.default_rng(seed + epoch).permutation(len(train_indices))]
        train_values = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            images, truth, mask, teacher = load_batch(records, batch, teacher_depth, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(images, (192, 256))
                loss, _components = rgbd_teacher_loss(prediction, truth, mask, teacher, loss_config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            train_values.append(float(loss.detach().cpu()))
        validation = run_validation(model, records, validation_indices, teacher_depth, device, loss_config, batch_size)
        history.append({"epoch": epoch + 1, "train_total": statistics.fmean(train_values), "validation": validation})
        print(json.dumps(history[-1]), flush=True)
        if validation["total"] < best_validation:
            best_validation = validation["total"]
            best_epoch = epoch + 1
            save_checkpoint(model, checkpoint_path)
    result = {
        "schema": "blindassist_dav2_rgbd_mobile_student_a4_r0_training_result",
        "protocol_sha256": sha256_file(args.protocol),
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
        "parameter_count": parameters,
        "selected_epoch": best_epoch,
        "best_validation_total": best_validation,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "p1_truth_opened": False,
    }
    (args.output_root / "training_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
