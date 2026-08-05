#!/usr/bin/env python3
"""Train the single frozen A3 temporal MobileNet depth student."""

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
from train_dav2_392_distilled_student_r0 import distillation_loss


def record_timestamp(record: dict[str, Any]) -> float:
    return float(str(record["frame_id"]).rsplit("_", 1)[1])


def temporal_pairs(records: list[dict[str, Any]], role: str) -> list[tuple[int, int]]:
    pairs = []
    for current in range(1, len(records)):
        previous = current - 1
        left = records[previous]
        right = records[current]
        delta_s = record_timestamp(right) - record_timestamp(left)
        if (
            left["role"] == role
            and right["role"] == role
            and left["video_id"] == right["video_id"]
            and 0.0 < delta_s <= 0.5
        ):
            pairs.append((previous, current))
    return pairs


def load_pair_batch(
    records: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
    teacher_depth: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = []
    targets = []
    for previous, current in pairs:
        for index in (previous, current):
            record = records[index]
            rgb_path = Path(str(record["rgb_path"]))
            bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
            if bgr is None or bgr.shape[:2] != (192, 256):
                raise OSError(f"cannot decode A3 training RGB: {rgb_path}")
            images.append(torch.from_numpy(bgr.transpose(2, 0, 1).copy()))
            targets.append(np.asarray(teacher_depth[index], dtype=np.float32))
    normalized = normalize_bgr_batch(images).to(device)
    target = torch.from_numpy(np.stack(targets)).to(device)
    return normalized, target


def combined_loss(
    prediction: torch.Tensor,
    teacher: torch.Tensor,
    per_frame_config: dict[str, Any],
    temporal_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    per_frame, components = distillation_loss(
        prediction,
        teacher,
        beta=float(per_frame_config["log_depth_smooth_l1_beta"]),
        gradient_weight=float(per_frame_config["log_depth_gradient_l1_weight"]),
        scale_weight=float(per_frame_config["median_log_scale_weight"]),
        minimum_depth=float(per_frame_config["depth_clamp_m"][0]),
        maximum_depth=float(per_frame_config["depth_clamp_m"][1]),
    )
    minimum_depth = float(per_frame_config["depth_clamp_m"][0])
    maximum_depth = float(per_frame_config["depth_clamp_m"][1])
    batch_pairs = prediction.shape[0] // 2
    prediction_log = torch.log(
        prediction.float().clamp(minimum_depth, maximum_depth)
    ).reshape(batch_pairs, 2, *prediction.shape[-2:])
    teacher_log = torch.log(
        teacher.float().clamp(minimum_depth, maximum_depth)
    ).reshape(batch_pairs, 2, *teacher.shape[-2:])
    prediction_delta = prediction_log[:, 1] - prediction_log[:, 0]
    teacher_delta = teacher_log[:, 1] - teacher_log[:, 0]
    temporal = functional.smooth_l1_loss(
        prediction_delta,
        teacher_delta,
        beta=float(temporal_config["log_depth_delta_smooth_l1_beta"]),
    )
    temporal_weight = float(
        temporal_config["log_depth_delta_smooth_l1_weight"]
    )
    total = per_frame + temporal_weight * temporal
    return total, {
        **components,
        "per_frame_total": components["total"],
        "temporal": float(temporal.detach().cpu()),
        "combined_total": float(total.detach().cpu()),
    }


def run_validation(
    model: TemporalMobileDepthStudent,
    records: list[dict[str, Any]],
    pairs: list[tuple[int, int]],
    teacher_depth: np.ndarray,
    device: torch.device,
    per_frame_config: dict[str, Any],
    temporal_config: dict[str, Any],
    pair_batch_size: int = 4,
) -> dict[str, float]:
    model.eval()
    values: dict[str, list[float]] = {}
    with torch.inference_mode():
        for start in range(0, len(pairs), pair_batch_size):
            batch_pairs = pairs[start : start + pair_batch_size]
            images, targets = load_pair_batch(
                records, batch_pairs, teacher_depth, device
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(images, (192, 256))
                _loss, components = combined_loss(
                    prediction, targets, per_frame_config, temporal_config
                )
            for key, value in components.items():
                values.setdefault(key, []).append(value)
    return {key: statistics.fmean(items) for key, items in values.items()}


def save_checkpoint(model: TemporalMobileDepthStudent, path: Path) -> None:
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
    parser.add_argument("--encoder-weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    teacher_manifest = json.loads(
        args.teacher_manifest.read_text(encoding="utf-8")
    )
    if protocol.get("status") != "FROZEN_BEFORE_A3_MODEL_INITIALIZATION_OR_TRAINING":
        raise ValueError("A3 protocol is not frozen")
    model_source = SCRIPT_DIR / "dav2_temporal_mobile_student_r0.py"
    trainer_source = Path(__file__).resolve()
    if sha256_file(model_source) != protocol["implementation"][
        "model_source_sha256"
    ]:
        raise ValueError("A3 model source hash mismatch")
    if sha256_file(trainer_source) != protocol["implementation"][
        "trainer_source_sha256"
    ]:
        raise ValueError("A3 trainer source hash mismatch")
    if sha256_file(args.teacher_manifest) != protocol["teacher_cache"][
        "manifest_sha256"
    ]:
        raise ValueError("A3 teacher manifest hash mismatch")
    if teacher_manifest.get("truth_inputs_opened") is not False:
        raise ValueError("A3 teacher truth firewall failed")
    teacher_path = Path(str(teacher_manifest["teacher_depth"]["path"]))
    if sha256_file(teacher_path) != protocol["teacher_cache"]["depth_sha256"]:
        raise ValueError("A3 teacher depth hash mismatch")
    if sha256_file(args.encoder_weights) != protocol["model"][
        "encoder_weights_sha256"
    ]:
        raise ValueError("A3 encoder weights hash mismatch")
    records = teacher_manifest["records"]
    for record in records:
        rgb_path = Path(str(record["rgb_path"]))
        if sha256_file(rgb_path) != record["rgb_sha256"]:
            raise ValueError(f"A3 RGB hash mismatch: {record['frame_id']}")
    train_pairs = temporal_pairs(records, "train")
    validation_pairs = temporal_pairs(records, "validation")
    if len(train_pairs) != int(protocol["temporal_pairs"]["train_pairs"]):
        raise ValueError("A3 train pair count mismatch")
    if len(validation_pairs) != int(
        protocol["temporal_pairs"]["validation_pairs"]
    ):
        raise ValueError("A3 validation pair count mismatch")
    teacher_depth = np.load(teacher_path, mmap_mode="r")
    if teacher_depth.shape != (3000, 192, 256):
        raise ValueError("A3 teacher cache shape mismatch")

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
        raise ValueError(f"A3 parameter cap exceeded: {parameters}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    pair_batch_size = int(training["pair_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])
    epochs = int(training["epochs"])
    per_frame_config = training["per_frame_loss"]
    temporal_config = training["temporal_loss"]
    args.output_root.mkdir(parents=True)
    checkpoint_path = args.output_root / "a3_temporal_mobile_student_best.pth"
    best_validation = math.inf
    best_epoch = None
    history = []
    started_training = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = [train_pairs[index] for index in rng.permutation(len(train_pairs))]
        optimizer.zero_grad(set_to_none=True)
        train_totals = []
        for batch_number, start in enumerate(
            range(0, len(order), pair_batch_size)
        ):
            batch_pairs = order[start : start + pair_batch_size]
            images, targets = load_pair_batch(
                records, batch_pairs, teacher_depth, device
            )
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(images, (192, 256))
                loss, components = combined_loss(
                    prediction,
                    targets,
                    per_frame_config,
                    temporal_config,
                )
                scaled_loss = loss / accumulation
            scaler.scale(scaled_loss).backward()
            train_totals.append(components["combined_total"])
            final_batch = start + pair_batch_size >= len(order)
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
            validation_pairs,
            teacher_depth,
            device,
            per_frame_config,
            temporal_config,
        )
        epoch_result = {
            "epoch": epoch + 1,
            "train_combined_total_mean": statistics.fmean(train_totals),
            "validation": validation,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result), flush=True)
        if validation["combined_total"] < best_validation:
            best_validation = validation["combined_total"]
            best_epoch = epoch + 1
            save_checkpoint(model, checkpoint_path)
    if best_epoch is None or not checkpoint_path.exists():
        raise RuntimeError("A3 failed to select a checkpoint")
    result = {
        "schema": "blindassist_dav2_temporal_mobile_student_a3_r0_training_result",
        "protocol_sha256": sha256_file(args.protocol),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "teacher_depth_sha256": sha256_file(teacher_path),
        "encoder_weights_sha256": sha256_file(args.encoder_weights),
        "model_source_sha256": sha256_file(model_source),
        "trainer_source_sha256": sha256_file(trainer_source),
        "truth_inputs_opened": False,
        "parameter_count": parameters,
        "train_pairs": len(train_pairs),
        "validation_pairs": len(validation_pairs),
        "seed": seed,
        "epochs_completed": epochs,
        "best_epoch": best_epoch,
        "best_validation_total": best_validation,
        "history": history,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
        },
        "training_duration_s": time.perf_counter() - started_training,
        "terminal": "A3_TEMPORAL_MOBILE_STUDENT_TRAINING_COMPLETE_P1_UNOPENED",
    }
    result_path = args.output_root / "training_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
