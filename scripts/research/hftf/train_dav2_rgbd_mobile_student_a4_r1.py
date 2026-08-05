#!/usr/bin/env python3
"""Train A4 R1 after deterministic zero-truth preflight exclusion."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dav2_temporal_mobile_student_r0 import (
    TemporalMobileDepthStudent,
    load_torchvision_encoder_weights,
    parameter_count,
)
from evaluate_dav2_model_variant_gate_r0 import sha256_file
from train_dav2_rgbd_mobile_student_a4_r0 import (
    load_batch,
    rgbd_teacher_loss,
    run_validation,
    save_checkpoint,
    truth_paths,
)


def eligible_indices(records: list[dict[str, object]], role: str) -> list[int]:
    eligible = []
    for index, record in enumerate(records):
        if record["role"] != role:
            continue
        depth_path, confidence_path = truth_paths(record)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        if depth is None or confidence is None or depth.shape != (192, 256):
            raise OSError(f"A4 R1 truth preflight decode failed: {record['frame_id']}")
        mask = (confidence == 2) & (depth >= 250) & (depth <= 6000)
        if np.any(mask):
            eligible.append(index)
    return eligible


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
    if protocol.get("status") != "FROZEN_BEFORE_A4_R1_MODEL_INITIALIZATION_OR_TRAINING":
        raise ValueError("A4 R1 protocol is not frozen")
    trainer_source = Path(__file__).resolve()
    model_source = SCRIPT_DIR / "dav2_temporal_mobile_student_r0.py"
    helper_source = SCRIPT_DIR / "train_dav2_rgbd_mobile_student_a4_r0.py"
    implementation = protocol["implementation"]
    for path, key in (
        (trainer_source, "trainer_source_sha256"),
        (model_source, "model_source_sha256"),
        (helper_source, "r0_helper_source_sha256"),
    ):
        if sha256_file(path) != implementation[key]:
            raise ValueError(f"A4 R1 source hash mismatch: {key}")
    if sha256_file(args.source_manifest) != protocol["data"]["source_manifest_sha256"]:
        raise ValueError("A4 R1 source manifest hash mismatch")
    if sha256_file(args.teacher_manifest) != protocol["teacher_cache"]["manifest_sha256"]:
        raise ValueError("A4 R1 teacher manifest hash mismatch")
    teacher_path = Path(str(teacher_manifest["teacher_depth"]["path"]))
    if sha256_file(teacher_path) != protocol["teacher_cache"]["depth_sha256"]:
        raise ValueError("A4 R1 teacher depth hash mismatch")
    if sha256_file(args.encoder_weights) != protocol["model"]["encoder_weights_sha256"]:
        raise ValueError("A4 R1 encoder weights hash mismatch")
    records = teacher_manifest["records"]
    for record in records:
        rgb_path = Path(str(record["rgb_path"]))
        if sha256_file(rgb_path) != record["rgb_sha256"]:
            raise ValueError(f"A4 R1 RGB hash mismatch: {record['frame_id']}")
    train_indices = eligible_indices(records, "train")
    validation_indices = eligible_indices(records, "validation")
    if [len(train_indices), len(validation_indices)] != protocol["data"]["eligible_role_counts"]:
        raise ValueError("A4 R1 eligible role counts mismatch")
    teacher_depth = np.load(teacher_path, mmap_mode="r")
    if teacher_depth.shape != (3000, 192, 256):
        raise ValueError("A4 R1 teacher cache shape mismatch")

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
        raise ValueError(f"A4 R1 parameter cap exceeded: {parameters}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    batch_size = int(training["batch_size"])
    loss_config = training["loss"]
    args.output_root.mkdir(parents=True)
    checkpoint_path = args.output_root / "a4_r1_rgbd_mobile_student_best.pth"
    best_validation = math.inf
    best_epoch = None
    history = []
    started = time.perf_counter()
    for epoch in range(int(training["epochs"])):
        model.train()
        order = [train_indices[index] for index in np.random.default_rng(seed + epoch).permutation(len(train_indices))]
        train_values = []
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            images, truth, mask, teacher = load_batch(records, batch, teacher_depth, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(images, (192, 256))
                loss, _parts = rgbd_teacher_loss(prediction, truth, mask, teacher, loss_config)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            train_values.append(float(loss.detach().cpu()))
        validation = run_validation(model, records, validation_indices, teacher_depth, device, loss_config, batch_size)
        row = {"epoch": epoch + 1, "train_total": statistics.fmean(train_values), "validation": validation}
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation["total"] < best_validation:
            best_validation = validation["total"]
            best_epoch = epoch + 1
            save_checkpoint(model, checkpoint_path)
    result = {
        "schema": "blindassist_dav2_rgbd_mobile_student_a4_r1_training_result",
        "protocol_sha256": sha256_file(args.protocol),
        "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": sha256_file(checkpoint_path)},
        "parameter_count": parameters,
        "eligible_role_counts": [len(train_indices), len(validation_indices)],
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
