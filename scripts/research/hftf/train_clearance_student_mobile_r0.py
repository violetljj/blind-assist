"""Train Clearance-Student Mobile R0 on the frozen A4 development RGB/metric stream."""

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

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clearance_student_mobile_r0 import (  # noqa: E402
    ClearanceStudentMobileR0,
    clearance_student_loss,
    normalize_bgr_batch,
    parameter_count,
)
from evaluate_dav2_model_variant_gate_r0 import sha256_file  # noqa: E402


def truth_paths(record: dict[str, Any]) -> tuple[Path, Path]:
    rgb = Path(str(record["rgb_path"]))
    root = rgb.parent.parent
    stem = rgb.stem
    return root / "lowres_depth" / f"{stem}.png", root / "confidence" / f"{stem}.png"


def load_batch(records: list[dict[str, Any]], indices: list[int], teacher: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    images, truths, teachers = [], [], []
    for index in indices:
        record = records[index]
        rgb = cv2.imread(str(record["rgb_path"]), cv2.IMREAD_COLOR)
        depth_path, confidence_path = truth_paths(record)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth_raw is None or confidence is None:
            raise OSError(f"S0 asset decode failed: {record['frame_id']}")
        depth = depth_raw.astype(np.float32) / 1000.0
        valid = (confidence == 2) & np.isfinite(depth) & (depth >= 0.25) & (depth <= 6.0)
        if not np.any(valid):
            raise ValueError(f"S0 frame has no metric truth: {record['frame_id']}")
        # Keep invalid pixels finite but outside the loss mask; clearance derivation
        # treats them conservatively as far/unknown.
        truths.append(np.where(np.isfinite(depth), depth, 8.0).astype(np.float32))
        images.append(torch.from_numpy(rgb.transpose(2, 0, 1).copy()))
        teachers.append(np.asarray(teacher[index], dtype=np.float32))
    return normalize_bgr_batch(images).to(device), torch.from_numpy(np.stack(truths)).to(device), torch.from_numpy(np.stack(teachers)).to(device)


def eligible_indices(records: list[dict[str, Any]], role: str) -> list[int]:
    eligible: list[int] = []
    for index, record in enumerate(records):
        if record["role"] != role:
            continue
        depth_path, confidence_path = truth_paths(record)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        confidence = cv2.imread(str(confidence_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None or confidence is None or depth_raw.shape != (192, 256):
            raise OSError(f"S0 eligibility decode failed: {record['frame_id']}")
        valid = (confidence == 2) & (depth_raw >= 250) & (depth_raw <= 6000)
        if np.any(valid):
            eligible.append(index)
    return eligible


def save_checkpoint(model: torch.nn.Module, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, partial)
    with partial.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, path)


def load_encoder_weights(model: ClearanceStudentMobileR0, path: Path) -> None:
    """Accept either torchvision feature-only or prior A4 full-student weights."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("encoder checkpoint is not a state dictionary")
    if any(str(key).startswith("encoder.") for key in state):
        state = {str(key).removeprefix("encoder."): value for key, value in state.items() if str(key).startswith("encoder.")}
    elif any(str(key).startswith("features.") for key in state):
        state = {str(key).removeprefix("features."): value for key, value in state.items() if str(key).startswith("features.")}
    else:
        raise ValueError("encoder checkpoint has no encoder/features prefix")
    model.encoder.load_state_dict(state, strict=True)


def run_epoch(model: torch.nn.Module, records: list[dict[str, Any]], indices: list[int], teacher: np.ndarray, device: torch.device, batch_size: int, loss_config: dict[str, Any], optimizer: torch.optim.Optimizer | None = None, scaler: torch.amp.GradScaler | None = None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    values: dict[str, list[float]] = {}
    for start in range(0, len(indices), batch_size):
        batch = indices[start : start + batch_size]
        images, truth, teacher_batch = load_batch(records, batch, teacher, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            pred = model(images, (192, 256))
            loss, parts = clearance_student_loss(pred, truth, teacher_batch, loss_config)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        for key, value in parts.items():
            values.setdefault(key, []).append(value)
    return {key: statistics.fmean(items) for key, items in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--encoder-weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(f"refusing to reuse output root: {args.output_root}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "clearance-student-mobile-r0":
        raise ValueError("S0 protocol mismatch")
    teacher_manifest = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    if teacher_manifest.get("truth_inputs_opened") is not False:
        raise ValueError("teacher truth firewall failed")
    teacher_path = Path(str(teacher_manifest["teacher_depth"]["path"]))
    if sha256_file(teacher_path) != teacher_manifest["teacher_depth"]["sha256"]:
        raise ValueError("teacher cache hash mismatch")
    records = teacher_manifest["records"]
    train_indices = eligible_indices(records, "train")
    valid_indices = eligible_indices(records, "validation")
    teacher = np.load(teacher_path, mmap_mode="r")
    if teacher.shape != (3000, 192, 256):
        raise ValueError("teacher cache shape mismatch")
    seed = 20260806
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClearanceStudentMobileR0(pretrained=False).to(device)
    load_encoder_weights(model, args.encoder_weights)
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    args.output_root.mkdir(parents=True)
    checkpoint = args.output_root / "clearance_student_mobile_r0_best.pth"
    history, best = [], math.inf
    best_epoch = None
    started = time.perf_counter()
    for epoch in range(args.epochs):
        order = [train_indices[i] for i in np.random.default_rng(seed + epoch).permutation(len(train_indices))]
        train = run_epoch(model, records, order, teacher, device, args.batch_size, {"depth_clamp_m": [0.25, 6.0]}, optimizer, scaler)
        with torch.inference_mode():
            valid = run_epoch(model, records, valid_indices, teacher, device, args.batch_size, {"depth_clamp_m": [0.25, 6.0]})
        row = {"epoch": epoch + 1, "train": train, "validation": valid}
        history.append(row); print(json.dumps(row), flush=True)
        if valid["total"] < best:
            best = valid["total"]; best_epoch = epoch + 1; save_checkpoint(model, checkpoint)
    result = {
        "schema": "blindassist_clearance_student_mobile_r0_training_result",
        "protocol_sha256": sha256_file(args.protocol),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "teacher_truth_inputs_opened": False,
        "parameter_count": params,
        "input_size": 384,
        "epochs": args.epochs,
        "selected_epoch": best_epoch,
        "best_validation_total": best,
        "training_seconds": time.perf_counter() - started,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "history": history,
        "terminal": "CLEARANCE_STUDENT_MOBILE_R0_TRAINING_COMPLETE_DEVELOPMENT_ONLY",
        "production_model_replacement_authorized": False,
    }
    (args.output_root / "training_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
