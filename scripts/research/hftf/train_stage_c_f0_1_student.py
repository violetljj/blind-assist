#!/usr/bin/env python3
"""Train one frozen F0.1 SANPO student arm and seed."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as tvf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_geometry_teacher_canary import _sha256
from verify_sanpo_pose_geometry_authority import _load_json


SCHEMA = "blindassist_hftf_stage_c_f0_1_student_arm_seed_training"
READY = "F0_1_SANPO_ARM_SEED_CHECKPOINT_FROZEN"
NOT_EVALUABLE = "F0_1_SANPO_ARM_SEED_TRAINING_NOT_EVALUABLE"
CONTRACT_SCHEMA = (
    "blindassist_hftf_stage_c_sanpo_student_training_execution_contract_f0_1"
)
CONTRACT_SHA256 = (
    "6466d20da8b6972241eff84dbcdc70bff2eebe546ff00503d97c74888fd26a60"
)
CORPUS_VALIDATION_SHA256 = (
    "78a3ebbdad3ed3309577488c5c944454570d8de8b5866d328649168aa72060c2"
)
STUDENT_SAMPLES_SHA256 = (
    "c1bad3b69e769151179762c49586b1b4fa6775a7cfd36a3f0ee9bccc2fbc585f"
)
PRETRAINED_SHA256 = (
    "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f"
)
ARMS = ("SF_CURRENT", "SF_FUTURE", "HIST_FUTURE")
SEEDS = (17, 29, 43)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("Student sample JSONL must contain objects")
        records.append(value)
    return records


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _arm_target(arm: str) -> tuple[str, bool]:
    if arm == "SF_CURRENT":
        return "current", True
    if arm == "SF_FUTURE":
        return "future", True
    if arm == "HIST_FUTURE":
        return "future", False
    raise ValueError(f"Unknown frozen arm: {arm}")


def _decode_label(label: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    known = np.asarray(label["known_target"], dtype=np.float32)
    risk_object = np.asarray(label["risk_target_nullable"], dtype=object)
    if known.shape != (2, 6, 6) or risk_object.shape != (2, 6, 6):
        raise ValueError("Student label shape must be [2,6,6]")
    if not np.isin(known, [0.0, 1.0]).all():
        raise ValueError("Known target must be binary")
    risk = np.zeros((2, 6, 6), dtype=np.float32)
    numeric = np.zeros((2, 6, 6), dtype=bool)
    for index in np.ndindex(risk_object.shape):
        value = risk_object[index]
        if value is None:
            continue
        if value not in (0, 1):
            raise ValueError("Known risk target must be binary")
        numeric[index] = True
        risk[index] = float(value)
    if not np.array_equal(numeric, known.astype(bool)):
        raise ValueError("Risk must be numeric iff KNOWN")
    return torch.from_numpy(risk), torch.from_numpy(known)


def _flip_targets(
    risk: torch.Tensor, known: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.flip(risk, dims=(1,)), torch.flip(known, dims=(1,))


def _sample_augmentation(
    seed: int, epoch: int, sample_id: str
) -> dict[str, Any]:
    digest = hashlib.sha256(
        f"{seed}:{epoch}:{sample_id}".encode("utf-8")
    ).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    operations = ["brightness", "contrast", "saturation", "hue"]
    generator.shuffle(operations)
    return {
        "horizontal_flip": generator.random() < 0.5,
        "brightness": generator.uniform(0.8, 1.2),
        "contrast": generator.uniform(0.8, 1.2),
        "saturation": generator.uniform(0.85, 1.15),
        "hue": generator.uniform(-0.02, 0.02),
        "operation_order": operations,
    }


def _resize_image(image: Image.Image) -> Image.Image:
    return tvf.resize(
        image.convert("RGB"),
        [192, 320],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def _transform_resized_image(
    image: Image.Image,
    augmentation: dict[str, Any] | None,
) -> torch.Tensor:
    value = image
    if augmentation is not None:
        if augmentation["horizontal_flip"]:
            value = tvf.hflip(value)
        for operation in augmentation["operation_order"]:
            value = getattr(tvf, f"adjust_{operation}")(
                value, augmentation[operation]
            )
    tensor = tvf.pil_to_tensor(value).float().div_(255.0)
    return tvf.normalize(tensor, MEAN, STD)


def _selected_history(
    record: dict[str, Any], repeat_anchor: bool
) -> list[dict[str, Any]]:
    history = record["history_rgb"]
    if (
        len(history) != 5
        or [item["relative_time_s"] for item in history]
        != [-0.8, -0.6, -0.4, -0.2, 0.0]
    ):
        raise ValueError("Frozen five-frame history is required")
    return [history[-1]] * 5 if repeat_anchor else history


def _validate_arm_history_images(
    records: list[dict[str, Any]], arm: str
) -> int:
    _, repeat_anchor = _arm_target(arm)
    expected_by_path: dict[str, str] = {}
    for record in records:
        for item in _selected_history(record, repeat_anchor):
            path = str(item["image_path"])
            expected_sha256 = str(item["image_sha256"])
            previous = expected_by_path.setdefault(path, expected_sha256)
            if previous != expected_sha256:
                raise ValueError(
                    f"Conflicting frozen history RGB hash reference: {path}"
                )
    for path, expected_sha256 in sorted(expected_by_path.items()):
        if _sha256(Path(path)) != expected_sha256:
            raise ValueError(f"Frozen history RGB hash mismatch: {path}")
    return len(expected_by_path)


class StudentDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        records: list[dict[str, Any]],
        arm: str,
        *,
        train: bool,
        seed: int,
    ) -> None:
        self.records = records
        self.arm = arm
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.target, self.repeat_anchor = _arm_target(arm)
        self._resized_image_cache: dict[str, tuple[str, Image.Image]] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def _load_resized_image(self, item: dict[str, Any]) -> Image.Image:
        path = str(item["image_path"])
        expected_sha256 = str(item["image_sha256"])
        cached = self._resized_image_cache.get(path)
        if cached is None:
            if _sha256(Path(path)) != expected_sha256:
                raise ValueError(f"Frozen history RGB hash mismatch: {path}")
            with Image.open(path) as image:
                resized = _resize_image(image)
            cached = (expected_sha256, resized)
            self._resized_image_cache[path] = cached
        elif cached[0] != expected_sha256:
            raise ValueError(
                f"Conflicting frozen history RGB hash reference: {path}"
            )
        return cached[1]

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        selected = _selected_history(record, self.repeat_anchor)
        augmentation = (
            _sample_augmentation(
                self.seed, self.epoch, str(record["sample_id"])
            )
            if self.train
            else None
        )
        if self.repeat_anchor:
            frame = _transform_resized_image(
                self._load_resized_image(selected[0]),
                augmentation,
            )
            images = [frame] * 5
        else:
            images = [
                _transform_resized_image(
                    self._load_resized_image(item),
                    augmentation,
                )
                for item in selected
            ]
        risk, known = _decode_label(record["labels"][self.target])
        if augmentation is not None and augmentation["horizontal_flip"]:
            risk, known = _flip_targets(risk, known)
        return torch.stack(images), risk, known


class TemporalStudent(nn.Module):
    def __init__(self, pretrained_path: Path | None) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        if pretrained_path is not None:
            state = torch.load(
                pretrained_path, map_location="cpu", weights_only=True
            )
            backbone.load_state_dict(state, strict=True)
        self.encoder = backbone.features
        self.temporal_depthwise = nn.Conv3d(
            576,
            576,
            kernel_size=(5, 1, 1),
            groups=576,
            bias=False,
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.Hardswish(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.head = nn.Linear(128, 144)

    def train(self, mode: bool = True) -> "TemporalStudent":
        super().train(mode)
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError("Model input must have shape Bx5x3xHxW")
        batch, time, channels, height, width = frames.shape
        features = self.encoder(
            frames.reshape(batch * time, channels, height, width)
        )
        _, feature_channels, feature_height, feature_width = features.shape
        features = features.reshape(
            batch,
            time,
            feature_channels,
            feature_height,
            feature_width,
        ).permute(0, 2, 1, 3, 4)
        fused = self.temporal_depthwise(features).squeeze(2)
        fused = self.pointwise(fused)
        pooled = self.pool(fused).flatten(1)
        output = self.head(self.dropout(pooled)).reshape(
            batch, 2, 2, 6, 6
        )
        return output[:, 0], output[:, 1]


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _class_weights(
    records: list[dict[str, Any]], target: str
) -> list[float]:
    positive = np.zeros(2, dtype=np.int64)
    negative = np.zeros(2, dtype=np.int64)
    for record in records:
        risk, known = _decode_label(record["labels"][target])
        for height in range(2):
            positive[height] += int(
                ((risk[height] == 1) & (known[height] == 1)).sum()
            )
            negative[height] += int(
                ((risk[height] == 0) & (known[height] == 1)).sum()
            )
    if np.any(positive == 0) or np.any(negative == 0):
        raise ValueError("Both risk classes are required in each height")
    return [
        float(np.clip(negative[index] / positive[index], 0.25, 20.0))
        for index in range(2)
    ]


def _losses(
    risk_logits: torch.Tensor,
    known_logits: torch.Tensor,
    risk_target: torch.Tensor,
    known_target: torch.Tensor,
    positive_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_risk = nnf.binary_cross_entropy_with_logits(
        risk_logits,
        risk_target,
        pos_weight=positive_weights,
        reduction="none",
    )
    denominator = known_target.sum().clamp_min(1.0)
    risk_loss = (raw_risk * known_target).sum() / denominator
    known_loss = nnf.binary_cross_entropy_with_logits(
        known_logits, known_target
    )
    return risk_loss + known_loss, risk_loss, known_loss


def _metric_counts(
    probabilities: torch.Tensor,
    risk: torch.Tensor,
    known: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, int]:
    prediction = probabilities >= threshold
    truth = risk >= 0.5
    mask = known >= 0.5
    return {
        "tp": int((prediction & truth & mask).sum()),
        "fp": int((prediction & ~truth & mask).sum()),
        "fn": int((~prediction & truth & mask).sum()),
        "tn": int((~prediction & ~truth & mask).sum()),
    }


def _metrics_from_counts(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn, tn = (
        counts["tp"],
        counts["fp"],
        counts["fn"],
        counts["tn"],
    )
    f1_denominator = 2 * tp + fp + fn
    recall_denominator = tp + fn
    fpr_denominator = fp + tn
    return {
        **counts,
        "f1": 2 * tp / f1_denominator if f1_denominator else 0.0,
        "recall": tp / recall_denominator if recall_denominator else 0.0,
        "false_positive_rate": (
            fp / fpr_denominator if fpr_denominator else 0.0
        ),
    }


@torch.no_grad()
def _evaluate(
    model: TemporalStudent,
    loader: DataLoader,
    device: torch.device,
    positive_weights: torch.Tensor,
) -> dict[str, Any]:
    model.eval()
    micro = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    per_height = [
        {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for _ in range(2)
    ]
    loss_sum = 0.0
    risk_loss_sum = 0.0
    known_loss_sum = 0.0
    known_correct = 0
    known_total = 0
    batches = 0
    for frames, risk, known in loader:
        frames = frames.to(device, non_blocking=True)
        risk = risk.to(device, non_blocking=True)
        known = known.to(device, non_blocking=True)
        risk_logits, known_logits = model(frames)
        loss, risk_loss, known_loss = _losses(
            risk_logits,
            known_logits,
            risk,
            known,
            positive_weights,
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite dev loss")
        probabilities = torch.sigmoid(risk_logits)
        counts = _metric_counts(probabilities, risk, known)
        for key in micro:
            micro[key] += counts[key]
        for height in range(2):
            height_counts = _metric_counts(
                probabilities[:, height],
                risk[:, height],
                known[:, height],
            )
            for key in per_height[height]:
                per_height[height][key] += height_counts[key]
        known_prediction = torch.sigmoid(known_logits) >= 0.5
        known_correct += int(
            (known_prediction == (known >= 0.5)).sum()
        )
        known_total += known.numel()
        loss_sum += float(loss)
        risk_loss_sum += float(risk_loss)
        known_loss_sum += float(known_loss)
        batches += 1
    return {
        "loss": loss_sum / batches,
        "risk_loss": risk_loss_sum / batches,
        "known_loss": known_loss_sum / batches,
        "risk_micro": _metrics_from_counts(micro),
        "risk_by_height": {
            name: _metrics_from_counts(per_height[index])
            for index, name in enumerate(("body", "head"))
        },
        "known_accuracy": known_correct / known_total,
    }


def _validate_inputs(
    contract_path: Path,
    corpus_validation_path: Path,
    student_samples_path: Path,
    pretrained_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if (
        _sha256(contract_path) != CONTRACT_SHA256
        or _sha256(corpus_validation_path) != CORPUS_VALIDATION_SHA256
        or _sha256(student_samples_path) != STUDENT_SAMPLES_SHA256
        or _sha256(pretrained_path) != PRETRAINED_SHA256
    ):
        raise ValueError("Frozen student-training input hash mismatch")
    contract = _load_json(contract_path)
    validation = _load_json(corpus_validation_path)
    records = _load_jsonl(student_samples_path)
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status")
        != "FROZEN_BEFORE_FIRST_F0_1_STUDENT_OPTIMIZATION_STEP"
        or validation.get("terminal")
        != "F0_1_SANPO_TRAIN_DEV_CORPUS_VALIDATED"
        or validation.get("authorization", {}).get(
            "training_execution_contract_may_be_frozen"
        )
        is not True
        or validation.get("authorization", {}).get(
            "heldout_target_materialization_authorized"
        )
        is not False
        or len(records) != 129
        or sum(record["role"] == "train" for record in records) != 90
        or sum(record["role"] == "dev" for record in records) != 39
        or any(record["role"] == "heldout" for record in records)
    ):
        raise ValueError("Frozen student-training authorization mismatch")
    runtime_contract = contract.get("runtime_contract", {})
    torchvision_version = __import__("torchvision").__version__
    if (
        torch.__version__ != runtime_contract.get("torch_version")
        or torchvision_version
        != runtime_contract.get("torchvision_version")
        or runtime_contract.get("device") != "cuda"
        or runtime_contract.get("precision") != "float32_no_amp"
        or runtime_contract.get("deterministic_algorithms") is not True
        or runtime_contract.get("dataloader_workers") != 0
    ):
        raise ValueError("Frozen student-training runtime mismatch")
    return contract, records


def _contract_parent_hashes(contract: dict[str, Any]) -> dict[str, str]:
    required = (
        "f0_protocol",
        "f0_1_protocol",
        "corpus_contract",
        "student_samples",
        "corpus_validation",
    )
    parents = contract.get("parents", {})
    hashes = {
        name: str(parents.get(name, {}).get("sha256", ""))
        for name in required
    }
    if any(
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in hashes.values()
    ):
        raise ValueError("Frozen execution contract parent hash is invalid")
    return hashes


def train(
    contract_path: Path,
    corpus_validation_path: Path,
    student_samples_path: Path,
    pretrained_path: Path,
    arm: str,
    seed: int,
    output_root: Path,
) -> dict[str, Any]:
    contract, records = _validate_inputs(
        contract_path,
        corpus_validation_path,
        student_samples_path,
        pretrained_path,
    )
    if arm not in ARMS or seed not in SEEDS:
        raise ValueError("Arm/seed is outside frozen execution set")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite training run: {output_root}")
    unique_input_image_count = _validate_arm_history_images(records, arm)
    parent_hashes = _contract_parent_hashes(contract)
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen CUDA training device is unavailable")
    _seed_everything(seed)
    target, _ = _arm_target(arm)
    train_records = [
        record for record in records if record["role"] == "train"
    ]
    dev_records = [
        record for record in records if record["role"] == "dev"
    ]
    train_dataset = StudentDataset(
        train_records, arm, train=True, seed=seed
    )
    dev_dataset = StudentDataset(
        dev_records, arm, train=False, seed=seed
    )
    positive_weight_values = _class_weights(train_records, target)
    device = torch.device("cuda")
    model = TemporalStudent(pretrained_path).to(device)
    parameter_count = _parameter_count(model)
    encoder_parameters = list(model.encoder.parameters())
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": 3e-5},
            {"params": head_parameters, "lr": 3e-4},
        ],
        weight_decay=1e-4,
    )
    positive_weights = torch.tensor(
        positive_weight_values,
        dtype=torch.float32,
        device=device,
    ).view(1, 2, 1, 1)
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    history: list[dict[str, Any]] = []
    best_f1 = -1.0
    best_epoch = -1
    best_metrics: dict[str, Any] | None = None
    best_model_state: dict[str, Any] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    all_finite = True
    for epoch in range(1, 31):
        train_dataset.set_epoch(epoch)
        generator = torch.Generator()
        generator.manual_seed(seed * 1000 + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
        )
        model.train()
        total_loss = 0.0
        total_risk_loss = 0.0
        total_known_loss = 0.0
        batches = 0
        for frames, risk, known in train_loader:
            frames = frames.to(device, non_blocking=True)
            risk = risk.to(device, non_blocking=True)
            known = known.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            risk_logits, known_logits = model(frames)
            loss, risk_loss, known_loss = _losses(
                risk_logits,
                known_logits,
                risk,
                known,
                positive_weights,
            )
            if not torch.isfinite(loss):
                all_finite = False
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5.0, error_if_nonfinite=True
            )
            if not torch.isfinite(gradient_norm):
                all_finite = False
                raise RuntimeError("Non-finite gradient norm")
            optimizer.step()
            if any(
                not torch.isfinite(parameter).all()
                for parameter in model.parameters()
            ):
                all_finite = False
                raise RuntimeError("Non-finite model parameter")
            total_loss += float(loss.detach())
            total_risk_loss += float(risk_loss.detach())
            total_known_loss += float(known_loss.detach())
            batches += 1
        dev_metrics = _evaluate(
            model, dev_loader, device, positive_weights
        )
        epoch_result = {
            "epoch": epoch,
            "train_loss": total_loss / batches,
            "train_risk_loss": total_risk_loss / batches,
            "train_known_loss": total_known_loss / batches,
            "dev": dev_metrics,
        }
        history.append(epoch_result)
        dev_f1 = float(dev_metrics["risk_micro"]["f1"])
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            best_epoch = epoch
            best_metrics = copy.deepcopy(dev_metrics)
            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        print(
            json.dumps(
                {
                    "arm": arm,
                    "seed": seed,
                    "epoch": epoch,
                    "train_loss": epoch_result["train_loss"],
                    "dev_f1": dev_f1,
                    "best_epoch": best_epoch,
                    "best_f1": best_f1,
                }
            ),
            flush=True,
        )
    if (
        best_model_state is None
        or best_optimizer_state is None
        or best_metrics is None
        or best_epoch < 1
        or len(history) != 30
    ):
        raise RuntimeError("Frozen checkpoint selection failed")
    checkpoint = {
        "schema": "blindassist_hftf_f0_1_student_checkpoint",
        "arm": arm,
        "seed": seed,
        "selected_epoch": best_epoch,
        "selected_dev_metrics": best_metrics,
        "selected_epoch_metrics": history[best_epoch - 1],
        "parameter_count": parameter_count,
        "unique_input_image_count": unique_input_image_count,
        "model_state_dict": best_model_state,
        "optimizer_state_dict": best_optimizer_state,
        "positive_weights_by_height": {
            "body": positive_weight_values[0],
            "head": positive_weight_values[1],
        },
        "contract_sha256": _sha256(contract_path),
        "corpus_validation_sha256": _sha256(corpus_validation_path),
        "student_samples_sha256": _sha256(student_samples_path),
        "pretrained_checkpoint_sha256": _sha256(pretrained_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "frozen_parent_sha256": parent_hashes,
    }
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial_root = Path(
        tempfile.mkdtemp(
            prefix=f"{output_root.name}.partial-",
            dir=output_root.parent,
        )
    )
    checkpoint_path = partial_root / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha256 = _sha256(checkpoint_path)
    report = {
        "schema": SCHEMA,
        "terminal": READY,
        "workflow_profile": "DEVELOPMENT_STANDARD",
        "claim_ceiling": "TRAIN_DEV_CHECKPOINT_SELECTION_ONLY",
        "arm": arm,
        "seed": seed,
        "target_horizon": target,
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": _sha256(contract_path),
        "corpus_validation_path": str(corpus_validation_path.resolve()),
        "corpus_validation_sha256": _sha256(corpus_validation_path),
        "student_samples_path": str(student_samples_path.resolve()),
        "student_samples_sha256": _sha256(student_samples_path),
        "pretrained_checkpoint_path": str(pretrained_path.resolve()),
        "pretrained_checkpoint_sha256": _sha256(pretrained_path),
        "implementation_sha256": _sha256(Path(__file__).resolve()),
        "frozen_parent_sha256": parent_hashes,
        "runtime": {
            "torch_version": torch.__version__,
            "torchvision_version": __import__("torchvision").__version__,
            "device": "cuda",
            "cuda_device_name": torch.cuda.get_device_name(0),
            "float32_no_amp": True,
            "deterministic_algorithms": True,
        },
        "train_record_count": len(train_records),
        "dev_record_count": len(dev_records),
        "heldout_record_count": 0,
        "parameter_count": parameter_count,
        "unique_input_image_count": unique_input_image_count,
        "positive_weights_by_height": {
            "body": positive_weight_values[0],
            "head": positive_weight_values[1],
        },
        "epochs_completed": len(history),
        "all_losses_gradients_and_parameters_finite": all_finite,
        "selected_epoch": best_epoch,
        "selected_dev_metrics": best_metrics,
        "checkpoint_file": "checkpoint.pt",
        "checkpoint_sha256": checkpoint_sha256,
        "history": history,
        "heldout_firewall": {
            "heldout_teacher_target_loaded": False,
            "heldout_rgb_loaded": False,
            "heldout_student_output_computed": False,
            "heldout_used_for_checkpoint_or_threshold": False,
        },
        "research_mainline_changed": False,
        "default_app_changed": False,
    }
    with (partial_root / "training_report.json").open(
        "x", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    partial_root.replace(output_root)
    return report


def _require_artifacts_output(path: Path) -> Path:
    artifacts_root = (
        Path(__file__).resolve().parents[3] / "artifacts.local"
    ).resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as error:
        raise ValueError(
            f"Output must stay under {artifacts_root}: {resolved}"
        ) from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--corpus-validation", type=Path, required=True)
    parser.add_argument("--student-samples", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        output_root = _require_artifacts_output(args.output_root)
        report = train(
            args.contract.resolve(),
            args.corpus_validation.resolve(),
            args.student_samples.resolve(),
            args.pretrained_checkpoint.resolve(),
            args.arm,
            args.seed,
            output_root,
        )
        print(
            json.dumps(
                {
                    "terminal": report["terminal"],
                    "arm": report["arm"],
                    "seed": report["seed"],
                    "selected_epoch": report["selected_epoch"],
                    "selected_dev_f1": report["selected_dev_metrics"][
                        "risk_micro"
                    ]["f1"],
                    "checkpoint_sha256": report["checkpoint_sha256"],
                    "output_root": str(output_root),
                }
            )
        )
        return 0
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        json.JSONDecodeError,
    ) as error:
        print(json.dumps({"ok": False, "error": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
