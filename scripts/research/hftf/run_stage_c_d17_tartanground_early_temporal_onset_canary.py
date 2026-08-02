#!/usr/bin/env python3
"""Train an early-temporal five-frame encoder on D16 onset fields.

D17 changes the representation, not the post-hoc head.  Each RGB frame passes
through only the first MobileNet block before ordered temporal differences are
encoded with a depthwise-separable 3D stem.  The fused current feature then
passes through the rest of MobileNet and a dense 6x6 onset head.

The current comparator has exactly the same parameters and initialization but
receives the current frame five times.  Its temporal differences are therefore
identically zero.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
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

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    binary_metrics,
    load_jsonl,
    sha256,
    summarize,
)
from train_stage_c_d5_tartanground_development_student import (
    MEAN,
    STD,
    augmentation,
)


SCHEMA = "blindassist_hftf_stage_c_d17_early_temporal_onset_canary_v0"
ARMS = ("current", "history")
TARGETS = ("near_body", "near_head", "far_body", "far_head")
SEED_CANARY = (17,)
EPOCHS = 30
BATCH_SIZE = 8
ENCODER_LEARNING_RATE = 2e-5
TEMPORAL_HEAD_LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-4
MOTION_RESIDUAL_SCALE = 0.25
PRIMARY_AUROC_FLOOR = 0.01
PRIMARY_AP_FLOOR = 0.005
PRIMARY_POSITIVE_FOLDS = 2
TARGET_BREADTH = 3
SAMPLE_NONINFERIORITY_FLOOR = -0.005
DEFAULT_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d16-tartanground-future-onset-v0/samples.jsonl"
)
DEFAULT_PRETRAINED = Path(
    "artifacts.local/models/hftf/torch/hub/checkpoints/"
    "mobilenet_v3_small-047dcff4.pth"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d17-tartanground-early-temporal-onset-canary-v0/"
    "report.json"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def transform_image(
    image: Image.Image,
    parameters: dict[str, Any] | None,
) -> torch.Tensor:
    value = image
    if parameters is not None:
        if parameters["horizontal_flip"]:
            value = tvf.hflip(value)
        for operation in parameters["operation_order"]:
            value = getattr(tvf, f"adjust_{operation}")(
                value,
                parameters[operation],
            )
    tensor = tvf.pil_to_tensor(value).float().div_(255.0)
    return tvf.normalize(tensor, MEAN, STD)


def decode_onset(
    record: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    target = record["future_onset_target"]
    onset = np.asarray(target["cell_onset"], dtype=np.float32)
    eligible = np.asarray(target["cell_eligible"], dtype=np.float32)
    if onset.shape != (2, 2, 6, 6):
        raise ValueError("D17 cell_onset must have shape [2,2,6,6]")
    if eligible.shape != onset.shape:
        raise ValueError("D17 cell eligibility shape mismatch")
    if not np.all((onset == 0.0) | (onset == 1.0)):
        raise ValueError("D17 onset values must be binary")
    if not np.all((eligible == 0.0) | (eligible == 1.0)):
        raise ValueError("D17 eligibility values must be binary")
    if np.any(onset > eligible):
        raise ValueError("D17 onset cell cannot be ineligible")
    return (
        torch.from_numpy(onset.reshape(4, 6, 6)),
        torch.from_numpy(eligible.reshape(4, 6, 6)),
    )


class OnsetDataset(
    Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]
):
    def __init__(
        self,
        records: list[dict[str, Any]],
        arm: str,
        *,
        train: bool,
        seed: int,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"Unknown arm: {arm}")
        self.records = records
        self.arm = arm
        self.train_mode = train
        self.seed = seed
        self.epoch = 0
        self.cache: dict[str, Image.Image] = {}

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def load_image(self, path: str) -> Image.Image:
        value = self.cache.get(path)
        if value is None:
            with Image.open(path) as source:
                value = tvf.resize(
                    source.convert("RGB"),
                    [128, 224],
                    interpolation=InterpolationMode.BILINEAR,
                    antialias=True,
                )
            self.cache[path] = value
        return value

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        history = record["history_rgb"]
        if len(history) != 5:
            raise ValueError("D17 requires exactly five RGB frames")
        selected = [history[-1]] * 5 if self.arm == "current" else history
        parameters = (
            augmentation(self.seed, self.epoch, record["sample_id"])
            if self.train_mode
            else None
        )
        frames = torch.stack(
            [
                transform_image(
                    self.load_image(str(item["image_path"])),
                    parameters,
                )
                for item in selected
            ]
        )
        onset, eligible = decode_onset(record)
        if parameters is not None and parameters["horizontal_flip"]:
            onset = torch.flip(onset, dims=(2,))
            eligible = torch.flip(eligible, dims=(2,))
        return frames, onset, eligible, index


class EarlyTemporalOnsetEncoder(nn.Module):
    """Fuse ordered low-level motion before the main spatial encoder."""

    def __init__(self, pretrained_path: Path) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        backbone.load_state_dict(
            torch.load(
                pretrained_path,
                map_location="cpu",
                weights_only=True,
            ),
            strict=True,
        )
        layers = list(backbone.features.children())
        self.low_encoder = layers[0]
        self.high_encoder = nn.Sequential(*layers[1:])
        self.temporal_motion = nn.Sequential(
            nn.Conv3d(
                16,
                16,
                kernel_size=(3, 5, 5),
                padding=(1, 2, 2),
                groups=16,
                bias=False,
            ),
            nn.GroupNorm(4, 16),
            nn.Hardswish(),
            nn.Conv3d(16, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.Hardswish(),
            nn.Conv3d(
                32,
                32,
                kernel_size=(4, 1, 1),
                groups=32,
                bias=False,
            ),
        )
        self.motion_output = nn.Conv2d(32, 16, kernel_size=1, bias=False)
        nn.init.zeros_(self.motion_output.weight)
        self.field_projection = nn.Sequential(
            nn.Conv2d(576, 128, kernel_size=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.Hardswish(),
        )
        self.field_head = nn.Conv2d(128, 4, kernel_size=1)

    def train(self, mode: bool = True) -> "EarlyTemporalOnsetEncoder":
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError("D17 input must have shape Bx5x3xHxW")
        batch, time, channels, height, width = frames.shape
        low = self.low_encoder(
            frames.reshape(batch * time, channels, height, width)
        )
        _, low_channels, low_height, low_width = low.shape
        if low_channels != 16:
            raise ValueError("Unexpected MobileNet low-level channel count")
        low = low.reshape(
            batch,
            time,
            low_channels,
            low_height,
            low_width,
        ).permute(0, 2, 1, 3, 4)
        ordered_motion = low[:, :, 1:] - low[:, :, :-1]
        motion_residual = self.temporal_motion(ordered_motion).squeeze(2)
        motion_residual = self.motion_output(motion_residual)
        fused = low[:, :, -1] + MOTION_RESIDUAL_SCALE * torch.tanh(
            motion_residual
        )
        encoded = self.high_encoder(fused)
        projected = self.field_projection(encoded)
        projected = nnf.interpolate(
            projected,
            size=(6, 6),
            mode="bilinear",
            align_corners=False,
        )
        return self.field_head(projected)


def model_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_records(records: list[dict[str, Any]]) -> None:
    if len(records) != 495:
        raise ValueError("D17 expects the exact 495-sample D16 corpus")
    sample_ids = [str(record["sample_id"]) for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("D17 sample IDs must be unique")
    folds = {int(record["environment_fold"]) for record in records}
    if folds != {0, 1, 2}:
        raise ValueError("D17 requires the inherited D16 folds 0,1,2")
    for record in records:
        decode_onset(record)
        history = record["history_rgb"]
        if len(history) != 5:
            raise ValueError("D17 record does not have five RGB frames")
        for frame in history:
            if not Path(str(frame["image_path"])).is_file():
                raise FileNotFoundError(str(frame["image_path"]))
    for fold in range(3):
        heldout = {
            str(record["environment"])
            for record in records
            if int(record["environment_fold"]) == fold
        }
        training = {
            str(record["environment"])
            for record in records
            if int(record["environment_fold"]) != fold
        }
        if heldout & training:
            raise ValueError("D17 environment leakage across folds")


def positive_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    positive = np.zeros(4, dtype=np.float64)
    eligible = np.zeros(4, dtype=np.float64)
    for record in records:
        onset, mask = decode_onset(record)
        positive += onset.numpy().sum(axis=(1, 2))
        eligible += mask.numpy().sum(axis=(1, 2))
    negative = eligible - positive
    if np.any(positive <= 0.0) or np.any(negative <= 0.0):
        raise ValueError("Every D17 training target needs both cell classes")
    return torch.from_numpy((negative / positive).astype(np.float32))


def environment_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    counts = Counter(str(record["environment"]) for record in records)
    total = len(records)
    environment_count = len(counts)
    return torch.tensor(
        [
            total / (environment_count * counts[str(record["environment"])])
            for record in records
        ],
        dtype=torch.float32,
    )


def masked_cell_loss(
    logits: torch.Tensor,
    onset: torch.Tensor,
    eligible: torch.Tensor,
    sample_weight: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    raw = nnf.binary_cross_entropy_with_logits(
        logits,
        onset,
        reduction="none",
        pos_weight=pos_weight.reshape(1, 4, 1, 1),
    )
    weight = eligible * sample_weight.reshape(-1, 1, 1, 1)
    denominator = weight.sum()
    if denominator <= 0.0:
        raise ValueError("D17 batch has no eligible onset cells")
    return (raw * weight).sum() / denominator


@torch.inference_mode()
def predict(
    model: nn.Module,
    records: list[dict[str, Any]],
    arm: str,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = OnsetDataset(records, arm, train=False, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model.eval()
    probability_rows = []
    onset_rows = []
    eligibility_rows = []
    for frames, onset, eligible, _ in loader:
        logits = model(frames.to(device, non_blocking=True))
        probability_rows.append(torch.sigmoid(logits).cpu().numpy())
        onset_rows.append(onset.numpy())
        eligibility_rows.append(eligible.numpy())
    return (
        np.concatenate(probability_rows),
        np.concatenate(onset_rows).astype(np.int64),
        np.concatenate(eligibility_rows).astype(bool),
    )


def metric_mean(rows: list[dict[str, float | None]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for metric in ("auroc", "average_precision"):
        values = [
            float(row[metric])
            for row in rows
            if row[metric] is not None
        ]
        output[metric] = float(np.mean(values)) if values else None
        output[f"{metric}_unit_count"] = len(values)
    return output


def evaluate_predictions(
    records: list[dict[str, Any]],
    probability: np.ndarray,
    onset: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    if probability.shape != (len(records), 4, 6, 6):
        raise ValueError("D17 probability shape mismatch")
    cell_by_target = {}
    sample_by_target = {}
    for target_index, target in enumerate(TARGETS):
        mask = eligible[:, target_index]
        cell_by_target[target] = binary_metrics(
            onset[:, target_index][mask],
            probability[:, target_index][mask],
        )
        sample_eligible = np.asarray(
            [
                bool(
                    record["future_onset_target"]["sample_eligible"][target]
                )
                for record in records
            ]
        )
        sample_target = np.asarray(
            [
                int(record["future_onset_target"]["sample_onset"][target])
                for record in records
            ],
            dtype=np.int64,
        )
        sample_score = probability[:, target_index].max(axis=(1, 2))
        sample_by_target[target] = binary_metrics(
            sample_target[sample_eligible],
            sample_score[sample_eligible],
        )

    environment_target_rows = []
    environments = sorted(
        {str(record["environment"]) for record in records}
    )
    for environment in environments:
        indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if str(record["environment"]) == environment
            ],
            dtype=np.int64,
        )
        for target_index, target in enumerate(TARGETS):
            mask = eligible[indices, target_index]
            row = binary_metrics(
                onset[indices, target_index][mask],
                probability[indices, target_index][mask],
            )
            environment_target_rows.append(
                {
                    "environment": environment,
                    "target": target,
                    **row,
                }
            )
    return {
        "cell_by_target": cell_by_target,
        "macro_cell": metric_mean(list(cell_by_target.values())),
        "sample_by_target": sample_by_target,
        "macro_sample": metric_mean(list(sample_by_target.values())),
        "environment_target_cell": environment_target_rows,
        "environment_macro_cell": metric_mean(environment_target_rows),
    }


def train_arm(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    pretrained_path: Path,
    arm: str,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], nn.Module]:
    seed_everything(seed)
    model = EarlyTemporalOnsetEncoder(pretrained_path).to(device)
    initial_sha256 = model_sha256(model)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    temporal_parameters = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(("temporal_motion.", "motion_output."))
    )
    dataset = OnsetDataset(
        train_records,
        arm,
        train=True,
        seed=seed,
    )
    env_weight = environment_weights(train_records)
    pos_weight = positive_weights(train_records).to(device)
    encoder_parameters = [
        *model.low_encoder.parameters(),
        *model.high_encoder.parameters(),
    ]
    temporal_head_parameters = [
        *model.temporal_motion.parameters(),
        *model.motion_output.parameters(),
        *model.field_projection.parameters(),
        *model.field_head.parameters(),
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": temporal_head_parameters,
                "lr": TEMPORAL_HEAD_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )
    epoch_rows = []
    for epoch in range(1, EPOCHS + 1):
        dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
        )
        model.train()
        loss_total = 0.0
        batch_count = 0
        for frames, onset, eligible, indices in loader:
            frames = frames.to(device, non_blocking=True)
            onset = onset.to(device, non_blocking=True)
            eligible = eligible.to(device, non_blocking=True)
            sample_weight = env_weight[indices].to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames)
            loss = masked_cell_loss(
                logits,
                onset,
                eligible,
                sample_weight,
                pos_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("D17 encountered a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            loss_total += float(loss.detach())
            batch_count += 1
        mean_loss = loss_total / batch_count
        epoch_rows.append({"epoch": epoch, "mean_train_loss": mean_loss})
        if epoch in {1, EPOCHS}:
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "seed": seed,
                        "epoch": epoch,
                        "mean_train_loss": mean_loss,
                    }
                ),
                flush=True,
            )
    probability, onset, eligible = predict(
        model,
        test_records,
        arm,
        seed,
        device,
    )
    metrics = evaluate_predictions(
        test_records,
        probability,
        onset,
        eligible,
    )
    diagnostic = {
        "initial_model_sha256": initial_sha256,
        "trainable_parameters": trainable_parameters,
        "temporal_parameters": temporal_parameters,
        "positive_weights": pos_weight.detach().cpu().tolist(),
        "first_epoch_loss": epoch_rows[0]["mean_train_loss"],
        "final_epoch_loss": epoch_rows[-1]["mean_train_loss"],
        "fixed_final_epoch": EPOCHS,
    }
    return metrics, diagnostic, model


def nested_metric(row: dict[str, Any], path: str) -> float:
    value: Any = row
    for part in path.split("."):
        value = value[part]
    if value is None:
        raise ValueError(f"D17 metric is not evaluable: {path}")
    return float(value)


def mean_metric_delta(
    units: list[dict[str, Any]],
    path: str,
) -> dict[str, Any]:
    by_fold = []
    for fold in range(3):
        values = [
            nested_metric(unit["history_minus_current"], path)
            for unit in units
            if int(unit["fold"]) == fold
        ]
        if not values:
            raise ValueError(f"D17 fold {fold} has no values for {path}")
        by_fold.append(float(np.mean(values)))
    return {
        **summarize(by_fold),
        "by_fold_seed_mean": by_fold,
    }


def build_gate(aggregate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    primary_auroc = aggregate["environment_macro_cell.auroc"]
    primary_ap = aggregate[
        "environment_macro_cell.average_precision"
    ]
    positive_targets = sum(
        aggregate[f"cell_by_target.{target}.auroc"]["mean"] > 0.0
        and aggregate[
            f"cell_by_target.{target}.average_precision"
        ]["mean"]
        > 0.0
        for target in TARGETS
    )
    sample_auroc = aggregate["macro_sample.auroc"]
    sample_ap = aggregate["macro_sample.average_precision"]
    checks = {
        "primary_auroc_effect": (
            primary_auroc["mean"] >= PRIMARY_AUROC_FLOOR
        ),
        "primary_ap_effect": primary_ap["mean"] >= PRIMARY_AP_FLOOR,
        "primary_auroc_positive_folds": (
            primary_auroc["positive_count"] >= PRIMARY_POSITIVE_FOLDS
        ),
        "primary_ap_positive_folds": (
            primary_ap["positive_count"] >= PRIMARY_POSITIVE_FOLDS
        ),
        "target_breadth": positive_targets >= TARGET_BREADTH,
        "sample_auroc_noninferiority": (
            sample_auroc["mean"] >= SAMPLE_NONINFERIORITY_FLOOR
        ),
        "sample_ap_noninferiority": (
            sample_ap["mean"] >= SAMPLE_NONINFERIORITY_FLOOR
        ),
    }
    return {
        "supported": all(checks.values()),
        "checks": checks,
        "observed_positive_targets_for_both_cell_metrics": positive_targets,
        "frozen_thresholds": {
            "primary_environment_macro_cell_auroc_mean_floor": (
                PRIMARY_AUROC_FLOOR
            ),
            "primary_environment_macro_cell_ap_mean_floor": PRIMARY_AP_FLOOR,
            "primary_positive_folds": PRIMARY_POSITIVE_FOLDS,
            "cell_targets_positive_on_both_metrics": TARGET_BREADTH,
            "macro_sample_metric_noninferiority_floor": (
                SAMPLE_NONINFERIORITY_FLOOR
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEED_CANARY),
    )
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("D17 output is non-overwriting")
    if not args.samples.is_file() or not args.pretrained.is_file():
        raise FileNotFoundError("D17 samples or pretrained weights missing")
    if not torch.cuda.is_available():
        raise RuntimeError("D17 requires CUDA")
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("D17 seeds must be non-empty and unique")
    records = load_jsonl(args.samples)
    validate_records(records)
    device = torch.device("cuda")
    metric_paths = [
        "environment_macro_cell.auroc",
        "environment_macro_cell.average_precision",
        "macro_cell.auroc",
        "macro_cell.average_precision",
        "macro_sample.auroc",
        "macro_sample.average_precision",
    ]
    metric_paths.extend(
        f"cell_by_target.{target}.{metric}"
        for target in TARGETS
        for metric in ("auroc", "average_precision")
    )
    units = []
    checkpoint_rows = []
    for fold in range(3):
        train_records = [
            record
            for record in records
            if int(record["environment_fold"]) != fold
        ]
        test_records = [
            record
            for record in records
            if int(record["environment_fold"]) == fold
        ]
        for seed in seeds:
            arm_metrics = {}
            arm_diagnostics = {}
            history_model = None
            for arm in ARMS:
                metrics, diagnostic, model = train_arm(
                    train_records,
                    test_records,
                    args.pretrained,
                    arm,
                    seed,
                    device,
                )
                arm_metrics[arm] = metrics
                arm_diagnostics[arm] = diagnostic
                if arm == "history":
                    history_model = model
            if (
                arm_diagnostics["current"]["initial_model_sha256"]
                != arm_diagnostics["history"]["initial_model_sha256"]
            ):
                raise ValueError("D17 paired arms did not share initialization")
            if (
                arm_diagnostics["current"]["trainable_parameters"]
                != arm_diagnostics["history"]["trainable_parameters"]
            ):
                raise ValueError("D17 paired arms differ in parameter count")
            delta: dict[str, Any] = {}
            for path in metric_paths:
                cursor = delta
                parts = path.split(".")
                for part in parts[:-1]:
                    cursor = cursor.setdefault(part, {})
                cursor[parts[-1]] = (
                    nested_metric(arm_metrics["history"], path)
                    - nested_metric(arm_metrics["current"], path)
                )
            heldout_environments = sorted(
                {str(record["environment"]) for record in test_records}
            )
            units.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "heldout_environments": heldout_environments,
                    "current": arm_metrics["current"],
                    "history": arm_metrics["history"],
                    "history_minus_current": delta,
                    "training": arm_diagnostics,
                }
            )
            if history_model is None:
                raise RuntimeError("D17 history model missing")
            checkpoint_path = (
                args.output.parent
                / "checkpoints"
                / f"fold-{fold}"
                / f"seed-{seed}-history.pt"
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "schema": SCHEMA,
                    "fold": fold,
                    "seed": seed,
                    "arm": "history",
                    "heldout_environments": heldout_environments,
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in history_model.state_dict().items()
                    },
                },
                checkpoint_path,
            )
            checkpoint_rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "path": str(checkpoint_path.resolve()),
                    "sha256": sha256(checkpoint_path),
                }
            )
            del history_model, model
            torch.cuda.empty_cache()
    aggregate = {
        path: mean_metric_delta(units, path)
        for path in metric_paths
    }
    gate = build_gate(aggregate)
    phase = "CANARY" if seeds == SEED_CANARY else "MULTI_SEED"
    status = (
        f"D17_EARLY_TEMPORAL_ONSET_{phase}_SUPPORTED"
        if gate["supported"]
        else f"D17_EARLY_TEMPORAL_ONSET_{phase}_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development synthetic onset representation canary",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "design": {
            "representation": (
                "ordered adjacent-frame differences after MobileNet block 0; "
                "depthwise-separable 3D fusion before all remaining encoder "
                "blocks; dense four-target 6x6 onset field"
            ),
            "comparator": (
                "identical model and initialization with current frame "
                "repeated five times, making ordered motion exactly zero"
            ),
            "folds": "inherited D16 three environment-heldout folds",
            "seeds": list(seeds),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "encoder_learning_rate": ENCODER_LEARNING_RATE,
            "temporal_head_learning_rate": TEMPORAL_HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "motion_residual_scale": MOTION_RESIDUAL_SCALE,
            "loss": (
                "cell-eligibility-masked, environment-balanced BCEWithLogits "
                "with train-fold per-target positive weights"
            ),
            "selection": "fixed final epoch; no heldout model selection",
            "primary_metric": (
                "heldout environment x target macro cell AUROC and AP"
            ),
        },
        "counts": {
            "samples": len(records),
            "environments": len(
                {str(record["environment"]) for record in records}
            ),
            "folds": 3,
            "seeds": len(seeds),
            "paired_units": len(units),
            "training_runs": len(units) * 2,
        },
        "device": str(device),
        "gate": gate,
        "units": units,
        "aggregate_fold_seed_mean_history_minus_current": aggregate,
        "history_checkpoints": checkpoint_rows,
        "next_action": (
            "expand the same frozen design to seeds 23 and 41"
            if gate["supported"] and phase == "CANARY"
            else (
                "transfer the frozen early-temporal encoder to THOR and JRDB"
                if gate["supported"]
                else (
                    "retain the onset task but stop this unaligned early-"
                    "temporal convolution candidate"
                )
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    sidecar.write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "gate": gate,
                "aggregate": aggregate,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
