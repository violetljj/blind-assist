#!/usr/bin/env python3
"""Run a trainable-tail equal-capacity THOR temporal student canary."""

from __future__ import annotations

import argparse
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
from torchvision.models import mobilenet_v3_small

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    DEFAULT_PRETRAINED,
    DEFAULT_SAMPLES,
    MEAN,
    STD,
    binary_metrics,
    load_jsonl,
    sha256,
    summarize,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d10_thor_magni_"
    "trainable_tail_equal_capacity_temporal_canary_v0"
)
DEFAULT_CACHE = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d10-thor-magni-trainable-rgb-cache-v0/"
    "history_rgb_uint8.npy"
)
SEED = 17
EPOCHS = 8
BATCH_SIZE = 24
BACKBONE_LEARNING_RATE = 2e-5
HEAD_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
UNFROZEN_BACKBONE_START = 9


class TrainableTailTemporalStudent(nn.Module):
    """Fine-tune late spatial features with a causal residual history path."""

    def __init__(self, pretrained: Path) -> None:
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        backbone.load_state_dict(
            torch.load(
                pretrained,
                map_location="cpu",
                weights_only=True,
            )
        )
        self.encoder = backbone.features
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        for module in self.encoder[UNFROZEN_BACKBONE_START:]:
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.temporal_residual_weight = nn.Parameter(
            torch.zeros(4, 576, 1, 1)
        )
        self.temporal_spatial = nn.Conv2d(
            576,
            576,
            kernel_size=3,
            padding=1,
            groups=576,
            bias=False,
        )
        nn.init.dirac_(
            self.temporal_spatial.weight,
            groups=576,
        )
        self.normalization = nn.GroupNorm(32, 576)
        self.projection = nn.Conv2d(576, 32, kernel_size=1)
        self.head = nn.Linear(32 * 4 * 7, 2)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encoder(frames)

    def forward(
        self,
        frames: torch.Tensor,
        *,
        arm: str,
    ) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1] != 5:
            raise ValueError("Expected normalized RGB [batch,5,3,128,224]")
        if arm == "current":
            current = self.encode(frames[:, -1])
            history = current[:, None].expand(-1, 5, -1, -1, -1)
        elif arm == "history":
            batch = frames.shape[0]
            history = self.encode(
                frames.reshape(-1, *frames.shape[2:])
            ).reshape(batch, 5, 576, 4, 7)
            current = history[:, -1]
        else:
            raise ValueError(f"Unknown arm: {arm}")
        residual = history[:, :4] - current[:, None]
        temporal = (
            residual
            * torch.tanh(self.temporal_residual_weight)[None]
        ).mean(dim=1)
        fused = current + self.temporal_spatial(temporal)
        hidden = nnf.hardswish(
            self.projection(self.normalization(fused))
        )
        return self.head(hidden.flatten(1))

    def train(self, mode: bool = True) -> "TrainableTailTemporalStudent":
        super().train(mode)
        if mode:
            for module in self.encoder.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def source_balanced_weights(sources: np.ndarray) -> np.ndarray:
    counts = Counter(str(value) for value in sources)
    values = np.asarray(
        [1.0 / counts[str(value)] for value in sources],
        dtype=np.float32,
    )
    return values / np.mean(values)


def normalize_rgb(values: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = (
        torch.from_numpy(np.array(values, copy=True))
        .to(device, non_blocking=True)
        .permute(0, 1, 4, 2, 3)
        .float()
        .div_(255.0)
    )
    mean = torch.asarray(
        MEAN,
        dtype=tensor.dtype,
        device=device,
    ).view(1, 1, 3, 1, 1)
    std = torch.asarray(
        STD,
        dtype=tensor.dtype,
        device=device,
    ).view(1, 1, 3, 1, 1)
    return (tensor - mean) / std


def evaluate_model(
    model: TrainableTailTemporalStudent,
    cache: np.ndarray,
    indices: np.ndarray,
    arm: str,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    output = []
    with torch.inference_mode():
        for start in range(0, len(indices), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            frames = normalize_rgb(cache[batch_indices], device)
            output.append(
                torch.sigmoid(
                    model(frames, arm=arm)
                ).cpu().numpy()
            )
    return np.concatenate(output)


def train_arm(
    cache: np.ndarray,
    labels: np.ndarray,
    sources: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    arm: str,
    pretrained: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    seed_everything(SEED)
    model = TrainableTailTemporalStudent(pretrained).to(device)
    backbone_parameters = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    head_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_parameters,
                "lr": BACKBONE_LEARNING_RATE,
            },
            {
                "params": head_parameters,
                "lr": HEAD_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )
    train_labels = labels[train_indices].astype(np.float32)
    positive = np.sum(train_labels, axis=0)
    negative = len(train_labels) - positive
    positive_weight = torch.from_numpy(
        (negative / np.maximum(positive, 1.0)).astype(np.float32)
    ).to(device)
    sample_weights = source_balanced_weights(sources[train_indices])

    epoch_losses = []
    for epoch in range(EPOCHS):
        model.train()
        generator = np.random.default_rng(SEED + epoch)
        order = generator.permutation(len(train_indices))
        total_loss = 0.0
        total_weight = 0
        for start in range(0, len(order), BATCH_SIZE):
            positions = order[start : start + BATCH_SIZE]
            batch_indices = train_indices[positions]
            frames = normalize_rgb(cache[batch_indices], device)
            target = torch.from_numpy(
                train_labels[positions]
            ).to(device)
            weight = torch.from_numpy(
                sample_weights[positions]
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames, arm=arm)
            per_target = nnf.binary_cross_entropy_with_logits(
                logits,
                target,
                pos_weight=positive_weight,
                reduction="none",
            )
            loss = torch.mean(per_target * weight[:, None])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(positions)
            total_weight += len(positions)
        epoch_losses.append(total_loss / total_weight)

    probability = evaluate_model(
        model,
        cache,
        test_indices,
        arm,
        device,
    )
    return probability, {
        "epoch_losses": epoch_losses,
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_parameter_count": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "trainable_backbone_parameter_count": sum(
            parameter.numel() for parameter in backbone_parameters
        ),
        "trainable_head_parameter_count": sum(
            parameter.numel() for parameter in head_parameters
        ),
    }


def metrics(
    labels: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    return {
        "proximity": binary_metrics(labels[:, 0], probability[:, 0]),
        "corridor": binary_metrics(labels[:, 1], probability[:, 1]),
    }


def metric_value(result: dict[str, Any], path: str) -> float:
    current: Any = result
    for key in path.split("."):
        current = current[key]
    if current is None:
        raise ValueError(f"Canary metric is not evaluable: {path}")
    return float(current)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or Path(str(args.output) + ".sha256").exists():
        raise ValueError("Refusing to overwrite D10 trainable-tail result")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: row["sample_id"])
    if len(records) != 1078:
        raise ValueError("Expected 1,078 THOR samples")
    cache_report_path = Path(str(args.cache) + ".json")
    cache_report = json.loads(
        cache_report_path.read_text(encoding="utf-8")
    )
    if (
        cache_report["inputs"]["samples_sha256"] != sha256(args.samples)
        or cache_report["design"]["sample_ids"]
        != [record["sample_id"] for record in records]
        or cache_report["output"]["sha256"] != sha256(args.cache)
    ):
        raise ValueError("RGB history cache binding mismatch")
    cache = np.load(args.cache, mmap_mode="r")
    if cache.shape != (1078, 5, 128, 224, 3):
        raise ValueError(f"Unexpected cache shape: {cache.shape}")

    folds = np.asarray([int(record["fold"]) for record in records])
    sources = np.asarray(
        [str(record["source_session_id"]) for record in records]
    )
    labels = np.asarray(
        [
            (
                int(record["target"]["future_proximity_le_1_25m"]),
                int(record["target"]["future_corridor_intrusion"]),
            )
            for record in records
        ],
        dtype=np.int64,
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    metric_paths = (
        "proximity.auroc",
        "proximity.average_precision",
        "corridor.auroc",
        "corridor.average_precision",
    )
    deltas: dict[str, list[float]] = {
        path: [] for path in metric_paths
    }
    fold_rows = []
    for fold in range(5):
        train_indices = np.flatnonzero(folds != fold)
        test_indices = np.flatnonzero(folds == fold)
        if set(sources[train_indices]) & set(sources[test_indices]):
            raise ValueError("Source-session leakage across D10 folds")
        arm_results = {}
        training = {}
        for arm in ("current", "history"):
            probability, diagnostic = train_arm(
                cache,
                labels,
                sources,
                train_indices,
                test_indices,
                arm,
                args.pretrained,
                device,
            )
            arm_results[arm] = metrics(
                labels[test_indices],
                probability,
            )
            training[arm] = diagnostic
        fold_delta = {}
        for path in metric_paths:
            value = (
                metric_value(arm_results["history"], path)
                - metric_value(arm_results["current"], path)
            )
            fold_delta[path] = value
            deltas[path].append(value)
        fold_rows.append(
            {
                "fold": fold,
                "train_sample_count": len(train_indices),
                "heldout_sample_count": len(test_indices),
                "train_source_count": len(set(sources[train_indices])),
                "heldout_source_count": len(set(sources[test_indices])),
                "heldout_sources": sorted(set(sources[test_indices])),
                "current": arm_results["current"],
                "history": arm_results["history"],
                "history_minus_current": fold_delta,
                "training": training,
            }
        )

    aggregate = {
        path: summarize(values)
        for path, values in deltas.items()
    }
    supported = all(
        aggregate[path]["mean"] is not None
        and float(aggregate[path]["mean"]) > 0.0
        and int(aggregate[path]["positive_count"]) >= 3
        for path in metric_paths
    )
    status = (
        "D10_TRAINABLE_TAIL_TEMPORAL_INCREMENT_SUPPORTED_EXPAND_MULTI_SEED"
        if supported
        else "D10_TRAINABLE_TAIL_TEMPORAL_INCREMENT_NOT_SUPPORTED_STOP"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development trainable representation canary",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "cache_path": str(args.cache.resolve()),
            "cache_sha256": cache_report["output"]["sha256"],
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "design": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "backbone_learning_rate": BACKBONE_LEARNING_RATE,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "unfrozen_backbone_start": UNFROZEN_BACKBONE_START,
            "batch_norm": "pretrained running statistics frozen",
            "split": "fixed SHA-256(source_session_id) modulo 5",
            "capacity_control": (
                "identical model parameters; current arm encodes one current "
                "frame and expands it to five features, history arm encodes "
                "five real frames"
            ),
            "selection": "fixed final epoch; no heldout selection",
            "success_gate": (
                "history-minus-current mean > 0 and at least 3/5 positive "
                "folds for proximity/corridor AUROC and AP"
            ),
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(set(sources)),
            "folds": 5,
            "training_units": 10,
        },
        "device": str(device),
        "folds": fold_rows,
        "aggregate_history_minus_current": aggregate,
        "next_action": (
            "expand unchanged model to seeds 23 and 41"
            if supported
            else (
                "close the current trainable-tail successor; do not tune "
                "epochs, unfreeze boundary, learning rate, or head"
            )
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256(args.output)
    Path(str(args.output) + ".sha256").write_text(
        f"{digest}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "status": status,
                "device": str(device),
                "aggregate_history_minus_current": aggregate,
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
