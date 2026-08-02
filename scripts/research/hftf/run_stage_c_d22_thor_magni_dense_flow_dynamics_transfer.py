#!/usr/bin/env python3
"""Replicate D20 dense-flow dynamics on THOR true-future onset."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, Dataset

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    binary_metrics,
    load_jsonl,
    sha256,
)
from extract_stage_c_d14_thor_magni_explicit_motion_features import (
    DEFAULT_RGB_CACHE,
)
from extract_stage_c_d18_tartanground_backward_raft_flow import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PAIRS_PER_SAMPLE,
    sample_id_digest,
)
from extract_stage_c_d22_thor_magni_backward_raft_flow import (
    EXPECTED_SAMPLES,
)
from run_stage_c_d13_thor_magni_future_onset_temporal_baseline import (
    DEFAULT_SAMPLES,
)
from run_stage_c_d17_tartanground_early_temporal_onset_canary import (
    BATCH_SIZE,
    DEFAULT_PRETRAINED,
    ENCODER_LEARNING_RATE,
    EPOCHS,
    TEMPORAL_HEAD_LEARNING_RATE,
    WEIGHT_DECAY,
    model_sha256,
    seed_everything,
)
from run_stage_c_d18_tartanground_flow_aligned_onset_canary import (
    backward_warp,
)
from run_stage_c_d20_tartanground_dense_flow_dynamics import (
    DenseFlowDynamicsOnsetEncoder,
    dense_dynamics_tensor,
)


SCHEMA = "blindassist_hftf_stage_c_d22_thor_magni_dense_flow_transfer_v0"
ARMS = ("current", "history")
TARGETS = ("proximity", "corridor")
SEED_CANARY = (17,)
DEFAULT_FLOW = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d22-thor-magni-backward-raft-flow-v0/"
    "current_to_history_flow_f16.npy"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d22-thor-magni-dense-flow-transfer-v0/"
    "report.json"
)


class ThorDenseFlowDataset(
    Dataset[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
        ]
    ]
):
    def __init__(
        self,
        records: list[dict[str, Any]],
        arm: str,
        rgb_cache_path: Path,
        flow_path: Path,
        *,
        train: bool,
        seed: int,
    ) -> None:
        if arm not in ARMS:
            raise ValueError(f"Unknown D22 arm: {arm}")
        self.records = records
        self.arm = arm
        self.train_mode = train
        self.seed = seed
        self.epoch = 0
        self.rgb = np.load(rgb_cache_path, mmap_mode="r")
        self.flow = np.load(flow_path, mmap_mode="r")
        if self.rgb.shape != (EXPECTED_SAMPLES, 5, 128, 224, 3):
            raise ValueError("D22 RGB cache shape mismatch")
        expected_flow = (
            EXPECTED_SAMPLES,
            PAIRS_PER_SAMPLE,
            2,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
        )
        if self.flow.shape != expected_flow or self.flow.dtype != np.float16:
            raise ValueError("D22 flow cache shape or dtype mismatch")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.records)

    def should_flip(self, record: dict[str, Any]) -> bool:
        if not self.train_mode:
            return False
        payload = (
            f"{self.seed}:{self.epoch}:{record['sample_id']}:horizontal"
        )
        digest = hashlib.sha256(payload.encode("utf-8")).digest()
        return bool(digest[0] & 1)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        record = self.records[index]
        cache_index = int(record["_d22_cache_index"])
        frames = torch.from_numpy(
            np.array(self.rgb[cache_index], copy=True)
        ).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
        if self.arm == "current":
            frames = frames[-1:].expand(5, -1, -1, -1).clone()
            flow = torch.zeros(
                PAIRS_PER_SAMPLE,
                2,
                OUTPUT_HEIGHT,
                OUTPUT_WIDTH,
                dtype=torch.float32,
            )
        else:
            flow = torch.from_numpy(
                np.array(
                    self.flow[cache_index],
                    dtype=np.float32,
                    copy=True,
                )
            )
        if self.should_flip(record):
            frames = torch.flip(frames, dims=(-1,))
            flow = torch.flip(flow, dims=(-1,))
            flow[:, 0].neg_()
        target = record["future_onset_target"]
        labels = torch.tensor(
            [
                float(target["proximity_onset"]),
                float(target["corridor_onset"]),
            ],
            dtype=torch.float32,
        )
        eligible = torch.tensor(
            [
                float(target["proximity_eligible"]),
                float(target["corridor_eligible"]),
            ],
            dtype=torch.float32,
        )
        return frames, flow, labels, eligible, index


class ThorDenseFlowDynamicsEncoder(DenseFlowDynamicsOnsetEncoder):
    def __init__(self, pretrained_path: Path) -> None:
        super().__init__(pretrained_path)
        self.field_head = nn.Identity()
        self.target_head = nn.Linear(128, len(TARGETS))

    def forward(
        self,
        frames: torch.Tensor,
        current_to_history_flow: torch.Tensor,
    ) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError("D22 input must have shape Bx5x3xHxW")
        batch, time, channels, height, width = frames.shape
        low = self.low_encoder(
            frames.reshape(batch * time, channels, height, width)
        )
        _, low_channels, low_height, low_width = low.shape
        if (
            low_channels != 16
            or low_height != OUTPUT_HEIGHT
            or low_width != OUTPUT_WIDTH
        ):
            raise ValueError("Unexpected D22 MobileNet low feature shape")
        low = low.reshape(
            batch,
            time,
            low_channels,
            low_height,
            low_width,
        )
        aligned_history, valid = backward_warp(
            low[:, :PAIRS_PER_SAMPLE],
            current_to_history_flow,
        )
        current = low[:, -1]
        dynamics = dense_dynamics_tensor(
            aligned_history,
            current,
            current_to_history_flow,
            valid,
        ).permute(0, 2, 1, 3, 4)
        motion_residual = self.temporal_motion(dynamics).squeeze(2)
        motion_residual = self.motion_output(motion_residual)
        fused = current + 0.25 * torch.tanh(motion_residual)
        encoded = self.high_encoder(fused)
        projected = self.field_projection(encoded)
        pooled = nnf.adaptive_avg_pool2d(projected, 1).flatten(1)
        return self.target_head(pooled)


def validate_inputs(
    records: list[dict[str, Any]],
    samples_path: Path,
    rgb_cache_path: Path,
    flow_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_ids = [str(record["sample_id"]) for record in records]
    if len(records) != EXPECTED_SAMPLES or len(set(expected_ids)) != len(
        expected_ids
    ):
        raise ValueError("D22 requires 1,078 unique D12 samples")
    if set(int(record["fold"]) for record in records) != set(range(5)):
        raise ValueError("D22 requires the inherited five D12 folds")
    for record in records:
        target = record["future_onset_target"]
        for name in TARGETS:
            if f"{name}_onset" not in target or f"{name}_eligible" not in target:
                raise ValueError("D22 future-onset target missing")

    rgb_report_path = Path(str(rgb_cache_path) + ".json")
    flow_report_path = flow_path.with_suffix(flow_path.suffix + ".json")
    if not rgb_report_path.is_file() or not flow_report_path.is_file():
        raise FileNotFoundError("D22 cache report missing")
    rgb_report = json.loads(rgb_report_path.read_text(encoding="utf-8"))
    flow_report = json.loads(flow_report_path.read_text(encoding="utf-8"))
    if rgb_report["design"]["sample_ids"] != expected_ids:
        raise ValueError("D22 RGB sample ordering mismatch")
    if rgb_report["output"]["sha256"] != sha256(rgb_cache_path):
        raise ValueError("D22 RGB SHA-256 mismatch")
    if flow_report["inputs"]["samples_sha256"] != sha256(samples_path):
        raise ValueError("D22 flow samples binding mismatch")
    if flow_report["counts"]["sample_ids_sha256"] != sample_id_digest(
        records
    ):
        raise ValueError("D22 flow sample ordering mismatch")
    if flow_report["output"]["sha256"] != sha256(flow_path):
        raise ValueError("D22 flow SHA-256 mismatch")
    return rgb_report, flow_report


def source_target_weights(
    records: list[dict[str, Any]],
) -> torch.Tensor:
    weights = np.zeros((len(records), len(TARGETS)), dtype=np.float32)
    sources = sorted({str(record["source_session_id"]) for record in records})
    for target_index, target_name in enumerate(TARGETS):
        key = f"{target_name}_eligible"
        for source in sources:
            indices = [
                index
                for index, record in enumerate(records)
                if str(record["source_session_id"]) == source
                and bool(record["future_onset_target"][key])
            ]
            if indices:
                weights[indices, target_index] = 1.0 / len(indices)
        eligible = weights[:, target_index] > 0
        if not np.any(eligible):
            raise ValueError("D22 target has no eligible training samples")
        weights[eligible, target_index] *= (
            float(np.sum(eligible)) / float(np.sum(weights[eligible, target_index]))
        )
    return torch.from_numpy(weights)


def positive_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    result = []
    for target_name in TARGETS:
        target_rows = [
            record["future_onset_target"]
            for record in records
            if bool(
                record["future_onset_target"][f"{target_name}_eligible"]
            )
        ]
        positive = sum(bool(row[f"{target_name}_onset"]) for row in target_rows)
        negative = len(target_rows) - positive
        if positive == 0 or negative == 0:
            raise ValueError("D22 train target is single-class")
        result.append(negative / positive)
    return torch.tensor(result, dtype=torch.float32)


def masked_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eligible: torch.Tensor,
    sample_weights: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    per_target = nnf.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    effective = eligible * sample_weights
    denominator = effective.sum(dim=0)
    evaluable = denominator > 0
    if not torch.any(evaluable):
        raise ValueError("D22 batch has no effective target weight")
    target_loss = (per_target * effective).sum(dim=0)[evaluable]
    return torch.mean(target_loss / denominator[evaluable])


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("D22 metric has no evaluable values")
    return float(np.mean(values))


def evaluate_predictions(
    records: list[dict[str, Any]],
    probability: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(
        [
            [
                int(record["future_onset_target"]["proximity_onset"]),
                int(record["future_onset_target"]["corridor_onset"]),
            ]
            for record in records
        ],
        dtype=np.int64,
    )
    eligible = np.asarray(
        [
            [
                bool(record["future_onset_target"]["proximity_eligible"]),
                bool(record["future_onset_target"]["corridor_eligible"]),
            ]
            for record in records
        ],
        dtype=bool,
    )
    sources = np.asarray(
        [str(record["source_session_id"]) for record in records]
    )
    by_target: dict[str, Any] = {}
    for target_index, target_name in enumerate(TARGETS):
        mask = eligible[:, target_index]
        pooled = binary_metrics(
            labels[mask, target_index],
            probability[mask, target_index],
        )
        source_rows = []
        for source in sorted(set(sources)):
            source_mask = mask & (sources == source)
            if len(set(labels[source_mask, target_index].tolist())) < 2:
                continue
            metrics = binary_metrics(
                labels[source_mask, target_index],
                probability[source_mask, target_index],
            )
            source_rows.append(
                {
                    "source_session_id": source,
                    "eligible_count": int(np.sum(source_mask)),
                    "positive_count": int(
                        np.sum(labels[source_mask, target_index])
                    ),
                    "auroc": float(metrics["auroc"]),
                    "average_precision": float(
                        metrics["average_precision"]
                    ),
                }
            )
        by_target[target_name] = {
            "pooled": {
                **pooled,
                "eligible_count": int(np.sum(mask)),
                "positive_count": int(np.sum(labels[mask, target_index])),
            },
            "source_macro": {
                "auroc": _mean([row["auroc"] for row in source_rows]),
                "average_precision": _mean(
                    [row["average_precision"] for row in source_rows]
                ),
                "evaluable_sources": len(source_rows),
            },
            "by_source": source_rows,
        }
    return {
        "source_macro": {
            metric: _mean(
                [
                    by_target[target]["source_macro"][metric]
                    for target in TARGETS
                ]
            )
            for metric in ("auroc", "average_precision")
        },
        "pooled_macro": {
            metric: _mean(
                [
                    by_target[target]["pooled"][metric]
                    for target in TARGETS
                ]
            )
            for metric in ("auroc", "average_precision")
        },
        "by_target": by_target,
    }


def nested_metric(payload: dict[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        value = value[part]
    return float(value)


@torch.inference_mode()
def predict(
    model: nn.Module,
    records: list[dict[str, Any]],
    arm: str,
    rgb_cache_path: Path,
    flow_path: Path,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    dataset = ThorDenseFlowDataset(
        records,
        arm,
        rgb_cache_path,
        flow_path,
        train=False,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model.eval()
    rows = []
    for frames, flow, _, _, _ in loader:
        logits = model(
            frames.to(device, non_blocking=True),
            flow.to(device, non_blocking=True),
        )
        rows.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(rows)


def train_arm(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    pretrained_path: Path,
    rgb_cache_path: Path,
    flow_path: Path,
    arm: str,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], nn.Module]:
    seed_everything(seed)
    model = ThorDenseFlowDynamicsEncoder(pretrained_path).to(device)
    initial_sha256 = model_sha256(model)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    dataset = ThorDenseFlowDataset(
        train_records,
        arm,
        rgb_cache_path,
        flow_path,
        train=True,
        seed=seed,
    )
    source_weight = source_target_weights(train_records)
    pos_weight = positive_weights(train_records).to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    *model.low_encoder.parameters(),
                    *model.high_encoder.parameters(),
                ],
                "lr": ENCODER_LEARNING_RATE,
            },
            {
                "params": [
                    *model.temporal_motion.parameters(),
                    *model.motion_output.parameters(),
                    *model.field_projection.parameters(),
                    *model.target_head.parameters(),
                ],
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
        for frames, flow, labels, eligible, indices in loader:
            frames = frames.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            eligible = eligible.to(device, non_blocking=True)
            sample_weight = source_weight[indices].to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames, flow)
            loss = masked_loss(
                logits,
                labels,
                eligible,
                sample_weight,
                pos_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("D22 encountered a non-finite loss")
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
    probability = predict(
        model,
        test_records,
        arm,
        rgb_cache_path,
        flow_path,
        seed,
        device,
    )
    metrics = evaluate_predictions(test_records, probability)
    diagnostic = {
        "initial_model_sha256": initial_sha256,
        "trainable_parameters": trainable_parameters,
        "positive_weights": pos_weight.detach().cpu().tolist(),
        "first_epoch_loss": epoch_rows[0]["mean_train_loss"],
        "final_epoch_loss": epoch_rows[-1]["mean_train_loss"],
        "fixed_final_epoch": EPOCHS,
    }
    return metrics, diagnostic, model


def summarize_delta(
    units: list[dict[str, Any]],
    path: str,
) -> dict[str, Any]:
    values = [
        nested_metric(unit["history_minus_current"], path)
        for unit in units
    ]
    by_fold = []
    for fold in sorted({int(unit["fold"]) for unit in units}):
        fold_values = [
            nested_metric(unit["history_minus_current"], path)
            for unit in units
            if int(unit["fold"]) == fold
        ]
        by_fold.append(_mean(fold_values))
    return {
        "count": len(values),
        "mean": _mean(values),
        "median": float(np.median(values)),
        "positive_count": int(sum(value > 0 for value in by_fold)),
        "by_fold_seed_mean": by_fold,
    }


def build_gate(aggregate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    positive_targets = sum(
        aggregate[f"by_target.{target}.source_macro.auroc"]["mean"] > 0
        and aggregate[
            f"by_target.{target}.source_macro.average_precision"
        ]["mean"]
        > 0
        for target in TARGETS
    )
    checks = {
        "primary_auroc_effect": (
            aggregate["source_macro.auroc"]["mean"] >= 0.01
        ),
        "primary_ap_effect": (
            aggregate["source_macro.average_precision"]["mean"] >= 0.005
        ),
        "primary_auroc_positive_folds": (
            aggregate["source_macro.auroc"]["positive_count"] >= 3
        ),
        "primary_ap_positive_folds": (
            aggregate["source_macro.average_precision"]["positive_count"] >= 3
        ),
        "target_breadth": positive_targets >= 2,
        "pooled_auroc_noninferiority": (
            aggregate["pooled_macro.auroc"]["mean"] >= -0.005
        ),
        "pooled_ap_noninferiority": (
            aggregate["pooled_macro.average_precision"]["mean"] >= -0.005
        ),
    }
    return {
        "frozen_thresholds": {
            "primary_source_macro_auroc_mean_floor": 0.01,
            "primary_source_macro_ap_mean_floor": 0.005,
            "primary_positive_folds": 3,
            "targets_positive_on_both_metrics": 2,
            "pooled_macro_metric_noninferiority_floor": -0.005,
        },
        "observed_positive_targets_for_both_metrics": positive_targets,
        "checks": checks,
        "supported": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--rgb-cache", type=Path, default=DEFAULT_RGB_CACHE)
    parser.add_argument("--pretrained", type=Path, default=DEFAULT_PRETRAINED)
    parser.add_argument("--flow", type=Path, default=DEFAULT_FLOW)
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
        raise FileExistsError("D22 output is non-overwriting")
    required = (
        args.samples,
        args.rgb_cache,
        args.pretrained,
        args.flow,
    )
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("D22 required input missing")
    if not torch.cuda.is_available():
        raise RuntimeError("D22 requires CUDA")
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("D22 seeds must be non-empty and unique")

    records = load_jsonl(args.samples)
    records.sort(key=lambda row: str(row["sample_id"]))
    rgb_report, flow_report = validate_inputs(
        records,
        args.samples,
        args.rgb_cache,
        args.flow,
    )
    for cache_index, record in enumerate(records):
        record["_d22_cache_index"] = cache_index
    device = torch.device("cuda")
    metric_paths = [
        "source_macro.auroc",
        "source_macro.average_precision",
        "pooled_macro.auroc",
        "pooled_macro.average_precision",
    ]
    metric_paths.extend(
        f"by_target.{target}.{scope}.{metric}"
        for target in TARGETS
        for scope in ("source_macro", "pooled")
        for metric in ("auroc", "average_precision")
    )

    units = []
    checkpoints = []
    for fold in range(5):
        train_records = [
            record for record in records if int(record["fold"]) != fold
        ]
        test_records = [
            record for record in records if int(record["fold"]) == fold
        ]
        for seed in seeds:
            arm_metrics: dict[str, Any] = {}
            arm_diagnostics: dict[str, Any] = {}
            history_model = None
            for arm in ARMS:
                metrics, diagnostic, model = train_arm(
                    train_records,
                    test_records,
                    args.pretrained,
                    args.rgb_cache,
                    args.flow,
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
                raise ValueError("D22 arms did not share initialization")
            if (
                arm_diagnostics["current"]["trainable_parameters"]
                != arm_diagnostics["history"]["trainable_parameters"]
            ):
                raise ValueError("D22 arm parameter count mismatch")
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
            heldout_sources = sorted(
                {
                    str(record["source_session_id"])
                    for record in test_records
                }
            )
            units.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "heldout_source_sessions": heldout_sources,
                    "current": arm_metrics["current"],
                    "history": arm_metrics["history"],
                    "history_minus_current": delta,
                    "training": arm_diagnostics,
                }
            )
            if history_model is None:
                raise RuntimeError("D22 history model missing")
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
                    "flow_sha256": flow_report["output"]["sha256"],
                    "heldout_source_sessions": heldout_sources,
                    "model_state_dict": {
                        name: value.detach().cpu()
                        for name, value in history_model.state_dict().items()
                    },
                },
                checkpoint_path,
            )
            checkpoints.append(
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
        path: summarize_delta(units, path) for path in metric_paths
    }
    gate = build_gate(aggregate)
    phase = "CANARY" if seeds == SEED_CANARY else "MULTI_SEED"
    status = (
        f"D22_THOR_MAGNI_DENSE_FLOW_TRANSFER_{phase}_SUPPORTED"
        if gate["supported"]
        else f"D22_THOR_MAGNI_DENSE_FLOW_TRANSFER_{phase}_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": (
                "Development independent-source representation transfer"
            ),
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "rgb_cache_path": str(args.rgb_cache.resolve()),
            "rgb_cache_sha256": rgb_report["output"]["sha256"],
            "rgb_cache_source_samples_sha256": rgb_report["inputs"][
                "samples_sha256"
            ],
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
            "flow_path": str(args.flow.resolve()),
            "flow_sha256": flow_report["output"]["sha256"],
        },
        "design": {
            "representation": (
                "D20 aligned low-level feature residual plus normalized "
                "dense flow x/y, magnitude, and warp validity at each of "
                "four history times"
            ),
            "targets": (
                "D12 current-negative true-future onset for proximity and "
                "corridor intrusion"
            ),
            "comparator": (
                "identical repeated-current model with zero flow and exact "
                "zero dynamics tensor"
            ),
            "folds": "inherited D12 five source-session-heldout folds",
            "seeds": list(seeds),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "encoder_learning_rate": ENCODER_LEARNING_RATE,
            "temporal_head_learning_rate": TEMPORAL_HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "selection": "fixed final epoch; no heldout model selection",
            "gate": (
                "source-session macro effect and fold consistency, both "
                "targets positive, pooled noninferiority"
            ),
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(
                {str(record["source_session_id"]) for record in records}
            ),
            "folds": 5,
            "seeds": len(seeds),
            "paired_units": len(units),
            "training_runs": len(units) * 2,
        },
        "device": str(device),
        "gate": gate,
        "units": units,
        "aggregate_fold_seed_mean_history_minus_current": aggregate,
        "history_checkpoints": checkpoints,
        "next_action": (
            "expand the same THOR transfer to seeds 23 and 41"
            if gate["supported"] and phase == "CANARY"
            else (
                "advance the replicated representation to real-event testing"
                if gate["supported"]
                else (
                    "retain D20 as TartanGround-local Development signal and "
                    "stop dense-flow transfer claims"
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
