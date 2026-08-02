#!/usr/bin/env python3
"""Pretrain D18 alignment on dense geometry fields before onset tuning."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v3_small

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
)
from extract_stage_c_d18_tartanground_backward_raft_flow import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PAIRS_PER_SAMPLE,
)
from run_stage_c_d17_tartanground_early_temporal_onset_canary import (
    ARMS,
    BATCH_SIZE,
    DEFAULT_PRETRAINED,
    DEFAULT_SAMPLES,
    ENCODER_LEARNING_RATE,
    EPOCHS,
    MOTION_RESIDUAL_SCALE,
    SEED_CANARY,
    TARGETS,
    TEMPORAL_HEAD_LEARNING_RATE,
    WEIGHT_DECAY,
    build_gate,
    environment_weights,
    evaluate_predictions,
    masked_cell_loss,
    mean_metric_delta,
    model_sha256,
    nested_metric,
    positive_weights,
    seed_everything,
    validate_records,
)
from run_stage_c_d18_tartanground_flow_aligned_onset_canary import (
    DEFAULT_FLOW,
    FlowAlignedOnsetDataset,
    backward_warp,
    validate_flow_binding,
)
from train_stage_c_d5_tartanground_development_student import augmentation


SCHEMA = "blindassist_hftf_stage_c_d19_geometry_dynamics_pretraining_v0"
PRETRAIN_EPOCHS = 15
ONSET_EPOCHS = EPOCHS - PRETRAIN_EPOCHS
TEACHER_CHANNELS = (
    "current_body",
    "current_head",
    "near_body",
    "near_head",
    "far_body",
    "far_head",
)
DEFAULT_BASE_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-development-corpus-v0/samples.jsonl"
)
DEFAULT_EXPANSION_SAMPLES = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-development-expansion-v1/samples.jsonl"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d19-tartanground-geometry-dynamics-pretraining-v0/"
    "report.json"
)


def decode_teacher_fields(
    labels: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    risk_rows = []
    known_rows = []
    for horizon in ("current", "near", "far"):
        row = labels[horizon]
        known = np.asarray(row["known_target"], dtype=np.float32)
        risk_object = np.asarray(
            row["risk_score_target_nullable"],
            dtype=object,
        )
        if known.shape != (3, 6, 6) or risk_object.shape != (3, 6, 6):
            raise ValueError("D19 teacher field shape mismatch")
        for height_index in (1, 2):
            channel_known = known[height_index]
            channel_risk = np.zeros((6, 6), dtype=np.float32)
            for index in np.ndindex((6, 6)):
                value = risk_object[height_index][index]
                if value is None:
                    if channel_known[index] != 0.0:
                        raise ValueError("D19 known teacher cell lacks risk")
                    continue
                number = float(value)
                if not 0.0 <= number <= 1.0:
                    raise ValueError("D19 teacher risk outside [0,1]")
                if channel_known[index] != 1.0:
                    raise ValueError("D19 numeric teacher risk is unknown")
                channel_risk[index] = number
            risk_rows.append(channel_risk)
            known_rows.append(channel_known)
    return (
        torch.from_numpy(np.stack(risk_rows)),
        torch.from_numpy(np.stack(known_rows)),
    )


class GeometryDynamicsDataset(FlowAlignedOnsetDataset):
    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        frames, flow, onset, eligible, _ = super().__getitem__(index)
        record = self.records[index]
        teacher_risk, teacher_known = decode_teacher_fields(
            record["_d19_teacher_labels"]
        )
        parameters = (
            augmentation(self.seed, self.epoch, record["sample_id"])
            if self.train_mode
            else None
        )
        if parameters is not None and parameters["horizontal_flip"]:
            teacher_risk = torch.flip(teacher_risk, dims=(2,))
            teacher_known = torch.flip(teacher_known, dims=(2,))
        return (
            frames,
            flow,
            onset,
            eligible,
            teacher_risk,
            teacher_known,
            index,
        )


class GeometryDynamicsOnsetEncoder(nn.Module):
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
        self.dynamics_head = nn.Conv2d(128, 6, kernel_size=1)
        self.onset_head = nn.Conv2d(128, 4, kernel_size=1)

    def train(
        self,
        mode: bool = True,
    ) -> "GeometryDynamicsOnsetEncoder":
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def project(
        self,
        frames: torch.Tensor,
        current_to_history_flow: torch.Tensor,
    ) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError("D19 input must have shape Bx5x3xHxW")
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
            raise ValueError("Unexpected D19 MobileNet low feature shape")
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
        temporal_enabled = (
            current_to_history_flow.abs().amax(dim=(1, 2, 3, 4)) > 0.0
        ).to(dtype=low.dtype)
        aligned_residual = (
            aligned_history - current.unsqueeze(1)
        ) * valid * temporal_enabled.reshape(-1, 1, 1, 1, 1)
        ordered_motion = aligned_residual.permute(0, 2, 1, 3, 4)
        motion_residual = self.temporal_motion(ordered_motion).squeeze(2)
        motion_residual = self.motion_output(motion_residual)
        fused = current + MOTION_RESIDUAL_SCALE * torch.tanh(
            motion_residual
        )
        encoded = self.high_encoder(fused)
        projected = self.field_projection(encoded)
        return nnf.interpolate(
            projected,
            size=(6, 6),
            mode="bilinear",
            align_corners=False,
        )

    def forward_dynamics(
        self,
        frames: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        return self.dynamics_head(self.project(frames, flow))

    def forward_onset(
        self,
        frames: torch.Tensor,
        flow: torch.Tensor,
    ) -> torch.Tensor:
        return self.onset_head(self.project(frames, flow))

    @torch.no_grad()
    def transfer_future_head_to_onset(self) -> None:
        self.onset_head.weight.copy_(self.dynamics_head.weight[2:6])
        self.onset_head.bias.copy_(self.dynamics_head.bias[2:6])


def teacher_positive_weights(
    records: list[dict[str, Any]],
) -> torch.Tensor:
    positive = np.zeros(6, dtype=np.float64)
    known = np.zeros(6, dtype=np.float64)
    for record in records:
        risk, mask = decode_teacher_fields(record["_d19_teacher_labels"])
        positive += (risk.numpy() >= 0.5).sum(axis=(1, 2))
        known += mask.numpy().sum(axis=(1, 2))
    negative = known - positive
    if np.any(positive <= 0.0) or np.any(negative <= 0.0):
        raise ValueError("Every D19 teacher channel needs both risk classes")
    return torch.from_numpy((negative / positive).astype(np.float32))


def masked_teacher_loss(
    logits: torch.Tensor,
    risk: torch.Tensor,
    known: torch.Tensor,
    sample_weight: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    binary_risk = (risk >= 0.5).to(dtype=logits.dtype)
    raw = nnf.binary_cross_entropy_with_logits(
        logits,
        binary_risk,
        reduction="none",
        pos_weight=pos_weight.reshape(1, 6, 1, 1),
    )
    weight = known * sample_weight.reshape(-1, 1, 1, 1)
    if weight.sum() <= 0.0:
        raise ValueError("D19 batch has no known teacher cells")
    return (raw * weight).sum() / weight.sum()


@torch.inference_mode()
def predict_onset(
    model: GeometryDynamicsOnsetEncoder,
    records: list[dict[str, Any]],
    arm: str,
    flow_path: Path,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = GeometryDynamicsDataset(
        records,
        arm,
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
    probability_rows = []
    onset_rows = []
    eligibility_rows = []
    for frames, flow, onset, eligible, _, _, _ in loader:
        logits = model.forward_onset(
            frames.to(device, non_blocking=True),
            flow.to(device, non_blocking=True),
        )
        probability_rows.append(torch.sigmoid(logits).cpu().numpy())
        onset_rows.append(onset.numpy())
        eligibility_rows.append(eligible.numpy())
    return (
        np.concatenate(probability_rows),
        np.concatenate(onset_rows).astype(np.int64),
        np.concatenate(eligibility_rows).astype(bool),
    )


def train_arm(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    pretrained_path: Path,
    flow_path: Path,
    arm: str,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any], nn.Module]:
    seed_everything(seed)
    model = GeometryDynamicsOnsetEncoder(pretrained_path).to(device)
    initial_sha256 = model_sha256(model)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    dataset = GeometryDynamicsDataset(
        train_records,
        arm,
        flow_path,
        train=True,
        seed=seed,
    )
    env_weight = environment_weights(train_records)
    onset_pos_weight = positive_weights(train_records).to(device)
    teacher_pos_weight = teacher_positive_weights(train_records).to(device)
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
                    *model.dynamics_head.parameters(),
                    *model.onset_head.parameters(),
                ],
                "lr": TEMPORAL_HEAD_LEARNING_RATE,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )
    stage_rows = []
    for global_epoch in range(1, EPOCHS + 1):
        stage = (
            "geometry_pretrain"
            if global_epoch <= PRETRAIN_EPOCHS
            else "onset_finetune"
        )
        if global_epoch == PRETRAIN_EPOCHS + 1:
            model.transfer_future_head_to_onset()
        dataset.set_epoch(global_epoch)
        generator = torch.Generator().manual_seed(
            seed * 1000 + global_epoch
        )
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
        for (
            frames,
            flow,
            onset,
            eligible,
            teacher_risk,
            teacher_known,
            indices,
        ) in loader:
            frames = frames.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            sample_weight = env_weight[indices].to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            if stage == "geometry_pretrain":
                logits = model.forward_dynamics(frames, flow)
                loss = masked_teacher_loss(
                    logits,
                    teacher_risk.to(device, non_blocking=True),
                    teacher_known.to(device, non_blocking=True),
                    sample_weight,
                    teacher_pos_weight,
                )
            else:
                logits = model.forward_onset(frames, flow)
                loss = masked_cell_loss(
                    logits,
                    onset.to(device, non_blocking=True),
                    eligible.to(device, non_blocking=True),
                    sample_weight,
                    onset_pos_weight,
                )
            if not torch.isfinite(loss):
                raise RuntimeError("D19 encountered a non-finite loss")
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
        stage_rows.append(
            {
                "epoch": global_epoch,
                "stage": stage,
                "mean_train_loss": mean_loss,
            }
        )
        if global_epoch in {
            1,
            PRETRAIN_EPOCHS,
            PRETRAIN_EPOCHS + 1,
            EPOCHS,
        }:
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "seed": seed,
                        "epoch": global_epoch,
                        "stage": stage,
                        "mean_train_loss": mean_loss,
                    }
                ),
                flush=True,
            )
    probability, onset, eligible = predict_onset(
        model,
        test_records,
        arm,
        flow_path,
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
        "teacher_positive_weights": (
            teacher_pos_weight.detach().cpu().tolist()
        ),
        "onset_positive_weights": onset_pos_weight.detach().cpu().tolist(),
        "geometry_first_loss": stage_rows[0]["mean_train_loss"],
        "geometry_final_loss": stage_rows[
            PRETRAIN_EPOCHS - 1
        ]["mean_train_loss"],
        "onset_first_loss": stage_rows[
            PRETRAIN_EPOCHS
        ]["mean_train_loss"],
        "onset_final_loss": stage_rows[-1]["mean_train_loss"],
        "geometry_pretrain_epochs": PRETRAIN_EPOCHS,
        "onset_finetune_epochs": ONSET_EPOCHS,
        "fixed_total_epochs": EPOCHS,
    }
    return metrics, diagnostic, model


def bind_teacher_labels(
    records: list[dict[str, Any]],
    base_samples: Path,
    expansion_samples: Path,
) -> None:
    teacher_records = load_jsonl(base_samples) + load_jsonl(
        expansion_samples
    )
    by_id = {
        str(record["sample_id"]): record["labels"]
        for record in teacher_records
    }
    if len(teacher_records) != 495 or len(by_id) != 495:
        raise ValueError("D19 expects 495 unique teacher records")
    if set(by_id) != {str(record["sample_id"]) for record in records}:
        raise ValueError("D19 teacher sample IDs do not match D16")
    for record in records:
        record["_d19_teacher_labels"] = by_id[str(record["sample_id"])]
        decode_teacher_fields(record["_d19_teacher_labels"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--base-samples",
        type=Path,
        default=DEFAULT_BASE_SAMPLES,
    )
    parser.add_argument(
        "--expansion-samples",
        type=Path,
        default=DEFAULT_EXPANSION_SAMPLES,
    )
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
        raise FileExistsError("D19 output is non-overwriting")
    required = (
        args.samples,
        args.base_samples,
        args.expansion_samples,
        args.pretrained,
    )
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("D19 required input missing")
    if not torch.cuda.is_available():
        raise RuntimeError("D19 requires CUDA")
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("D19 seeds must be non-empty and unique")
    records = load_jsonl(args.samples)
    validate_records(records)
    flow_report = validate_flow_binding(records, args.samples, args.flow)
    bind_teacher_labels(records, args.base_samples, args.expansion_samples)
    for flow_index, record in enumerate(records):
        record["_d18_flow_index"] = flow_index
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
                raise ValueError("D19 arms did not share initialization")
            if (
                arm_diagnostics["current"]["trainable_parameters"]
                != arm_diagnostics["history"]["trainable_parameters"]
            ):
                raise ValueError("D19 arm parameter count mismatch")
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
                raise RuntimeError("D19 history model missing")
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
        f"D19_GEOMETRY_DYNAMICS_PRETRAINING_{phase}_SUPPORTED"
        if gate["supported"]
        else f"D19_GEOMETRY_DYNAMICS_PRETRAINING_{phase}_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development synthetic dynamics-pretraining canary",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "base_samples_path": str(args.base_samples.resolve()),
            "base_samples_sha256": sha256(args.base_samples),
            "expansion_samples_path": str(
                args.expansion_samples.resolve()
            ),
            "expansion_samples_sha256": sha256(args.expansion_samples),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
            "flow_path": str(args.flow.resolve()),
            "flow_sha256": flow_report["output"]["sha256"],
        },
        "design": {
            "representation": (
                "D18 RAFT-aligned early temporal encoder with dense "
                "current/near/far body/head geometry-field pretraining"
            ),
            "schedule": (
                "15 geometry-field epochs; copy four future head channels "
                "into onset head; 15 onset fine-tuning epochs"
            ),
            "total_epoch_control": (
                "same 30 total epochs as D18 for both paired arms"
            ),
            "teacher_channels": list(TEACHER_CHANNELS),
            "comparator": (
                "identical repeated-current model with zero flow"
            ),
            "folds": "inherited D16 three environment-heldout folds",
            "seeds": list(seeds),
            "batch_size": BATCH_SIZE,
            "encoder_learning_rate": ENCODER_LEARNING_RATE,
            "temporal_head_learning_rate": TEMPORAL_HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "selection": "fixed final epoch; no heldout model selection",
            "gate": "identical to D18",
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
            "expand the same geometry-pretrained design to seeds 23 and 41"
            if gate["supported"] and phase == "CANARY"
            else (
                "transfer the frozen representation to THOR and JRDB"
                if gate["supported"]
                else (
                    "stop current geometry-field pretraining schedule; "
                    "test explicit future-dynamics prediction"
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
