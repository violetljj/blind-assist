#!/usr/bin/env python3
"""Evaluate RAFT-aligned early temporal fusion on D16 onset fields."""

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
    EXPECTED_SAMPLES,
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PAIRS_PER_SAMPLE,
    sample_id_digest,
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
    OnsetDataset,
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
from train_stage_c_d5_tartanground_development_student import augmentation


SCHEMA = "blindassist_hftf_stage_c_d18_flow_aligned_onset_canary_v0"
DEFAULT_FLOW = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d18-tartanground-backward-raft-flow-v0/"
    "current_to_history_flow_f16.npy"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d18-tartanground-flow-aligned-onset-canary-v0/"
    "report.json"
)


class FlowAlignedOnsetDataset(OnsetDataset):
    def __init__(
        self,
        records: list[dict[str, Any]],
        arm: str,
        flow_path: Path,
        *,
        train: bool,
        seed: int,
    ) -> None:
        super().__init__(records, arm, train=train, seed=seed)
        self.flows = np.load(flow_path, mmap_mode="r")
        expected = (
            EXPECTED_SAMPLES,
            PAIRS_PER_SAMPLE,
            2,
            OUTPUT_HEIGHT,
            OUTPUT_WIDTH,
        )
        if self.flows.shape != expected or self.flows.dtype != np.float16:
            raise ValueError("D18 flow cache shape or dtype mismatch")

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
        frames, onset, eligible, _ = super().__getitem__(index)
        record = self.records[index]
        flow_index = int(record["_d18_flow_index"])
        if self.arm == "current":
            flow = torch.zeros(
                PAIRS_PER_SAMPLE,
                2,
                OUTPUT_HEIGHT,
                OUTPUT_WIDTH,
                dtype=torch.float32,
            )
        else:
            flow = torch.from_numpy(
                np.array(self.flows[flow_index], dtype=np.float32, copy=True)
            )
            parameters = (
                augmentation(self.seed, self.epoch, record["sample_id"])
                if self.train_mode
                else None
            )
            if parameters is not None and parameters["horizontal_flip"]:
                flow = torch.flip(flow, dims=(-1,))
                flow[:, 0].neg_()
        return frames, flow, onset, eligible, index


def backward_warp(
    source_history: torch.Tensor,
    current_to_history_flow: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample history at coordinates reached from each current pixel."""
    if source_history.ndim != 5:
        raise ValueError("D18 history features must be BxTxCxHxW")
    batch, time, channels, height, width = source_history.shape
    if current_to_history_flow.shape != (batch, time, 2, height, width):
        raise ValueError("D18 backward-flow shape mismatch")
    y, x = torch.meshgrid(
        torch.arange(
            height,
            device=source_history.device,
            dtype=source_history.dtype,
        ),
        torch.arange(
            width,
            device=source_history.device,
            dtype=source_history.dtype,
        ),
        indexing="ij",
    )
    flow = current_to_history_flow.to(dtype=source_history.dtype)
    source_x = x.reshape(1, 1, height, width) + flow[:, :, 0]
    source_y = y.reshape(1, 1, height, width) + flow[:, :, 1]
    valid = (
        (source_x >= 0.0)
        & (source_x <= width - 1)
        & (source_y >= 0.0)
        & (source_y <= height - 1)
    )
    grid_x = 2.0 * source_x / (width - 1) - 1.0
    grid_y = 2.0 * source_y / (height - 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    warped = nnf.grid_sample(
        source_history.reshape(batch * time, channels, height, width),
        grid.reshape(batch * time, height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape(batch, time, channels, height, width)
    return warped, valid.unsqueeze(2).to(dtype=source_history.dtype)


class FlowAlignedEarlyTemporalOnsetEncoder(nn.Module):
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

    def train(
        self,
        mode: bool = True,
    ) -> "FlowAlignedEarlyTemporalOnsetEncoder":
        super().train(mode)
        if mode:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward(
        self,
        frames: torch.Tensor,
        current_to_history_flow: torch.Tensor,
    ) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError("D18 input must have shape Bx5x3xHxW")
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
            raise ValueError("Unexpected D18 MobileNet low feature shape")
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
        projected = nnf.interpolate(
            projected,
            size=(6, 6),
            mode="bilinear",
            align_corners=False,
        )
        return self.field_head(projected)


@torch.inference_mode()
def predict(
    model: nn.Module,
    records: list[dict[str, Any]],
    arm: str,
    flow_path: Path,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = FlowAlignedOnsetDataset(
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
    for frames, flow, onset, eligible, _ in loader:
        logits = model(
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
    model = FlowAlignedEarlyTemporalOnsetEncoder(pretrained_path).to(device)
    initial_sha256 = model_sha256(model)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    dataset = FlowAlignedOnsetDataset(
        train_records,
        arm,
        flow_path,
        train=True,
        seed=seed,
    )
    env_weight = environment_weights(train_records)
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
                    *model.field_head.parameters(),
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
        for frames, flow, onset, eligible, indices in loader:
            frames = frames.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            onset = onset.to(device, non_blocking=True)
            eligible = eligible.to(device, non_blocking=True)
            sample_weight = env_weight[indices].to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(frames, flow)
            loss = masked_cell_loss(
                logits,
                onset,
                eligible,
                sample_weight,
                pos_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("D18 encountered a non-finite loss")
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
        "positive_weights": pos_weight.detach().cpu().tolist(),
        "first_epoch_loss": epoch_rows[0]["mean_train_loss"],
        "final_epoch_loss": epoch_rows[-1]["mean_train_loss"],
        "fixed_final_epoch": EPOCHS,
    }
    return metrics, diagnostic, model


def validate_flow_binding(
    records: list[dict[str, Any]],
    samples_path: Path,
    flow_path: Path,
) -> dict[str, Any]:
    report_path = flow_path.with_suffix(flow_path.suffix + ".json")
    if not flow_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("D18 flow cache or report missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_shape = [
        EXPECTED_SAMPLES,
        PAIRS_PER_SAMPLE,
        2,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
    ]
    if (
        report["inputs"]["samples_sha256"] != sha256(samples_path)
        or report["counts"]["sample_ids_sha256"]
        != sample_id_digest(records)
        or report["output"]["shape"] != expected_shape
        or report["output"]["sha256"] != sha256(flow_path)
    ):
        raise ValueError("D18 flow cache binding mismatch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
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
        raise FileExistsError("D18 output is non-overwriting")
    if not args.samples.is_file() or not args.pretrained.is_file():
        raise FileNotFoundError("D18 samples or pretrained weights missing")
    if not torch.cuda.is_available():
        raise RuntimeError("D18 requires CUDA")
    seeds = tuple(int(seed) for seed in args.seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("D18 seeds must be non-empty and unique")
    records = load_jsonl(args.samples)
    validate_records(records)
    flow_report = validate_flow_binding(records, args.samples, args.flow)
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
                raise ValueError("D18 arms did not share initialization")
            if (
                arm_diagnostics["current"]["trainable_parameters"]
                != arm_diagnostics["history"]["trainable_parameters"]
            ):
                raise ValueError("D18 arm parameter count mismatch")
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
                raise RuntimeError("D18 history model missing")
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
        f"D18_FLOW_ALIGNED_ONSET_{phase}_SUPPORTED"
        if gate["supported"]
        else f"D18_FLOW_ALIGNED_ONSET_{phase}_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": "Development synthetic aligned-representation canary",
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
            "flow_path": str(args.flow.resolve()),
            "flow_sha256": flow_report["output"]["sha256"],
        },
        "design": {
            "representation": (
                "fixed RAFT current-to-history backward flow warps four "
                "MobileNet block-0 history maps into current coordinates; "
                "aligned residuals use the D17 early 3D temporal stem"
            ),
            "comparator": (
                "identical repeated-current model with zero flow and exact "
                "zero aligned residual"
            ),
            "controlled_against_d17": (
                "same samples, folds, targets, initialization, parameter "
                "count, optimization, epochs, loss, evaluation, and gate"
            ),
            "folds": "inherited D16 three environment-heldout folds",
            "seeds": list(seeds),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "encoder_learning_rate": ENCODER_LEARNING_RATE,
            "temporal_head_learning_rate": TEMPORAL_HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "motion_residual_scale": MOTION_RESIDUAL_SCALE,
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
            "expand the same flow-aligned design to seeds 23 and 41"
            if gate["supported"] and phase == "CANARY"
            else (
                "transfer the aligned representation to THOR and JRDB"
                if gate["supported"]
                else (
                    "stop RGB optical-flow alignment plus early residual "
                    "fusion; test geometry-teacher dynamics instead"
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
