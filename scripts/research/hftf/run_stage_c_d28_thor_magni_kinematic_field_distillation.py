#!/usr/bin/env python3
"""Distill D27 static and kinematic distance fields into RGB students."""

from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    load_jsonl,
    sha256,
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
    dense_dynamics_tensor,
)
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_FLOW,
    DEFAULT_RGB_CACHE,
    DEFAULT_SAMPLES,
    ThorDenseFlowDataset,
    ThorDenseFlowDynamicsEncoder,
    validate_inputs,
)
from run_stage_c_d25_thor_magni_time_to_entry import (
    nested,
    source_weights,
    summarize_delta,
)
from run_stage_c_d26_thor_magni_counterfactual_collision_field import (
    DEFAULT_D8_SAMPLES,
    DIRECTION_NAMES,
    prepare_records,
)
from evaluate_stage_c_d27_thor_magni_kinematic_information_ceiling import (
    DEFAULT_D26_REPORT,
    DISTANCE_CAP_M,
    evaluate_scores,
    metric_paths as truth_metric_paths,
)
from extract_stage_c_d18_tartanground_backward_raft_flow import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PAIRS_PER_SAMPLE,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d28_thor_magni_"
    "kinematic_field_distillation_v0"
)
ARMS = ("current", "history")
SEEDS = (17,)
FOLDS = tuple(range(5))
HORIZON_COUNT = 4
SMOOTH_L1_BETA = 0.05
DEFAULT_D27_REPORT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d27-thor-magni-kinematic-information-ceiling-v0/"
    "report.json"
)
DEFAULT_D27_SCORES = DEFAULT_D27_REPORT.with_name("oracle_scores.npz")
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d28-thor-magni-kinematic-field-distillation-v0/"
    "report.json"
)


def bind_teacher_scores(
    records: list[dict[str, Any]],
    score_path: Path,
) -> dict[str, str]:
    payload = np.load(score_path)
    sample_ids = [str(value) for value in payload["sample_ids"]]
    expected_ids = [str(record["sample_id"]) for record in records]
    if sample_ids != expected_ids:
        raise ValueError("D28 teacher score sample ordering mismatch")
    current = np.asarray(payload["current_static"], dtype=np.float32)
    history = np.asarray(
        payload["history_kinematic"],
        dtype=np.float32,
    )
    expected_shape = (
        len(records),
        len(DIRECTION_NAMES),
        HORIZON_COUNT,
    )
    if current.shape != expected_shape or history.shape != expected_shape:
        raise ValueError("D28 teacher score shape mismatch")
    if (
        not np.isfinite(current).all()
        or not np.isfinite(history).all()
        or np.min(current) < -DISTANCE_CAP_M - 1e-6
        or np.max(current) > 1e-6
        or np.min(history) < -DISTANCE_CAP_M - 1e-6
        or np.max(history) > 1e-6
    ):
        raise ValueError("D28 teacher score range mismatch")
    for index, record in enumerate(records):
        record["_d28_current_teacher"] = current[index]
        record["_d28_history_teacher"] = history[index]
    return {
        "sample_ids_first": sample_ids[0],
        "sample_ids_last": sample_ids[-1],
    }


class D28FieldDistillationDataset(ThorDenseFlowDataset):
    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        flipped = self.should_flip(self.records[index])
        frames, flow, _, _, inherited_index = super().__getitem__(index)
        key = (
            "_d28_current_teacher"
            if self.arm == "current"
            else "_d28_history_teacher"
        )
        target = torch.from_numpy(
            np.asarray(self.records[index][key], dtype=np.float32)
        ).div(DISTANCE_CAP_M)
        if flipped:
            target = target.flip(0)
        return frames, flow, target, inherited_index


class D28FieldDistillationEncoder(ThorDenseFlowDynamicsEncoder):
    def __init__(self, pretrained_path: Path) -> None:
        super().__init__(pretrained_path)
        self.target_head = nn.Linear(
            128 * 4 * 7,
            len(DIRECTION_NAMES) * HORIZON_COUNT,
        )

    def forward(
        self,
        frames: torch.Tensor,
        current_to_history_flow: torch.Tensor,
    ) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError("D28 input must have shape Bx5x3xHxW")
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
            raise ValueError("Unexpected D28 MobileNet low feature shape")
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
        if projected.shape[1:] != (128, 4, 7):
            raise ValueError("D28 projected spatial shape mismatch")
        raw = self.target_head(projected.flatten(1)).reshape(
            batch,
            len(DIRECTION_NAMES),
            HORIZON_COUNT,
        )
        bounded = -torch.sigmoid(raw)
        return torch.cummax(bounded, dim=2).values


def teacher_fit(
    records: list[dict[str, Any]],
    prediction_m: np.ndarray,
    arm: str,
) -> dict[str, Any]:
    key = (
        "_d28_current_teacher"
        if arm == "current"
        else "_d28_history_teacher"
    )
    target = np.asarray([record[key] for record in records])
    error = np.abs(prediction_m - target)
    by_source = []
    for source in sorted(
        {str(record["source_session_id"]) for record in records}
    ):
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["source_session_id"]) == source
        ]
        by_source.append(
            {
                "source_session_id": source,
                "eligible_count": len(indices),
                "mae_m": float(np.mean(error[indices])),
            }
        )
    return {
        "source_macro_mae_m": float(
            np.mean([row["mae_m"] for row in by_source])
        ),
        "pooled_mae_m": float(np.mean(error)),
        "by_source": by_source,
    }


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
    dataset = D28FieldDistillationDataset(
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
    for frames, flow, _, _ in loader:
        rows.append(
            model(
                frames.to(device, non_blocking=True),
                flow.to(device, non_blocking=True),
            )
            .cpu()
            .numpy()
        )
    return np.concatenate(rows) * DISTANCE_CAP_M


def evaluate(
    records: list[dict[str, Any]],
    prediction_m: np.ndarray,
    arm: str,
) -> dict[str, Any]:
    result = evaluate_scores(records, prediction_m)
    result["teacher_fit"] = teacher_fit(
        records,
        prediction_m,
        arm,
    )
    return result


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
    model = D28FieldDistillationEncoder(pretrained_path).to(device)
    initial_sha256 = model_sha256(model)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    dataset = D28FieldDistillationDataset(
        train_records,
        arm,
        rgb_cache_path,
        flow_path,
        train=True,
        seed=seed,
    )
    sample_weight = source_weights(train_records)
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
        total_loss = 0.0
        batch_count = 0
        for frames, flow, target, indices in loader:
            frames = frames.to(device, non_blocking=True)
            flow = flow.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weights = sample_weight[indices].to(
                device,
                non_blocking=True,
            )
            optimizer.zero_grad(set_to_none=True)
            prediction = model(frames, flow)
            losses = nnf.smooth_l1_loss(
                prediction,
                target,
                beta=SMOOTH_L1_BETA,
                reduction="none",
            ).mean(dim=(1, 2))
            loss = torch.sum(losses * weights) / torch.sum(weights)
            if not torch.isfinite(loss):
                raise RuntimeError("D28 encountered a non-finite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            total_loss += float(loss.detach())
            batch_count += 1
        mean_loss = total_loss / batch_count
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
    prediction_m = predict(
        model,
        test_records,
        arm,
        rgb_cache_path,
        flow_path,
        seed,
        device,
    )
    metrics = evaluate(test_records, prediction_m, arm)
    diagnostics = {
        "initial_model_sha256": initial_sha256,
        "trainable_parameters": trainable_parameters,
        "fixed_final_epoch": EPOCHS,
        "first_epoch_loss": epoch_rows[0]["mean_train_loss"],
        "final_epoch_loss": epoch_rows[-1]["mean_train_loss"],
    }
    return metrics, diagnostics, model


def metric_paths() -> list[str]:
    return [
        *truth_metric_paths(),
        "teacher_fit.source_macro_mae_m",
        "teacher_fit.pooled_mae_m",
    ]


def build_gate(
    aggregate: dict[str, dict[str, Any]],
    monotonicity_violations: int,
) -> dict[str, Any]:
    source_auroc = aggregate[
        "source_macro_direction_horizon_macro.auroc"
    ]
    source_ap = aggregate[
        "source_macro_direction_horizon_macro.average_precision"
    ]
    pooled_auroc = aggregate[
        "pooled_direction_horizon_macro.auroc"
    ]
    pooled_ap = aggregate[
        "pooled_direction_horizon_macro.average_precision"
    ]
    safe_choice = aggregate["safe_choice.source_macro_accuracy"]
    teacher_mae = aggregate["teacher_fit.source_macro_mae_m"]
    auroc_positive_directions = sum(
        aggregate[
            f"by_direction.{name}."
            "source_macro_horizon_macro.auroc"
        ]["mean"]
        > 0
        for name in DIRECTION_NAMES
    )
    ap_positive_directions = sum(
        aggregate[
            f"by_direction.{name}."
            "source_macro_horizon_macro.average_precision"
        ]["mean"]
        > 0
        for name in DIRECTION_NAMES
    )
    checks = {
        "source_macro_auroc_effect": source_auroc["mean"] >= 0.010,
        "source_macro_ap_effect": source_ap["mean"] >= 0.005,
        "source_macro_auroc_positive_folds": (
            source_auroc["positive_folds"] >= 3
        ),
        "source_macro_ap_positive_folds": (
            source_ap["positive_folds"] >= 3
        ),
        "auroc_positive_directions": auroc_positive_directions >= 2,
        "ap_positive_directions": ap_positive_directions >= 2,
        "safe_choice_effect": safe_choice["mean"] >= 0.020,
        "safe_choice_positive_folds": (
            safe_choice["positive_folds"] >= 3
        ),
        "pooled_auroc_noninferiority": pooled_auroc["mean"] >= -0.005,
        "pooled_ap_noninferiority": pooled_ap["mean"] >= -0.005,
        "teacher_mae_noninferiority": teacher_mae["mean"] <= 0.25,
        "monotonicity_exact": monotonicity_violations == 0,
    }
    return {
        "frozen_thresholds": {
            "source_macro_auroc_mean_floor": 0.010,
            "source_macro_ap_mean_floor": 0.005,
            "positive_folds": 3,
            "positive_directions": 2,
            "safe_choice_mean_delta_floor": 0.020,
            "safe_choice_positive_folds": 3,
            "pooled_noninferiority_floor": -0.005,
            "history_minus_current_teacher_mae_ceiling_m": 0.25,
            "monotonicity_violations": 0,
        },
        "direction_breadth": {
            "auroc_positive_directions": auroc_positive_directions,
            "ap_positive_directions": ap_positive_directions,
        },
        "checks": checks,
        "supported": all(checks.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--d8-samples",
        type=Path,
        default=DEFAULT_D8_SAMPLES,
    )
    parser.add_argument(
        "--rgb-cache",
        type=Path,
        default=DEFAULT_RGB_CACHE,
    )
    parser.add_argument("--flow", type=Path, default=DEFAULT_FLOW)
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument(
        "--d27-report",
        type=Path,
        default=DEFAULT_D27_REPORT,
    )
    parser.add_argument(
        "--teacher-scores",
        type=Path,
        default=DEFAULT_D27_SCORES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("D28 report output is non-overwriting")
    if not torch.cuda.is_available():
        raise RuntimeError("D28 requires CUDA")

    d27 = json.loads(args.d27_report.read_text(encoding="utf-8"))
    if d27["status"] != (
        "D27_THOR_MAGNI_HISTORY_KINEMATIC_INFORMATION_CEILING_SUPPORTED"
    ):
        raise ValueError("D28 requires the supported D27 terminal")
    if d27["inputs"]["samples_sha256"] != sha256(args.samples):
        raise ValueError("D28 D27 samples binding mismatch")
    if d27["inputs"]["d8_samples_sha256"] != sha256(args.d8_samples):
        raise ValueError("D28 D27 D8 binding mismatch")
    if d27["inputs"]["oracle_scores_sha256"] != sha256(
        args.teacher_scores
    ):
        raise ValueError("D28 D27 teacher-score binding mismatch")

    d12_records = load_jsonl(args.samples)
    validate_inputs(
        d12_records,
        args.samples,
        args.rgb_cache,
        args.flow,
    )
    records = prepare_records(
        d12_records,
        load_jsonl(args.d8_samples),
    )
    score_binding = bind_teacher_scores(records, args.teacher_scores)

    device = torch.device("cuda")
    units = []
    checkpoints = []
    paths = metric_paths()
    for fold in FOLDS:
        train_records = [
            record for record in records if int(record["fold"]) != fold
        ]
        test_records = [
            record for record in records if int(record["fold"]) == fold
        ]
        for seed in SEEDS:
            arm_metrics = {}
            arm_diagnostics = {}
            heldout_sources = sorted(
                {
                    str(record["source_session_id"])
                    for record in test_records
                }
            )
            for arm in ARMS:
                metrics, diagnostics, model = train_arm(
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
                arm_diagnostics[arm] = diagnostics
                checkpoint_path = (
                    args.output.parent
                    / "checkpoints"
                    / f"fold-{fold}"
                    / f"seed-{seed}-{arm}.pt"
                )
                checkpoint_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )
                model.cpu()
                torch.save(
                    {
                        "schema": SCHEMA,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "heldout_source_sessions": heldout_sources,
                        "model_state_dict": model.state_dict(),
                    },
                    checkpoint_path,
                )
                checkpoints.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "path": str(checkpoint_path.resolve()),
                        "sha256": sha256(checkpoint_path),
                    }
                )
                del model
                gc.collect()
                torch.cuda.empty_cache()
            if (
                arm_diagnostics["current"]["initial_model_sha256"]
                != arm_diagnostics["history"]["initial_model_sha256"]
            ):
                raise ValueError("D28 arms did not share initialization")
            delta = {}
            for path in paths:
                cursor = delta
                parts = path.split(".")
                for part in parts[:-1]:
                    cursor = cursor.setdefault(part, {})
                cursor[parts[-1]] = (
                    nested(arm_metrics["history"], path)
                    - nested(arm_metrics["current"], path)
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
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "seed": seed,
                        "source_macro_auroc_delta": delta[
                            "source_macro_direction_horizon_macro"
                        ]["auroc"],
                        "source_macro_ap_delta": delta[
                            "source_macro_direction_horizon_macro"
                        ]["average_precision"],
                        "safe_choice_delta": delta["safe_choice"][
                            "source_macro_accuracy"
                        ],
                        "teacher_mae_delta_m": delta["teacher_fit"][
                            "source_macro_mae_m"
                        ],
                    }
                ),
                flush=True,
            )

    aggregate = {
        path: summarize_delta(units, path) for path in paths
    }
    monotonicity_violations = sum(
        int(unit[arm]["monotonicity_violations"])
        for unit in units
        for arm in ARMS
    )
    gate = build_gate(aggregate, monotonicity_violations)
    status = (
        "D28_THOR_MAGNI_KINEMATIC_FIELD_DISTILLATION_INCREMENT_SUPPORTED"
        if gate["supported"]
        else (
            "D28_THOR_MAGNI_KINEMATIC_FIELD_DISTILLATION_"
            "INCREMENT_NOT_SUPPORTED"
        )
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": (
                "Development source-heldout object-motion "
                "teacher-student canary"
            ),
            "human_event_truth": False,
            "promotion": False,
            "app_or_safety": False,
        },
        "inputs": {
            "samples_path": str(args.samples.resolve()),
            "samples_sha256": sha256(args.samples),
            "d8_samples_path": str(args.d8_samples.resolve()),
            "d8_samples_sha256": sha256(args.d8_samples),
            "rgb_cache_path": str(args.rgb_cache.resolve()),
            "rgb_cache_sha256": sha256(args.rgb_cache),
            "flow_path": str(args.flow.resolve()),
            "flow_sha256": sha256(args.flow),
            "pretrained_path": str(args.pretrained.resolve()),
            "pretrained_sha256": sha256(args.pretrained),
            "d27_report_path": str(args.d27_report.resolve()),
            "d27_report_sha256": sha256(args.d27_report),
            "teacher_scores_path": str(args.teacher_scores.resolve()),
            "teacher_scores_sha256": sha256(args.teacher_scores),
            **score_binding,
        },
        "design": {
            "current_teacher": "D27 current-static distance field",
            "history_teacher": "D27 history-kinematic distance field",
            "score_range_m": [-DISTANCE_CAP_M, 0.0],
            "loss": "source-balanced Smooth L1",
            "smooth_l1_beta_normalized": SMOOTH_L1_BETA,
            "smooth_l1_beta_m": (
                SMOOTH_L1_BETA * DISTANCE_CAP_M
            ),
            "representation": (
                "D22 dense-flow dynamics plus flattened 128x4x7 spatial "
                "feature and monotone 3x4 distance head"
            ),
            "selection": "fixed final epoch",
            "folds": 5,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        "counts": {
            "samples": len(records),
            "source_sessions": len(
                {str(record["source_session_id"]) for record in records}
            ),
            "paired_units": len(units),
            "training_runs": len(units) * 2,
            "monotonicity_violations": monotonicity_violations,
        },
        "gate": gate,
        "aggregate_fold_mean_history_minus_current": aggregate,
        "units": units,
        "checkpoints": checkpoints,
        "next_action": (
            "freeze multi-seed and independent-source replication"
            if gate["supported"]
            else (
                "retain D27 information ceiling and stop this direct "
                "whole-frame teacher-distillation recipe"
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
                "primary_aggregate": {
                    key: aggregate[key]
                    for key in (
                        (
                            "source_macro_direction_horizon_macro."
                            "auroc"
                        ),
                        (
                            "source_macro_direction_horizon_macro."
                            "average_precision"
                        ),
                        "safe_choice.source_macro_accuracy",
                        "teacher_fit.source_macro_mae_m",
                    )
                },
                "report_sha256": digest,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
