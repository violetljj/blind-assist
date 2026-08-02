#!/usr/bin/env python3
"""Train a three-direction THOR-MAGNI counterfactual collision field."""

from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    binary_metrics,
    load_jsonl,
    sha256,
)
from evaluate_stage_c_d24_thor_magni_proximity_event_ablation import (
    DEFAULT_D8_SAMPLES,
    infer_scene_column,
)
from materialize_stage_c_d8_thor_magni_local_route_supervision import (
    FUTURE_HORIZON_SECONDS,
    FUTURE_SAMPLE_SECONDS,
    PROXIMITY_THRESHOLD_M,
    read_scenario,
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
    HORIZONS,
    HORIZON_NAMES,
    entry_bin,
    nested,
    source_weights,
    summarize_delta,
)
from extract_stage_c_d18_tartanground_backward_raft_flow import (
    OUTPUT_HEIGHT,
    OUTPUT_WIDTH,
    PAIRS_PER_SAMPLE,
)


SCHEMA = (
    "blindassist_hftf_stage_c_d26_thor_magni_"
    "counterfactual_collision_field_v0"
)
ARMS = ("current", "history")
SEEDS = (17,)
FOLDS = tuple(range(5))
DIRECTION_DEGREES = (-30.0, 0.0, 30.0)
DIRECTION_NAMES = ("left", "center", "right")
CLASS_COUNT = 5
EXPECTED_CLASS_COUNTS = (
    (78, 44, 35, 33, 340),
    (60, 41, 34, 32, 363),
    (72, 63, 46, 28, 321),
)
CLASS_CENTERS_SECONDS = np.asarray(
    (0.25, 0.75, 1.25, 1.75, 2.50),
    dtype=np.float64,
)
DEFAULT_D25_REPORT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d25-thor-magni-time-to-entry-v0/report.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d26-thor-magni-counterfactual-collision-field-v0/"
    "report.json"
)


def rotate(vector: np.ndarray, degrees: float) -> np.ndarray:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return np.asarray(
        (
            cosine * vector[0] - sine * vector[1],
            sine * vector[0] + cosine * vector[1],
        ),
        dtype=np.float64,
    )


def counterfactual_entry_offsets(
    record: dict[str, Any],
    trajectory_cache: dict[tuple[str, str], dict[str, Any]],
) -> list[float | None]:
    path = Path(str(record["scenario_csv_path"]))
    camera_body = str(record["camera_body"])
    key = (str(path.resolve()), camera_body)
    data = trajectory_cache.get(key)
    if data is None:
        data = read_scenario(
            path,
            camera_body,
            infer_scene_column(path, camera_body),
        )
        trajectory_cache[key] = data
    matches = np.flatnonzero(
        data["frames"] == int(record["qtm_frame"])
    )
    if len(matches) != 1:
        raise ValueError(
            f"D26 QTM anchor is not unique: {record['sample_id']}"
        )
    index = int(matches[0])
    before = index - 25
    after = index + 25
    if before < 0 or after >= len(data["times"]):
        raise ValueError("D26 velocity support is out of range")
    velocity = (
        data["camera"][after, :2] - data["camera"][before, :2]
    ) / (data["times"][after] - data["times"][before])
    speed = float(np.linalg.norm(velocity))
    if not np.isfinite(speed) or speed < 0.25:
        raise ValueError("D26 wearer speed does not match D8 eligibility")
    forward = velocity / speed
    directions = [
        rotate(forward, degrees) for degrees in DIRECTION_DEGREES
    ]
    end_time = data["times"][index] + FUTURE_HORIZON_SECONDS
    future_end = int(
        np.searchsorted(data["times"], end_time, side="right")
    )
    local = np.diff(
        data["times"][max(0, index - 20): index + 21]
    )
    step = max(
        1,
        int(round(FUTURE_SAMPLE_SECONDS / np.median(local))),
    )
    first: list[float | None] = [None, None, None]
    origin = data["camera"][index, :2]
    for future_index in range(index, future_end, step):
        delta_time = float(
            data["times"][future_index] - data["times"][index]
        )
        other_positions = [
            positions[future_index, :2]
            for positions in data["others"].values()
            if np.isfinite(positions[future_index, :2]).all()
        ]
        if not other_positions:
            continue
        for direction_index, direction in enumerate(directions):
            if first[direction_index] is not None:
                continue
            candidate = origin + speed * delta_time * direction
            minimum = min(
                float(np.linalg.norm(position - candidate))
                for position in other_positions
            )
            if minimum <= PROXIMITY_THRESHOLD_M:
                first[direction_index] = delta_time
    return first


def prepare_records(
    d12_records: list[dict[str, Any]],
    d8_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    d8_by_id = {str(record["sample_id"]): record for record in d8_records}
    if len(d8_by_id) != len(d8_records):
        raise ValueError("D26 D8 sample IDs are not unique")
    trajectory_cache: dict[tuple[str, str], dict[str, Any]] = {}
    prepared = []
    for cache_index, record in enumerate(d12_records):
        record["_d22_cache_index"] = cache_index
        if not bool(
            record["future_onset_target"]["proximity_eligible"]
        ):
            continue
        sample_id = str(record["sample_id"])
        if sample_id not in d8_by_id:
            raise ValueError(f"D26 D12-to-D8 join failed: {sample_id}")
        offsets = counterfactual_entry_offsets(
            d8_by_id[sample_id],
            trajectory_cache,
        )
        if any(
            value is not None
            and not 0.0 <= value <= FUTURE_HORIZON_SECONDS + 1e-6
            for value in offsets
        ):
            raise ValueError("D26 entry offset is out of range")
        record["_d26_entry_offsets_seconds"] = offsets
        record["_d26_entry_bins"] = [
            entry_bin(value) for value in offsets
        ]
        prepared.append(record)
    if len(prepared) != 530:
        raise ValueError("D26 requires exact 530 eligible anchors")
    counts = tuple(
        tuple(
            int(value)
            for value in np.bincount(
                [
                    int(record["_d26_entry_bins"][direction_index])
                    for record in prepared
                ],
                minlength=CLASS_COUNT,
            )
        )
        for direction_index in range(len(DIRECTION_NAMES))
    )
    if counts != EXPECTED_CLASS_COUNTS:
        raise ValueError(f"D26 frozen class census mismatch: {counts}")
    exact_nonredundant = sum(
        len(
            {
                (
                    2.5
                    if value is None
                    else round(float(value), 6)
                )
                for value in record["_d26_entry_offsets_seconds"]
            }
        )
        > 1
        for record in prepared
    )
    class_nonredundant = sum(
        len(set(int(value) for value in record["_d26_entry_bins"])) > 1
        for record in prepared
    )
    binary_disagreement = sum(
        len(
            {
                int(value) < CLASS_COUNT - 1
                for value in record["_d26_entry_bins"]
            }
        )
        > 1
        for record in prepared
    )
    if (
        exact_nonredundant != 287
        or class_nonredundant != 271
        or binary_disagreement != 231
    ):
        raise ValueError("D26 frozen direction-opportunity census mismatch")
    for fold in FOLDS:
        rows = [
            record for record in prepared if int(record["fold"]) == fold
        ]
        for direction_index in range(len(DIRECTION_NAMES)):
            for horizon_index in range(len(HORIZONS)):
                positives = sum(
                    int(record["_d26_entry_bins"][direction_index])
                    <= horizon_index
                    for record in rows
                )
                if positives == 0 or positives == len(rows):
                    raise ValueError(
                        "D26 fold/direction/horizon is not evaluable"
                    )
    return prepared


def swap_direction_labels(
    labels: torch.Tensor,
    flipped: bool,
) -> torch.Tensor:
    if labels.shape != (len(DIRECTION_NAMES),):
        raise ValueError("D26 direction labels must have shape [3]")
    return labels.flip(0) if flipped else labels


class D26CollisionFieldDataset(ThorDenseFlowDataset):
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
        labels = torch.tensor(
            self.records[index]["_d26_entry_bins"],
            dtype=torch.long,
        )
        return (
            frames,
            flow,
            swap_direction_labels(labels, flipped),
            inherited_index,
        )


class D26CollisionFieldEncoder(ThorDenseFlowDynamicsEncoder):
    def __init__(self, pretrained_path: Path) -> None:
        super().__init__(pretrained_path)
        self.target_head = nn.Linear(
            128 * 4 * 7,
            len(DIRECTION_NAMES) * CLASS_COUNT,
        )

    def forward(
        self,
        frames: torch.Tensor,
        current_to_history_flow: torch.Tensor,
    ) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError("D26 input must have shape Bx5x3xHxW")
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
            raise ValueError("Unexpected D26 MobileNet low feature shape")
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
            raise ValueError("D26 projected spatial shape mismatch")
        logits = self.target_head(projected.flatten(1))
        return logits.reshape(
            batch,
            len(DIRECTION_NAMES),
            CLASS_COUNT,
        )


def direction_class_weights(
    records: list[dict[str, Any]],
) -> torch.Tensor:
    result = []
    for direction_index in range(len(DIRECTION_NAMES)):
        counts = np.bincount(
            [
                int(record["_d26_entry_bins"][direction_index])
                for record in records
            ],
            minlength=CLASS_COUNT,
        )
        if np.any(counts == 0):
            raise ValueError("D26 training fold is missing a class")
        result.append(
            len(records) / (CLASS_COUNT * counts.astype(np.float64))
        )
    return torch.from_numpy(np.asarray(result, dtype=np.float32))


def cumulative_probabilities(
    class_probability: np.ndarray,
) -> np.ndarray:
    probability = np.asarray(class_probability, dtype=np.float64)
    expected = (None, len(DIRECTION_NAMES), CLASS_COUNT)
    if probability.ndim != 3 or probability.shape[1:] != expected[1:]:
        raise ValueError("D26 probability must have shape Nx3x5")
    if not np.isfinite(probability).all():
        raise ValueError("D26 probability contains non-finite values")
    if not np.allclose(probability.sum(axis=2), 1.0, atol=1e-5):
        raise ValueError("D26 class probability does not sum to one")
    cumulative = np.cumsum(probability[:, :, :4], axis=2)
    if np.any(np.diff(cumulative, axis=2) < -1e-12):
        raise ValueError("D26 cumulative probability is not monotone")
    return cumulative


def evaluate(
    records: list[dict[str, Any]],
    class_probability: np.ndarray,
) -> dict[str, Any]:
    cumulative = cumulative_probabilities(class_probability)
    by_direction = {}
    for direction_index, direction_name in enumerate(DIRECTION_NAMES):
        by_horizon = {}
        for horizon_index, horizon_name in enumerate(HORIZON_NAMES):
            target = np.asarray(
                [
                    int(record["_d26_entry_bins"][direction_index])
                    <= horizon_index
                    for record in records
                ],
                dtype=np.int64,
            )
            score = cumulative[:, direction_index, horizon_index]
            by_source = []
            for source in sorted(
                {
                    str(record["source_session_id"])
                    for record in records
                }
            ):
                indices = [
                    index
                    for index, record in enumerate(records)
                    if str(record["source_session_id"]) == source
                ]
                metric = binary_metrics(target[indices], score[indices])
                if metric["auroc"] is None:
                    continue
                by_source.append(
                    {
                        "source_session_id": source,
                        "eligible_count": len(indices),
                        "positive_count": int(np.sum(target[indices])),
                        "auroc": float(metric["auroc"]),
                        "average_precision": float(
                            metric["average_precision"]
                        ),
                        "brier": float(
                            np.mean(
                                (
                                    score[indices]
                                    - target[indices].astype(np.float64)
                                )
                                ** 2
                            )
                        ),
                    }
                )
            pooled = binary_metrics(target, score)
            by_horizon[horizon_name] = {
                "seconds": HORIZONS[horizon_index],
                "by_source": by_source,
                "source_macro": {
                    "auroc": float(
                        np.mean([row["auroc"] for row in by_source])
                    ),
                    "average_precision": float(
                        np.mean(
                            [
                                row["average_precision"]
                                for row in by_source
                            ]
                        )
                    ),
                    "brier": float(
                        np.mean([row["brier"] for row in by_source])
                    ),
                    "evaluable_sources": len(by_source),
                },
                "pooled": {
                    "auroc": float(pooled["auroc"]),
                    "average_precision": float(
                        pooled["average_precision"]
                    ),
                    "brier": float(
                        np.mean(
                            (score - target.astype(np.float64)) ** 2
                        )
                    ),
                    "eligible_count": len(target),
                    "positive_count": int(np.sum(target)),
                },
            }
        by_direction[direction_name] = {
            "source_macro_horizon_macro": {
                metric: float(
                    np.mean(
                        [
                            by_horizon[name]["source_macro"][metric]
                            for name in HORIZON_NAMES
                        ]
                    )
                )
                for metric in ("auroc", "average_precision", "brier")
            },
            "pooled_horizon_macro": {
                metric: float(
                    np.mean(
                        [
                            by_horizon[name]["pooled"][metric]
                            for name in HORIZON_NAMES
                        ]
                    )
                )
                for metric in ("auroc", "average_precision", "brier")
            },
            "by_horizon": by_horizon,
        }
    source_macro = {
        metric: float(
            np.mean(
                [
                    by_direction[name][
                        "source_macro_horizon_macro"
                    ][metric]
                    for name in DIRECTION_NAMES
                ]
            )
        )
        for metric in ("auroc", "average_precision", "brier")
    }
    pooled = {
        metric: float(
            np.mean(
                [
                    by_direction[name]["pooled_horizon_macro"][metric]
                    for name in DIRECTION_NAMES
                ]
            )
        )
        for metric in ("auroc", "average_precision", "brier")
    }

    truth_bins = np.asarray(
        [record["_d26_entry_bins"] for record in records],
        dtype=np.int64,
    )
    truth_time = np.asarray(
        [
            [
                2.5 if value is None else float(value)
                for value in record["_d26_entry_offsets_seconds"]
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    expected_time = np.sum(
        class_probability * CLASS_CENTERS_SECONDS[None, None, :],
        axis=2,
    )
    nonredundant = np.asarray(
        [
            len(set(round(float(value), 6) for value in row)) > 1
            for row in truth_time
        ],
        dtype=bool,
    )
    predicted_direction = np.argmax(expected_time, axis=1)
    safe_time = np.max(truth_time, axis=1)
    correct = (
        np.abs(
            truth_time[np.arange(len(records)), predicted_direction]
            - safe_time
        )
        <= 1e-6
    )
    safe_by_source = []
    for source in sorted(
        {str(record["source_session_id"]) for record in records}
    ):
        indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if str(record["source_session_id"]) == source
                and bool(nonredundant[index])
            ],
            dtype=np.int64,
        )
        if len(indices) == 0:
            continue
        safe_by_source.append(
            {
                "source_session_id": source,
                "eligible_count": len(indices),
                "accuracy": float(np.mean(correct[indices])),
            }
        )
    safe_choice = {
        "source_macro_accuracy": float(
            np.mean([row["accuracy"] for row in safe_by_source])
        ),
        "pooled_accuracy": float(np.mean(correct[nonredundant])),
        "eligible_count": int(np.sum(nonredundant)),
        "evaluable_sources": len(safe_by_source),
        "by_source": safe_by_source,
    }
    return {
        "source_macro_direction_horizon_macro": source_macro,
        "pooled_direction_horizon_macro": pooled,
        "safe_choice": safe_choice,
        "by_direction": by_direction,
        "monotonicity_violations": int(
            np.sum(np.diff(cumulative, axis=2) < -1e-12)
        ),
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
    dataset = D26CollisionFieldDataset(
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
        logits = model(
            frames.to(device, non_blocking=True),
            flow.to(device, non_blocking=True),
        )
        rows.append(torch.softmax(logits, dim=2).cpu().numpy())
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
    model = D26CollisionFieldEncoder(pretrained_path).to(device)
    initial_sha256 = model_sha256(model)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    dataset = D26CollisionFieldDataset(
        train_records,
        arm,
        rgb_cache_path,
        flow_path,
        train=True,
        seed=seed,
    )
    sample_weight = source_weights(train_records)
    class_weight = direction_class_weights(train_records).to(device)
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
            logits = model(frames, flow)
            losses = torch.stack(
                [
                    nnf.cross_entropy(
                        logits[:, direction_index],
                        target[:, direction_index],
                        weight=class_weight[direction_index],
                        reduction="none",
                    )
                    for direction_index in range(len(DIRECTION_NAMES))
                ],
                dim=1,
            ).mean(dim=1)
            loss = torch.sum(losses * weights) / torch.sum(weights)
            if not torch.isfinite(loss):
                raise RuntimeError("D26 encountered a non-finite loss")
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
    probability = predict(
        model,
        test_records,
        arm,
        rgb_cache_path,
        flow_path,
        seed,
        device,
    )
    metrics = evaluate(test_records, probability)
    diagnostics = {
        "initial_model_sha256": initial_sha256,
        "trainable_parameters": trainable_parameters,
        "fixed_final_epoch": EPOCHS,
        "first_epoch_loss": epoch_rows[0]["mean_train_loss"],
        "final_epoch_loss": epoch_rows[-1]["mean_train_loss"],
        "class_weights": class_weight.detach().cpu().tolist(),
    }
    return metrics, diagnostics, model


def metric_paths() -> list[str]:
    result = [
        f"{scope}.{metric}"
        for scope in (
            "source_macro_direction_horizon_macro",
            "pooled_direction_horizon_macro",
        )
        for metric in ("auroc", "average_precision", "brier")
    ]
    result.extend(
        ("safe_choice.source_macro_accuracy", "safe_choice.pooled_accuracy")
    )
    for direction_name in DIRECTION_NAMES:
        result.extend(
            f"by_direction.{direction_name}.{scope}.{metric}"
            for scope in (
                "source_macro_horizon_macro",
                "pooled_horizon_macro",
            )
            for metric in ("auroc", "average_precision", "brier")
        )
        result.extend(
            (
                f"by_direction.{direction_name}.by_horizon."
                f"{horizon_name}.{scope}.{metric}"
            )
            for horizon_name in HORIZON_NAMES
            for scope in ("source_macro", "pooled")
            for metric in ("auroc", "average_precision", "brier")
        )
    return result


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
            "monotonicity_violations": 0,
        },
        "direction_breadth": {
            "auroc_positive_directions": auroc_positive_directions,
            "ap_positive_directions": ap_positive_directions,
        },
        "checks": checks,
        "supported": all(checks.values()),
    }


def validate_binding(
    d25_path: Path,
    samples_path: Path,
    rgb_cache_path: Path,
    flow_path: Path,
    pretrained_path: Path,
) -> dict[str, Any]:
    d25 = json.loads(d25_path.read_text(encoding="utf-8"))
    if d25["status"] != (
        "D25_THOR_MAGNI_TIME_TO_ENTRY_INCREMENT_NOT_SUPPORTED"
    ):
        raise ValueError("D26 requires the completed D25 terminal")
    actual = {
        "samples_sha256": sha256(samples_path),
        "rgb_cache_sha256": sha256(rgb_cache_path),
        "flow_sha256": sha256(flow_path),
        "pretrained_sha256": sha256(pretrained_path),
    }
    for key, digest in actual.items():
        if str(d25["inputs"][key]) != digest:
            raise ValueError(f"D26 D25 input binding mismatch: {key}")
    return d25


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
        "--d25-report",
        type=Path,
        default=DEFAULT_D25_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("D26 report output is non-overwriting")
    if not torch.cuda.is_available():
        raise RuntimeError("D26 requires CUDA")

    d12_records = load_jsonl(args.samples)
    validate_inputs(
        d12_records,
        args.samples,
        args.rgb_cache,
        args.flow,
    )
    d8_records = load_jsonl(args.d8_samples)
    records = prepare_records(d12_records, d8_records)
    d25 = validate_binding(
        args.d25_report,
        args.samples,
        args.rgb_cache,
        args.flow,
        args.pretrained,
    )

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
                raise ValueError("D26 arms did not share initialization")
            if (
                arm_diagnostics["current"]["trainable_parameters"]
                != arm_diagnostics["history"]["trainable_parameters"]
            ):
                raise ValueError("D26 arm parameter count mismatch")
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
        "D26_THOR_MAGNI_COUNTERFACTUAL_COLLISION_FIELD_INCREMENT_SUPPORTED"
        if gate["supported"]
        else (
            "D26_THOR_MAGNI_COUNTERFACTUAL_COLLISION_FIELD_"
            "INCREMENT_NOT_SUPPORTED"
        )
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": (
                "Development source-heldout action-conditioned tracked-body "
                "future-field canary"
            ),
            "source_native_geometric_proxy": True,
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
            "d25_report_path": str(args.d25_report.resolve()),
            "d25_report_sha256": sha256(args.d25_report),
            "d25_status": d25["status"],
        },
        "design": {
            "teacher": (
                "constant-speed candidate wearer paths at -30/0/+30 degrees "
                "against source-recorded future trajectories of other bodies"
            ),
            "target": (
                "three directions x five first-entry-time classes at 1.25m"
            ),
            "representation": (
                "D22 dense-flow dynamics with flattened 128x4x7 spatial "
                "field feature and a 3x5 linear head"
            ),
            "comparison": (
                "independently trained equal-capacity current and history "
                "arms from identical initialization"
            ),
            "loss": (
                "source-balanced, direction-equal, per-direction "
                "inverse-frequency class-balanced cross entropy"
            ),
            "selection": "fixed final epoch",
            "directions_degrees": list(DIRECTION_DEGREES),
            "horizons_seconds": list(HORIZONS),
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
            "class_counts_by_direction": [
                list(values) for values in EXPECTED_CLASS_COUNTS
            ],
            "exact_time_direction_nonredundant_samples": 287,
            "time_class_direction_nonredundant_samples": 271,
            "binary_direction_disagreement_samples": 231,
            "paired_units": len(units),
            "training_runs": len(units) * 2,
            "monotonicity_violations": monotonicity_violations,
        },
        "gate": gate,
        "aggregate_fold_mean_history_minus_current": aggregate,
        "units": units,
        "checkpoints": checkpoints,
        "next_action": (
            "freeze a real-sequence or independent-source directional event test"
            if gate["supported"]
            else (
                "retain earlier bounded signals and stop the current "
                "counterfactual tracked-body field successor"
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
                        "pooled_direction_horizon_macro.auroc",
                        (
                            "pooled_direction_horizon_macro."
                            "average_precision"
                        ),
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
