#!/usr/bin/env python3
"""Train an ordinal THOR-MAGNI proximity time-to-entry canary."""

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

from evaluate_stage_c_d8_thor_magni_rgb_history_screen import (
    binary_metrics,
    load_jsonl,
    sha256,
)
from evaluate_stage_c_d24_thor_magni_proximity_event_ablation import (
    DEFAULT_D8_SAMPLES,
    build_crossing_offsets,
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
from run_stage_c_d22_thor_magni_dense_flow_dynamics_transfer import (
    DEFAULT_FLOW,
    DEFAULT_RGB_CACHE,
    DEFAULT_SAMPLES,
    ThorDenseFlowDataset,
    ThorDenseFlowDynamicsEncoder,
    validate_inputs,
)


SCHEMA = "blindassist_hftf_stage_c_d25_thor_magni_time_to_entry_v0"
ARMS = ("current", "history")
SEEDS = (17,)
FOLDS = tuple(range(5))
HORIZONS = (0.5, 1.0, 1.5, 2.0)
HORIZON_NAMES = ("0_5", "1_0", "1_5", "2_0")
CLASS_COUNT = 5
DEFAULT_D23_REPORT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d23-thor-magni-proximity-multiseed-v0/report.json"
)
DEFAULT_D24_REPORT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d24-thor-magni-proximity-event-ablation-v0/report.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d25-thor-magni-time-to-entry-v0/report.json"
)


def entry_bin(offset: float | None) -> int:
    if offset is None:
        return 4
    if not 0.0 <= offset <= HORIZONS[-1] + 1e-6:
        raise ValueError("D25 entry offset is outside [0,2] seconds")
    for index, horizon in enumerate(HORIZONS):
        if offset <= horizon + 1e-9:
            return index
    raise ValueError("D25 failed to assign entry-time bin")


def prepare_records(
    records: list[dict[str, Any]],
    crossing_offsets: dict[str, float],
) -> list[dict[str, Any]]:
    prepared = []
    for cache_index, record in enumerate(records):
        record["_d22_cache_index"] = cache_index
        target = record["future_onset_target"]
        if not bool(target["proximity_eligible"]):
            continue
        sample_id = str(record["sample_id"])
        offset = crossing_offsets.get(sample_id)
        if bool(target["proximity_onset"]) != (offset is not None):
            raise ValueError("D25 onset/crossing binding mismatch")
        record["_d25_entry_offset_seconds"] = offset
        record["_d25_entry_bin"] = entry_bin(offset)
        prepared.append(record)
    if len(prepared) != 530:
        raise ValueError("D25 requires exact 530 eligible anchors")
    counts = np.bincount(
        [int(record["_d25_entry_bin"]) for record in prepared],
        minlength=CLASS_COUNT,
    )
    if counts.tolist() != [61, 32, 35, 29, 373]:
        raise ValueError(f"D25 frozen class census mismatch: {counts}")
    for fold in FOLDS:
        fold_records = [
            record for record in prepared if int(record["fold"]) == fold
        ]
        for horizon_index in range(len(HORIZONS)):
            positives = sum(
                int(record["_d25_entry_bin"]) <= horizon_index
                for record in fold_records
            )
            if positives == 0 or positives == len(fold_records):
                raise ValueError("D25 fold/horizon is not binary-evaluable")
    return prepared


class D25TimeToEntryDataset(ThorDenseFlowDataset):
    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        frames, flow, _, _, inherited_index = super().__getitem__(index)
        target = torch.tensor(
            int(self.records[index]["_d25_entry_bin"]),
            dtype=torch.long,
        )
        return frames, flow, target, inherited_index


class D25TimeToEntryEncoder(ThorDenseFlowDynamicsEncoder):
    def __init__(self, pretrained_path: Path) -> None:
        super().__init__(pretrained_path)
        self.target_head = nn.Linear(128, CLASS_COUNT)


def source_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    weights = np.zeros(len(records), dtype=np.float32)
    for source in sorted(
        {str(record["source_session_id"]) for record in records}
    ):
        indices = [
            index
            for index, record in enumerate(records)
            if str(record["source_session_id"]) == source
        ]
        weights[indices] = 1.0 / len(indices)
    weights *= len(weights) / float(np.sum(weights))
    return torch.from_numpy(weights)


def class_weights(records: list[dict[str, Any]]) -> torch.Tensor:
    counts = np.bincount(
        [int(record["_d25_entry_bin"]) for record in records],
        minlength=CLASS_COUNT,
    )
    if np.any(counts == 0):
        raise ValueError("D25 training fold is missing an entry-time class")
    return torch.from_numpy(
        (len(records) / (CLASS_COUNT * counts)).astype(np.float32)
    )


def cumulative_probabilities(
    class_probability: np.ndarray,
) -> np.ndarray:
    probability = np.asarray(class_probability, dtype=np.float64)
    if probability.ndim != 2 or probability.shape[1] != CLASS_COUNT:
        raise ValueError("D25 probability must have shape Nx5")
    if not np.isfinite(probability).all():
        raise ValueError("D25 probability contains non-finite values")
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("D25 class probability does not sum to one")
    cumulative = np.cumsum(probability[:, :4], axis=1)
    if np.any(np.diff(cumulative, axis=1) < -1e-12):
        raise ValueError("D25 cumulative probability is not monotone")
    return cumulative


def evaluate(
    records: list[dict[str, Any]],
    class_probability: np.ndarray,
) -> dict[str, Any]:
    cumulative = cumulative_probabilities(class_probability)
    by_horizon = {}
    for horizon_index, name in enumerate(HORIZON_NAMES):
        target = np.asarray(
            [
                int(record["_d25_entry_bin"]) <= horizon_index
                for record in records
            ],
            dtype=np.int64,
        )
        score = cumulative[:, horizon_index]
        by_source = []
        for source in sorted(
            {str(record["source_session_id"]) for record in records}
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
        by_horizon[name] = {
            "seconds": HORIZONS[horizon_index],
            "by_source": by_source,
            "source_macro": {
                "auroc": float(
                    np.mean([row["auroc"] for row in by_source])
                ),
                "average_precision": float(
                    np.mean(
                        [row["average_precision"] for row in by_source]
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
                    np.mean((score - target.astype(np.float64)) ** 2)
                ),
                "eligible_count": len(target),
                "positive_count": int(np.sum(target)),
            },
        }
    source_macro = {
        metric: float(
            np.mean(
                [
                    by_horizon[name]["source_macro"][metric]
                    for name in HORIZON_NAMES
                ]
            )
        )
        for metric in ("auroc", "average_precision", "brier")
    }
    pooled = {
        metric: float(
            np.mean(
                [
                    by_horizon[name]["pooled"][metric]
                    for name in HORIZON_NAMES
                ]
            )
        )
        for metric in ("auroc", "average_precision", "brier")
    }
    return {
        "source_macro_horizon_macro": source_macro,
        "pooled_horizon_macro": pooled,
        "by_horizon": by_horizon,
        "monotonicity_violations": int(
            np.sum(np.diff(cumulative, axis=1) < -1e-12)
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
    dataset = D25TimeToEntryDataset(
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
        rows.append(torch.softmax(logits, dim=1).cpu().numpy())
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
    model = D25TimeToEntryEncoder(pretrained_path).to(device)
    initial_sha256 = model_sha256(model)
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    dataset = D25TimeToEntryDataset(
        train_records,
        arm,
        rgb_cache_path,
        flow_path,
        train=True,
        seed=seed,
    )
    sample_weight = source_weights(train_records)
    class_weight = class_weights(train_records).to(device)
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
            losses = nnf.cross_entropy(
                logits,
                target,
                weight=class_weight,
                reduction="none",
            )
            loss = torch.sum(losses * weights) / torch.sum(weights)
            if not torch.isfinite(loss):
                raise RuntimeError("D25 encountered a non-finite loss")
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
        "class_weights": [
            float(value) for value in class_weight.detach().cpu()
        ],
    }
    return metrics, diagnostics, model


def nested(payload: dict[str, Any], path: str) -> float:
    value: Any = payload
    for part in path.split("."):
        value = value[part]
    return float(value)


def summarize_delta(
    units: list[dict[str, Any]],
    path: str,
) -> dict[str, Any]:
    values = [
        nested(unit["history_minus_current"], path) for unit in units
    ]
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "positive_folds": int(sum(value > 0 for value in values)),
        "negative_folds": int(sum(value < 0 for value in values)),
        "folds": [int(unit["fold"]) for unit in units],
        "values": values,
    }


def build_gate(
    aggregate: dict[str, dict[str, Any]],
    monotonicity_violations: int,
) -> dict[str, Any]:
    source_auroc = aggregate[
        "source_macro_horizon_macro.auroc"
    ]
    source_ap = aggregate[
        "source_macro_horizon_macro.average_precision"
    ]
    pooled_auroc = aggregate["pooled_horizon_macro.auroc"]
    pooled_ap = aggregate[
        "pooled_horizon_macro.average_precision"
    ]
    horizon_auroc_positive = sum(
        aggregate[f"by_horizon.{name}.source_macro.auroc"]["mean"] > 0
        for name in HORIZON_NAMES
    )
    horizon_ap_positive = sum(
        aggregate[
            f"by_horizon.{name}.source_macro.average_precision"
        ]["mean"]
        > 0
        for name in HORIZON_NAMES
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
        "auroc_positive_horizons": horizon_auroc_positive >= 3,
        "ap_positive_horizons": horizon_ap_positive >= 3,
        "pooled_auroc_noninferiority": pooled_auroc["mean"] >= -0.005,
        "pooled_ap_noninferiority": pooled_ap["mean"] >= -0.005,
        "monotonicity_exact": monotonicity_violations == 0,
    }
    return {
        "frozen_thresholds": {
            "source_macro_auroc_mean_floor": 0.010,
            "source_macro_ap_mean_floor": 0.005,
            "positive_folds": 3,
            "positive_horizons": 3,
            "pooled_noninferiority_floor": -0.005,
            "monotonicity_violations": 0,
        },
        "horizon_breadth": {
            "auroc_positive_horizons": horizon_auroc_positive,
            "ap_positive_horizons": horizon_ap_positive,
        },
        "checks": checks,
        "supported": all(checks.values()),
    }


def metric_paths() -> list[str]:
    result = [
        f"{scope}.{metric}"
        for scope in (
            "source_macro_horizon_macro",
            "pooled_horizon_macro",
        )
        for metric in ("auroc", "average_precision", "brier")
    ]
    result.extend(
        f"by_horizon.{name}.{scope}.{metric}"
        for name in HORIZON_NAMES
        for scope in ("source_macro", "pooled")
        for metric in ("auroc", "average_precision", "brier")
    )
    return result


def validate_bindings(
    d23_path: Path,
    d24_path: Path,
    samples_path: Path,
    rgb_cache_path: Path,
    flow_path: Path,
    pretrained_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    d23 = json.loads(d23_path.read_text(encoding="utf-8"))
    d24 = json.loads(d24_path.read_text(encoding="utf-8"))
    if d23["status"] != (
        "D23_THOR_MAGNI_PROXIMITY_MULTI_SEED_ROBUSTNESS_SUPPORTED"
    ):
        raise ValueError("D25 requires the supported D23 representation")
    if d24["status"] != (
        "D24_THOR_MAGNI_PROXIMITY_EVENT_DYNAMICS_NOT_SUPPORTED"
    ):
        raise ValueError("D25 requires the completed D24 event ablation")
    actual = {
        "samples_sha256": sha256(samples_path),
        "rgb_cache_sha256": sha256(rgb_cache_path),
        "flow_sha256": sha256(flow_path),
        "pretrained_sha256": sha256(pretrained_path),
    }
    for key, digest in actual.items():
        if str(d23["inputs"][key]) != digest:
            raise ValueError(f"D25 D23 input binding mismatch: {key}")
    return d23, d24


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
        "--d23-report",
        type=Path,
        default=DEFAULT_D23_REPORT,
    )
    parser.add_argument(
        "--d24-report",
        type=Path,
        default=DEFAULT_D24_REPORT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    sidecar = args.output.with_suffix(args.output.suffix + ".sha256")
    if args.output.exists() or sidecar.exists():
        raise FileExistsError("D25 report output is non-overwriting")
    if not torch.cuda.is_available():
        raise RuntimeError("D25 requires CUDA")

    full_records = load_jsonl(args.samples)
    validate_inputs(
        full_records,
        args.samples,
        args.rgb_cache,
        args.flow,
    )
    d8_records = load_jsonl(args.d8_samples)
    crossing_offsets = build_crossing_offsets(full_records, d8_records)
    records = prepare_records(full_records, crossing_offsets)
    d23, d24 = validate_bindings(
        args.d23_report,
        args.d24_report,
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
            arm_models = {}
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
                arm_models[arm] = model
            if (
                arm_diagnostics["current"]["initial_model_sha256"]
                != arm_diagnostics["history"]["initial_model_sha256"]
            ):
                raise ValueError("D25 arms did not share initialization")
            if (
                arm_diagnostics["current"]["trainable_parameters"]
                != arm_diagnostics["history"]["trainable_parameters"]
            ):
                raise ValueError("D25 arm parameter count mismatch")
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
            for arm in ARMS:
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
                torch.save(
                    {
                        "schema": SCHEMA,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "heldout_source_sessions": heldout_sources,
                        "model_state_dict": {
                            name: value.detach().cpu()
                            for name, value in arm_models[
                                arm
                            ].state_dict().items()
                        },
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
            print(
                json.dumps(
                    {
                        "fold": fold,
                        "seed": seed,
                        "source_macro_auroc_delta": delta[
                            "source_macro_horizon_macro"
                        ]["auroc"],
                        "source_macro_ap_delta": delta[
                            "source_macro_horizon_macro"
                        ]["average_precision"],
                    }
                ),
                flush=True,
            )
            del arm_models
            torch.cuda.empty_cache()

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
        "D25_THOR_MAGNI_TIME_TO_ENTRY_INCREMENT_SUPPORTED"
        if gate["supported"]
        else "D25_THOR_MAGNI_TIME_TO_ENTRY_INCREMENT_NOT_SUPPORTED"
    )
    report = {
        "schema": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "status": status,
        "authority": {
            "role": (
                "Development source-heldout ordinal time-to-entry "
                "representation canary"
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
            "d23_report_path": str(args.d23_report.resolve()),
            "d23_report_sha256": sha256(args.d23_report),
            "d24_report_path": str(args.d24_report.resolve()),
            "d24_report_sha256": sha256(args.d24_report),
            "d23_status": d23["status"],
            "d24_status": d24["status"],
        },
        "design": {
            "target": (
                "five-class first proximity-entry time: 0-.5, .5-1, "
                "1-1.5, 1.5-2 seconds, or no entry within 2 seconds"
            ),
            "cumulative_outputs": [
                f"P(T<={horizon})" for horizon in HORIZONS
            ],
            "comparison": (
                "independently trained equal-capacity current and history "
                "arms from identical initialization"
            ),
            "loss": (
                "source-balanced and five-class inverse-frequency-balanced "
                "cross entropy"
            ),
            "selection": "fixed final epoch",
            "folds": 5,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        "counts": {
            "full_samples": len(full_records),
            "eligible_samples": len(records),
            "source_sessions": len(
                {str(record["source_session_id"]) for record in records}
            ),
            "class_counts": [
                int(value)
                for value in np.bincount(
                    [
                        int(record["_d25_entry_bin"])
                        for record in records
                    ],
                    minlength=CLASS_COUNT,
                )
            ],
            "cumulative_positive_counts": [
                sum(
                    int(record["_d25_entry_bin"]) <= horizon_index
                    for record in records
                )
                for horizon_index in range(len(HORIZONS))
            ],
            "paired_units": len(units),
            "training_runs": len(units) * 2,
            "monotonicity_violations": monotonicity_violations,
        },
        "gate": gate,
        "aggregate_fold_mean_history_minus_current": aggregate,
        "units": units,
        "checkpoints": checkpoints,
        "next_action": (
            "freeze a real-sequence ordinal event decision test"
            if gate["supported"]
            else (
                "retain D23 binary representation robustness and stop the "
                "current dense-flow time-to-entry successor"
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
                        "source_macro_horizon_macro.auroc",
                        (
                            "source_macro_horizon_macro."
                            "average_precision"
                        ),
                        "pooled_horizon_macro.auroc",
                        "pooled_horizon_macro.average_precision",
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
