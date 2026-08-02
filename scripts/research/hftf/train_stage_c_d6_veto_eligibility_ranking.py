#!/usr/bin/env python3
"""Rank false alerts among frozen-baseline active critical cells."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

from train_stage_c_d5_tartanground_development_student import (
    EarlyPairStem,
    HftfDataset,
    TemporalStudent,
    binary_metrics,
    load_jsonl,
    seed_everything,
    sha256,
)

RANKING_MODES = ("pair_only", "confidence_residual")


class VetoEligibilityStudent(nn.Module):
    """Predict false-alert eligibility from an early RGB pair."""

    def __init__(self, *, zero_head: bool = False) -> None:
        super().__init__()
        self.stem = EarlyPairStem()
        self.pool = nn.AdaptiveAvgPool2d((1, 6))
        self.head = nn.Conv1d(
            128,
            3 * 3 * 6,
            kernel_size=1,
            bias=not zero_head,
        )
        if zero_head:
            nn.init.zeros_(self.head.weight)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or frames.shape[1:3] != (5, 3):
            raise ValueError("Expected Bx5x3xHxW input")
        current = frames[:, -1]
        baseline = frames[:, :-1].mean(dim=1)
        delta = current - baseline
        pair = torch.cat(
            (current, baseline, delta, delta.abs()),
            dim=1,
        )
        output = self.head(self.pool(self.stem(pair)).squeeze(2))
        output = output.reshape(frames.shape[0], 3, 3, 6, 6)
        return output.permute(0, 1, 2, 4, 3)


def eligibility_targets(
    reference_risk_logits: torch.Tensor,
    reference_known_logits: torch.Tensor,
    risk: torch.Tensor,
    known: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return false-alert targets and the frozen eligible-cell mask."""
    eligible = (
        (known > 0.5)
        & (reference_risk_logits.sigmoid() >= 0.5)
        & (reference_known_logits.sigmoid() >= 0.5)
    )
    critical = torch.zeros_like(eligible)
    critical[:, 1:, 1:] = True
    eligible &= critical
    false_alert = (risk < 0.5).to(risk.dtype)
    return false_alert, eligible


def masked_veto_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eligible: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    if not torch.any(eligible):
        return logits.sum() * 0.0
    raw = nnf.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weight,
        reduction="none",
    )
    return raw[eligible].mean()


def reference_predictions(
    reference: TemporalStudent,
    frames: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    repeated_current = frames[:, -1:].expand(
        -1,
        5,
        -1,
        -1,
        -1,
    ).contiguous()
    return reference(repeated_current)


def compose_ranking_logits(
    pair_logits: torch.Tensor,
    reference_risk_logits: torch.Tensor,
    ranking_mode: str,
) -> torch.Tensor:
    if ranking_mode == "pair_only":
        return pair_logits
    if ranking_mode == "confidence_residual":
        return -reference_risk_logits.detach() + pair_logits
    raise ValueError(f"Unknown ranking mode: {ranking_mode}")


def evaluate(
    student: VetoEligibilityStudent,
    reference: TemporalStudent,
    loader: DataLoader,
    device: torch.device,
    ranking_mode: str,
) -> dict[str, Any]:
    arrays = collect_ranking_arrays(
        student,
        reference,
        loader,
        device,
        ranking_mode,
    )
    candidate = binary_metrics(
        arrays["probability"],
        arrays["target"],
        arrays["eligible"],
    )
    comparator = binary_metrics(
        arrays["confidence"],
        arrays["target"],
        arrays["eligible"],
    )
    return {
        "candidate": candidate,
        "baseline_inverse_risk_confidence": comparator,
        "candidate_auroc_delta": (
            float(candidate["auroc"] - comparator["auroc"])
            if candidate["auroc"] is not None
            and comparator["auroc"] is not None
            else None
        ),
        "candidate_average_precision_delta": (
            float(
                candidate["average_precision"]
                - comparator["average_precision"]
            )
            if candidate["average_precision"] is not None
            and comparator["average_precision"] is not None
            else None
        ),
        "false_alert_prevalence": (
            candidate["tp"] + candidate["fn"]
        )
        / max(candidate["known_cells"], 1),
    }


def collect_ranking_arrays(
    student: VetoEligibilityStudent,
    reference: TemporalStudent,
    loader: DataLoader,
    device: torch.device,
    ranking_mode: str,
) -> dict[str, np.ndarray]:
    student.eval()
    reference.eval()
    probabilities = []
    confidence_scores = []
    targets = []
    masks = []
    with torch.no_grad():
        for frames, risk, known in loader:
            frames = frames.to(device, non_blocking=True)
            risk = risk.to(device, non_blocking=True)
            known = known.to(device, non_blocking=True)
            reference_risk, reference_known = reference_predictions(
                reference,
                frames,
            )
            target, eligible = eligibility_targets(
                reference_risk,
                reference_known,
                risk,
                known,
            )
            logits = compose_ranking_logits(
                student(frames),
                reference_risk,
                ranking_mode,
            )
            probabilities.append(logits.sigmoid().cpu().numpy())
            confidence_scores.append(
                (1.0 - reference_risk.sigmoid()).cpu().numpy()
            )
            targets.append(target.cpu().numpy())
            masks.append(eligible.cpu().numpy())
    probability = np.concatenate(probabilities)
    confidence = np.concatenate(confidence_scores)
    target = np.concatenate(targets)
    mask = np.concatenate(masks)
    return {
        "probability": probability,
        "confidence": confidence,
        "target": target,
        "eligible": mask,
    }


def load_reference(
    pretrained_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TemporalStudent, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if (
        checkpoint.get("arm") != "single"
        or checkpoint.get("architecture") != "directional"
        or checkpoint.get("temporal_mode", "joint") != "joint"
    ):
        raise ValueError("Expected directional single joint reference")
    reference = TemporalStudent(
        pretrained_path,
        architecture="directional",
        temporal_mode="joint",
    )
    reference.load_state_dict(checkpoint["model_state_dict"], strict=True)
    reference.to(device).eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    return reference, checkpoint


def count_training_targets(
    reference: TemporalStudent,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, int | float]:
    positive = 0
    negative = 0
    reference.eval()
    with torch.no_grad():
        for frames, risk, known in loader:
            frames = frames.to(device, non_blocking=True)
            risk = risk.to(device, non_blocking=True)
            known = known.to(device, non_blocking=True)
            reference_risk, reference_known = reference_predictions(
                reference,
                frames,
            )
            target, eligible = eligibility_targets(
                reference_risk,
                reference_known,
                risk,
                known,
            )
            positive += int(((target >= 0.5) & eligible).sum())
            negative += int(((target < 0.5) & eligible).sum())
    if positive == 0 or negative == 0:
        raise ValueError("Training eligible cells need both classes")
    return {
        "eligible_cells": positive + negative,
        "false_alert_cells": positive,
        "true_alert_cells": negative,
        "false_alert_prevalence": positive / (positive + negative),
        "positive_weight": float(
            np.clip(negative / positive, 0.25, 20.0)
        ),
    }


def train(
    samples_path: Path,
    pretrained_path: Path,
    reference_checkpoint_path: Path,
    output_root: Path,
    seed: int,
    epochs: int,
    learning_rate: float,
    ranking_mode: str,
) -> dict[str, Any]:
    if epochs <= 0 or learning_rate <= 0.0:
        raise ValueError("Epochs and learning rate must be positive")
    if ranking_mode not in RANKING_MODES:
        raise ValueError(f"Unknown ranking mode: {ranking_mode}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Development run")
    seed_everything(seed)
    records = load_jsonl(samples_path)
    train_records = [row for row in records if row["role"] == "train"]
    dev_records = [row for row in records if row["role"] == "dev"]
    train_dataset = HftfDataset(
        train_records,
        "history",
        train=True,
        seed=seed,
    )
    train_count_dataset = HftfDataset(
        train_records,
        "history",
        train=False,
        seed=seed,
    )
    dev_dataset = HftfDataset(
        dev_records,
        "history",
        train=False,
        seed=seed,
    )
    device = torch.device("cuda")
    reference, reference_checkpoint = load_reference(
        pretrained_path,
        reference_checkpoint_path,
        device,
    )
    student = VetoEligibilityStudent(
        zero_head=ranking_mode == "confidence_residual"
    ).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=learning_rate,
        weight_decay=1e-4,
    )
    count_loader = DataLoader(
        train_count_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    training_targets = count_training_targets(
        reference,
        count_loader,
        device,
    )
    positive_weight = torch.tensor(
        training_targets["positive_weight"],
        dtype=torch.float32,
        device=device,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    dev_environment_loaders = {
        environment: DataLoader(
            HftfDataset(
                [
                    row
                    for row in dev_records
                    if row["environment"] == environment
                ],
                "history",
                train=False,
                seed=seed,
            ),
            batch_size=8,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        for environment in sorted(
            {row["environment"] for row in dev_records}
        )
    }

    best_score = -1.0
    best_epoch = -1
    best_state = None
    best_metrics = None
    best_environment_metrics = None
    history = []
    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(seed * 1000 + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
        )
        student.train()
        loss_total = 0.0
        batches = 0
        eligible_total = 0
        for frames, risk, known in train_loader:
            frames = frames.to(device, non_blocking=True)
            risk = risk.to(device, non_blocking=True)
            known = known.to(device, non_blocking=True)
            with torch.no_grad():
                reference_risk, reference_known = (
                    reference_predictions(reference, frames)
                )
                target, eligible = eligibility_targets(
                    reference_risk,
                    reference_known,
                    risk,
                    known,
                )
            optimizer.zero_grad(set_to_none=True)
            logits = compose_ranking_logits(
                student(frames),
                reference_risk,
                ranking_mode,
            )
            loss = masked_veto_loss(
                logits,
                target,
                eligible,
                positive_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite veto ranking loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                5.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
            loss_total += float(loss.detach())
            eligible_total += int(eligible.sum())
            batches += 1

        aggregate = evaluate(
            student,
            reference,
            dev_loader,
            device,
            ranking_mode,
        )
        by_environment = {
            environment: evaluate(
                student,
                reference,
                loader,
                device,
                ranking_mode,
            )
            for environment, loader in dev_environment_loaders.items()
        }
        environment_aurocs = [
            metrics["candidate"]["auroc"]
            for metrics in by_environment.values()
        ]
        if any(value is None for value in environment_aurocs):
            raise ValueError(
                "Every dev environment needs both eligibility classes"
            )
        selection_score = float(np.mean(environment_aurocs))
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_total / batches,
                "train_eligible_cells_seen": eligible_total,
                "selection_environment_macro_auroc": selection_score,
                "dev": aggregate,
                "dev_by_environment": by_environment,
            }
        )
        if selection_score > best_score:
            best_score = selection_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in student.state_dict().items()
            }
            best_metrics = copy.deepcopy(aggregate)
            best_environment_metrics = copy.deepcopy(by_environment)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "selection_environment_macro_auroc": (
                        selection_score
                    ),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                }
            ),
            flush=True,
        )
    if (
        best_state is None
        or best_metrics is None
        or best_environment_metrics is None
    ):
        raise RuntimeError("No veto eligibility checkpoint selected")

    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema": "blindassist_hftf_veto_eligibility_checkpoint_v1",
        "seed": seed,
        "ranking_mode": ranking_mode,
        "selected_epoch": best_epoch,
        "selection_environment_macro_auroc": best_score,
        "model_state_dict": best_state,
    }
    checkpoint_path = output_root / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    report = {
        "schema": "blindassist_hftf_stage_c_d6_veto_eligibility_ranking_v1",
        "status": "DEVELOPMENT_VETO_ELIGIBILITY_RANKING_COMPLETE",
        "policy": {
            "outcome_open": True,
            "repairable": True,
            "one_shot": False,
            "promotion_evidence": False,
            "system_output_connected": False,
            "threshold_search": False,
        },
        "task": {
            "eligible": (
                "teacher-known future body/head cells with frozen "
                "baseline risk>=0.5 and known>=0.5"
            ),
            "positive_class": "teacher-negative false alert",
            "comparator": "1 - frozen baseline risk probability",
            "ranking_mode": ranking_mode,
        },
        "seed": seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "selection_metric": (
            "environment_macro_veto_eligibility_auroc"
        ),
        "trainable_parameters": sum(
            parameter.numel() for parameter in student.parameters()
        ),
        "training_targets": training_targets,
        "train_sample_count": len(train_records),
        "dev_sample_count": len(dev_records),
        "samples_sha256": sha256(samples_path),
        "pretrained_sha256": sha256(pretrained_path),
        "reference_checkpoint_path": str(
            reference_checkpoint_path.resolve()
        ),
        "reference_checkpoint_sha256": sha256(
            reference_checkpoint_path
        ),
        "reference_selected_epoch": reference_checkpoint[
            "selected_epoch"
        ],
        "selected_epoch": best_epoch,
        "selected_selection_score": best_score,
        "selected_dev_metrics": best_metrics,
        "selected_dev_metrics_by_environment": (
            best_environment_metrics
        ),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
        },
        "history": history,
    }
    report_path = output_root / "report.json"
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument(
        "--reference-checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--ranking-mode",
        choices=RANKING_MODES,
        default="pair_only",
    )
    args = parser.parse_args()
    report = train(
        args.samples,
        args.pretrained,
        args.reference_checkpoint,
        args.output_root,
        args.seed,
        args.epochs,
        args.learning_rate,
        args.ranking_mode,
    )
    print(
        json.dumps(
            {
                "selected_epoch": report["selected_epoch"],
                "selected_selection_score": report[
                    "selected_selection_score"
                ],
                "candidate": report["selected_dev_metrics"][
                    "candidate"
                ],
                "baseline_inverse_risk_confidence": report[
                    "selected_dev_metrics"
                ]["baseline_inverse_risk_confidence"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
