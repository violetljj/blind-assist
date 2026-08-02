#!/usr/bin/env python3
"""Train one real-phase-supervised early-pair canary on held-out SANPO."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

from evaluate_stage_c_d6_sanpo_real_veto_ranking import (
    DEFAULT_CANDIDATE_ROOT,
    DEFAULT_MANIFEST,
    DEFAULT_PRETRAINED,
    build_event_windows,
    passed_alertable_pairs,
    phase_ranking,
    phase_unit_rows,
    ranking_comparison,
)
from export_stage_c_d6_veto_review_candidates import PublicHistoryDataset
from train_stage_c_d5_tartanground_development_student import (
    seed_everything,
    sha256,
)
from train_stage_c_d6_veto_eligibility_ranking import (
    VetoEligibilityStudent,
    compose_ranking_logits,
    load_reference,
    reference_predictions,
)


SEED = 17
MODEL_FOLD = 0
HELDOUT_FOLD = 0
FOLD_COUNT = 5
EPOCHS = 20
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8


def stable_fold_assignments(
    rows: list[dict[str, Any]],
    fold_count: int = FOLD_COUNT,
) -> dict[str, int]:
    phases_by_session: dict[str, set[str]] = {}
    for row in rows:
        phases_by_session.setdefault(
            str(row["source_session_id"]),
            set(),
        ).add(str(row["phase"]))
    strata = ([], [])
    for session, phases in phases_by_session.items():
        if "positive_alertable" in phases:
            strata[0].append(session)
        elif phases == {"negative_event"}:
            strata[1].append(session)
        else:
            raise ValueError(
                f"Unexpected session phases: {session} {sorted(phases)}"
            )
    assignments = {}
    for sessions in strata:
        ordered = sorted(
            sessions,
            key=lambda value: (
                hashlib.sha256(value.encode("utf-8")).hexdigest(),
                value,
            ),
        )
        for index, session in enumerate(ordered):
            assignments[session] = index % fold_count
    return assignments


def scored(windows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in windows
        if row["false_alert_target"] is not None
    ]


def phase_group_weights(
    windows: list[dict[str, Any]],
) -> np.ndarray:
    """Balance target classes and source-session phase groups."""
    groups = [
        (
            int(float(row["false_alert_target"])),
            str(row["source_session_id"]),
            str(row["phase"]),
        )
        for row in windows
    ]
    group_counts = Counter(groups)
    groups_per_class = Counter(group[0] for group in group_counts)
    if set(groups_per_class) != {0, 1}:
        raise ValueError("Training windows must contain both target classes")
    weights = np.asarray(
        [
            0.5
            / groups_per_class[group[0]]
            / group_counts[group]
            for group in groups
        ],
        dtype=np.float32,
    )
    if not np.isclose(float(weights.sum()), 1.0):
        raise RuntimeError("Training weights do not sum to one")
    return weights


def central_eligible(
    risk_logits: torch.Tensor,
    known_logits: torch.Tensor,
) -> torch.Tensor:
    eligible = (
        (risk_logits.sigmoid() >= 0.5)
        & (known_logits.sigmoid() >= 0.5)
    )
    central = torch.zeros_like(eligible)
    central[:, 1:, 1:, 2:4, :] = True
    return eligible & central


def cache_reference(
    reference: torch.nn.Module,
    dataset: PublicHistoryDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    risk = torch.empty(
        (len(dataset), 3, 3, 6, 6),
        dtype=torch.float32,
    )
    known = torch.empty_like(risk)
    reference.eval()
    with torch.no_grad():
        for frames, indices in loader:
            frames = frames.to(device, non_blocking=True)
            batch_risk, batch_known = reference_predictions(
                reference,
                frames,
            )
            risk[indices] = batch_risk.cpu()
            known[indices] = batch_known.cpu()
    return risk, known


def initial_residual_max_abs(
    student: VetoEligibilityStudent,
    dataset: PublicHistoryDataset,
    device: torch.device,
) -> float:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    maximum = 0.0
    student.eval()
    with torch.no_grad():
        for frames, _ in loader:
            residual = student(frames.to(device))
            maximum = max(
                maximum,
                float(residual.abs().max().item()),
            )
    return maximum


def train_student(
    student: VetoEligibilityStudent,
    dataset: PublicHistoryDataset,
    reference_risk: torch.Tensor,
    reference_known: torch.Tensor,
    weights: np.ndarray,
    device: torch.device,
) -> list[dict[str, float | int]]:
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    target = torch.tensor(
        [
            float(row["false_alert_target"])
            for row in dataset.windows
        ],
        dtype=torch.float32,
    )
    sample_weight = torch.from_numpy(weights)
    history = []
    for epoch in range(1, EPOCHS + 1):
        generator = torch.Generator().manual_seed(SEED * 1000 + epoch)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=True,
        )
        student.train()
        weighted_loss_total = 0.0
        weight_total = 0.0
        eligible_cells = 0
        eligible_windows = 0
        for frames, indices in loader:
            frames = frames.to(device, non_blocking=True)
            risk = reference_risk[indices].to(
                device,
                non_blocking=True,
            )
            known = reference_known[indices].to(
                device,
                non_blocking=True,
            )
            eligible = central_eligible(risk, known)
            residual = student(frames)
            logits = compose_ranking_logits(
                residual,
                risk,
                "confidence_residual",
            )
            batch_target = target[indices].to(
                device,
                non_blocking=True,
            )
            raw = nnf.binary_cross_entropy_with_logits(
                logits,
                batch_target[:, None, None, None, None].expand_as(
                    logits
                ),
                reduction="none",
            )
            counts = eligible.flatten(1).sum(dim=1)
            valid = counts > 0
            if not torch.any(valid):
                continue
            per_window = (
                (raw * eligible).flatten(1).sum(dim=1)
                / counts.clamp_min(1)
            )
            batch_weight = sample_weight[indices].to(
                device,
                non_blocking=True,
            )
            weighted_sum = (
                per_window[valid] * batch_weight[valid]
            ).sum()
            denominator = batch_weight[valid].sum()
            loss = weighted_sum * (
                len(dataset) / int(valid.sum().item())
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scalar_weight = float(denominator.item())
            weighted_loss_total += float(weighted_sum.item())
            weight_total += scalar_weight
            eligible_cells += int(counts[valid].sum().item())
            eligible_windows += int(valid.sum().item())
        if weight_total <= 0.0:
            raise RuntimeError("No eligible training windows")
        row = {
            "epoch": epoch,
            "weighted_train_loss": weighted_loss_total / weight_total,
            "eligible_window_observations": eligible_windows,
            "eligible_cell_observations": eligible_cells,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
    return history


def evaluate(
    student: VetoEligibilityStudent,
    dataset: PublicHistoryDataset,
    reference_risk: torch.Tensor,
    reference_known: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    probability = np.zeros(
        (len(dataset), 3, 3, 6, 6),
        dtype=np.float32,
    )
    comparator = np.zeros_like(probability)
    known_probability = reference_known.sigmoid().numpy()
    eligible = central_eligible(
        reference_risk,
        reference_known,
    ).numpy()
    student.eval()
    with torch.no_grad():
        for frames, indices in loader:
            frames = frames.to(device, non_blocking=True)
            residual = student(frames)
            risk = reference_risk[indices].to(
                device,
                non_blocking=True,
            )
            logits = compose_ranking_logits(
                residual,
                risk,
                "confidence_residual",
            )
            probability[indices.numpy()] = logits.sigmoid().cpu().numpy()
    comparator[:] = 1.0 - reference_risk.sigmoid().numpy()
    targets = np.zeros_like(probability)
    for index, row in enumerate(dataset.windows):
        targets[index].fill(float(row["false_alert_target"]))
    cell = ranking_comparison(
        probability,
        comparator,
        targets,
        eligible,
    )
    phase_rows = phase_unit_rows(
        dataset.windows,
        probability,
        comparator,
        known_probability,
        eligible,
    )
    event_p95 = phase_ranking(
        phase_rows,
        {"negative_event", "positive_alertable"},
        "p95",
    )
    pairs = passed_alertable_pairs(phase_rows)
    return {
        "cell_ranking": cell,
        "event_phase_p95_ranking": event_p95,
        "positive_passed_vs_alertable": pairs,
        "phase_units": phase_rows,
    }


def expansion_decision(result: dict[str, Any]) -> dict[str, Any]:
    ranking = result["event_phase_p95_ranking"]
    auroc_delta = ranking["candidate_auroc_delta"]
    ap_delta = ranking["candidate_average_precision_delta"]
    supported = (
        auroc_delta is not None
        and ap_delta is not None
        and float(auroc_delta) > 0.0
        and float(ap_delta) > 0.0
    )
    return {
        "criterion": (
            "held-out event-phase p95 AUROC delta > 0 AND "
            "average-precision delta > 0"
        ),
        "supported_to_expand": supported,
        "terminal": (
            "D6_REAL_PHASE_SUPERVISED_EARLY_PAIR_CANARY_"
            "INCREMENT_SUPPORTED_TO_EXPAND"
            if supported
            else
            "D6_REAL_PHASE_SUPERVISED_EARLY_PAIR_CANARY_"
            "INCREMENT_NOT_SUPPORTED_STOP"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--candidate-root",
        type=Path,
        default=DEFAULT_CANDIDATE_ROOT,
    )
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=DEFAULT_PRETRAINED,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        raise ValueError("Refusing to overwrite real-phase canary")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this Development canary")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        int(manifest["event_count"]) != 30
        or sum(len(event["frames"]) for event in manifest["events"])
        != 1920
    ):
        raise ValueError("Expected the 30-event / 1,920-frame SANPO view")
    all_windows = build_event_windows(args.manifest, manifest)
    scored_windows = scored(all_windows)
    assignments = stable_fold_assignments(scored_windows, FOLD_COUNT)
    train_windows = [
        row
        for row in scored_windows
        if assignments[str(row["source_session_id"])] != HELDOUT_FOLD
    ]
    heldout_windows = [
        row
        for row in scored_windows
        if assignments[str(row["source_session_id"])] == HELDOUT_FOLD
    ]
    if {
        str(row["source_session_id"]) for row in train_windows
    } & {
        str(row["source_session_id"]) for row in heldout_windows
    }:
        raise RuntimeError("Source-session split leaked")

    candidate_report_path = (
        args.candidate_root
        / f"seed-{SEED}"
        / f"fold-{MODEL_FOLD}"
        / "report.json"
    )
    candidate_report = json.loads(
        candidate_report_path.read_text(encoding="utf-8")
    )
    reference_checkpoint_path = Path(
        candidate_report["reference_checkpoint_path"]
    )
    seed_everything(SEED)
    device = torch.device("cuda")
    reference, _ = load_reference(
        args.pretrained,
        reference_checkpoint_path,
        device,
    )
    train_dataset = PublicHistoryDataset(train_windows)
    heldout_dataset = PublicHistoryDataset(heldout_windows)
    train_risk, train_known = cache_reference(
        reference,
        train_dataset,
        device,
        BATCH_SIZE,
    )
    heldout_risk, heldout_known = cache_reference(
        reference,
        heldout_dataset,
        device,
        BATCH_SIZE,
    )
    student = VetoEligibilityStudent(zero_head=True).to(device)
    initial_max_abs = initial_residual_max_abs(
        student,
        heldout_dataset,
        device,
    )
    if initial_max_abs != 0.0:
        raise RuntimeError("Zero-initialized residual is not exact")
    weights = phase_group_weights(train_windows)
    history = train_student(
        student,
        train_dataset,
        train_risk,
        train_known,
        weights,
        device,
    )
    heldout = evaluate(
        student,
        heldout_dataset,
        heldout_risk,
        heldout_known,
        device,
    )
    decision = expansion_decision(heldout)

    args.output_root.mkdir(parents=True)
    checkpoint_path = args.output_root / "checkpoint.pt"
    torch.save(
        {
            "schema": (
                "blindassist_hftf_stage_c_d6_sanpo_real_phase_"
                "early_pair_checkpoint_v0"
            ),
            "seed": SEED,
            "model_fold": MODEL_FOLD,
            "heldout_fold": HELDOUT_FOLD,
            "model_state_dict": student.state_dict(),
        },
        checkpoint_path,
    )
    report = {
        "schema": (
            "blindassist_hftf_stage_c_d6_sanpo_real_phase_"
            "early_pair_canary_v0"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "REAL_PHASE_EARLY_PAIR_CANARY_COMPLETE",
        "decision": decision,
        "policy": {
            "data_role": "consumed_development",
            "source_session_heldout": True,
            "engineering_failure_burns_scientific_cohort": False,
            "heldout_used_for_training_or_selection": False,
            "fixed_final_epoch_evaluation": True,
            "threshold_search": False,
            "architecture_search": False,
            "seed_search": False,
            "fold_search": False,
            "system_output_connected": False,
            "promotion_or_safety_evidence": False,
        },
        "design": {
            "seed": SEED,
            "model_fold": MODEL_FOLD,
            "heldout_fold": HELDOUT_FOLD,
            "fold_count": FOLD_COUNT,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "model": (
                "EarlyPairStem + direction-preserving field residual "
                "over frozen inverse-risk comparator"
            ),
            "train_loss": (
                "class-balanced and source-session-phase-balanced "
                "window-mean BCE on central baseline-active cells"
            ),
            "primary": (
                "held-out event-phase p95 AUROC and average-precision "
                "increments over frozen inverse-risk comparator"
            ),
            "secondary": (
                "held-out positive passed-minus-alertable p95 direction"
            ),
        },
        "inputs": {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": sha256(args.manifest),
            "candidate_report_path": str(
                candidate_report_path.resolve()
            ),
            "candidate_report_sha256": sha256(
                candidate_report_path
            ),
            "reference_checkpoint_path": str(
                reference_checkpoint_path.resolve()
            ),
            "reference_checkpoint_sha256": sha256(
                reference_checkpoint_path
            ),
            "pretrained_sha256": sha256(args.pretrained),
        },
        "split": {
            "source_fold_assignments": assignments,
            "train_source_session_count": len(
                {
                    str(row["source_session_id"])
                    for row in train_windows
                }
            ),
            "heldout_source_session_count": len(
                {
                    str(row["source_session_id"])
                    for row in heldout_windows
                }
            ),
            "train_window_count": len(train_windows),
            "heldout_window_count": len(heldout_windows),
            "heldout_source_sessions": sorted(
                {
                    str(row["source_session_id"])
                    for row in heldout_windows
                }
            ),
        },
        "initial_equivalence": {
            "maximum_absolute_pair_residual": initial_max_abs,
            "exact": initial_max_abs == 0.0,
        },
        "training_history": history,
        "heldout": heldout,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256(checkpoint_path),
        },
        "evidence_limit": (
            "Single seed17/model-fold0/heldout-fold0 canary on consumed "
            "SANPO Development. A positive result only permits the "
            "predeclared remaining-fold replication; it does not "
            "support App, promotion, production, or safety claims."
        ),
    }
    report_path = args.output_root / "report.json"
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
    digest = sha256(report_path)
    Path(str(report_path) + ".sha256").write_text(
        f"{digest}  {report_path.name}\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "decision": decision,
                "event_phase_p95": heldout[
                    "event_phase_p95_ranking"
                ],
                "positive_pairs": heldout[
                    "positive_passed_vs_alertable"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
