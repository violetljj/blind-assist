#!/usr/bin/env python3
"""R24 opportunity-aware successor to the frozen R23 TARO ranker.

R23 improved source-family holdout macro utility, but its ordinary pairwise
gain target did not reliably cover parents where some candidate could beat
both generic pose diversity and the passive-coverage comparator.  R24 adds one
materially new teacher signal: a candidate-level strict-opportunity label.  It
keeps the source-time student inputs, query-conditioned architecture, LOFO
firewall, uncertainty treatment, and utility no-regret bound from R23.

The passive comparator and candidate depth remain teacher-side only.  They are
never inputs to inference, uncertainty, or held-source selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_query_conditioned_no_regret_ranker as r23


SCHEMA = "blindassist.taro.task_evidence_opportunity_aware_no_regret_ranker.v1"
SEEDS = (24011, 24029, 24043)
EPOCHS = 220
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
REGRESSION_LOSS_WEIGHT = 0.25
OPPORTUNITY_LOSS_WEIGHT = 0.5
GRADIENT_CLIP_NORM = 5.0
OPPORTUNITY_LCB_Z = 1.0
MIN_OPPORTUNITY_PROBABILITY = 0.5
MAX_POSITIVE_CLASS_WEIGHT = 20.0


class OpportunityAwareUtilityRanker(nn.Module):
    """R23 interaction encoder with an additional strict-opportunity head."""

    def __init__(self) -> None:
        super().__init__()
        self.task_encoder = nn.Sequential(
            nn.Linear(r23.STATIC_TOKEN_WIDTH, r23.STATIC_HIDDEN),
            nn.GELU(),
            nn.Linear(r23.STATIC_HIDDEN, r23.EMBED_DIM),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(r23.GEOMETRY_TOKEN_WIDTH, r23.GEOMETRY_HIDDEN),
            nn.GELU(),
            nn.Linear(r23.GEOMETRY_HIDDEN, r23.EMBED_DIM),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(r23.POSE_TRANSFORMED_WIDTH, r23.POSE_HIDDEN),
            nn.GELU(),
            nn.Linear(r23.POSE_HIDDEN, r23.EMBED_DIM),
        )
        self.query_embedding = nn.Parameter(torch.empty(r23.QUERY_COUNT, r23.EMBED_DIM))
        nn.init.normal_(self.query_embedding, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(r23.EMBED_DIM, r23.ATTENTION_HEADS, batch_first=True)
        self.task_norm = nn.LayerNorm(r23.EMBED_DIM)
        self.candidate_norm = nn.LayerNorm(r23.EMBED_DIM)
        self.fusion = nn.Sequential(
            nn.Linear(r23.EMBED_DIM * 5, r23.FUSION_HIDDEN[0]),
            nn.GELU(),
            nn.Linear(r23.FUSION_HIDDEN[0], r23.FUSION_HIDDEN[1]),
            nn.GELU(),
            nn.Linear(r23.FUSION_HIDDEN[1], 3),
        )

    def forward(
        self,
        pose: torch.Tensor,
        task_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        generic_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query_embedding.unsqueeze(0)
        pose_token = self.pose_encoder(pose)
        task = self.task_norm(self.task_encoder(task_tokens) + query)
        candidate = self.candidate_norm(self.candidate_encoder(candidate_tokens) + query + pose_token.unsqueeze(1))
        attended_task, _weights = self.cross_attention(candidate, task, task, need_weights=False)
        pooled = torch.cat(
            (
                candidate.mean(dim=1),
                attended_task.mean(dim=1),
                (candidate * attended_task).mean(dim=1),
                (candidate * task).mean(dim=1),
                pose_token,
            ),
            dim=1,
        )
        output = self.fusion(pooled)
        mean = generic_base + r23.RESIDUAL_SCALE * torch.tanh(output[:, 0])
        log_variance = torch.clamp(output[:, 1], min=-5.0, max=1.0)
        return mean, log_variance, output[:, 2]


def _passive_index(records: Sequence[scorer.CandidateRecord], indices: Sequence[int]) -> int:
    return max(
        indices,
        key=lambda index: (
            float(records[index].coverage),
            -records[index].pair.gap_s,
            records[index].pair.neighbor.frame_id,
        ),
    )


def opportunity_targets(
    records: Sequence[scorer.CandidateRecord],
    sources: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Teacher-only labels for beating both frozen comparators."""
    shared.require(len(records) == len(sources), "R24 opportunity source alignment drift")
    labels = np.zeros(len(records), dtype=np.float32)
    by_reference: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (record, source) in enumerate(zip(records, sources, strict=True)):
        shared.require(record.target_gain is not None and record.coverage is not None, "R24 teacher row incomplete")
        by_reference[(source, record.reference_id)].append(index)
    positive_reference_count = 0
    for indices in by_reference.values():
        generic = r23._generic_index(records, indices)
        passive = _passive_index(records, indices)
        comparator = max(int(records[generic].target_gain), int(records[passive].target_gain))
        local_positive = False
        for index in indices:
            if int(records[index].target_gain) > comparator:
                labels[index] = 1.0
                local_positive = True
        positive_reference_count += int(local_positive)

    _targets, base_weights = r23._balanced_candidate_targets(records, sources)
    positive_mass = float(np.sum(base_weights * labels))
    negative_mass = float(np.sum(base_weights * (1.0 - labels)))
    positive_class_weight = min(MAX_POSITIVE_CLASS_WEIGHT, negative_mass / max(positive_mass, 1e-9))
    class_weights = np.where(labels > 0.5, positive_class_weight, 1.0).astype(np.float32)
    weights = base_weights * class_weights
    weights *= len(weights) / np.sum(weights)
    audit = {
        "candidate_count": len(records),
        "reference_count": len(by_reference),
        "positive_candidate_count": int(np.sum(labels)),
        "positive_reference_count": positive_reference_count,
        "positive_class_weight": positive_class_weight,
        "teacher_comparator": "max(generic target gain, passive-coverage target gain)",
    }
    shared.require(audit["positive_candidate_count"] > 0, "R24 opportunity teacher has no positives")
    return labels, weights, audit


def train_ranker(
    records: Sequence[scorer.CandidateRecord],
    sources: Sequence[str],
    transform: r23.StructuredFeatureTransform,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[OpportunityAwareUtilityRanker, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    inputs = r23._torch_inputs(transform.apply(records))
    high_np, low_np, pair_weights_np = r21._pair_indices(records, sources)
    utility_targets_np, candidate_weights_np = r23._balanced_candidate_targets(records, sources)
    opportunity_targets_np, opportunity_weights_np, opportunity_audit = opportunity_targets(records, sources)
    high = torch.from_numpy(high_np)
    low = torch.from_numpy(low_np)
    pair_weights = torch.from_numpy(pair_weights_np)
    utility_targets = torch.from_numpy(utility_targets_np)
    candidate_weights = torch.from_numpy(candidate_weights_np)
    strict_targets = torch.from_numpy(opportunity_targets_np)
    strict_weights = torch.from_numpy(opportunity_weights_np)
    model = OpportunityAwareUtilityRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_rank = float("nan")
    final_regression = float("nan")
    final_opportunity = float("nan")
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        mean, log_variance, opportunity_logit = model(*inputs)
        rank_loss = torch.mean(F.softplus(-(mean[high] - mean[low])) * pair_weights)
        regression_rows = 0.5 * (torch.exp(-log_variance) * (mean - utility_targets) ** 2 + log_variance)
        regression_loss = torch.mean(regression_rows * candidate_weights)
        opportunity_rows = F.binary_cross_entropy_with_logits(opportunity_logit, strict_targets, reduction="none")
        opportunity_loss = torch.mean(opportunity_rows * strict_weights)
        loss = rank_loss + REGRESSION_LOSS_WEIGHT * regression_loss + OPPORTUNITY_LOSS_WEIGHT * opportunity_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        final_rank = float(rank_loss.detach().cpu())
        final_regression = float(regression_loss.detach().cpu())
        final_opportunity = float(opportunity_loss.detach().cpu())
    state_bytes = b"".join(value.detach().cpu().numpy().tobytes() for _name, value in sorted(model.state_dict().items()))
    return model, {
        "seed": seed,
        "epochs": epochs,
        "pair_count": len(high_np),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "opportunity_teacher_audit": opportunity_audit,
        "final_pairwise_loss": final_rank,
        "final_heteroscedastic_regression_loss": final_regression,
        "final_opportunity_classification_loss": final_opportunity,
        "model_state_sha256": hashlib.sha256(state_bytes).hexdigest().upper(),
    }


def ensemble_predictions(
    records: Sequence[scorer.CandidateRecord],
    transform: r23.StructuredFeatureTransform,
    models: Sequence[OpportunityAwareUtilityRanker],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    inputs = r23._torch_inputs(transform.apply(records))
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    opportunity_logits: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            mean, log_variance, opportunity_logit = model(*inputs)
            means.append(mean.cpu().numpy().astype(np.float64))
            log_variances.append(log_variance.cpu().numpy().astype(np.float64))
            opportunity_logits.append(opportunity_logit.cpu().numpy().astype(np.float64))
    return np.stack(means), np.stack(log_variances), np.stack(opportunity_logits)


def opportunity_no_regret_gate(
    records: Sequence[scorer.CandidateRecord],
    ensemble_means: np.ndarray,
    ensemble_log_variances: np.ndarray,
    ensemble_opportunity_logits: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Use source-time probability and utility bounds; never teacher targets."""
    shared.require(ensemble_means.shape == ensemble_log_variances.shape, "R24 utility ensemble shape mismatch")
    shared.require(ensemble_means.shape == ensemble_opportunity_logits.shape, "R24 opportunity ensemble shape mismatch")
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    selected_scores = np.full(len(records), -1e9, dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    learned_override_count = 0
    probabilities = 1.0 / (1.0 + np.exp(-ensemble_opportunity_logits))
    for reference, indices in sorted(by_reference.items()):
        generic = r23._generic_index(records, indices)
        rows: list[tuple[float, float, int, float, float, float, float]] = []
        for index in indices:
            if index == generic:
                continue
            delta = ensemble_means[:, index] - ensemble_means[:, generic]
            utility_mean = float(np.mean(delta))
            utility_epistemic_variance = float(np.var(delta, ddof=1)) if len(delta) > 1 else 0.0
            utility_aleatoric_variance = float(
                np.mean(np.exp(ensemble_log_variances[:, index]) + np.exp(ensemble_log_variances[:, generic]))
            )
            utility_std = float(
                np.sqrt(max(0.0, utility_epistemic_variance + r23.ALEATORIC_GATE_WEIGHT * utility_aleatoric_variance))
            )
            utility_lcb = utility_mean - r23.LCB_Z * utility_std - r23.MIN_NORMALIZED_ADVANTAGE
            candidate_probabilities = probabilities[:, index]
            probability_mean = float(np.mean(candidate_probabilities))
            probability_std = float(np.std(candidate_probabilities, ddof=1)) if len(candidate_probabilities) > 1 else 0.0
            probability_lcb = probability_mean - OPPORTUNITY_LCB_Z * probability_std
            rows.append(
                (
                    probability_lcb,
                    utility_lcb,
                    index,
                    probability_mean,
                    probability_std,
                    utility_mean,
                    utility_std,
                )
            )
        best = max(
            rows,
            key=lambda row: (
                row[0],
                row[1],
                -records[row[2]].pair.translation_m,
                records[row[2]].pair.neighbor.frame_id,
            ),
        ) if rows else None
        admitted = best is not None and best[0] > MIN_OPPORTUNITY_PROBABILITY and best[1] > 0.0
        if admitted:
            selected = best[2]
            learned_override_count += 1
            decision = "OPPORTUNITY_AWARE_OVERRIDE"
        else:
            selected = generic
            decision = "GENERIC_FALLBACK"
        selected_scores[selected] = 1.0
        receipts.append(
            {
                "reference_frame_id": reference,
                "decision": decision,
                "generic_neighbor_frame_id": records[generic].pair.neighbor.frame_id,
                "selected_neighbor_frame_id": records[selected].pair.neighbor.frame_id,
                "best_non_generic_opportunity_probability_mean": best[3] if best is not None else 0.0,
                "best_non_generic_opportunity_probability_std": best[4] if best is not None else 0.0,
                "best_non_generic_opportunity_probability_lcb": best[0] if best is not None else float("-inf"),
                "best_non_generic_utility_mean_advantage": best[5] if best is not None else 0.0,
                "best_non_generic_utility_std": best[6] if best is not None else 0.0,
                "best_non_generic_utility_lcb": best[1] if best is not None else float("-inf"),
            }
        )
    return selected_scores, {
        "reference_count": len(by_reference),
        "learned_override_count": learned_override_count,
        "generic_fallback_count": len(by_reference) - learned_override_count,
        "selection_receipt_sha256": hashlib.sha256(shared.canonical_json_bytes(receipts)).hexdigest().upper(),
        "receipts": receipts,
    }


def run_lofo(
    datasets: Mapping[str, tuple[list[scorer.CandidateRecord], dict[str, Any], int]],
) -> dict[str, Any]:
    audit = r23.dataset_audit(datasets)
    source_opportunity_audit: dict[str, Any] = {}
    for source, rows in datasets.items():
        _labels, _weights, source_opportunity_audit[source] = opportunity_targets(rows[0], [source] * len(rows[0]))
    folds: dict[str, Any] = {}
    parameter_count: int | None = None
    for held_source in r21.SOURCE_NAMES:
        train_sources = [source for source in r21.SOURCE_NAMES if source != held_source]
        train_records: list[scorer.CandidateRecord] = []
        train_source_rows: list[str] = []
        for source in train_sources:
            records = datasets[source][0]
            train_records.extend(records)
            train_source_rows.extend([source] * len(records))
        transform = r23.StructuredFeatureTransform.fit(train_records)
        models: list[OpportunityAwareUtilityRanker] = []
        training_receipts: list[dict[str, Any]] = []
        for seed in SEEDS:
            model, receipt = train_ranker(train_records, train_source_rows, transform, seed)
            models.append(model)
            training_receipts.append(receipt)
        parameter_count = training_receipts[0]["parameter_count"]
        held_records = datasets[held_source][0]
        means, log_variances, opportunity_logits = ensemble_predictions(held_records, transform, models)
        ungated_scores = np.mean(means, axis=0)
        gated_scores, gate = opportunity_no_regret_gate(
            held_records,
            means,
            log_variances,
            opportunity_logits,
        )
        folds[held_source] = {
            "held_source_excluded_from_normalizer_fit": True,
            "held_source_excluded_from_model_fit": True,
            "held_source_target_excluded_from_stopping_uncertainty_and_selection": True,
            "held_source_opportunity_label_excluded_from_fit_and_selection": True,
            "source_qualified_parent_sets_disjoint": True,
            "train_sources": train_sources,
            "training_candidate_count": len(train_records),
            "training_parent_count": len(
                {(source, record.parent_id) for source, record in zip(train_source_rows, train_records, strict=True)}
            ),
            "training_receipts": training_receipts,
            "normalizer_sha256": transform.receipt_sha256(),
            "ungated_utility_metrics": r21.fold_metrics(held_records, ungated_scores),
            "opportunity_no_regret_gate": gate,
            "metrics": r21.fold_metrics(held_records, gated_scores),
        }
    all_checks = all(all(fold["metrics"]["checks"].values()) for fold in folds.values())
    terminal = (
        "TASK_EVIDENCE_OPPORTUNITY_AWARE_NO_REGRET_LOFO_PASS"
        if all_checks
        else "STOP_TASK_EVIDENCE_OPPORTUNITY_AWARE_NO_REGRET_LOFO_FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "successor_of": r23.SCHEMA,
        "mode": "CONSUMED_THREE_SOURCE_DEVELOPMENT_LEAVE_ONE_SOURCE_FAMILY_OUT",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside frozen body/path capsules; UNKNOWN remains unknown.",
        "candidate_table_audit": audit,
        "source_opportunity_teacher_audit": source_opportunity_audit,
        "frozen_model_family": {
            "architecture": "R23 query-conditioned cross-attention utility distribution plus strict-opportunity classification head",
            "parameter_count": parameter_count,
            "parameter_target_range": [100000, 1000000],
            "student_feature_contract_identical_to_r23": True,
            "neighbor_depth_in_student_input": False,
            "new_teacher_supervision": "candidate target gain strictly exceeds both generic and passive-coverage target gains",
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "R23 pairwise and heteroscedastic utility losses plus class-balanced strict-opportunity BCE",
            "opportunity_loss_weight": OPPORTUNITY_LOSS_WEIGHT,
            "gate": {
                "minimum_opportunity_probability_lcb": MIN_OPPORTUNITY_PROBABILITY,
                "opportunity_lcb_z": OPPORTUNITY_LCB_Z,
                "utility_lcb_z": r23.LCB_Z,
                "utility_aleatoric_gate_weight": r23.ALEATORIC_GATE_WEIGHT,
                "minimum_normalized_utility_advantage": r23.MIN_NORMALIZED_ADVANTAGE,
                "fallback": "generic pose-diversity candidate",
            },
        },
        "sources": {
            source: {
                "disposition": "CONSUMED_DEVELOPMENT",
                "candidate_count": len(rows[0]),
                "parent_count": len({record.parent_id for record in rows[0]}),
                "geometry_abstention_count": rows[2],
                "source_receipt": rows[1],
            }
            for source, rows in datasets.items()
        },
        "folds": folds,
        "terminal": terminal,
        "fresh_confirmation_source_lock_authorized": terminal == "TASK_EVIDENCE_OPPORTUNITY_AWARE_NO_REGRET_LOFO_PASS",
        "android_candidate_authorized": False,
        "read_boundary": {
            "rgb_payload_decodes": 0,
            "neighbor_depth_or_passive_coverage_in_student_input": False,
            "held_source_target_or_opportunity_label_in_fit_or_selection": False,
            "network_requests": 0,
            "r11_reads": 0,
        },
        "claim_ceiling": "Consumed three-source Development and source-family holdout evidence only; not fresh Confirmation, collision correctness, Android, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(shared.canonical_json_bytes(result)).hexdigest().upper()
    return result


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets = {
        "TUM_RGBD": r21._build_tum_records(r23.structured_candidate_features),
        "BONN_RGBD_DYNAMIC": r21._build_bonn_records(bonn_root, r23.structured_candidate_features),
        "ARKITSCENES": r21._build_arkit_records(arkit_root, r23.structured_candidate_features),
    }
    return run_lofo(datasets)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bonn-root", type=Path, default=shared.DEFAULT_BONN_ROOT)
    parser.add_argument("--arkit-root", type=Path, default=r21.DEFAULT_ARKIT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.bonn_root.resolve(), args.arkit_root.resolve())
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
