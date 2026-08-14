#!/usr/bin/env python3
"""R23 query-conditioned TARO utility ranker with a no-regret gate.

This is a consumed-Development experiment.  Candidate neighbor depth is used
only as the teacher target.  The student sees the reference evidence tensor,
candidate-relative pose, intrinsics-derived candidate geometry, and nothing
from candidate depth.  Each leave-one-source-family-out fold fits on two source
families and evaluates the third without held-source targets entering feature
normalization, model fitting, stopping, uncertainty, or selection.

Unlike R22's flattened tensor MLP, R23 keeps nine body/path queries as tokens.
Candidate geometry cross-attends to the reference task tokens, predicts a
normalized utility distribution, and may override the generic pose-diversity
candidate only when a frozen conservative advantage bound is positive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_query_conditioned_no_regret_ranker.v1"
QUERY_COUNT = 9
CELL_SHAPE = (
    len(oracle.ALONG_BIN_EDGES_M) - 1,
    len(oracle.ACROSS_BIN_EDGES_M) - 1,
    len(oracle.HEIGHT_BIN_EDGES_M) - 1,
)
GEOMETRY_CHANNELS = ("visible", "parallax", "occluded_parallax", "far_parallax")
STATIC_TOKEN_WIDTH = int(np.prod(CELL_SHAPE))
GEOMETRY_TOKEN_WIDTH = STATIC_TOKEN_WIDTH * len(GEOMETRY_CHANNELS)
BASE_FEATURE_COUNT = len(scorer.FEATURE_NAMES)
STATIC_FEATURE_COUNT = QUERY_COUNT * STATIC_TOKEN_WIDTH
GEOMETRY_FEATURE_COUNT = QUERY_COUNT * GEOMETRY_TOKEN_WIDTH
TOTAL_FEATURE_COUNT = BASE_FEATURE_COUNT + STATIC_FEATURE_COUNT + GEOMETRY_FEATURE_COUNT

# Only compact source-time pose and aggregate frustum features enter the global
# candidate token.  Query structure remains in its own tokens below.
POSE_FEATURE_INDICES = tuple(range(14))
POSE_TRANSFORMED_WIDTH = len(POSE_FEATURE_INDICES) * 3
EMBED_DIM = 48
ATTENTION_HEADS = 4
STATIC_HIDDEN = 96
GEOMETRY_HIDDEN = 160
POSE_HIDDEN = 64
FUSION_HIDDEN = (96, 48)
RESIDUAL_SCALE = 1.0
SEEDS = (23011, 23029, 23041)
EPOCHS = 220
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
REGRESSION_LOSS_WEIGHT = 0.25
GRADIENT_CLIP_NORM = 5.0

# Frozen before the real R23 LOFO run.  Aleatoric uncertainty is down-weighted
# because candidate depth contains irreducible observation noise; ensemble
# disagreement is the primary OOD signal for the policy gate.
LCB_Z = 1.0
ALEATORIC_GATE_WEIGHT = 0.05
MIN_NORMALIZED_ADVANTAGE = 0.0


def structured_candidate_features(
    context: scorer.ReferenceContext,
    pair: Any,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build source-time base, task-token, and candidate-token features."""
    base_features, analytic = scorer.source_time_candidate_features(context, pair)
    shared.require(context.static.shape == (QUERY_COUNT, *CELL_SHAPE), "R23 static task tensor shape drift")
    relative = np.linalg.inv(context.row.reference.camera_to_world) @ pair.neighbor.camera_to_world
    inverse = np.linalg.inv(relative)
    points_rows: list[np.ndarray] = []
    along_rows: list[np.ndarray] = []
    for query in context.queries:
        centers, along = scorer._cell_centers(query)
        points_rows.append(centers)
        along_rows.append(along)
    points_ref = np.stack(points_rows, axis=0)
    along = np.stack(along_rows, axis=0)
    unknown = ~context.static.reshape(QUERY_COUNT, -1)
    points_neighbor = points_ref @ inverse[:3, :3].T + inverse[:3, 3]
    ref_z = points_ref[..., 2]
    neighbor_z = points_neighbor[..., 2]
    width, height = tum.LOW_SIZE_WH
    k = context.intrinsics
    ref_u = k[0, 0] * points_ref[..., 0] / np.maximum(ref_z, 1e-9) + k[0, 2]
    ref_v = k[1, 1] * points_ref[..., 1] / np.maximum(ref_z, 1e-9) + k[1, 2]
    nei_u = k[0, 0] * points_neighbor[..., 0] / np.maximum(neighbor_z, 1e-9) + k[0, 2]
    nei_v = k[1, 1] * points_neighbor[..., 1] / np.maximum(neighbor_z, 1e-9) + k[1, 2]
    visible = (
        unknown
        & (neighbor_z >= adapter.DEPTH_RANGE_M[0])
        & (neighbor_z <= adapter.DEPTH_RANGE_M[1])
        & (nei_u >= 0.0)
        & (nei_u < width)
        & (nei_v >= 0.0)
        & (nei_v < height)
    )
    parallax_weight = np.clip(np.sqrt((nei_u - ref_u) ** 2 + (nei_v - ref_v) ** 2) / 20.0, 0.0, 1.0)
    ref_col = np.clip(np.rint(ref_u).astype(np.int64), 0, width - 1)
    ref_row = np.clip(np.rint(ref_v).astype(np.int64), 0, height - 1)
    sampled = context.low_depth[ref_row, ref_col]
    sample_valid = context.valid[ref_row, ref_col]
    occluded = visible & sample_valid & (sampled + 0.05 < ref_z)
    far_weight = np.clip(along / adapter.HORIZON_M, 0.0, 1.0)
    geometry = np.stack(
        (
            visible.astype(np.float64),
            parallax_weight * visible,
            parallax_weight * occluded,
            parallax_weight * visible * far_weight,
        ),
        axis=-1,
    ).reshape(QUERY_COUNT, GEOMETRY_TOKEN_WIDTH)
    result = np.concatenate((base_features, context.static.reshape(-1), geometry.reshape(-1))).astype(np.float64)
    shared.require(result.shape == (TOTAL_FEATURE_COUNT,), "R23 structured feature width drift")
    shared.require(np.all(np.isfinite(result)), "R23 structured feature non-finite")
    return result, analytic


def _reference_blocks(raw: np.ndarray, records: Sequence[scorer.CandidateRecord]) -> tuple[np.ndarray, np.ndarray]:
    z = np.zeros_like(raw)
    unit = np.zeros_like(raw)
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    for indices in by_reference.values():
        local = raw[indices]
        mean = np.mean(local, axis=0)
        scale = np.std(local, axis=0)
        scale[scale < 1e-9] = 1.0
        z[indices] = (local - mean) / scale
        low = np.min(local, axis=0)
        span = np.max(local, axis=0) - low
        span[span < 1e-9] = 1.0
        unit[indices] = (local - low) / span
    return z, unit


@dataclass(frozen=True)
class ModelInputs:
    pose: np.ndarray
    task_tokens: np.ndarray
    candidate_tokens: np.ndarray
    generic_base: np.ndarray


class StructuredFeatureTransform:
    def __init__(self, pose_mean: np.ndarray, pose_scale: np.ndarray):
        self.pose_mean = pose_mean
        self.pose_scale = pose_scale

    @classmethod
    def fit(cls, records: Sequence[scorer.CandidateRecord]) -> "StructuredFeatureTransform":
        shared.require(bool(records), "R23 transform fit records empty")
        base = np.stack([record.features[:BASE_FEATURE_COUNT] for record in records]).astype(np.float64)
        pose = base[:, POSE_FEATURE_INDICES]
        mean = np.mean(pose, axis=0)
        scale = np.std(pose, axis=0)
        scale[scale < 1e-9] = 1.0
        return cls(mean, scale)

    def apply(self, records: Sequence[scorer.CandidateRecord]) -> ModelInputs:
        raw = np.stack([record.features for record in records]).astype(np.float64)
        shared.require(raw.shape[1] == TOTAL_FEATURE_COUNT, "R23 transform feature width drift")
        base = raw[:, :BASE_FEATURE_COUNT]
        pose_raw = base[:, POSE_FEATURE_INDICES]
        pose_z, pose_unit = _reference_blocks(pose_raw, records)
        pose = np.concatenate(((pose_raw - self.pose_mean) / self.pose_scale, pose_z, pose_unit), axis=1)
        static_start = BASE_FEATURE_COUNT
        geometry_start = static_start + STATIC_FEATURE_COUNT
        task = raw[:, static_start:geometry_start].reshape(-1, QUERY_COUNT, STATIC_TOKEN_WIDTH)
        candidate = raw[:, geometry_start:].reshape(-1, QUERY_COUNT, GEOMETRY_TOKEN_WIDTH)
        translation_position = POSE_FEATURE_INDICES.index(scorer.FEATURE_NAMES.index("translation_m"))
        rotation_position = POSE_FEATURE_INDICES.index(scorer.FEATURE_NAMES.index("rotation_deg"))
        generic_base = pose_unit[:, translation_position] + r21.BASE_ROTATION_WEIGHT * pose_unit[:, rotation_position]
        shared.require(pose.shape[1] == POSE_TRANSFORMED_WIDTH, "R23 pose transform width drift")
        shared.require(
            np.all(np.isfinite(pose))
            and np.all(np.isfinite(task))
            and np.all(np.isfinite(candidate))
            and np.all(np.isfinite(generic_base)),
            "R23 transformed feature non-finite",
        )
        return ModelInputs(
            pose.astype(np.float32),
            task.astype(np.float32),
            candidate.astype(np.float32),
            generic_base.astype(np.float32),
        )

    def receipt_sha256(self) -> str:
        return hashlib.sha256(self.pose_mean.tobytes() + self.pose_scale.tobytes()).hexdigest().upper()


class QueryConditionedUtilityRanker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.task_encoder = nn.Sequential(
            nn.Linear(STATIC_TOKEN_WIDTH, STATIC_HIDDEN),
            nn.GELU(),
            nn.Linear(STATIC_HIDDEN, EMBED_DIM),
        )
        self.candidate_encoder = nn.Sequential(
            nn.Linear(GEOMETRY_TOKEN_WIDTH, GEOMETRY_HIDDEN),
            nn.GELU(),
            nn.Linear(GEOMETRY_HIDDEN, EMBED_DIM),
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(POSE_TRANSFORMED_WIDTH, POSE_HIDDEN),
            nn.GELU(),
            nn.Linear(POSE_HIDDEN, EMBED_DIM),
        )
        self.query_embedding = nn.Parameter(torch.empty(QUERY_COUNT, EMBED_DIM))
        nn.init.normal_(self.query_embedding, mean=0.0, std=0.02)
        self.cross_attention = nn.MultiheadAttention(EMBED_DIM, ATTENTION_HEADS, batch_first=True)
        self.task_norm = nn.LayerNorm(EMBED_DIM)
        self.candidate_norm = nn.LayerNorm(EMBED_DIM)
        self.fusion = nn.Sequential(
            nn.Linear(EMBED_DIM * 5, FUSION_HIDDEN[0]),
            nn.GELU(),
            nn.Linear(FUSION_HIDDEN[0], FUSION_HIDDEN[1]),
            nn.GELU(),
            nn.Linear(FUSION_HIDDEN[1], 2),
        )

    def forward(
        self,
        pose: torch.Tensor,
        task_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        generic_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        mean = generic_base + RESIDUAL_SCALE * torch.tanh(output[:, 0])
        log_variance = torch.clamp(output[:, 1], min=-5.0, max=1.0)
        return mean, log_variance


def _balanced_candidate_targets(
    records: Sequence[scorer.CandidateRecord],
    sources: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    shared.require(len(records) == len(sources), "R23 target source alignment drift")
    targets = np.zeros(len(records), dtype=np.float64)
    by_reference: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, (record, source) in enumerate(zip(records, sources, strict=True)):
        shared.require(record.target_gain is not None, "R23 teacher target missing")
        by_reference[(source, record.reference_id)].append(index)
    for indices in by_reference.values():
        local = np.asarray([float(records[index].target_gain) for index in indices], dtype=np.float64)
        low = float(np.min(local))
        span = float(np.max(local) - low)
        targets[indices] = 0.0 if span < 1e-9 else (local - low) / span

    source_parents: dict[str, set[str]] = defaultdict(set)
    parent_references: dict[tuple[str, str], set[str]] = defaultdict(set)
    reference_counts: dict[tuple[str, str], int] = defaultdict(int)
    for record, source in zip(records, sources, strict=True):
        source_parents[source].add(record.parent_id)
        parent_references[(source, record.parent_id)].add(record.reference_id)
        reference_counts[(source, record.reference_id)] += 1
    source_count = len(source_parents)
    weights = np.asarray(
        [
            1.0
            / (
                source_count
                * len(source_parents[source])
                * len(parent_references[(source, record.parent_id)])
                * reference_counts[(source, record.reference_id)]
            )
            for record, source in zip(records, sources, strict=True)
        ],
        dtype=np.float64,
    )
    weights *= len(weights) / np.sum(weights)
    return targets.astype(np.float32), weights.astype(np.float32)


def _torch_inputs(inputs: ModelInputs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.from_numpy(inputs.pose),
        torch.from_numpy(inputs.task_tokens),
        torch.from_numpy(inputs.candidate_tokens),
        torch.from_numpy(inputs.generic_base),
    )


def train_ranker(
    records: Sequence[scorer.CandidateRecord],
    sources: Sequence[str],
    transform: StructuredFeatureTransform,
    seed: int,
    epochs: int = EPOCHS,
) -> tuple[QueryConditionedUtilityRanker, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    model_inputs = transform.apply(records)
    pose, task, candidate, base = _torch_inputs(model_inputs)
    high_np, low_np, pair_weights_np = r21._pair_indices(records, sources)
    targets_np, candidate_weights_np = _balanced_candidate_targets(records, sources)
    high = torch.from_numpy(high_np)
    low = torch.from_numpy(low_np)
    pair_weights = torch.from_numpy(pair_weights_np)
    targets = torch.from_numpy(targets_np)
    candidate_weights = torch.from_numpy(candidate_weights_np)
    model = QueryConditionedUtilityRanker()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    final_rank = float("nan")
    final_regression = float("nan")
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        mean, log_variance = model(pose, task, candidate, base)
        rank_loss = torch.mean(F.softplus(-(mean[high] - mean[low])) * pair_weights)
        regression_rows = 0.5 * (torch.exp(-log_variance) * (mean - targets) ** 2 + log_variance)
        regression_loss = torch.mean(regression_rows * candidate_weights)
        loss = rank_loss + REGRESSION_LOSS_WEIGHT * regression_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        final_rank = float(rank_loss.detach().cpu())
        final_regression = float(regression_loss.detach().cpu())
    state_bytes = b"".join(value.detach().cpu().numpy().tobytes() for _name, value in sorted(model.state_dict().items()))
    return model, {
        "seed": seed,
        "epochs": epochs,
        "pair_count": len(high_np),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_pairwise_loss": final_rank,
        "final_heteroscedastic_regression_loss": final_regression,
        "model_state_sha256": hashlib.sha256(state_bytes).hexdigest().upper(),
    }


def ensemble_predictions(
    records: Sequence[scorer.CandidateRecord],
    transform: StructuredFeatureTransform,
    models: Sequence[QueryConditionedUtilityRanker],
) -> tuple[np.ndarray, np.ndarray]:
    inputs = _torch_inputs(transform.apply(records))
    means: list[np.ndarray] = []
    log_variances: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            mean, log_variance = model(*inputs)
            means.append(mean.cpu().numpy().astype(np.float64))
            log_variances.append(log_variance.cpu().numpy().astype(np.float64))
    return np.stack(means), np.stack(log_variances)


def _generic_index(records: Sequence[scorer.CandidateRecord], indices: Sequence[int]) -> int:
    return max(
        indices,
        key=lambda index: (
            records[index].pair.translation_m,
            records[index].pair.rotation_deg,
            -records[index].pair.gap_s,
            records[index].pair.neighbor.frame_id,
        ),
    )


def no_regret_gate(
    records: Sequence[scorer.CandidateRecord],
    ensemble_means: np.ndarray,
    ensemble_log_variances: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Freeze one selection per reference without reading teacher targets."""
    shared.require(ensemble_means.shape == ensemble_log_variances.shape, "R23 ensemble shape mismatch")
    shared.require(ensemble_means.shape[1] == len(records), "R23 ensemble record alignment mismatch")
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    selected_scores = np.full(len(records), -1e9, dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    learned_override_count = 0
    for reference, indices in sorted(by_reference.items()):
        generic = _generic_index(records, indices)
        candidate_rows: list[tuple[float, int, float, float, float]] = []
        for index in indices:
            if index == generic:
                continue
            delta = ensemble_means[:, index] - ensemble_means[:, generic]
            mean_advantage = float(np.mean(delta))
            epistemic_variance = float(np.var(delta, ddof=1)) if len(delta) > 1 else 0.0
            aleatoric_variance = float(
                np.mean(np.exp(ensemble_log_variances[:, index]) + np.exp(ensemble_log_variances[:, generic]))
            )
            total_std = float(np.sqrt(max(0.0, epistemic_variance + ALEATORIC_GATE_WEIGHT * aleatoric_variance)))
            lcb = mean_advantage - LCB_Z * total_std - MIN_NORMALIZED_ADVANTAGE
            candidate_rows.append((lcb, index, mean_advantage, float(np.sqrt(epistemic_variance)), float(np.sqrt(aleatoric_variance))))
        best = max(
            candidate_rows,
            key=lambda row: (row[0], -records[row[1]].pair.translation_m, records[row[1]].pair.neighbor.frame_id),
        ) if candidate_rows else None
        if best is not None and best[0] > 0.0:
            selected = best[1]
            learned_override_count += 1
            decision = "LEARNED_OVERRIDE"
            lcb, _index, mean_advantage, epistemic_std, aleatoric_std = best
        else:
            selected = generic
            decision = "GENERIC_FALLBACK"
            lcb = best[0] if best is not None else float("-inf")
            mean_advantage = best[2] if best is not None else 0.0
            epistemic_std = best[3] if best is not None else 0.0
            aleatoric_std = best[4] if best is not None else 0.0
        selected_scores[selected] = 1.0
        receipts.append(
            {
                "reference_frame_id": reference,
                "decision": decision,
                "generic_neighbor_frame_id": records[generic].pair.neighbor.frame_id,
                "selected_neighbor_frame_id": records[selected].pair.neighbor.frame_id,
                "best_non_generic_mean_advantage": mean_advantage,
                "best_non_generic_epistemic_std": epistemic_std,
                "best_non_generic_aleatoric_std": aleatoric_std,
                "best_non_generic_lcb": lcb,
            }
        )
    return selected_scores, {
        "reference_count": len(by_reference),
        "learned_override_count": learned_override_count,
        "generic_fallback_count": len(by_reference) - learned_override_count,
        "selection_receipt_sha256": hashlib.sha256(shared.canonical_json_bytes(receipts)).hexdigest().upper(),
        "receipts": receipts,
    }


def dataset_audit(
    datasets: Mapping[str, tuple[list[scorer.CandidateRecord], dict[str, Any], int]],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    all_identity_rows: list[tuple[str, str, str, str]] = []
    for source, (records, _receipt, abstentions) in datasets.items():
        by_reference: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            shared.require(record.target_gain is not None, "R23 audit target missing")
            by_reference[record.reference_id].append(index)
            all_identity_rows.append((source, record.parent_id, record.reference_id, record.pair.neighbor.frame_id))
        strict_improvement_references = 0
        non_tied_references = 0
        for indices in by_reference.values():
            generic = _generic_index(records, indices)
            gains = [int(records[index].target_gain) for index in indices]
            non_tied_references += int(max(gains) > min(gains))
            strict_improvement_references += int(max(gains) > int(records[generic].target_gain))
        rows[source] = {
            "candidate_count": len(records),
            "reference_count": len(by_reference),
            "parent_count": len({record.parent_id for record in records}),
            "geometry_abstention_count": abstentions,
            "non_tied_teacher_reference_count": non_tied_references,
            "strict_improvement_over_generic_reference_count": strict_improvement_references,
            "feature_width": TOTAL_FEATURE_COUNT,
        }
    duplicate_count = len(all_identity_rows) - len(set(all_identity_rows))
    checks = {
        "three_source_families_present": set(rows) == set(r21.SOURCE_NAMES),
        "minimum_total_candidates_1000": sum(row["candidate_count"] for row in rows.values()) >= 1000,
        "minimum_total_parents_40": sum(row["parent_count"] for row in rows.values()) >= 40,
        "every_source_has_non_tied_teacher_references": all(row["non_tied_teacher_reference_count"] > 0 for row in rows.values()),
        "every_source_has_strict_improvement_denominator": all(row["strict_improvement_over_generic_reference_count"] > 0 for row in rows.values()),
        "candidate_identity_unique_within_source": duplicate_count == 0,
        "neighbor_depth_excluded_from_student_features": True,
    }
    shared.require(all(checks.values()), "R23 candidate-table audit failed")
    return {
        "sources": rows,
        "total_candidate_count": sum(row["candidate_count"] for row in rows.values()),
        "source_qualified_parent_count": sum(row["parent_count"] for row in rows.values()),
        "duplicate_candidate_identity_count": duplicate_count,
        "checks": checks,
    }


def run_lofo(
    datasets: Mapping[str, tuple[list[scorer.CandidateRecord], dict[str, Any], int]],
) -> dict[str, Any]:
    audit = dataset_audit(datasets)
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
        transform = StructuredFeatureTransform.fit(train_records)
        models: list[QueryConditionedUtilityRanker] = []
        training_receipts: list[dict[str, Any]] = []
        for seed in SEEDS:
            model, receipt = train_ranker(train_records, train_source_rows, transform, seed)
            models.append(model)
            training_receipts.append(receipt)
        parameter_count = training_receipts[0]["parameter_count"]
        held_records = datasets[held_source][0]
        means, log_variances = ensemble_predictions(held_records, transform, models)
        ungated_scores = np.mean(means, axis=0)
        gated_scores, gate = no_regret_gate(held_records, means, log_variances)
        folds[held_source] = {
            "held_source_excluded_from_normalizer_fit": True,
            "held_source_excluded_from_model_fit": True,
            "held_source_target_excluded_from_stopping_uncertainty_and_selection": True,
            "source_qualified_parent_sets_disjoint": True,
            "train_sources": train_sources,
            "training_candidate_count": len(train_records),
            "training_parent_count": len(
                {(source, record.parent_id) for source, record in zip(train_source_rows, train_records, strict=True)}
            ),
            "training_receipts": training_receipts,
            "normalizer_sha256": transform.receipt_sha256(),
            "ungated_metrics": r21.fold_metrics(held_records, ungated_scores),
            "no_regret_gate": gate,
            "metrics": r21.fold_metrics(held_records, gated_scores),
        }
    all_checks = all(all(fold["metrics"]["checks"].values()) for fold in folds.values())
    terminal = (
        "TASK_EVIDENCE_QUERY_CONDITIONED_NO_REGRET_LOFO_PASS"
        if all_checks
        else "STOP_TASK_EVIDENCE_QUERY_CONDITIONED_NO_REGRET_LOFO_FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "CONSUMED_THREE_SOURCE_DEVELOPMENT_LEAVE_ONE_SOURCE_FAMILY_OUT",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside frozen body/path capsules; UNKNOWN remains unknown.",
        "candidate_table_audit": audit,
        "frozen_model_family": {
            "architecture": "nine task tokens and nine candidate-geometry tokens with pose-conditioned four-head cross-attention, heteroscedastic utility head, and three-seed no-regret ensemble",
            "parameter_count": parameter_count,
            "parameter_target_range": [100000, 1000000],
            "feature_contract": {
                "base_feature_names": list(scorer.FEATURE_NAMES),
                "pose_feature_names": [scorer.FEATURE_NAMES[index] for index in POSE_FEATURE_INDICES],
                "task_token_shape": [QUERY_COUNT, STATIC_TOKEN_WIDTH],
                "candidate_token_shape": [QUERY_COUNT, GEOMETRY_TOKEN_WIDTH],
                "candidate_geometry_channels": list(GEOMETRY_CHANNELS),
                "raw_feature_count": TOTAL_FEATURE_COUNT,
                "neighbor_depth_in_student_input": False,
                "teacher_target": "per-reference normalized novel evidence gain from candidate neighbor depth",
            },
            "embedding_dim": EMBED_DIM,
            "attention_heads": ATTENTION_HEADS,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "source-parent-reference-balanced pairwise softplus plus heteroscedastic normalized-utility regression",
            "uncertainty": "ensemble advantage disagreement plus down-weighted predicted aleatoric variance",
            "no_regret_gate": {
                "lcb_z": LCB_Z,
                "aleatoric_gate_weight": ALEATORIC_GATE_WEIGHT,
                "minimum_normalized_advantage": MIN_NORMALIZED_ADVANTAGE,
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
        "fresh_confirmation_source_lock_authorized": terminal == "TASK_EVIDENCE_QUERY_CONDITIONED_NO_REGRET_LOFO_PASS",
        "android_candidate_authorized": False,
        "read_boundary": {
            "rgb_payload_decodes": 0,
            "neighbor_depth_in_student_input": False,
            "held_source_target_in_fit_or_selection": False,
            "network_requests": 0,
            "r11_reads": 0,
        },
        "claim_ceiling": "Consumed three-source Development and source-family holdout evidence only; not fresh Confirmation, collision correctness, Android, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(shared.canonical_json_bytes(result)).hexdigest().upper()
    return result


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets = {
        "TUM_RGBD": r21._build_tum_records(structured_candidate_features),
        "BONN_RGBD_DYNAMIC": r21._build_bonn_records(bonn_root, structured_candidate_features),
        "ARKITSCENES": r21._build_arkit_records(arkit_root, structured_candidate_features),
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
