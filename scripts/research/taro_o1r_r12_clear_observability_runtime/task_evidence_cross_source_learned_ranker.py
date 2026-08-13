#!/usr/bin/env python3
"""Cross-source learned TARO ranker with leave-one-source-family-out gates.

All task targets used here are already consumed Development. Each fold fits its
normalizer and ranker on two source families and evaluates the third without
using held-source targets for fitting, hyperparameters, or stopping. The scorer
is a bounded nonlinear residual over the generic pose-diversity baseline.
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

from scripts.research.taro_o1r_r12_clear_observability_runtime import arkitscenes_balanced_pose_source_frontdoor as arkit
from scripts.research.taro_o1r_r12_clear_observability_runtime import balanced_pose_source_frontdoor as balanced
from scripts.research.taro_o1r_r12_clear_observability_runtime import positive_oracle_canary as bonn
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_arkitscenes_opportunity_confirmation as opportunity
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_oracle_canary as oracle
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as shared
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import tum_balanced_pose_source_frontdoor as tum


SCHEMA = "blindassist.taro.task_evidence_cross_source_learned_ranker.v1"
REPO_ROOT = Path(__file__).resolve().parents[3]
TUM_MANIFESTS = tum.DEFAULT_MANIFESTS + (
    REPO_ROOT / "docs/research/taro/TARO_TASK_EVIDENCE_TASK_OUTCOME_BLIND_TUM_CONFIRMATION_COHORT_R0_2026-08-13.json",
)
DEFAULT_ARKIT_ROOT = REPO_ROOT / "artifacts.local/datasets/assistive-geometry-b0-arkitscenes-20260809-r2"
SOURCE_NAMES = ("TUM_RGBD", "BONN_RGBD_DYNAMIC", "ARKITSCENES")
SEEDS = (12013, 12031, 12047)
HIDDEN_WIDTHS = (32, 16)
EPOCHS = 300
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.001
RESIDUAL_SCALE = 0.75
BASE_ROTATION_WEIGHT = 0.05
MIN_OPPORTUNITY_PARENTS = 4
MIN_STRICT_WIN_PARENTS = 3
MIN_STRICT_WIN_FRACTION = 0.5


def _build_tum_records(feature_fn: Any = scorer.source_time_candidate_features) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    frames, assets, source = tum.load_outcome_blind_roster(TUM_MANIFESTS, verify_archive_hashes=False)
    selected, capability = balanced.select_pose_capable_references(frames, oracle.MAX_REFERENCES_PER_PARENT)
    reference_observations = scorer._load_observations([row.reference.frame_id for row in selected], assets)
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    abstained = 0
    for row in selected:
        low, points, valid, _coverage = reference_observations[row.reference.frame_id]
        intrinsics = bonn._scaled_intrinsics(assets[row.reference.frame_id].intrinsics, tum.NATIVE_SIZE_WH, tum.LOW_SIZE_WH)
        queries = oracle._queries(row.reference, low, intrinsics)
        if queries is None:
            abstained += 1
            continue
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, intrinsics, queries, static)
        contexts[row.reference.frame_id] = context
        for pair in oracle.pose_proposal_pairs(row):
            features, analytic = feature_fn(context, pair)
            records.append(scorer.CandidateRecord(row.reference.parent_id, "CONSUMED_DEVELOPMENT", row.reference.frame_id, pair, features, analytic))
    observations = scorer._load_observations([record.pair.neighbor.frame_id for record in records], assets)
    scorer._attach_targets(records, contexts, observations)
    return records, {"source": source, "capability": capability}, abstained


def _build_bonn_records(root: Path, feature_fn: Any = scorer.source_time_candidate_features) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    contexts, records, capability, abstained = shared._bonn_contexts_and_records(root)
    if feature_fn is not scorer.source_time_candidate_features:
        for record in records:
            record.features, record.analytic = feature_fn(contexts[record.reference_id], record.pair)
    shared._attach_bonn_targets(records, contexts)
    return records, {"capability": capability}, abstained


def _build_arkit_records(root: Path, feature_fn: Any = scorer.source_time_candidate_features) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    frames, assets, source = arkit.load_outcome_blind_roster(root)
    selected, capability = balanced.select_pose_capable_references(frames, opportunity.MAX_REFERENCES_PER_PARENT)
    contexts: dict[str, scorer.ReferenceContext] = {}
    records: list[scorer.CandidateRecord] = []
    abstained = 0
    for row in selected:
        low, points, valid, _coverage, intrinsics = opportunity._load_observation(assets[row.reference.frame_id])
        queries = oracle._queries(row.reference, low, intrinsics)
        if queries is None:
            abstained += 1
            continue
        static = oracle.query_evidence_cells(points, valid, queries)
        context = scorer.ReferenceContext(row, low, points, valid, intrinsics, queries, static)
        contexts[row.reference.frame_id] = context
        for pair in oracle.pose_proposal_pairs(row):
            features, analytic = feature_fn(context, pair)
            records.append(scorer.CandidateRecord(row.reference.parent_id, "CONSUMED_DEVELOPMENT", row.reference.frame_id, pair, features, analytic))
    observations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for frame_id in sorted({record.pair.neighbor.frame_id for record in records}):
        low, points, valid, coverage, _intrinsics = opportunity._load_observation(assets[frame_id])
        observations[frame_id] = (low, points, valid, coverage)
    scorer._attach_targets(records, contexts, observations)
    return records, {"source": source, "capability": capability}, abstained


def _reference_blocks(records: Sequence[scorer.CandidateRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.stack([record.features for record in records]).astype(np.float64)
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
    return raw, z, unit


class FeatureTransform:
    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean = mean
        self.scale = scale

    @classmethod
    def fit(cls, records: Sequence[scorer.CandidateRecord]) -> "FeatureTransform":
        raw = np.stack([record.features for record in records]).astype(np.float64)
        mean = np.mean(raw, axis=0)
        scale = np.std(raw, axis=0)
        scale[scale < 1e-9] = 1.0
        return cls(mean, scale)

    def apply(self, records: Sequence[scorer.CandidateRecord]) -> tuple[np.ndarray, np.ndarray]:
        raw, z, unit = _reference_blocks(records)
        transformed = np.concatenate(((raw - self.mean) / self.scale, z, unit), axis=1)
        translation_index = scorer.FEATURE_NAMES.index("translation_m")
        rotation_index = scorer.FEATURE_NAMES.index("rotation_deg")
        base = unit[:, translation_index] + BASE_ROTATION_WEIGHT * unit[:, rotation_index]
        shared.require(np.all(np.isfinite(transformed)) and np.all(np.isfinite(base)), "learned feature transform non-finite")
        return transformed.astype(np.float32), base.astype(np.float32)


class BoundedResidualRanker(nn.Module):
    def __init__(self, feature_count: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_count, HIDDEN_WIDTHS[0]),
            nn.Tanh(),
            nn.Linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1]),
            nn.Tanh(),
            nn.Linear(HIDDEN_WIDTHS[1], 1),
        )

    def forward(self, features: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        return base + RESIDUAL_SCALE * torch.tanh(self.network(features).squeeze(-1))


def _pair_indices(records: Sequence[scorer.CandidateRecord], sources: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    rows: list[tuple[int, int, str, str]] = []
    for indices in by_reference.values():
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                left_gain = int(records[left].target_gain)
                right_gain = int(records[right].target_gain)
                if left_gain == right_gain:
                    continue
                high, low = (left, right) if left_gain > right_gain else (right, left)
                rows.append((high, low, sources[high], records[high].parent_id))
    shared.require(bool(rows), "cross-source pairwise dataset empty")
    counts: dict[tuple[str, str], int] = defaultdict(int)
    parents_by_source: dict[str, set[str]] = defaultdict(set)
    for _high, _low, source, parent in rows:
        counts[(source, parent)] += 1
        parents_by_source[source].add(parent)
    source_count = len(parents_by_source)
    weights = np.asarray([
        1.0 / (source_count * len(parents_by_source[source]) * counts[(source, parent)])
        for _high, _low, source, parent in rows
    ], dtype=np.float64)
    weights *= len(weights) / np.sum(weights)
    return (
        np.asarray([row[0] for row in rows], dtype=np.int64),
        np.asarray([row[1] for row in rows], dtype=np.int64),
        weights.astype(np.float32),
    )


def train_ranker(
    records: Sequence[scorer.CandidateRecord],
    sources: Sequence[str],
    transform: FeatureTransform,
    seed: int,
) -> tuple[BoundedResidualRanker, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    features_np, base_np = transform.apply(records)
    high_np, low_np, weights_np = _pair_indices(records, sources)
    features = torch.from_numpy(features_np)
    base = torch.from_numpy(base_np)
    high = torch.from_numpy(high_np)
    low = torch.from_numpy(low_np)
    weights = torch.from_numpy(weights_np)
    model = BoundedResidualRanker(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_value = float("nan")
    for _epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        scores = model(features, base)
        loss = torch.mean(F.softplus(-(scores[high] - scores[low])) * weights)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
    state_bytes = b"".join(value.detach().cpu().numpy().tobytes() for _name, value in sorted(model.state_dict().items()))
    return model, {"seed": seed, "final_pairwise_loss": loss_value, "model_state_sha256": hashlib.sha256(state_bytes).hexdigest().upper()}


def ensemble_scores(
    records: Sequence[scorer.CandidateRecord],
    transform: FeatureTransform,
    models: Sequence[BoundedResidualRanker],
) -> np.ndarray:
    features_np, base_np = transform.apply(records)
    features = torch.from_numpy(features_np)
    base = torch.from_numpy(base_np)
    with torch.no_grad():
        rows = [model(features, base).cpu().numpy().astype(np.float64) for model in models]
    return np.mean(np.stack(rows), axis=0)


def fold_metrics(records: Sequence[scorer.CandidateRecord], scores: Sequence[float]) -> dict[str, Any]:
    macro, per_parent = shared._selection_metrics(records, scores)
    opportunity_parents, strict_parents, opportunity_rows = opportunity._opportunity_counts(records, scores)
    reference_count = sum(row["reference_count"] for row in per_parent.values())
    required_strict = max(MIN_STRICT_WIN_PARENTS, int(np.ceil(MIN_STRICT_WIN_FRACTION * opportunity_parents)))
    checks = {
        "minimum_evaluated_references": reference_count >= 16,
        "minimum_evaluated_parents": len(per_parent) >= 4,
        "minimum_opportunity_parents": opportunity_parents >= MIN_OPPORTUNITY_PARENTS,
        "opportunity_denominated_strict_win_gate": strict_parents >= required_strict,
        "ranker_parent_macro_beats_passive": macro["ranker"] > macro["passive"],
        "ranker_parent_macro_beats_generic": macro["ranker"] > macro["generic"],
    }
    return {
        "parent_macro": macro,
        "reference_count": reference_count,
        "parent_count": len(per_parent),
        "opportunity_parent_count": opportunity_parents,
        "strict_win_parent_count": strict_parents,
        "required_strict_win_parent_count": required_strict,
        "checks": checks,
        "per_parent": per_parent,
        "opportunity_per_parent": opportunity_rows,
    }


def run_lofo(
    datasets: Mapping[str, tuple[list[scorer.CandidateRecord], dict[str, Any], int]],
    schema: str = SCHEMA,
    feature_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    for held_source in SOURCE_NAMES:
        train_sources = [name for name in SOURCE_NAMES if name != held_source]
        train_records: list[scorer.CandidateRecord] = []
        train_source_rows: list[str] = []
        for source in train_sources:
            records = datasets[source][0]
            train_records.extend(records)
            train_source_rows.extend([source] * len(records))
        transform = FeatureTransform.fit(train_records)
        models = []
        receipts = []
        for seed in SEEDS:
            model, receipt = train_ranker(train_records, train_source_rows, transform, seed)
            models.append(model)
            receipts.append(receipt)
        held_records = datasets[held_source][0]
        scores = ensemble_scores(held_records, transform, models)
        folds[held_source] = {
            "held_source_excluded_from_normalizer_fit": True,
            "held_source_excluded_from_model_fit": True,
            "train_sources": train_sources,
            "training_candidate_count": len(train_records),
            "training_parent_count": len({(source, record.parent_id) for source, record in zip(train_source_rows, train_records, strict=True)}),
            "training_receipts": receipts,
            "normalizer_sha256": hashlib.sha256(transform.mean.tobytes() + transform.scale.tobytes()).hexdigest().upper(),
            "metrics": fold_metrics(held_records, scores),
        }
    all_checks = all(all(value["metrics"]["checks"].values()) for value in folds.values())
    terminal = "TASK_EVIDENCE_CROSS_SOURCE_LEARNED_RANKER_LOFO_PASS" if all_checks else "STOP_TASK_EVIDENCE_CROSS_SOURCE_LEARNED_RANKER_LOFO_FAIL"
    result = {
        "schema": schema,
        "mode": "CONSUMED_THREE_SOURCE_DEVELOPMENT_LEAVE_ONE_SOURCE_FAMILY_OUT",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside frozen body/path capsules; UNKNOWN remains unknown.",
        "frozen_model_family": {
            "architecture": "generic-pose base plus bounded two-hidden-layer tanh residual scorer",
            "feature_contract": dict(feature_contract or {"raw_training_standardized": list(scorer.FEATURE_NAMES), "per_reference_zscore": True, "per_reference_minmax": True, "neighbor_depth_in_input": False}),
            "hidden_widths": list(HIDDEN_WIDTHS),
            "residual_scale": RESIDUAL_SCALE,
            "base_rotation_weight": BASE_ROTATION_WEIGHT,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "parent-and-source-balanced within-reference pairwise softplus",
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
        "fresh_confirmation_source_lock_authorized": terminal == "TASK_EVIDENCE_CROSS_SOURCE_LEARNED_RANKER_LOFO_PASS",
        "android_candidate_authorized": False,
        "read_boundary": {"rgb_payload_decodes": 0, "neighbor_depth_in_scorer_input": False, "held_source_target_in_fit": False, "network_requests": 0, "r11_reads": 0},
        "claim_ceiling": "Consumed three-source Development and source-family holdout evidence only; not fresh Confirmation, collision correctness, Android, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(shared.canonical_json_bytes(result)).hexdigest().upper()
    return result


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets: dict[str, tuple[list[scorer.CandidateRecord], dict[str, Any], int]] = {
        "TUM_RGBD": _build_tum_records(),
        "BONN_RGBD_DYNAMIC": _build_bonn_records(bonn_root),
        "ARKITSCENES": _build_arkit_records(arkit_root),
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
    parser.add_argument("--arkit-root", type=Path, default=DEFAULT_ARKIT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.bonn_root.resolve(), args.arkit_root.resolve())
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
