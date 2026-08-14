#!/usr/bin/env python3
"""R29 selective no-regret gate over the R27 reprojection proposal.

R27 supplies a fixed analytic task-visibility proposal.  R29 learns only
whether to accept that proposal instead of the generic pose-diversity choice.
Every leave-one-source-family-out fold trains on the other two source families;
held-source targets never enter feature normalization, fitting, or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_background_corrected_reprojection_scorer as r28
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_reprojection_visibility_scorer as r27
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_rgb_query_interaction_ranker as r25


SCHEMA = "blindassist.taro.task_evidence_selective_reprojection_gate.v1"
SEEDS = (29011, 29027, 29041, 29059, 29077)
HIDDEN_WIDTHS = (64, 32)
EPOCHS = 500
LEARNING_RATE = 0.005
WEIGHT_DECAY = 0.001
ENSEMBLE_LCB_Z = 1.0
RAW_PROPOSAL_MINIMUM_NOVEL_CELL_ADVANTAGE = 1.0
BASE_FEATURE_NAMES = (
    "reprojection_novel_cell_count",
    "unexplained_warp_hole_cell_count",
    "photometric_inconsistent_cell_count",
    "novel_appearance_strength_sum",
    "candidate_visible_unknown_cell_count",
    "direct_warp_coverage_fraction",
    "explained_warp_coverage_fraction",
    "photometric_residual_threshold",
    "background_corrected_excess_novel_cell_count",
    "task_novel_enrichment_ratio",
    "translation_m",
    "rotation_deg",
)
PAIR_FEATURE_NAMES = tuple(
    f"{prefix}_{name}"
    for prefix in ("proposal_minus_generic_over_reference_std", "proposal_reference_z", "generic_reference_z")
    for name in BASE_FEATURE_NAMES
)


@dataclass(frozen=True)
class ProposalExample:
    source: str
    parent_id: str
    reference_id: str
    generic_index: int
    proposal_index: int
    features: np.ndarray
    label: float | None


class OverrideGate(nn.Module):
    def __init__(self, input_width: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_width, HIDDEN_WIDTHS[0]),
            nn.GELU(),
            nn.LayerNorm(HIDDEN_WIDTHS[0]),
            nn.Linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1]),
            nn.GELU(),
            nn.Linear(HIDDEN_WIDTHS[1], 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


@dataclass(frozen=True)
class FeatureTransform:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, examples: Sequence[ProposalExample]) -> "FeatureTransform":
        r21.shared.require(bool(examples), "R29 training examples empty")
        features = np.stack([example.features for example in examples]).astype(np.float64)
        mean = np.mean(features, axis=0)
        scale = np.std(features, axis=0)
        scale[scale < 1e-6] = 1.0
        return cls(mean, scale)

    def apply(self, examples: Sequence[ProposalExample]) -> np.ndarray:
        if not examples:
            return np.empty((0, len(PAIR_FEATURE_NAMES)), dtype=np.float32)
        values = (np.stack([example.features for example in examples]) - self.mean) / self.scale
        values = np.clip(values, -6.0, 6.0)
        r21.shared.require(
            values.shape[1] == len(PAIR_FEATURE_NAMES) and np.all(np.isfinite(values)),
            "R29 transformed feature drift",
        )
        return values.astype(np.float32)


def _record_vector(record: scorer.CandidateRecord) -> np.ndarray:
    analytic = r28.background_corrected_analytic(record.analytic)
    values = np.asarray(
        [
            analytic["reprojection_novel_cell_count"],
            analytic["unexplained_warp_hole_cell_count"],
            analytic["photometric_inconsistent_cell_count"],
            analytic["novel_appearance_strength_sum"],
            analytic["candidate_visible_unknown_cell_count"],
            analytic["direct_warp_coverage_fraction"],
            analytic["explained_warp_coverage_fraction"],
            analytic["photometric_residual_threshold"],
            analytic["background_corrected_excess_novel_cell_count"],
            analytic["task_novel_enrichment_ratio"],
            record.pair.translation_m,
            record.pair.rotation_deg,
        ],
        dtype=np.float64,
    )
    r21.shared.require(
        values.shape == (len(BASE_FEATURE_NAMES),) and np.all(np.isfinite(values)),
        "R29 record feature drift",
    )
    return values


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


def proposal_examples(
    source: str,
    records: Sequence[scorer.CandidateRecord],
    include_labels: bool,
) -> list[ProposalExample]:
    """Build only source-time proposal/generic features unless labels requested."""

    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    output: list[ProposalExample] = []
    for reference_id, indices in sorted(by_reference.items()):
        vectors = np.stack([_record_vector(records[index]) for index in indices])
        mean = np.mean(vectors, axis=0)
        scale = np.std(vectors, axis=0)
        scale[scale < 1e-6] = 1.0
        generic = _generic_index(records, indices)
        proposal = max(
            indices,
            key=lambda index: (
                records[index].analytic["reprojection_novel_cell_count"],
                records[index].analytic["novel_appearance_strength_sum"],
                records[index].pair.translation_m,
                records[index].pair.neighbor.frame_id,
            ),
        )
        raw_advantage = float(
            records[proposal].analytic["reprojection_novel_cell_count"]
            - records[generic].analytic["reprojection_novel_cell_count"]
        )
        if proposal == generic or raw_advantage < RAW_PROPOSAL_MINIMUM_NOVEL_CELL_ADVANTAGE:
            continue
        proposal_vector = _record_vector(records[proposal])
        generic_vector = _record_vector(records[generic])
        features = np.concatenate(
            (
                (proposal_vector - generic_vector) / scale,
                (proposal_vector - mean) / scale,
                (generic_vector - mean) / scale,
            )
        )
        label: float | None = None
        if include_labels:
            r21.shared.require(
                records[proposal].target_gain is not None and records[generic].target_gain is not None,
                "R29 training target absent",
            )
            label = float(int(records[proposal].target_gain) > int(records[generic].target_gain))
        output.append(
            ProposalExample(
                source,
                records[proposal].parent_id,
                reference_id,
                generic,
                proposal,
                features,
                label,
            )
        )
    return output


def _balanced_weights(examples: Sequence[ProposalExample]) -> np.ndarray:
    groups = Counter((example.source, example.parent_id) for example in examples)
    labels = Counter(int(float(example.label)) for example in examples)
    r21.shared.require(set(labels) == {0, 1}, "R29 training label support incomplete")
    values = np.asarray(
        [
            1.0 / groups[(example.source, example.parent_id)]
            * (len(examples) / (2.0 * labels[int(float(example.label))]))
            for example in examples
        ],
        dtype=np.float32,
    )
    values *= len(values) / float(np.sum(values))
    return values


def train_gate(
    examples: Sequence[ProposalExample],
    transform: FeatureTransform,
    seed: int,
) -> tuple[OverrideGate, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    features = torch.from_numpy(transform.apply(examples))
    labels = torch.tensor([float(example.label) for example in examples], dtype=torch.float32)
    weights = torch.from_numpy(_balanced_weights(examples))
    model = OverrideGate(features.shape[1])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    final_loss = float("nan")
    for _epoch in range(EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        rows = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = torch.mean(rows * weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach())
    state = b"".join(
        value.detach().cpu().numpy().tobytes()
        for _name, value in sorted(model.state_dict().items())
    )
    return model, {
        "seed": seed,
        "epochs": EPOCHS,
        "training_example_count": len(examples),
        "positive_example_count": sum(float(example.label) > 0.5 for example in examples),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "final_balanced_bce": final_loss,
        "model_state_sha256": hashlib.sha256(state).hexdigest().upper(),
    }


def ensemble_logits(
    examples: Sequence[ProposalExample],
    transform: FeatureTransform,
    models: Sequence[OverrideGate],
) -> np.ndarray:
    features = torch.from_numpy(transform.apply(examples))
    if not examples:
        return np.empty((len(models), 0), dtype=np.float64)
    with torch.no_grad():
        return np.stack(
            [model(features).cpu().numpy().astype(np.float64) for model in models]
        )


def gated_selection_scores(
    records: Sequence[scorer.CandidateRecord],
    examples: Sequence[ProposalExample],
    logits: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    r21.shared.require(logits.shape[1] == len(examples), "R29 prediction alignment drift")
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    scores = np.zeros(len(records), dtype=np.float64)
    decisions: dict[str, dict[str, Any]] = {}
    accepted = 0
    for column, example in enumerate(examples):
        mean_logit = float(np.mean(logits[:, column]))
        epistemic_std = float(np.std(logits[:, column], ddof=0))
        lcb = mean_logit - ENSEMBLE_LCB_Z * epistemic_std
        accept = lcb > 0.0
        selected = example.proposal_index if accept else example.generic_index
        scores[selected] = 1.0
        accepted += int(accept)
        decisions[example.reference_id] = {
            "proposal_neighbor_id": records[example.proposal_index].pair.neighbor.frame_id,
            "generic_neighbor_id": records[example.generic_index].pair.neighbor.frame_id,
            "ensemble_mean_logit": mean_logit,
            "ensemble_epistemic_std": epistemic_std,
            "lower_confidence_logit": lcb,
            "accepted": accept,
        }
    for reference_id, indices in by_reference.items():
        if reference_id in decisions:
            continue
        scores[_generic_index(records, indices)] = 1.0
    rows = [{"reference_id": key, **value} for key, value in sorted(decisions.items())]
    return scores, {
        "raw_proposal_count": len(examples),
        "accepted_override_count": accepted,
        "rejected_override_count": len(examples) - accepted,
        "generic_total_count": len(by_reference) - accepted,
        "decision_receipt_sha256": hashlib.sha256(r21.shared.canonical_json_bytes(rows)).hexdigest().upper(),
    }


def _build_tum_records() -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = r25.RgbStore(r25._tum_rgb_assets(r21.TUM_MANIFESTS), "frozen TUM rgb.txt identities and source manifests")

    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return r27.reprojection_visibility_features(
            context, pair, store.planes(pair.reference), store.planes(pair.neighbor)
        )

    try:
        records, source, abstained = r21._build_tum_records(feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def _build_bonn_records(root: Path) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = r25.RgbStore({}, "Bonn rgb.txt-associated direct RGB paths")

    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return r27.reprojection_visibility_features(
            context, pair, store.planes(pair.reference), store.planes(pair.neighbor)
        )

    try:
        records, source, abstained = r21._build_bonn_records(root, feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def _build_arkit_records(root: Path) -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = r25.RgbStore(r25._arkit_rgb_assets(root), "ARKitScenes manifest lowres_wide identities")

    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return r27.reprojection_visibility_features(
            context, pair, store.planes(pair.reference), store.planes(pair.neighbor)
        )

    try:
        records, source, abstained = r21._build_arkit_records(root, feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def run_lofo(
    datasets: Mapping[str, tuple[list[scorer.CandidateRecord], dict[str, Any], int]],
) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    parameter_count: int | None = None
    for held_source in r21.SOURCE_NAMES:
        training_examples: list[ProposalExample] = []
        for source in r21.SOURCE_NAMES:
            if source == held_source:
                continue
            training_examples.extend(proposal_examples(source, datasets[source][0], include_labels=True))
        transform = FeatureTransform.fit(training_examples)
        models: list[OverrideGate] = []
        training_receipts: list[dict[str, Any]] = []
        for seed in SEEDS:
            model, receipt = train_gate(training_examples, transform, seed)
            models.append(model)
            training_receipts.append(receipt)
            parameter_count = receipt["parameter_count"]
        held_records = datasets[held_source][0]
        held_examples = proposal_examples(held_source, held_records, include_labels=False)
        predictions = ensemble_logits(held_examples, transform, models)
        gated_scores, gate_receipt = gated_selection_scores(
            held_records, held_examples, predictions
        )
        ungated_scores, ungated_receipt = r27.primary_selection_scores(held_records)
        folds[held_source] = {
            "training_sources": [source for source in r21.SOURCE_NAMES if source != held_source],
            "training_parent_count": len({(example.source, example.parent_id) for example in training_examples}),
            "training_example_count": len(training_examples),
            "training_positive_count": sum(float(example.label) > 0.5 for example in training_examples),
            "training_receipts": training_receipts,
            "held_parent_count": len({record.parent_id for record in held_records}),
            "held_candidate_count": len(held_records),
            "held_target_in_feature_transform_fit_or_gate_selection": False,
            "gate_receipt": gate_receipt,
            "ungated_r27_receipt": ungated_receipt,
            "metrics": r21.fold_metrics(held_records, gated_scores),
            "ungated_r27_metrics": r21.fold_metrics(held_records, ungated_scores),
        }
    passed = all(all(fold["metrics"]["checks"].values()) for fold in folds.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "CONSUMED_THREE_SOURCE_DEVELOPMENT_LEAVE_ONE_SOURCE_FAMILY_OUT",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside nine frozen body/path capsules; UNKNOWN remains unknown.",
        "proposal": {
            "mechanism": "R27 fixed query-aligned RGB-D-to-RGB z-buffer reprojection visibility",
            "minimum_raw_novel_cell_advantage": RAW_PROPOSAL_MINIMUM_NOVEL_CELL_ADVANTAGE,
            "candidate_depth_in_input": False,
        },
        "gate": {
            "architecture": "proposal-versus-generic reference-relative feature differences into 64-32 GELU MLP, five-seed ensemble",
            "base_feature_names": list(BASE_FEATURE_NAMES),
            "pair_feature_names": list(PAIR_FEATURE_NAMES),
            "parameter_count_per_model": parameter_count,
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "source-parent-balanced and class-balanced binary cross entropy for proposal target gain strictly exceeding generic",
            "acceptance": "ensemble mean logit minus one epistemic standard deviation is positive",
            "target_in_gate_input": False,
        },
        "sources": {
            source: {
                "candidate_count": len(rows[0]),
                "parent_count": len({record.parent_id for record in rows[0]}),
                "geometry_abstention_count": rows[2],
                "source_receipt": rows[1],
            }
            for source, rows in datasets.items()
        },
        "folds": folds,
        "terminal": (
            "TASK_EVIDENCE_SELECTIVE_REPROJECTION_GATE_LOFO_PASS"
            if passed
            else "STOP_TASK_EVIDENCE_SELECTIVE_REPROJECTION_GATE_LOFO_FAIL"
        ),
        "fresh_confirmation_source_lock_authorized": passed,
        "android_candidate_authorized": False,
        "read_boundary": {
            "candidate_depth_in_scorer_or_gate_input": False,
            "held_source_target_in_gate_fit_or_selection": False,
            "candidate_rgb_in_scorer_input": True,
            "network_requests": 0,
        },
        "claim_ceiling": "Consumed three-source Development and source-family holdout evidence only. A PASS would authorize a fresh task-outcome-blind confirmation lock, not collision correctness, Android, product, default-App, navigation, or safety claims.",
    }
    result["content_sha256"] = hashlib.sha256(r21.shared.canonical_json_bytes(result)).hexdigest().upper()
    return result


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets = {
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
    parser.add_argument("--bonn-root", type=Path, default=r21.shared.DEFAULT_BONN_ROOT)
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
