#!/usr/bin/env python3
"""R30 strict-opportunity selective gate over the R27 proposal.

R29 optimized proposal gain over generic but the frozen breadth gate requires a
proposal to beat both generic and passive.  R30 changes only the training label
to that strict opportunity event.  Passive candidate depth coverage remains
training-label-side and never enters held-source features or selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_selective_reprojection_gate as r29
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_reprojection_visibility_scorer as r27
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer


SCHEMA = "blindassist.taro.task_evidence_strict_opportunity_reprojection_gate.v1"


def strict_proposal_examples(
    source: str,
    records: Sequence[scorer.CandidateRecord],
    include_labels: bool,
) -> list[r29.ProposalExample]:
    base = r29.proposal_examples(source, records, include_labels=False)
    if not include_labels:
        return base
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    output: list[r29.ProposalExample] = []
    for example in base:
        indices = by_reference[example.reference_id]
        passive = max(
            indices,
            key=lambda index: (
                float(records[index].coverage),
                -records[index].pair.gap_s,
                records[index].pair.neighbor.frame_id,
            ),
        )
        proposal_target = records[example.proposal_index].target_gain
        generic_target = records[example.generic_index].target_gain
        passive_target = records[passive].target_gain
        r21.shared.require(
            proposal_target is not None and generic_target is not None and passive_target is not None,
            "R30 training target absent",
        )
        label = float(int(proposal_target) > max(int(generic_target), int(passive_target)))
        output.append(
            r29.ProposalExample(
                example.source,
                example.parent_id,
                example.reference_id,
                example.generic_index,
                example.proposal_index,
                example.features,
                label,
            )
        )
    return output


def run_lofo(
    datasets: Mapping[str, tuple[list[scorer.CandidateRecord], dict[str, Any], int]],
) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    parameter_count: int | None = None
    for held_source in r21.SOURCE_NAMES:
        training_examples: list[r29.ProposalExample] = []
        for source in r21.SOURCE_NAMES:
            if source == held_source:
                continue
            training_examples.extend(
                strict_proposal_examples(source, datasets[source][0], include_labels=True)
            )
        transform = r29.FeatureTransform.fit(training_examples)
        models: list[r29.OverrideGate] = []
        training_receipts: list[dict[str, Any]] = []
        for seed in r29.SEEDS:
            model, receipt = r29.train_gate(training_examples, transform, seed)
            models.append(model)
            training_receipts.append(receipt)
            parameter_count = receipt["parameter_count"]
        held_records = datasets[held_source][0]
        held_examples = strict_proposal_examples(
            held_source, held_records, include_labels=False
        )
        predictions = r29.ensemble_logits(held_examples, transform, models)
        scores, gate_receipt = r29.gated_selection_scores(
            held_records, held_examples, predictions
        )
        ungated_scores, ungated_receipt = r27.primary_selection_scores(held_records)
        folds[held_source] = {
            "training_sources": [source for source in r21.SOURCE_NAMES if source != held_source],
            "training_parent_count": len(
                {(example.source, example.parent_id) for example in training_examples}
            ),
            "training_example_count": len(training_examples),
            "training_strict_positive_count": sum(
                float(example.label) > 0.5 for example in training_examples
            ),
            "training_receipts": training_receipts,
            "held_parent_count": len({record.parent_id for record in held_records}),
            "held_candidate_count": len(held_records),
            "held_target_or_passive_coverage_in_feature_transform_fit_or_gate_selection": False,
            "gate_receipt": gate_receipt,
            "ungated_r27_receipt": ungated_receipt,
            "metrics": r21.fold_metrics(held_records, scores),
            "ungated_r27_metrics": r21.fold_metrics(held_records, ungated_scores),
        }
    passed = all(all(fold["metrics"]["checks"].values()) for fold in folds.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "CONSUMED_THREE_SOURCE_DEVELOPMENT_LEAVE_ONE_SOURCE_FAMILY_OUT",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside nine frozen body/path capsules; UNKNOWN remains unknown.",
        "proposal": {
            "mechanism": "R27 fixed query-aligned RGB-D-to-RGB z-buffer reprojection visibility",
            "minimum_raw_novel_cell_advantage": r29.RAW_PROPOSAL_MINIMUM_NOVEL_CELL_ADVANTAGE,
            "candidate_depth_in_input": False,
        },
        "gate": {
            "architecture": "R29 proposal-versus-generic reference-relative 64-32 GELU MLP, five-seed ensemble",
            "base_feature_names": list(r29.BASE_FEATURE_NAMES),
            "pair_feature_names": list(r29.PAIR_FEATURE_NAMES),
            "parameter_count_per_model": parameter_count,
            "seeds": list(r29.SEEDS),
            "epochs": r29.EPOCHS,
            "learning_rate": r29.LEARNING_RATE,
            "weight_decay": r29.WEIGHT_DECAY,
            "loss": "source-parent-balanced and class-balanced binary cross entropy",
            "positive_label": "proposal target gain strictly exceeds both generic and passive target gains",
            "passive_coverage_or_target_in_held_input": False,
            "acceptance": "ensemble mean logit minus one epistemic standard deviation is positive",
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
            "TASK_EVIDENCE_STRICT_OPPORTUNITY_REPROJECTION_GATE_LOFO_PASS"
            if passed
            else "STOP_TASK_EVIDENCE_STRICT_OPPORTUNITY_REPROJECTION_GATE_LOFO_FAIL"
        ),
        "fresh_confirmation_source_lock_authorized": passed,
        "android_candidate_authorized": False,
        "read_boundary": {
            "candidate_depth_in_scorer_or_gate_input": False,
            "held_source_target_or_passive_coverage_in_gate_fit_or_selection": False,
            "candidate_rgb_in_scorer_input": True,
            "network_requests": 0,
        },
        "claim_ceiling": "Consumed three-source Development and source-family holdout evidence only. A PASS would authorize a fresh task-outcome-blind confirmation lock, not collision correctness, Android, product, default-App, navigation, or safety claims.",
    }
    result["content_sha256"] = hashlib.sha256(r21.shared.canonical_json_bytes(result)).hexdigest().upper()
    return result


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets = {
        "TUM_RGBD": r29._build_tum_records(),
        "BONN_RGBD_DYNAMIC": r29._build_bonn_records(bonn_root),
        "ARKITSCENES": r29._build_arkit_records(arkit_root),
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
