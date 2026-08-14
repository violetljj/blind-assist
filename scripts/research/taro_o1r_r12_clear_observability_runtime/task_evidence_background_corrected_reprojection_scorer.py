#!/usr/bin/env python3
"""R28 background-corrected query reprojection visibility scorer.

R27 exposed transferable query-aligned correspondence signal but overrode the
generic policy too broadly on dynamic Bonn scenes.  R28 subtracts the number of
task-cell novelty hits expected from the candidate's global reprojection failure
rate.  No target-derived parameter or source identity enters the policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_cross_source_learned_ranker as r21
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pose_scorer_canary as scorer
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_reprojection_visibility_scorer as r27
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_rgb_query_interaction_ranker as r25


SCHEMA = "blindassist.taro.task_evidence_background_corrected_reprojection_scorer.v1"
MINIMUM_OVERRIDE_EXCESS_CELL_ADVANTAGE = 1.0


def background_corrected_analytic(analytic: Mapping[str, float]) -> dict[str, float]:
    """Remove task novelty expected from global correspondence failure."""

    global_novel_fraction = float(
        np.clip(
            1.0
            - float(analytic["explained_warp_coverage_fraction"])
            + float(analytic["direct_warp_coverage_fraction"])
            * (1.0 - r27.PHOTOMETRIC_RESIDUAL_QUANTILE),
            0.0,
            1.0,
        )
    )
    visible = float(analytic["candidate_visible_unknown_cell_count"])
    observed = float(analytic["reprojection_novel_cell_count"])
    expected = visible * global_novel_fraction
    excess = observed - expected
    output = dict(analytic)
    output.update(
        {
            "global_reprojection_novel_fraction": global_novel_fraction,
            "background_expected_task_novel_cell_count": expected,
            "background_corrected_excess_novel_cell_count": excess,
            "task_novel_enrichment_ratio": observed / max(expected, 1.0),
        }
    )
    return output


def background_corrected_features(
    context: scorer.ReferenceContext,
    pair: Any,
    reference_planes: np.ndarray,
    candidate_planes: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    base_features, base_analytic = r27.reprojection_visibility_features(
        context, pair, reference_planes, candidate_planes
    )
    analytic = background_corrected_analytic(base_analytic)
    features = np.concatenate(
        (
            base_features,
            np.asarray(
                [
                    analytic["global_reprojection_novel_fraction"],
                    analytic["background_expected_task_novel_cell_count"],
                    analytic["background_corrected_excess_novel_cell_count"],
                    analytic["task_novel_enrichment_ratio"],
                ],
                dtype=np.float64,
            ),
        )
    )
    r21.shared.require(
        features.shape == (12,) and np.all(np.isfinite(features)),
        "R28 feature drift",
    )
    return features, analytic


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


def primary_selection_scores(
    records: Sequence[scorer.CandidateRecord],
) -> tuple[np.ndarray, dict[str, Any]]:
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    scores = np.zeros(len(records), dtype=np.float64)
    receipts: list[dict[str, Any]] = []
    overrides = 0
    for reference_id, indices in sorted(by_reference.items()):
        generic = _generic_index(records, indices)
        best = max(
            indices,
            key=lambda index: (
                records[index].analytic["background_corrected_excess_novel_cell_count"],
                records[index].analytic["task_novel_enrichment_ratio"],
                records[index].analytic["reprojection_novel_cell_count"],
                records[index].pair.translation_m,
                records[index].pair.neighbor.frame_id,
            ),
        )
        advantage = float(
            records[best].analytic["background_corrected_excess_novel_cell_count"]
            - records[generic].analytic["background_corrected_excess_novel_cell_count"]
        )
        selected = best if advantage >= MINIMUM_OVERRIDE_EXCESS_CELL_ADVANTAGE else generic
        scores[selected] = 1.0
        overrides += int(selected != generic)
        receipts.append(
            {
                "reference_id": reference_id,
                "generic_neighbor_id": records[generic].pair.neighbor.frame_id,
                "selected_neighbor_id": records[selected].pair.neighbor.frame_id,
                "background_corrected_excess_advantage": advantage,
                "overrode_generic": selected != generic,
            }
        )
    return scores, {
        "reference_count": len(by_reference),
        "generic_override_count": overrides,
        "generic_fallback_count": len(by_reference) - overrides,
        "selection_receipt_sha256": hashlib.sha256(r21.shared.canonical_json_bytes(receipts)).hexdigest().upper(),
    }


def ungated_scores(records: Sequence[scorer.CandidateRecord]) -> np.ndarray:
    return np.asarray(
        [record.analytic["background_corrected_excess_novel_cell_count"] for record in records],
        dtype=np.float64,
    )


def _build_tum_records() -> tuple[list[scorer.CandidateRecord], dict[str, Any], int]:
    store = r25.RgbStore(r25._tum_rgb_assets(r21.TUM_MANIFESTS), "frozen TUM rgb.txt identities and source manifests")

    def feature(context: scorer.ReferenceContext, pair: Any) -> tuple[np.ndarray, dict[str, float]]:
        return background_corrected_features(
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
        return background_corrected_features(
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
        return background_corrected_features(
            context, pair, store.planes(pair.reference), store.planes(pair.neighbor)
        )

    try:
        records, source, abstained = r21._build_arkit_records(root, feature)
        source["rgb_signal_receipt"] = store.receipt()
        return records, source, abstained
    finally:
        store.close()


def evaluate(bonn_root: Path, arkit_root: Path) -> dict[str, Any]:
    datasets = {
        "TUM_RGBD": _build_tum_records(),
        "BONN_RGBD_DYNAMIC": _build_bonn_records(bonn_root),
        "ARKITSCENES": _build_arkit_records(arkit_root),
    }
    source_results: dict[str, Any] = {}
    for source, (records, receipt, abstentions) in datasets.items():
        scores, selection = primary_selection_scores(records)
        source_results[source] = {
            "candidate_count": len(records),
            "parent_count": len({record.parent_id for record in records}),
            "geometry_abstention_count": abstentions,
            "source_receipt": receipt,
            "selection": selection,
            "primary_metrics": r21.fold_metrics(records, scores),
            "ungated_metrics": r21.fold_metrics(records, ungated_scores(records)),
        }
    passed = all(all(row["primary_metrics"]["checks"].values()) for row in source_results.values())
    terminal = (
        "TASK_EVIDENCE_BACKGROUND_CORRECTED_REPROJECTION_THREE_SOURCE_PASS"
        if passed
        else "STOP_TASK_EVIDENCE_BACKGROUND_CORRECTED_REPROJECTION_TRANSFER_FAIL"
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "CONSUMED_THREE_SOURCE_DEVELOPMENT_FIXED_BACKGROUND_CORRECTION",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside nine frozen body/path capsules; UNKNOWN remains unknown.",
        "mechanism": {
            "name": "BACKGROUND_CORRECTED_QUERY_REPROJECTION_VISIBILITY",
            "expected_task_novelty": "candidate-visible unknown task cells multiplied by global reprojection novelty fraction",
            "score": "observed query-aligned reprojection novelty minus expected background novelty",
            "minimum_override_excess_cell_advantage": MINIMUM_OVERRIDE_EXCESS_CELL_ADVANTAGE,
            "source_identity_in_policy": False,
            "candidate_depth_in_scorer_input": False,
            "training_steps": 0,
            "parameters_fit_from_targets": 0,
        },
        "sources": source_results,
        "terminal": terminal,
        "fresh_confirmation_source_lock_authorized": passed,
        "android_candidate_authorized": False,
        "read_boundary": {
            "reference_rgb_and_depth_in_scorer_input": True,
            "candidate_rgb_in_scorer_input": True,
            "candidate_depth_in_scorer_input": False,
            "candidate_depth_used_only_after_selection_for_consumed_development_metric": True,
            "network_requests": 0,
        },
        "claim_ceiling": "Consumed three-source Development evidence only. A PASS would authorize a fresh task-outcome-blind source lock, not collision correctness, Android, product, default-App, navigation, or safety claims.",
    }
    result["content_sha256"] = hashlib.sha256(r21.shared.canonical_json_bytes(result)).hexdigest().upper()
    return result


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
