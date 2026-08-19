#!/usr/bin/env python3
"""R33 fusion of R31 reliability tokens and R32 source-anchored geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.research.taro_o1r_r12_clear_observability_runtime import (
    task_evidence_multi_candidate_reliability_consistency as r31,
)


SCHEMA = "blindassist.taro.task_evidence_geometry_fused_reliability_consistency.v1"
R32_FEATURE_NAMES = (
    "predicted_novel_cell_count",
    "predicted_observed_cell_count",
    "predicted_candidate_valid_fraction",
    "predicted_candidate_stable_fraction",
    "reference_anchor_median_abs_log_error",
    "reference_anchor_p90_abs_log_error",
    "translation_m_duplicate",
    "rotation_deg_duplicate",
    "gap_s_duplicate",
)
R32_AVAILABILITY_NAME = "r32_source_anchored_geometry_available"


class R33Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R33Error(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def load_r32_lookup(path: Path) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    require(path.is_file(), f"R33 R32 feature cache absent: {path}")
    with np.load(path, allow_pickle=False) as value:
        references = value["reference_ids"].astype(str)
        neighbors = value["neighbor_ids"].astype(str)
        features = value["features"].astype(np.float64)
        source = str(value["source"])
        schema = str(value["schema"])
    require(features.shape == (len(references), len(R32_FEATURE_NAMES)), "R33 R32 feature width drift")
    lookup = {
        (reference, neighbor): row
        for reference, neighbor, row in zip(references, neighbors, features, strict=True)
    }
    require(len(lookup) == len(references), "R33 duplicate R32 candidate identity")
    return lookup, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
        "schema": schema,
        "source": source,
        "candidate_count": len(lookup),
    }


def augment_dataset(
    dataset: r31.SourceDataset,
    lookup: Mapping[tuple[str, str], np.ndarray],
    require_complete: bool,
) -> dict[str, Any]:
    available = 0
    available_parents: set[str] = set()
    missing_parents: set[str] = set()
    for record in dataset.records:
        row = lookup.get((record.reference_id, record.pair.neighbor.frame_id))
        if row is None:
            geometry = np.zeros(len(R32_FEATURE_NAMES), dtype=np.float64)
            availability = 0.0
            missing_parents.add(record.parent_id)
        else:
            geometry = np.asarray(row, dtype=np.float64)
            availability = 1.0
            available += 1
            available_parents.add(record.parent_id)
        record.features = np.concatenate(
            (np.asarray(record.features, dtype=np.float64), geometry, np.asarray([availability]))
        )
    if require_complete:
        require(available == len(dataset.records), f"R33 complete R32 join missing: {dataset.source}")
    require(
        all(record.features.shape == (len(r31.GLOBAL_FEATURE_NAMES) + len(R32_FEATURE_NAMES) + 1,) for record in dataset.records),
        "R33 augmented feature width drift",
    )
    return {
        "candidate_count": len(dataset.records),
        "available_candidate_count": available,
        "missing_candidate_count": len(dataset.records) - available,
        "available_parent_ids": sorted(available_parents),
        "missing_parent_ids": sorted(missing_parents - available_parents),
        "missing_semantics": "UNKNOWN_R32_AND_MASK_ZERO_R31_PRESERVED",
    }


def run(
    r31_cache_root: Path,
    r32_main_root: Path,
    r32_supplement_root: Path,
    prediction_root: Path | None,
) -> dict[str, Any]:
    datasets: dict[str, r31.SourceDataset] = {}
    joins: dict[str, Any] = {}
    receipts: dict[str, Any] = {}
    for source in r31.SOURCE_NAMES:
        dataset = r31.load_cache(r31_cache_root, source)
        feature_root = (
            r32_main_root
            if source in {"ARKITSCENES", "BONN_RGBD_DYNAMIC"}
            else r32_supplement_root
        )
        lookup, receipt = load_r32_lookup(feature_root / f"{source.lower()}-v1.npz")
        joins[source] = augment_dataset(
            dataset,
            lookup,
            require_complete=source != "TUM_RGBD",
        )
        receipts[source] = receipt
        datasets[source] = dataset
    result = r31.run_lofo(datasets, prediction_root)
    result.pop("content_sha256", None)
    result["schema"] = SCHEMA
    result["mode"] = "CONSUMED_FOUR_SOURCE_DEVELOPMENT_GEOMETRY_FUSED_LEAVE_ONE_SOURCE_FAMILY_OUT"
    result["mechanism"]["name"] = "R31_RELIABILITY_CONSISTENCY_PLUS_R32_SOURCE_ANCHORED_MONOCULAR_GEOMETRY"
    result["mechanism"]["r32_feature_names"] = list(R32_FEATURE_NAMES)
    result["mechanism"]["r32_availability_feature"] = R32_AVAILABILITY_NAME
    result["mechanism"]["candidate_sensor_depth_in_scorer_input"] = False
    result["mechanism"]["candidate_rgb_derived_depth_in_scorer_input"] = True
    result["r32_feature_cache_receipts"] = receipts
    result["r32_join_receipts"] = joins
    passed = bool(result["consumed_development_pass"])
    result["terminal"] = (
        "TARO_R33_FOUR_SOURCE_GEOMETRY_FUSED_LOFO_PASS"
        if passed
        else "STOP_TARO_R33_FOUR_SOURCE_GEOMETRY_FUSED_LOFO_FAIL"
    )
    result["read_boundary"]["candidate_rgb_derived_depth_in_scorer_input"] = True
    result["read_boundary"]["candidate_sensor_depth_in_scorer_input"] = False
    result["read_boundary"]["missing_r32_feature_semantics"] = "UNKNOWN_WITH_EXPLICIT_AVAILABILITY_MASK"
    result["claim_ceiling"] = "Consumed four-source geometry-fused LOFO Development only. A PASS authorizes a new untouched confirmation lock, not Android, product, collision, navigation, deployment, or safety claims."
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r31-cache-root", type=Path, required=True)
    parser.add_argument("--r32-main-root", type=Path, required=True)
    parser.add_argument("--r32-supplement-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        args.r31_cache_root.resolve(),
        args.r32_main_root.resolve(),
        args.r32_supplement_root.resolve(),
        args.prediction_root.resolve() if args.prediction_root is not None else None,
    )
    r31._write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
