#!/usr/bin/env python3
"""R35 source-regime expert policy on consumed Development evidence.

The policy does not claim source-family generalization.  It binds one fixed
expert to each already-known sensor/data regime, evaluates every mapped regime,
and may only authorize a parent-disjoint confirmation inside a mapped regime.
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
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_multi_candidate_reliability_consistency as r31
from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_reprojection_visibility_scorer as r27


SCHEMA = "blindassist.taro.task_evidence_source_regime_expert_policy.v2"
SOURCE_POLICY = {
    "ARKITSCENES": "SOURCE_ANCHORED_MONOCULAR_GEOMETRY",
    "BONN_RGBD_DYNAMIC": "GEOMETRY_OPPORTUNITY_FLOOR_THEN_UTILITY_LCB",
    "TUM_RGBD": "OPPORTUNITY_MARGIN_TOP",
    "OPENLORIS_HOME": "R31_V7_RELIABILITY_CONSISTENCY",
}


class R35Error(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R35Error(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _geometry_scores(
    source: str,
    records: Sequence[Any],
    main_root: Path,
    supplement_root: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    root = main_root if source in {"ARKITSCENES", "BONN_RGBD_DYNAMIC"} else supplement_root
    path = root / f"{source.lower()}-v1.npz"
    require(path.is_file(), f"R35 geometry cache absent: {path}")
    with np.load(path, allow_pickle=False) as value:
        lookup = {
            (reference, neighbor): float(row[0])
            for reference, neighbor, row in zip(
                value["reference_ids"].astype(str),
                value["neighbor_ids"].astype(str),
                value["features"],
                strict=True,
            )
        }
    scores = np.asarray(
        [lookup.get((record.reference_id, record.pair.neighbor.frame_id), np.nan) for record in records],
        dtype=np.float64,
    )
    return scores, {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "available_candidate_count": int(np.sum(np.isfinite(scores))),
    }


def select(
    records: Sequence[Any],
    utility: np.ndarray,
    opportunity: np.ndarray,
    geometry: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if policy == "R31_V7_RELIABILITY_CONSISTENCY":
        return r31.gated_scores(records, utility, opportunity)
    output = np.zeros(len(records), dtype=np.float64)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.reference_id].append(index)
    utility_mean = np.mean(utility, axis=0)
    receipts: list[dict[str, Any]] = []
    for reference_id, indices in sorted(groups.items()):
        generic = r27._generic_index(records, indices)
        utility_proposal = max(
            indices,
            key=lambda index: (utility_mean[index], records[index].pair.neighbor.frame_id),
        )
        utility_margin = utility[:, utility_proposal] - utility[:, generic]
        utility_lcb = float(np.mean(utility_margin) - r31.LCB_Z * np.std(utility_margin, ddof=0))
        opportunity_baseline = opportunity[:, generic]
        opportunity_lcbs = {
            index: float(
                np.mean(opportunity[:, index] - opportunity_baseline)
                - r31.LCB_Z * np.std(opportunity[:, index] - opportunity_baseline, ddof=0)
            )
            for index in indices
        }
        opportunity_proposal = max(
            indices,
            key=lambda index: (opportunity_lcbs[index], records[index].pair.neighbor.frame_id),
        )
        geometry_available = all(np.isfinite(geometry[index]) for index in indices)
        geometry_proposal = (
            max(indices, key=lambda index: (geometry[index], records[index].pair.neighbor.frame_id))
            if geometry_available
            else generic
        )
        geometry_exists = (
            geometry_proposal != generic and geometry[geometry_proposal] > geometry[generic]
        )
        geometry_opportunity = opportunity[:, geometry_proposal] - opportunity_baseline
        geometry_opportunity_lcb = float(
            np.mean(geometry_opportunity) - r31.LCB_Z * np.std(geometry_opportunity, ddof=0)
        )
        if policy == "SOURCE_ANCHORED_MONOCULAR_GEOMETRY":
            selected = geometry_proposal if geometry_exists else generic
        elif policy == "GEOMETRY_OPPORTUNITY_FLOOR_THEN_UTILITY_LCB":
            selected = (
                geometry_proposal
                if geometry_exists and geometry_opportunity_lcb > -1.25
                else utility_proposal
                if utility_proposal != generic and utility_lcb > 0.0
                else generic
            )
        elif policy == "OPPORTUNITY_MARGIN_TOP":
            selected = (
                opportunity_proposal
                if opportunity_proposal != generic and opportunity_lcbs[opportunity_proposal] > 0.0
                else generic
            )
        else:
            raise R35Error(f"unsupported source-regime policy: {policy}")
        output[selected] = 1.0
        receipts.append(
            {
                "reference_id": reference_id,
                "generic_neighbor_id": records[generic].pair.neighbor.frame_id,
                "selected_neighbor_id": records[selected].pair.neighbor.frame_id,
                "policy": policy,
            }
        )
    return output, {
        "reference_count": len(groups),
        "decision_receipt_sha256": hashlib.sha256(canonical_json_bytes(receipts)).hexdigest().upper(),
    }


def evaluate(
    cache_root: Path,
    prediction_root: Path,
    geometry_main_root: Path,
    geometry_supplement_root: Path,
) -> dict[str, Any]:
    folds: dict[str, Any] = {}
    for source in r31.SOURCE_NAMES:
        dataset = r31.load_cache(cache_root, source)
        prediction_path = prediction_root / f"{source.lower()}-v1.npz"
        require(prediction_path.is_file(), f"R35 prediction cache absent: {prediction_path}")
        with np.load(prediction_path, allow_pickle=False) as value:
            utility = value["utility_ensemble"].astype(np.float64)
            opportunity = value["opportunity_ensemble"].astype(np.float64)
        geometry, geometry_receipt = _geometry_scores(
            source, dataset.records, geometry_main_root, geometry_supplement_root
        )
        policy = SOURCE_POLICY[source]
        scores, selection = select(dataset.records, utility, opportunity, geometry, policy)
        metrics = r21.fold_metrics(dataset.records, scores)
        folds[source] = {
            "policy": policy,
            "policy_selected_from_consumed_source_regime_outcomes": True,
            "prediction_receipt": {
                "path": str(prediction_path),
                "bytes": prediction_path.stat().st_size,
                "sha256": sha256_file(prediction_path),
            },
            "geometry_receipt": geometry_receipt,
            "selection": selection,
            "metrics": metrics,
        }
    passed = all(all(row["metrics"]["checks"].values()) for row in folds.values())
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "mode": "CONSUMED_DEVELOPMENT_SOURCE_REGIME_MAPPED_EXPERTS",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside nine frozen body/path capsules; UNKNOWN remains unknown.",
        "source_policy": SOURCE_POLICY,
        "source_policy_parameters": {
            "BONN_RGBD_DYNAMIC": {
                "geometry_opportunity_lcb_floor": -1.25,
                "utility_lcb_floor": 0.0,
                "fit_role": "CONSUMED_BONN_DEVELOPMENT",
            }
        },
        "folds": folds,
        "terminal": "TARO_R35_SOURCE_REGIME_DEVELOPMENT_PASS" if passed else "STOP_TARO_R35_SOURCE_REGIME_DEVELOPMENT_FAIL",
        "consumed_development_pass": passed,
        "fresh_parent_confirmation_authorized": passed,
        "fresh_source_confirmation_authorized": False,
        "read_boundary": {
            "candidate_depth_in_r31_scorer_input": False,
            "inference_anchor_uses_target_coverage": False,
            "source_regime_identity_in_policy": True,
            "all_policy_choices_fit_on_consumed_outcomes": True,
        },
        "claim_ceiling": "Source-regime-mapped consumed Development only. A PASS authorizes a new parent-disjoint confirmation inside an already mapped regime; it is not fresh-source, broad-generalization, Android, product, deployment, collision, navigation, or safety evidence.",
        "android_candidate_authorized": False,
        "product_authorized": False,
        "safety_authorized": False,
    }
    result["content_sha256"] = hashlib.sha256(canonical_json_bytes(result)).hexdigest().upper()
    return result


def write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--geometry-main-root", type=Path, required=True)
    parser.add_argument("--geometry-supplement-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.cache_root.resolve(),
        args.prediction_root.resolve(),
        args.geometry_main_root.resolve(),
        args.geometry_supplement_root.resolve(),
    )
    if args.output is not None:
        write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
