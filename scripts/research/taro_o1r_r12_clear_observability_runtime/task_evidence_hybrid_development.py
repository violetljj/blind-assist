#!/usr/bin/env python3
"""Consumed multi-source Development for a guarded TARO task-evidence policy.

TUM FIT and Bonn task-evidence outcomes are both consumed Development here.
The frozen candidate family keeps generic pose diversity as a guardrail and
allows task geometry to choose only inside a bounded translation band, or to
make a bounded normalized trade. A policy must beat passive and generic in each
source family before a fresh-source confirmation roster may be locked.
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

from scripts.research.taro_o1r_r12_clear_observability_runtime import task_evidence_pairwise_ranker_bonn_confirmation as shared


SCHEMA = "blindassist.taro.task_evidence_hybrid_development.v1"
TASK_TERMS = ("visible_unknown", "unknown_parallax", "occluded_parallax", "far_unknown_parallax")
TRANSLATION_BANDS_M = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40)
TASK_WEIGHTS = (0.02, 0.05, 0.10, 0.20, 0.40, 0.80)
ROTATION_WEIGHTS = (0.0, 0.01, 0.05)
MIN_STRICT_WIN_PARENTS_PER_SOURCE = 4


def candidate_policy_specs() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for task in TASK_TERMS:
        for band in TRANSLATION_BANDS_M:
            policies.append({"family": "TRANSLATION_GUARDRAIL_TASK", "task_term": task, "translation_band_m": band})
        for task_weight in TASK_WEIGHTS:
            for rotation_weight in ROTATION_WEIGHTS:
                policies.append({
                    "family": "NORMALIZED_POSE_TASK_BLEND",
                    "task_term": task,
                    "task_weight": task_weight,
                    "rotation_weight": rotation_weight,
                })
    return policies


def _unit_interval(values: np.ndarray) -> np.ndarray:
    low = float(np.min(values))
    high = float(np.max(values))
    if high - low < 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values - low) / (high - low)


def policy_scores(records: Sequence[Any], policy: Mapping[str, Any]) -> np.ndarray:
    scores = np.full(len(records), -1e12, dtype=np.float64)
    by_reference: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_reference[record.reference_id].append(index)
    for indices in by_reference.values():
        translation = np.asarray([records[index].pair.translation_m for index in indices], dtype=np.float64)
        rotation = np.asarray([records[index].pair.rotation_deg for index in indices], dtype=np.float64)
        task = np.asarray([records[index].analytic[str(policy["task_term"])] for index in indices], dtype=np.float64)
        translation_unit = _unit_interval(translation)
        rotation_unit = _unit_interval(rotation)
        task_unit = _unit_interval(task)
        if policy["family"] == "TRANSLATION_GUARDRAIL_TASK":
            admitted = translation >= float(np.max(translation)) - float(policy["translation_band_m"]) - 1e-12
            local = np.where(admitted, task_unit + 1e-6 * translation_unit + 1e-9 * rotation_unit, -1e12)
        elif policy["family"] == "NORMALIZED_POSE_TASK_BLEND":
            local = translation_unit + float(policy["task_weight"]) * task_unit + float(policy["rotation_weight"]) * rotation_unit
        else:
            raise shared.RankerError(f"unknown hybrid policy family: {policy['family']}")
        scores[indices] = local
    shared.require(np.all(np.isfinite(scores)), "hybrid policy score non-finite")
    return scores


def _source_metrics(records: Sequence[Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    macro, rows = shared._selection_metrics(records, policy_scores(records, policy))
    strict = sum(row["strict_win_reference_count"] > 0 for row in rows.values())
    return {
        "parent_macro": macro,
        "strict_win_parent_count": strict,
        "parent_count": len(rows),
        "reference_count": sum(row["reference_count"] for row in rows.values()),
        "per_parent": rows,
    }


def policy_is_admissible(metrics: Mapping[str, Mapping[str, Any]]) -> bool:
    for source in ("TUM_RGBD", "BONN_RGBD_DYNAMIC"):
        row = metrics[source]
        macro = row["parent_macro"]
        if int(row["strict_win_parent_count"]) < MIN_STRICT_WIN_PARENTS_PER_SOURCE:
            return False
        if not (float(macro["ranker"]) > float(macro["passive"]) and float(macro["ranker"]) > float(macro["generic"])):
            return False
    return True


def _selection_value(row: Mapping[str, Any]) -> tuple[float, float, int, str]:
    improvements = []
    all_macros = []
    strict = 0
    for source in ("TUM_RGBD", "BONN_RGBD_DYNAMIC"):
        metrics = row["metrics"][source]
        macro = metrics["parent_macro"]
        improvements.append((float(macro["ranker"]) - float(macro["generic"])) / max(float(macro["generic"]), 1e-9))
        all_macros.append(float(macro["ranker"]))
        strict += int(metrics["strict_win_parent_count"])
    return min(improvements), float(np.mean(improvements)), strict, json.dumps(row["policy"], sort_keys=True)


def evaluate(bonn_root: Path) -> dict[str, Any]:
    tum_records, tum_source, tum_abstained = shared._build_tum_fit()
    bonn_contexts, bonn_records, bonn_capability, bonn_abstained = shared._bonn_contexts_and_records(bonn_root)
    shared._attach_bonn_targets(bonn_records, bonn_contexts)

    candidates = []
    for policy in candidate_policy_specs():
        metrics = {
            "TUM_RGBD": _source_metrics(tum_records, policy),
            "BONN_RGBD_DYNAMIC": _source_metrics(bonn_records, policy),
        }
        candidates.append({"policy": policy, "metrics": metrics, "admissible": policy_is_admissible(metrics)})
    admissible = [row for row in candidates if row["admissible"]]
    selected = max(admissible, key=_selection_value) if admissible else None
    terminal = "TASK_EVIDENCE_HYBRID_DEVELOPMENT_PASS" if selected is not None else "STOP_TASK_EVIDENCE_HYBRID_NO_CROSS_SOURCE_POLICY"
    compact_candidates = [
        {
            "policy": row["policy"],
            "admissible": row["admissible"],
            "metrics": {
                source: {
                    "parent_macro": metrics["parent_macro"],
                    "strict_win_parent_count": metrics["strict_win_parent_count"],
                    "parent_count": metrics["parent_count"],
                    "reference_count": metrics["reference_count"],
                }
                for source, metrics in row["metrics"].items()
            },
        }
        for row in candidates
    ]
    result = {
        "schema": SCHEMA,
        "mode": "CONSUMED_MULTI_SOURCE_DEVELOPMENT",
        "task_definition": "Select one pose-valid extra frame that maximizes novel observed cells inside frozen body/path capsules; UNKNOWN remains unknown.",
        "candidate_family_frozen_before_run": {
            "task_terms": list(TASK_TERMS),
            "translation_bands_m": list(TRANSLATION_BANDS_M),
            "task_weights": list(TASK_WEIGHTS),
            "rotation_weights": list(ROTATION_WEIGHTS),
            "candidate_count": len(candidates),
        },
        "selection_policy": "require every source family to beat passive and generic with at least four strict-win parents; then maximize worst-source relative gain over generic",
        "sources": {
            "TUM_RGBD": {"disposition": "CONSUMED_FIT_DEVELOPMENT", "source": tum_source, "geometry_abstention_count": tum_abstained},
            "BONN_RGBD_DYNAMIC": {"disposition": "CONSUMED_NEW_TASK_DEVELOPMENT", "pose_capability": bonn_capability, "geometry_abstention_count": bonn_abstained},
        },
        "candidate_summaries": compact_candidates,
        "admissible_candidate_count": len(admissible),
        "selected": selected,
        "terminal": terminal,
        "fresh_confirmation_source_lock_authorized": selected is not None,
        "android_candidate_authorized": False,
        "read_boundary": {"rgb_payload_decodes": 0, "network_requests": 0, "r11_reads": 0},
        "claim_ceiling": "Consumed TUM/Bonn Development only; not fresh Confirmation, collision correctness, Android, product, default-App, or safety evidence.",
    }
    result["content_sha256"] = hashlib.sha256(shared.canonical_json_bytes(result)).hexdigest().upper()
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
    parser.add_argument("--bonn-root", type=Path, default=shared.DEFAULT_BONN_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(args.bonn_root.resolve())
    if args.output is not None:
        _write_exclusive(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
