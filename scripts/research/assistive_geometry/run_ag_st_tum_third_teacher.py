#!/usr/bin/env python3
"""Evaluate DepthART as an incremental third Teacher on fresh TUM parents.

The two-Teacher R1 recipe remains the fallback.  FIT parents choose between an
OR-style coverage witness and an AND-style precision witness.  Held-out parents
are opened once after that choice.  A third Teacher is promoted only when its
FIT-selected branch also improves the corresponding held-out objective.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from ag_st_depthart_teacher import DepthArtMetricTeacher  # noqa: E402
from ag_st_tum_rgbd import load_tum_role_payloads  # noqa: E402
from build_ag_st_multiteacher_factor_labels import (  # noqa: E402
    PAIR_DISAGREEMENT_SCALE,
    _compact_curve,
    _split_error,
    robust_observed_scale,
    teacher_pair_quality,
)
from run_ag_st_stage0a import _error_metrics, compute_selective_metrics, sha256_file  # noqa: E402
from run_ag_st_tum_cross_source import (  # noqa: E402
    FROZEN_ACCEPT_THRESHOLD,
    MINIMUM_EVALUATION_COVERAGE,
    _attach_multiview_quality,
    _attach_secondary_teacher,
    _group_payloads,
    _load_mapanything,
    _mapanything_infer_parent,
    _metric_records,
    _preprocess_secondary_depth,
    build_depth_label_payload,
    cross_source_passes,
    evaluate_frozen_threshold,
)
from train_ag_st_masked_student import (  # noqa: E402
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
)


DEFAULT_COHORT = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_TUM_THIRD_TEACHER_COHORT_R2_2026-08-10.json"
)
DEFAULT_MAPANYTHING_MODEL = REPO_ROOT / "artifacts.local/models/map-anything-apache"
DEFAULT_DAV2_REPO = REPO_ROOT / "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main"
DEFAULT_DAV2_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-st-tum-third-teacher-r2"
SCHEMA = "blindassist_ag_st_tum_third_teacher_result_v1"
VARIANTS = ("two_teacher", "three_union", "three_consensus")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def combine_third_teacher_quality(
    geometry_quality: np.ndarray,
    two_quality: np.ndarray,
    two_valid: np.ndarray,
    depthart_pair_quality: np.ndarray,
    depthart_pair_valid: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    geometry = np.clip(np.asarray(geometry_quality, dtype=np.float32), 0.0, 1.0)
    two = np.clip(np.asarray(two_quality, dtype=np.float32), 0.0, 1.0)
    two_ok = np.asarray(two_valid, dtype=np.bool_)
    third_pair = np.clip(np.asarray(depthart_pair_quality, dtype=np.float32), 0.0, 1.0)
    third_ok = np.asarray(depthart_pair_valid, dtype=np.bool_)
    depthart = np.sqrt(geometry * third_pair).astype(np.float32)
    depthart[~third_ok] = 0.0
    union_valid = two_ok | third_ok
    union = np.maximum(np.where(two_ok, two, 0.0), depthart).astype(np.float32)
    union[~union_valid] = 0.0
    consensus_valid = two_ok & third_ok
    consensus = np.sqrt(two * depthart).astype(np.float32)
    consensus[~consensus_valid] = 0.0
    return {
        "two_teacher": (two, two_ok),
        "three_union": (union, union_valid),
        "three_consensus": (consensus, consensus_valid),
    }


def _attach_depthart_teacher(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    teacher = DepthArtMetricTeacher(args.depthart_source, args.depthart_checkpoint, args.device)
    for record in records:
        payload = record["payload"]
        raw = teacher.infer(payload.load_rgb(), payload.intrinsics)
        primary = record["primary_depth_m"]
        depth = _preprocess_secondary_depth(payload, raw, (primary.shape[1], primary.shape[0]))
        scale, support = robust_observed_scale(record["observed_depth_m"], depth, minimum_support=512)
        depth = (depth * scale).astype(np.float32)
        valid = np.isfinite(depth) & (depth > 0)
        disagreement, pair_quality, pair_valid = teacher_pair_quality(
            primary,
            record["primary_valid"],
            depth,
            valid,
            disagreement_scale=PAIR_DISAGREEMENT_SCALE,
        )
        record["dav2_pair_relative_disagreement"] = record["pair_relative_disagreement"]
        record["dav2_pair_quality"] = record["pair_quality"]
        record["dav2_pair_valid"] = record["pair_valid"]
        record["two_teacher_quality"] = record["combined_quality"]
        record["depthart_depth_m"] = depth
        record["depthart_valid"] = valid
        record["depthart_anchor_scale"] = scale
        record["depthart_anchor_support"] = support
        record["depthart_pair_relative_disagreement"] = disagreement
        record["depthart_pair_quality"] = pair_quality
        record["depthart_pair_valid"] = pair_valid
        record["variant_quality"] = combine_third_teacher_quality(
            record["geometry_quality"],
            record["two_teacher_quality"],
            record["dav2_pair_valid"],
            pair_quality,
            pair_valid,
        )
    receipt = {
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "frame_count": len(records),
    }
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def _records_for_variant(records: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    require(variant in VARIANTS, f"unknown third-Teacher variant: {variant}")
    output: list[dict[str, Any]] = []
    for record in records:
        quality, valid = record["variant_quality"][variant]
        row = dict(record)
        row["combined_quality"] = quality
        row["pair_valid"] = valid
        output.append(row)
    return output


def evaluate_variants(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        variant: evaluate_frozen_threshold(_records_for_variant(records, variant))
        for variant in VARIANTS
    }


def select_variant_on_fit(evaluations: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = evaluations["two_teacher"]
    union = evaluations["three_union"]
    consensus = evaluations["three_consensus"]
    baseline_overall = baseline["overall"]
    union_overall = union["overall"]
    consensus_overall = consensus["overall"]
    union_improves = bool(
        cross_source_passes(union)
        and union_overall["coverage"] > baseline_overall["coverage"]
        and union_overall["accepted"]["mae_m"] <= baseline_overall["accepted"]["mae_m"]
    )
    consensus_improves = bool(
        cross_source_passes(consensus)
        and consensus_overall["coverage"] >= MINIMUM_EVALUATION_COVERAGE
        and consensus_overall["accepted"]["mae_m"] < baseline_overall["accepted"]["mae_m"]
    )
    selected = "three_union" if union_improves else (
        "three_consensus" if consensus_improves else "two_teacher"
    )
    return {
        "selected": selected,
        "evaluation_unopened_at_selection": True,
        "union_fit_no_regret": union_improves,
        "consensus_fit_precision_gain": consensus_improves,
    }


def heldout_promotes(
    selected: str,
    evaluations: dict[str, dict[str, Any]],
) -> bool:
    if selected == "two_teacher":
        return False
    baseline = evaluations["two_teacher"]
    candidate = evaluations[selected]
    if not cross_source_passes(candidate):
        return False
    baseline_overall = baseline["overall"]
    candidate_overall = candidate["overall"]
    if selected == "three_union":
        return bool(
            candidate_overall["coverage"] > baseline_overall["coverage"]
            and candidate_overall["accepted"]["mae_m"]
            <= baseline_overall["accepted"]["mae_m"]
        )
    return bool(
        candidate_overall["coverage"] >= MINIMUM_EVALUATION_COVERAGE
        and candidate_overall["accepted"]["mae_m"]
        < baseline_overall["accepted"]["mae_m"]
    )


def _selected_label(record: dict[str, Any], variant: str) -> dict[str, np.ndarray]:
    quality, valid = record["variant_quality"][variant]
    row = dict(record)
    row["combined_quality"] = quality
    row["pair_valid"] = valid
    if variant == "three_union":
        row["pair_quality"] = np.maximum(record["dav2_pair_quality"], record["depthart_pair_quality"])
        row["pair_relative_disagreement"] = np.minimum(
            record["dav2_pair_relative_disagreement"],
            record["depthart_pair_relative_disagreement"],
        )
    elif variant == "three_consensus":
        row["pair_quality"] = np.sqrt(
            record["dav2_pair_quality"] * record["depthart_pair_quality"]
        ).astype(np.float32)
        row["pair_relative_disagreement"] = np.maximum(
            record["dav2_pair_relative_disagreement"],
            record["depthart_pair_relative_disagreement"],
        )
    label = build_depth_label_payload(row)
    label.update(
        {
            "dav2_teacher_depth_m_hw": record["secondary_depth_m"],
            "dav2_pair_valid_hw": record["dav2_pair_valid"],
            "dav2_pair_quality_hw": record["dav2_pair_quality"],
            "depthart_teacher_depth_m_hw": record["depthart_depth_m"],
            "depthart_pair_valid_hw": record["depthart_pair_valid"],
            "depthart_pair_quality_hw": record["depthart_pair_quality"],
            "selected_teacher_variant": np.asarray(variant),
        }
    )
    # The selected witness, not an unselected Teacher, supplies the disagreement
    # component of uncertainty on Teacher-added pixels.
    teacher_added = label["provenance_code_hw"] == 2
    metric = label["metric_depth_m_hw"]
    score = label["quality_score_hw"]
    base = (0.015 + 0.02 * np.maximum(metric, 0.0)) * (1.0 + 2.5 * (1.0 - score))
    disagreement_m = np.abs(record["primary_depth_m"] - record["secondary_depth_m"])
    if variant == "three_union":
        disagreement_m = np.minimum(
            disagreement_m,
            np.abs(record["primary_depth_m"] - record["depthart_depth_m"]),
        )
    elif variant == "three_consensus":
        disagreement_m = np.maximum(
            disagreement_m,
            np.abs(record["primary_depth_m"] - record["depthart_depth_m"]),
        )
    uncertainty = label["depth_uncertainty_proxy_m_hw"].copy()
    uncertainty[teacher_added] = np.sqrt(
        np.square(base[teacher_added]) + np.square(0.5 * disagreement_m[teacher_added])
    )
    label["depth_uncertainty_proxy_m_hw"] = uncertainty.astype(np.float32)
    return label


def _teacher_error(records: list[dict[str, Any]], key: str, valid_key: str) -> dict[str, Any]:
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    for record in records:
        mask = record["hidden_mask"] & record[valid_key]
        truth.append(record["truth_depth_m"][mask])
        prediction.append(record[key][mask])
    return _error_metrics(np.concatenate(truth), np.concatenate(prediction))


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    require(args.cohort_manifest.is_file(), "fresh TUM third-Teacher cohort missing")
    require(not args.output_dir.exists(), f"output directory already exists: {args.output_dir}")
    fit_payloads, fit_receipt = load_tum_role_payloads(args.cohort_manifest, "fit")
    evaluation_payloads, evaluation_receipt = load_tum_role_payloads(args.cohort_manifest, "evaluation")
    require(
        not (set(fit_receipt["parent_ids"]) & set(evaluation_receipt["parent_ids"])),
        "fresh TUM FIT/evaluation overlap",
    )

    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch_module, map_model, model_dtype, preprocess_inputs, geometry_functions = _load_mapanything(args)
    role_records: dict[str, list[dict[str, Any]]] = {"fit": [], "evaluation": []}
    for role_index, (role, payloads) in enumerate(
        (("fit", fit_payloads), ("evaluation", evaluation_payloads))
    ):
        for parent_index, parent_payloads in enumerate(_group_payloads(payloads).values()):
            role_records[role].extend(
                _mapanything_infer_parent(
                    args,
                    role_index * 17 + parent_index,
                    parent_payloads,
                    torch_module,
                    map_model,
                    model_dtype,
                    preprocess_inputs,
                    geometry_functions,
                )
            )
        _attach_multiview_quality(role_records[role])
    del map_model
    gc.collect()
    torch.cuda.empty_cache()

    all_records = role_records["fit"] + role_records["evaluation"]
    _attach_secondary_teacher(args, all_records)
    gc.collect()
    torch.cuda.empty_cache()
    depthart_execution = _attach_depthart_teacher(args, all_records)

    fit_variants = evaluate_variants(role_records["fit"])
    fit_selection = select_variant_on_fit(fit_variants)
    evaluation_variants = evaluate_variants(role_records["evaluation"])
    selected = fit_selection["selected"]
    third_promoted = heldout_promotes(selected, evaluation_variants)
    baseline_supported = cross_source_passes(evaluation_variants["two_teacher"])
    materialized_variant = selected if third_promoted else "two_teacher"
    labels_materialized = bool(third_promoted or baseline_supported)
    args.output_dir.mkdir(parents=True)

    frame_receipts: list[dict[str, Any]] = []
    totals = {"pixels": 0, "source": 0, "teacher": 0, "metric": 0}
    if labels_materialized:
        for role, records in role_records.items():
            for record in records:
                label = _selected_label(record, materialized_variant)
                output = args.output_dir / f"{role}__{record['frame_id']}.npz"
                np.savez_compressed(output, **label)
                source = label["source_native_valid_hw"]
                metric = label["metric_depth_valid_hw"]
                teacher = ~source & metric
                totals["pixels"] += int(metric.size)
                totals["source"] += int(source.sum())
                totals["teacher"] += int(teacher.sum())
                totals["metric"] += int(metric.sum())
                frame_receipts.append(
                    {
                        "role": role,
                        "parent_id": record["parent_id"],
                        "frame_id": record["frame_id"],
                        "output_path": str(output.resolve()),
                        "output_bytes": output.stat().st_size,
                        "source_native_coverage": float(np.mean(source)),
                        "teacher_added_coverage": float(np.mean(teacher)),
                        "metric_depth_coverage": float(np.mean(metric)),
                    }
                )

    risk_coverage: dict[str, Any] = {}
    for role, records in role_records.items():
        risk_coverage[role] = {}
        for variant in VARIANTS:
            metrics = compute_selective_metrics(_metric_records(_records_for_variant(records, variant)))
            risk_coverage[role][variant] = {"compact_curve": _compact_curve(metrics), "full": metrics}

    teacher_error = {
        role: {
            "primary_mapanything": _teacher_error(records, "primary_depth_m", "primary_valid"),
            "secondary_dav2": _teacher_error(records, "secondary_depth_m", "secondary_valid"),
            "third_depthart": _teacher_error(records, "depthart_depth_m", "depthart_valid"),
        }
        for role, records in role_records.items()
    }
    denominator = totals["pixels"]
    return {
        "schema": SCHEMA,
        "status": (
            "THIRD_TEACHER_PROMOTED_DEPTH_LABELS_MATERIALIZED"
            if third_promoted
            else (
                "THIRD_TEACHER_NOT_PROMOTED_TWO_TEACHER_LABELS_MATERIALIZED"
                if labels_materialized
                else "FRESH_TUM_SELECTIVE_RISK_NOT_SUPPORTED"
            )
        ),
        "mode": "WILD_LAB_FIT_SELECTED_HELDOUT_THIRD_TEACHER_EVALUATION",
        "question": "Does frozen DepthART add no-regret coverage or precision evidence to the MapAnything plus DA2 SuperTeacher on fresh TUM parents?",
        "cohort": {
            "manifest_path": str(args.cohort_manifest.resolve()),
            "manifest_sha256": sha256_file(args.cohort_manifest),
            "fit": fit_receipt,
            "evaluation": evaluation_receipt,
            "evaluation_opened_after_fit_selection": True,
        },
        "teachers": {
            "primary": {"model_id": "facebook/map-anything-apache", "role": "LABEL_PROPOSER"},
            "secondary": {"model_id": "depth-anything-v2-metric-hypersim-vits", "role": "INDEPENDENT_WITNESS"},
            "third": {
                "model_id": "DepthART-S metric indoor",
                "checkpoint_sha256": sha256_file(args.depthart_checkpoint),
                "role": "INDEPENDENT_WITNESS_NOT_TRUTH",
            },
        },
        "frozen_threshold": FROZEN_ACCEPT_THRESHOLD,
        "variants": {
            "two_teacher": "R1 geometry quality times MapAnything-DA2 agreement",
            "three_union": "accept the stronger independent DA2 or DepthART witness",
            "three_consensus": "require both independent witnesses and take their geometric mean",
        },
        "fit": {"variants": fit_variants, "selection": fit_selection},
        "evaluation": {
            "variants": evaluation_variants,
            "selected_from_fit": selected,
            "third_teacher_promoted": third_promoted,
            "materialized_variant": materialized_variant if labels_materialized else None,
        },
        "teacher_error_diagnostic": teacher_error,
        "risk_coverage": risk_coverage,
        "materialization": {
            "labels_materialized": labels_materialized,
            "frame_count": len(frame_receipts),
            "source_native_coverage": totals["source"] / denominator if denominator else None,
            "teacher_added_coverage": totals["teacher"] / denominator if denominator else None,
            "metric_depth_coverage": totals["metric"] / denominator if denominator else None,
            "unknown_depth_coverage": 1.0 - totals["metric"] / denominator if denominator else None,
            "support_boundary_obstacle": "ALL_UNKNOWN_TUM_GRAVITY_BASIS_UNVERIFIED",
        },
        "execution": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "depthart": depthart_execution,
            "elapsed_seconds": time.monotonic() - started,
        },
        "frame_receipts": frame_receipts,
        "decision": {
            "third_teacher_is_required_to_beat_primary": False,
            "student_training_authorized": False,
            "next_execution": "Add a non-TUM metric RGB-D source and a gravity-verified source for support/boundary before student training.",
        },
        "claim_boundary": "Fresh-parent TUM/Kinect depth pseudo-label and third-Teacher complementarity evidence only; not a third sensor domain, support/boundary truth, formal F1 authorization, task utility, deployment, product, or safety evidence.",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--mapanything-model", type=Path, default=DEFAULT_MAPANYTHING_MODEL)
    parser.add_argument("--dav2-repo", type=Path, default=DEFAULT_DAV2_REPO)
    parser.add_argument("--dav2-checkpoint", type=Path, default=DEFAULT_DAV2_CHECKPOINT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dav2-precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--dav2-input-size", type=int, default=518)
    parser.add_argument("--longest-side", type=int, default=336)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--mask-modulus", type=int, default=4)
    parser.add_argument("--minimum-hidden-pixels", type=int, default=512)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        result_path = args.output_dir / "result.json"
        _write_json_exclusive(result_path, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result": str(result_path),
                    "fit_selection": result["fit"]["selection"],
                    "evaluation": result["evaluation"],
                    "materialization": result["materialization"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        failure = {
            "schema": "blindassist_ag_st_tum_third_teacher_failure_v1",
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "failure.json"
        if not failure_path.exists():
            _write_json_exclusive(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
