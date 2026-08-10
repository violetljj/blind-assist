#!/usr/bin/env python3
"""Evaluate a source-blind MapAnything/DA2 boundary agreement gate."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from build_ag_st_factor_labels import (
    TEACHER_C_QUALITY,
    _pairwise_point_to_plane_edges,
    backproject_depth_grid,
    compute_dense_normals,
)
from download_b0_arkitscenes_assets import require, sha256_file
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE0_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-stage0a-mapanything-apache-train16-block64-r1"
)
DEFAULT_LABEL_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-superteacher-factor-labels-multiteacher-train16-r2"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-multiteacher-boundary-gate-r2/result.json"
)
POINT_PLANE_SEED_THRESHOLD_M = 0.30
AGREEMENT_RADIUS_PX = 2
SIGNALS = (
    "source",
    "primary",
    "secondary",
    "spatial_consensus",
    "local_quality_consensus",
    "consensus",
)


def depth_boundary_seed(
    depth_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    normals, normal_valid = compute_dense_normals(depth, mask, intrinsics)
    points = backproject_depth_grid(depth, intrinsics)
    point_plane_edge, _, neighbor_count = _pairwise_point_to_plane_edges(
        points,
        normals,
        normal_valid,
        mask,
    )
    return mask & (neighbor_count > 0) & (point_plane_edge >= POINT_PLANE_SEED_THRESHOLD_M)


def teacher_boundary_consensus(
    primary_seed: np.ndarray,
    secondary_seed: np.ndarray,
    quality_valid: np.ndarray,
) -> np.ndarray:
    primary = np.asarray(primary_seed, dtype=np.bool_)
    secondary = np.asarray(secondary_seed, dtype=np.bool_)
    quality = np.asarray(quality_valid, dtype=np.bool_)
    require(primary.shape == secondary.shape == quality.shape, "boundary gate shape mismatch")
    primary_near = maximum_filter(primary.astype(np.uint8), size=5) > 0
    secondary_near = maximum_filter(secondary.astype(np.uint8), size=5) > 0
    return quality & ((primary & secondary_near) | (secondary & primary_near))


def _counts(predicted: np.ndarray, target: np.ndarray, evaluable: np.ndarray) -> dict[str, int]:
    predicted_eval = np.asarray(predicted, dtype=np.bool_) & evaluable
    target_eval = np.asarray(target, dtype=np.bool_) & evaluable
    target_near = maximum_filter(target_eval.astype(np.uint8), size=5) > 0
    predicted_near = maximum_filter(predicted_eval.astype(np.uint8), size=5) > 0
    return {
        "predicted": int(np.sum(predicted_eval)),
        "target": int(np.sum(target_eval)),
        "precision_hit": int(np.sum(predicted_eval & target_near)),
        "recall_hit": int(np.sum(target_eval & predicted_near)),
    }


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    return {
        **counts,
        "precision_within_2px": counts["precision_hit"] / max(1, counts["predicted"]),
        "recall_within_2px": counts["recall_hit"] / max(1, counts["target"]),
    }


def _add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] += int(value)


def _label_name(stage0_path: Path) -> str:
    first, remainder = stage0_path.stem.split("_", 1)
    require(remainder.startswith(first + "_"), f"unexpected Stage0 frame identity: {stage0_path.name}")
    return remainder + ".npz"


def run(stage0_dir: Path, label_dir: Path) -> dict[str, Any]:
    require(stage0_dir.is_dir() and label_dir.is_dir(), "boundary gate input directory missing")
    stage0_paths = sorted(stage0_dir.glob("*.npz"))
    require(len(stage0_paths) == 48, "boundary gate requires frozen 48 Stage0 frames")
    totals = {
        name: defaultdict(int)
        for name in SIGNALS
    }
    parent_totals: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {
            name: defaultdict(int)
            for name in SIGNALS
        }
    )
    frames: list[dict[str, Any]] = []
    for stage0_path in stage0_paths:
        label_path = label_dir / _label_name(stage0_path)
        require(label_path.is_file(), f"multi-Teacher label missing: {label_path.name}")
        with np.load(stage0_path) as source, np.load(label_path) as labels:
            truth = np.asarray(source["truth_depth_m"], dtype=np.float32)
            source_valid = np.asarray(source["source_valid"], dtype=np.bool_)
            hidden = np.asarray(source["hidden_mask"], dtype=np.bool_)
            intrinsics = np.asarray(labels["intrinsics_output"], dtype=np.float64)
            require(
                truth.shape == hidden.shape == labels["primary_teacher_depth_m_hw"].shape,
                "boundary gate frame shape mismatch",
            )
            target = (
                np.asarray(labels["boundary_probability_pseudo_hw"], dtype=np.float32) >= 0.5
            )
            truth_evaluable = (
                np.asarray(labels["physical_boundary_valid_diagnostic_hw"], dtype=np.bool_)
                & np.asarray(labels["source_native_valid_hw"], dtype=np.bool_)
            )
            evaluable = hidden & truth_evaluable
            primary_depth = np.asarray(labels["primary_teacher_depth_m_hw"], dtype=np.float32)
            secondary_depth = np.asarray(labels["secondary_teacher_depth_m_hw"], dtype=np.float32)
            primary_valid = np.isfinite(primary_depth) & (primary_depth > 0)
            secondary_valid = (
                np.asarray(labels["secondary_teacher_valid_hw"], dtype=np.bool_)
                & np.isfinite(secondary_depth)
                & (secondary_depth > 0)
            )
            primary_seed = depth_boundary_seed(primary_depth, primary_valid, intrinsics)
            secondary_seed = depth_boundary_seed(secondary_depth, secondary_valid, intrinsics)
            source_seed = depth_boundary_seed(truth, source_valid, intrinsics)
            pair_valid = np.asarray(labels["teacher_pair_valid_hw"], dtype=np.bool_)
            pair_quality_valid = (
                pair_valid
                & (np.asarray(labels["teacher_pair_quality_hw"], dtype=np.float32) >= TEACHER_C_QUALITY)
            )
            spatial_consensus = teacher_boundary_consensus(
                primary_seed,
                secondary_seed,
                pair_valid,
            )
            local_quality_valid = pair_valid & (
                maximum_filter(pair_quality_valid.astype(np.uint8), size=5) > 0
            )
            local_quality_consensus = teacher_boundary_consensus(
                primary_seed,
                secondary_seed,
                local_quality_valid,
            )
            consensus = teacher_boundary_consensus(
                primary_seed,
                secondary_seed,
                pair_quality_valid,
            )
            frame_counts = {
                "source": _counts(source_seed, target, evaluable),
                "primary": _counts(primary_seed, target, evaluable),
                "secondary": _counts(secondary_seed, target, evaluable),
                "spatial_consensus": _counts(spatial_consensus, target, evaluable),
                "local_quality_consensus": _counts(local_quality_consensus, target, evaluable),
                "consensus": _counts(consensus, target, evaluable),
            }
            parent_id = stage0_path.stem.split("_", 1)[0]
            for name, counts in frame_counts.items():
                _add_counts(totals[name], counts)
                _add_counts(parent_totals[parent_id][name], counts)
            frames.append(
                {
                    "parent_id": parent_id,
                    "stage0_frame": stage0_path.name,
                    "label_frame": label_path.name,
                    "hidden_evaluable_pixels": int(np.sum(evaluable)),
                    "pair_quality_accepted_pixels": int(np.sum(pair_quality_valid & hidden)),
                    "metrics": {name: _metrics(counts) for name, counts in frame_counts.items()},
                }
            )
    result_metrics = {name: _metrics(dict(counts)) for name, counts in totals.items()}
    parents: list[dict[str, Any]] = []
    evaluable_parent_metrics: list[dict[str, float | int]] = []
    for parent_id in sorted(parent_totals):
        metrics = {
            name: _metrics(dict(counts))
            for name, counts in parent_totals[parent_id].items()
        }
        if int(metrics["consensus"]["target"]) >= 20:
            evaluable_parent_metrics.append(metrics["consensus"])
        parents.append({"parent_id": parent_id, "metrics": metrics})
    macro_precision = float(
        np.mean([row["precision_within_2px"] for row in evaluable_parent_metrics])
    )
    macro_recall = float(
        np.mean([row["recall_within_2px"] for row in evaluable_parent_metrics])
    )
    consensus_total = result_metrics["consensus"]
    gates = {
        "evaluable_parent_count_ge_12": len(evaluable_parent_metrics) >= 12,
        "hidden_source_target_pixels_ge_500": int(consensus_total["target"]) >= 500,
        "consensus_predicted_pixels_ge_100": int(consensus_total["predicted"]) >= 100,
        "parent_macro_precision_ge_0p50": macro_precision >= 0.50,
        "parent_macro_recall_ge_0p20": macro_recall >= 0.20,
    }
    passed = all(gates.values())
    return {
        "schema": "blindassist_ag_st_multiteacher_boundary_gate_v1",
        "status": "MULTITEACHER_BOUNDARY_GATE_PASS" if passed else "MULTITEACHER_BOUNDARY_GATE_FAIL",
        "question": "Does independent MapAnything/DA2 geometric-boundary agreement recover the frozen source-derived physical-boundary seed on source pixels hidden from Teacher inference?",
        "inputs": {
            "stage0_result": str((stage0_dir / "result.json").resolve()),
            "stage0_result_sha256": sha256_file(stage0_dir / "result.json"),
            "multiteacher_result": str((label_dir / "result.json").resolve()),
            "multiteacher_result_sha256": sha256_file(label_dir / "result.json"),
            "parent_count": len(parent_totals),
            "frame_count": len(frames),
        },
        "frozen_gate": {
            "teacher_boundary_seed": "camera-space point-to-plane residual >=0.30 m",
            "agreement_radius_px": AGREEMENT_RADIUS_PX,
            "pair_quality_threshold": TEACHER_C_QUALITY,
            "source_reference_boundary": "frozen factor-builder boundary_probability_pseudo_hw >=0.5 with source-native physical-boundary validity",
            "evaluation_mask": "deterministically hidden source-valid pixels only",
        },
        "metrics": result_metrics,
        "failure_attribution": {
            "source_seed_contract": result_metrics["source"],
            "spatial_consensus_without_quality": result_metrics["spatial_consensus"],
            "spatial_consensus_with_local_2px_quality": result_metrics["local_quality_consensus"],
            "pixelwise_quality_consensus": result_metrics["consensus"],
            "interpretation_rule": "If spatial consensus remains weak without pixelwise quality, the failure is Teacher boundary localization/content rather than only the depth-quality gate.",
        },
        "evaluable_parent_count": len(evaluable_parent_metrics),
        "parent_macro_consensus_precision_within_2px": macro_precision,
        "parent_macro_consensus_recall_within_2px": macro_recall,
        "gates": gates,
        "parents": parents,
        "frames": frames,
        "decision": {
            "teacher_filled_boundary_materialization_authorized": passed,
            "if_failed": "Keep teacher-filled boundary UNKNOWN; source-native and source-exact boundary contracts remain unchanged.",
            "complete_truth_required": False,
        },
        "claim_boundary": (
            "TRAIN-only source-depth hidden-reference diagnosis for two frozen Teachers. "
            "No reducer/task outcome, formal F1, real-world, product, deployment, or safety claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"boundary gate output exists: {args.output}")
    result = run(args.stage0_dir, args.label_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "evaluable_parent_count",
                    "parent_macro_consensus_precision_within_2px",
                    "parent_macro_consensus_recall_within_2px",
                    "gates",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
