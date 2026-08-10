#!/usr/bin/env python3
"""Test whether independent RGB edges can precision-gate Teacher boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter, sobel


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
MAPANYTHING_ROOT = REPO_ROOT / "artifacts.local/tools/map-anything"
sys.path[:0] = [str(MODULE_DIR), str(MAPANYTHING_ROOT)]

from arkitscenes_truth_reader import parse_trajectory  # noqa: E402
from build_ag_st_factor_labels import TEACHER_C_QUALITY  # noqa: E402
from download_b0_arkitscenes_assets import require, sha256_file  # noqa: E402
from evaluate_ag_st_multiteacher_boundary_gate import (  # noqa: E402
    _add_counts,
    _counts,
    _label_name,
    _metrics,
    depth_boundary_seed,
    teacher_boundary_consensus,
)
from run_ag_st_stage0a import (  # noqa: E402
    load_factor_source_frame,
    resolve_trajectory_path,
    select_train_videos,
)


DEFAULT_STAGE0_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-stage0a-mapanything-apache-train16-block64-r1"
)
DEFAULT_LABEL_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-superteacher-factor-labels-multiteacher-train16-r2"
)
DEFAULT_OUTPUT = REPO_ROOT / "artifacts.local/experiments/ag-st-rgb-boundary-gate-r0/result.json"
RGB_EDGE_QUANTILE = 0.90
RGB_EDGE_MINIMUM = 0.05


def rgb_edge_map(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    image = np.asarray(rgb, dtype=np.float32) / 255.0
    require(image.ndim == 3 and image.shape[2] == 3, "RGB edge input invalid")
    gray = image @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
    smooth = gaussian_filter(gray, sigma=1.0)
    gradient = np.hypot(sobel(smooth, axis=0), sobel(smooth, axis=1)) / 8.0
    threshold = max(RGB_EDGE_MINIMUM, float(np.quantile(gradient, RGB_EDGE_QUANTILE)))
    return gradient.astype(np.float32), gradient >= threshold, threshold


def run(stage0_dir: Path, label_dir: Path) -> dict[str, Any]:
    from mapanything.utils.cropping import crop_resize_if_necessary

    stage0_result_path = stage0_dir / "result.json"
    require(stage0_result_path.is_file() and label_dir.is_dir(), "RGB boundary input missing")
    stage0_result = json.loads(stage0_result_path.read_text(encoding="utf-8"))
    manifest_path = Path(stage0_result["source"]["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parents = [str(value) for value in stage0_result["source"]["parents"]]
    videos = {str(row["video_id"]): row for row in select_train_videos(manifest, parents)}
    trajectories = {
        parent: parse_trajectory(resolve_trajectory_path(video))
        for parent, video in videos.items()
    }
    frame_indices = {
        str(frame["frame_stem"]): int(frame["frame_index"])
        for parent in stage0_result["parent_runs"]
        for frame in parent["frame_summaries"]
    }
    names = (
        "primary_rgb",
        "secondary_rgb",
        "union_rgb",
        "teacher_consensus_rgb",
    )
    totals = {name: defaultdict(int) for name in names}
    parent_totals: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: {name: defaultdict(int) for name in names}
    )
    frames: list[dict[str, Any]] = []
    for stage0_path in sorted(stage0_dir.glob("*.npz")):
        label_path = label_dir / _label_name(stage0_path)
        require(label_path.is_file(), f"RGB boundary label missing: {label_path.name}")
        frame_stem = label_path.stem
        parent_id = frame_stem.split("_", 1)[0]
        source_frame = load_factor_source_frame(
            videos[parent_id],
            frame_indices[frame_stem],
            trajectories[parent_id],
        )
        with np.load(stage0_path) as source, np.load(label_path) as labels:
            shape = labels["primary_teacher_depth_m_hw"].shape
            processed = crop_resize_if_necessary(
                image=source_frame["rgb_upright"],
                resolution=(shape[1], shape[0]),
                depthmap=source_frame["depth_m_upright"],
                intrinsics=source_frame["intrinsics_upright"],
                additional_quantities=[source_frame["depth_valid_upright"].astype(np.uint8)],
            )
            rgb = np.asarray(processed[0].convert("RGB"))
            require(rgb.shape[:2] == shape, "RGB boundary preprocessing shape drift")
            _, rgb_edge, rgb_threshold = rgb_edge_map(rgb)
            rgb_near = maximum_filter(rgb_edge.astype(np.uint8), size=5) > 0

            hidden = np.asarray(source["hidden_mask"], dtype=np.bool_)
            target = np.asarray(labels["boundary_probability_pseudo_hw"], dtype=np.float32) >= 0.5
            evaluable = hidden & np.asarray(
                labels["physical_boundary_valid_diagnostic_hw"], dtype=np.bool_
            ) & np.asarray(labels["source_native_valid_hw"], dtype=np.bool_)
            intrinsics = np.asarray(labels["intrinsics_output"], dtype=np.float64)
            primary_depth = np.asarray(labels["primary_teacher_depth_m_hw"], dtype=np.float32)
            secondary_depth = np.asarray(labels["secondary_teacher_depth_m_hw"], dtype=np.float32)
            primary_valid = np.isfinite(primary_depth) & (primary_depth > 0)
            secondary_valid = np.asarray(labels["secondary_teacher_valid_hw"], dtype=np.bool_) & (
                np.isfinite(secondary_depth) & (secondary_depth > 0)
            )
            primary = depth_boundary_seed(primary_depth, primary_valid, intrinsics)
            secondary = depth_boundary_seed(secondary_depth, secondary_valid, intrinsics)
            quality = np.asarray(labels["teacher_pair_valid_hw"], dtype=np.bool_) & (
                np.asarray(labels["teacher_pair_quality_hw"], dtype=np.float32) >= TEACHER_C_QUALITY
            )
            teacher_consensus = teacher_boundary_consensus(primary, secondary, quality)
            predictions = {
                "primary_rgb": primary & rgb_near,
                "secondary_rgb": secondary & rgb_near,
                "union_rgb": (primary | secondary) & rgb_near & quality,
                "teacher_consensus_rgb": teacher_consensus & rgb_near,
            }
            frame_counts = {
                name: _counts(prediction, target, evaluable)
                for name, prediction in predictions.items()
            }
            for name, counts in frame_counts.items():
                _add_counts(totals[name], counts)
                _add_counts(parent_totals[parent_id][name], counts)
            frames.append(
                {
                    "parent_id": parent_id,
                    "frame_stem": frame_stem,
                    "rgb_edge_threshold": rgb_threshold,
                    "rgb_edge_coverage": float(np.mean(rgb_edge)),
                    "metrics": {name: _metrics(counts) for name, counts in frame_counts.items()},
                }
            )
    metrics = {name: _metrics(dict(counts)) for name, counts in totals.items()}
    parents_out: list[dict[str, Any]] = []
    macro: list[dict[str, float | int]] = []
    for parent_id in sorted(parent_totals):
        values = {name: _metrics(dict(counts)) for name, counts in parent_totals[parent_id].items()}
        if int(values["teacher_consensus_rgb"]["target"]) >= 20:
            macro.append(values["teacher_consensus_rgb"])
        parents_out.append({"parent_id": parent_id, "metrics": values})
    macro_precision = float(np.mean([row["precision_within_2px"] for row in macro]))
    macro_recall = float(np.mean([row["recall_within_2px"] for row in macro]))
    selected = metrics["teacher_consensus_rgb"]
    gates = {
        "evaluable_parent_count_ge_8": len(macro) >= 8,
        "predicted_pixels_ge_100": int(selected["predicted"]) >= 100,
        "parent_macro_precision_ge_0p50": macro_precision >= 0.50,
        "parent_macro_recall_ge_0p10": macro_recall >= 0.10,
    }
    passed = all(gates.values())
    return {
        "schema": "blindassist_ag_st_rgb_boundary_gate_v1",
        "status": "RGB_BOUNDARY_GATE_PASS" if passed else "RGB_BOUNDARY_GATE_FAIL",
        "question": "Can a frozen zero-model RGB edge gate rescue precision of MapAnything/DA2 boundary agreement before paying for a segmentation foundation model?",
        "inputs": {
            "stage0_result_sha256": sha256_file(stage0_result_path),
            "multiteacher_result_sha256": sha256_file(label_dir / "result.json"),
            "source_manifest_sha256": sha256_file(manifest_path),
            "parent_count": len(parent_totals),
            "frame_count": len(frames),
        },
        "frozen_gate": {
            "rgb": "luma Gaussian sigma=1 plus Sobel magnitude, max(0.05, frame q90), 2px dilation",
            "teacher": "MapAnything/DA2 point-to-plane seed agreement within 2px plus frozen pair quality >=0.30",
            "evaluation": "source-native factor boundary on source pixels hidden from Teacher inference",
        },
        "metrics": metrics,
        "evaluable_parent_count": len(macro),
        "parent_macro_precision_within_2px": macro_precision,
        "parent_macro_recall_within_2px": macro_recall,
        "gates": gates,
        "parents": parents_out,
        "frames": frames,
        "decision": {
            "rgb_boundary_evidence_supported": passed,
            "segmentation_foundation_model_execution_justified": passed,
            "teacher_filled_boundary_materialization_authorized": False,
            "if_failed": "Keep teacher-filled boundary UNKNOWN and do not spend the next execution on SAM-style mask refinement.",
        },
        "claim_boundary": "TRAIN-only zero-model RGB-edge diagnostic; no formal F1, real-world, task, deployment, product, or safety claim.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0-dir", type=Path, default=DEFAULT_STAGE0_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"RGB boundary output exists: {args.output}")
    result = run(args.stage0_dir, args.label_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "evaluable_parent_count", "parent_macro_precision_within_2px", "parent_macro_recall_within_2px", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
