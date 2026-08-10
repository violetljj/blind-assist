#!/usr/bin/env python3
"""Fresh-view ICL canary for a source-exact metric depth boundary contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter

from build_ag_st_factor_labels import (
    PROVENANCE_SOURCE_NATIVE,
    TIER_A_SOURCE,
    backproject_depth_grid,
    compute_geometric_factors,
)
from download_b0_arkitscenes_assets import require, sha256_file
from run_ag_st_icl_mesh_support_identity import parse_global_pose_text
from run_ag_st_icl_pixel_boundary_canary import (
    DEFAULT_IDENTITY_RESULT,
    canonical_camera_to_world,
    downsample_exact_depth,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "artifacts.local/downloads/ag-st-icl-boundary-r1"
DEFAULT_ARCHIVE = DEFAULT_SOURCE_ROOT / "living_room_traj1_frei_png.tar.gz"
DEFAULT_SELECTED_ROOT = DEFAULT_SOURCE_ROOT / "selected12"
DEFAULT_POSES = DEFAULT_SOURCE_ROOT / "livingRoom1n.gt.sim"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-st-icl-fresh-depth-boundary-r2"
SELECTED_INDICES = (0, 88, 175, 263, 351, 438, 526, 613, 701, 789, 876, 964)
EXACT_GEOMETRY_GAP_M = 0.06


def exact_depth_boundary_target(
    depth: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a complete 4-neighbour boundary target over exact-depth pairs."""
    points = backproject_depth_grid(depth, intrinsics)
    target = np.zeros(depth.shape, dtype=np.bool_)
    evaluable = np.zeros(depth.shape, dtype=np.bool_)
    for first, second in (
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
    ):
        pair_valid = valid[first] & valid[second]
        gap = np.linalg.norm(points[first] - points[second], axis=-1)
        transition = pair_valid & (gap >= EXACT_GEOMETRY_GAP_M)
        evaluable[first] |= pair_valid
        evaluable[second] |= pair_valid
        target[first] |= transition
        target[second] |= transition
    return target, evaluable


def boundary_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    evaluable: np.ndarray,
) -> dict[str, Any]:
    predicted_eval = np.asarray(predicted, dtype=np.bool_) & evaluable
    target_eval = np.asarray(target, dtype=np.bool_) & evaluable
    target_near = maximum_filter(target_eval.astype(np.uint8), size=5) > 0
    predicted_near = maximum_filter(predicted_eval.astype(np.uint8), size=5) > 0
    predicted_count = int(np.sum(predicted_eval))
    target_count = int(np.sum(target_eval))
    return {
        "predicted_seed_pixels": predicted_count,
        "exact_target_pixels": target_count,
        "precision_within_2px": int(np.sum(predicted_eval & target_near)) / max(1, predicted_count),
        "recall_within_2px": int(np.sum(target_eval & predicted_near)) / max(1, target_count),
    }


def run(
    archive: Path,
    selected_root: Path,
    pose_path: Path,
    identity_result_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for path in (archive, pose_path, identity_result_path):
        require(path.is_file(), f"ICL fresh boundary source missing: {path}")
    require(not output_dir.exists(), f"ICL fresh boundary output exists: {output_dir}")
    require(
        sha256_file(archive) == "75D5F87EBAF313F6DDF9D1750815C277E3B16DB8ABD68A950F6A3665A49F2403",
        "ICL trajectory-1 archive hash drift",
    )
    require(
        sha256_file(pose_path) == "672FB9BFAB2FF7B4CA3A1CD5DC06DF3EFE16370DAAC654463AE3082F9851AFEB",
        "ICL trajectory-1 pose hash drift",
    )
    identity = json.loads(identity_result_path.read_text(encoding="utf-8"))
    exact_floor_height = float(identity["mesh"]["exact_floor_world_height_m"])
    poses = parse_global_pose_text(pose_path.read_text(encoding="utf-8"))
    require(len(poses) == 965, "ICL trajectory-1 pose count drift")
    output_dir.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for index in SELECTED_INDICES:
        depth_path = selected_root / f"depth/{index}.png"
        rgb_path = selected_root / f"rgb/{index}.png"
        require(depth_path.is_file() and rgb_path.is_file(), f"ICL fresh RGB-D missing: {index}")
        depth, valid, intrinsics = downsample_exact_depth(depth_path)
        target, evaluable = exact_depth_boundary_target(depth, valid, intrinsics)
        pose_icl = poses[index]
        camera_height = float(pose_icl[1, 3]) - exact_floor_height
        factors = compute_geometric_factors(
            depth,
            valid,
            intrinsics,
            canonical_camera_to_world(pose_icl),
            np.where(valid, 0.99, 0.0).astype(np.float32),
            np.where(valid, TIER_A_SOURCE, 0).astype(np.uint8),
            np.where(valid, PROVENANCE_SOURCE_NATIVE, 0).astype(np.uint8),
            np.where(valid, 0.002, np.inf).astype(np.float32),
            support_camera_height_override_m=camera_height,
            support_plane_residual_override_m=0.01,
        )
        predicted = factors["evidence_truth_valid_hw"] & (
            factors["boundary_probability_pseudo_hw"] >= 0.5
        )
        frame_metrics = boundary_metrics(predicted, target, evaluable)
        frame_evaluable = bool(
            float(np.mean(evaluable)) >= 0.80
            and frame_metrics["exact_target_pixels"] >= 20
        )
        if frame_evaluable:
            metrics.append(frame_metrics)
        quality_tier = np.where(evaluable, TIER_A_SOURCE, 0).astype(np.uint8)
        provenance = np.where(evaluable, PROVENANCE_SOURCE_NATIVE, 0).astype(np.uint8)
        output_path = output_dir / f"icl_lr_kt1_{index:04d}.npz"
        np.savez_compressed(
            output_path,
            boundary_probability_truth_hw=target.astype(np.float32),
            boundary_truth_valid_hw=evaluable.astype(np.uint8),
            boundary_unknown_hw=(~evaluable).astype(np.uint8),
            boundary_quality_tier_hw=quality_tier,
            boundary_provenance_hw=provenance,
            geometric_seed_hw=predicted.astype(np.uint8),
        )
        rows.append(
            {
                "pose_index": index,
                "frame_evaluable": frame_evaluable,
                "camera_height_m": camera_height,
                "truth_evaluable_coverage": float(np.mean(evaluable)),
                "metrics": frame_metrics,
                "output": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "depth_sha256": sha256_file(depth_path),
                "rgb_sha256": sha256_file(rgb_path),
            }
        )
    require(metrics, "ICL fresh boundary has no evaluable frames")
    macro_precision = float(np.mean([row["precision_within_2px"] for row in metrics]))
    macro_recall = float(np.mean([row["recall_within_2px"] for row in metrics]))
    exact_pixels = int(sum(row["exact_target_pixels"] for row in metrics))
    gates = {
        "evaluable_frame_count_ge_10": len(metrics) >= 10,
        "exact_target_pixel_count_ge_500": exact_pixels >= 500,
        "macro_precision_within_2px_ge_0p50": macro_precision >= 0.50,
        "macro_recall_within_2px_ge_0p60": macro_recall >= 0.60,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_icl_fresh_depth_boundary_canary_v1",
        "status": "ICL_FRESH_EXACT_DEPTH_BOUNDARY_PASS" if passed else "ICL_FRESH_EXACT_DEPTH_BOUNDARY_FAIL",
        "source": {
            "dataset": "ICL-NUIM living room lr kt1",
            "archive": str(archive.resolve()),
            "archive_sha256": sha256_file(archive),
            "global_pose_sha256": sha256_file(pose_path),
            "identity_result_sha256": sha256_file(identity_result_path),
        },
        "selection": {
            "pose_indices": list(SELECTED_INDICES),
            "rule": "12 rounded evenly spaced indices over the 965-pose trajectory, frozen before reading pixel outputs",
            "fresh_against_trajectory_0_canary": True,
        },
        "target_contract": {
            "positive": "A valid horizontal or vertical exact-depth neighbour pair has at least 0.06 m camera-space 3D gap; both pixels are positive.",
            "negative": "An evaluable pixel has valid exact-depth neighbour evidence but no qualifying local gap.",
            "unknown": "A pixel has no valid horizontal or vertical exact-depth neighbour.",
            "teacher_output_used_as_truth": False,
        },
        "frame_count": len(rows),
        "evaluable_frame_count": len(metrics),
        "exact_target_pixel_count": exact_pixels,
        "macro_precision_within_2px": macro_precision,
        "macro_recall_within_2px": macro_recall,
        "gates": gates,
        "frames": rows,
        "decision": {
            "source_exact_dense_boundary_labels_materialized": True,
            "current_geometric_seed_supported_on_fresh_views": passed,
            "source_exact_synthetic_boundary_training_authorized": passed,
            "teacher_generated_dense_boundary_training_authorized": False,
            "teacher_boundary_requirement": "Independent Teacher/source agreement and uncertainty gating remain required before teacher-filled boundary pixels can enter training.",
            "complete_factor_truth_required": False,
        },
        "claim_boundary": (
            "Fresh-view source-exact synthetic metric-depth boundary evidence in a second trajectory of one ICL scene. "
            "It validates a local geometric label contract, not real-world, task, product, or safety behavior."
        ),
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--global-poses", type=Path, default=DEFAULT_POSES)
    parser.add_argument("--identity-result", type=Path, default=DEFAULT_IDENTITY_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(
        args.archive,
        args.selected_root,
        args.global_poses,
        args.identity_result,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "evaluable_frame_count",
                    "exact_target_pixel_count",
                    "macro_precision_within_2px",
                    "macro_recall_within_2px",
                    "gates",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
