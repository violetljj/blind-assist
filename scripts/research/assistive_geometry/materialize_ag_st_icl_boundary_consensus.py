#!/usr/bin/env python3
"""Materialize positive-only AG-ST boundary anchors from frozen ICL evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from build_ag_st_factor_labels import (
    PROVENANCE_SOURCE_NATIVE,
    TIER_A_SOURCE,
    compute_geometric_factors,
)
from download_b0_arkitscenes_assets import require, sha256_file
from run_ag_st_icl_mesh_support_identity import parse_global_pose_text
from run_ag_st_icl_pixel_boundary_canary import (
    DEFAULT_GLOBAL_POSES,
    DEFAULT_IDENTITY_RESULT,
    DEFAULT_MESH,
    DEFAULT_SELECTED_ROOT,
    canonical_camera_to_world,
    downsample_exact_depth,
    exact_mesh_boundary_target,
    load_mesh_surface_samples,
    mesh_surface_labels,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CANARY_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-icl-pixel-boundary-r0/result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-icl-boundary-consensus-positive-r0"
)


def consensus_positive(exact_target: np.ndarray, geometric_seed: np.ndarray) -> np.ndarray:
    """Keep exact-mesh positives only when a geometric seed lies within two pixels."""
    require(exact_target.shape == geometric_seed.shape, "boundary consensus shape mismatch")
    seed_near = maximum_filter(geometric_seed.astype(np.uint8), size=5) > 0
    return np.asarray(exact_target, dtype=np.bool_) & seed_near


def materialize(
    canary_result_path: Path,
    selected_root: Path,
    mesh_path: Path,
    pose_path: Path,
    identity_result_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    for path in (canary_result_path, mesh_path, pose_path, identity_result_path):
        require(path.is_file(), f"ICL boundary consensus input missing: {path}")
    require(not output_dir.exists(), f"ICL boundary consensus output exists: {output_dir}")
    canary = json.loads(canary_result_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_result_path.read_text(encoding="utf-8"))
    require(canary.get("status") == "ICL_PIXEL_EXACT_BOUNDARY_FAIL", "unexpected canary status")
    require(
        canary.get("decision", {}).get("boundary_training_authorized") is False,
        "failed boundary canary cannot authorize dense boundary training",
    )
    poses = parse_global_pose_text(pose_path.read_text(encoding="utf-8"))
    frame_identity = {int(row["pose_index"]): row for row in identity["frames"]}
    exact_floor_height = float(identity["mesh"]["exact_floor_world_height_m"])
    mesh = load_mesh_surface_samples(mesh_path)
    tree = cKDTree(mesh["points_world"])

    output_dir.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    for row in canary["frames"]:
        if not bool(row["frame_evaluable"]):
            continue
        index = int(row["pose_index"])
        depth_path = selected_root / f"depth/{index}.png"
        depth, valid, intrinsics = downsample_exact_depth(depth_path)
        pose_icl = poses[index]
        labels, _, points_world = mesh_surface_labels(
            depth,
            valid,
            intrinsics,
            pose_icl,
            tree,
            mesh["surface_ids"],
        )
        exact_target, target_evaluable = exact_mesh_boundary_target(labels, points_world, valid)
        camera_height = float(pose_icl[1, 3]) - exact_floor_height
        require(
            bool(frame_identity[index]["support_identity_evaluable"]),
            "materialized frame lost support identity",
        )
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
        geometric_seed = factors["evidence_truth_valid_hw"] & (
            factors["boundary_probability_pseudo_hw"] >= 0.5
        )
        positive = consensus_positive(exact_target, geometric_seed)
        require(np.all(~positive | exact_target), "consensus escaped exact target")
        require(np.all(~positive | target_evaluable), "consensus escaped exact evaluability")
        tier = np.where(positive, TIER_A_SOURCE, 0).astype(np.uint8)
        provenance = np.where(positive, PROVENANCE_SOURCE_NATIVE, 0).astype(np.uint8)
        output_path = output_dir / f"icl_lr_kt0_{index:04d}.npz"
        np.savez_compressed(
            output_path,
            boundary_positive_hw=positive.astype(np.uint8),
            boundary_supervision_valid_hw=positive.astype(np.uint8),
            boundary_unknown_hw=(~positive).astype(np.uint8),
            boundary_quality_tier_hw=tier,
            boundary_provenance_hw=provenance,
            exact_mesh_target_hw=exact_target.astype(np.uint8),
            geometric_seed_hw=geometric_seed.astype(np.uint8),
            mesh_surface_id_hw=labels.astype(np.int32),
        )
        rows.append(
            {
                "pose_index": index,
                "output": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "exact_target_pixels": int(np.sum(exact_target)),
                "consensus_positive_pixels": int(np.sum(positive)),
                "unknown_pixels": int(np.sum(~positive)),
            }
        )
    require(rows, "no frozen-evaluable ICL frames materialized")
    result = {
        "schema": "blindassist_ag_st_icl_boundary_consensus_positive_v1",
        "status": "ICL_BOUNDARY_CONSENSUS_POSITIVE_MATERIALIZED",
        "source_canary_result": str(canary_result_path.resolve()),
        "source_canary_result_sha256": sha256_file(canary_result_path),
        "frame_count": len(rows),
        "exact_target_pixel_count": sum(row["exact_target_pixels"] for row in rows),
        "consensus_positive_pixel_count": sum(row["consensus_positive_pixels"] for row in rows),
        "frames": rows,
        "label_contract": {
            "positive": "exact mesh-surface transition with >=0.06 m gap and a geometric seed within 2 px",
            "negative": "none",
            "unknown": "every pixel not meeting the positive contract",
            "provenance": "source-native exact synthetic RGB-D/mesh plus deterministic geometric agreement",
        },
        "decision": {
            "positive_boundary_anchor_available": True,
            "dense_boundary_training_authorized": False,
            "reason": "The frozen dense canary failed precision and frame-count gates; absence of this selective exact target is not a negative label.",
        },
        "claim_boundary": (
            "Positive-only source-exact synthetic boundary anchors on one ICL scene. "
            "All non-positive pixels remain UNKNOWN; no real-world, task, product, or safety claim."
        ),
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canary-result", type=Path, default=DEFAULT_CANARY_RESULT)
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--global-poses", type=Path, default=DEFAULT_GLOBAL_POSES)
    parser.add_argument("--identity-result", type=Path, default=DEFAULT_IDENTITY_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = materialize(
        args.canary_result,
        args.selected_root,
        args.mesh,
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
                    "frame_count",
                    "exact_target_pixel_count",
                    "consensus_positive_pixel_count",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
