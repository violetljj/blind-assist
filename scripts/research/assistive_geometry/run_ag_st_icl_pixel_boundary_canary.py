#!/usr/bin/env python3
"""External pixel canary for AG-ST boundary seeds on ICL-NUIM exact RGB-D/mesh."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from build_ag_st_factor_labels import (
    PROVENANCE_SOURCE_NATIVE,
    TIER_A_SOURCE,
    backproject_depth_grid,
    compute_geometric_factors,
)
from download_b0_arkitscenes_assets import require, sha256_file
from run_ag_st_icl_mesh_support_identity import (
    DEFAULT_GLOBAL_POSES,
    DEFAULT_MESH,
    DEFAULT_MESH_ARCHIVE,
    INTRINSICS,
    _triangle_samples,
    parse_global_pose_text,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RGBD_ARCHIVE = (
    REPO_ROOT
    / "artifacts.local/downloads/ag-st-icl-boundary-r0/living_room_traj0_frei_png.tar.gz"
)
DEFAULT_SELECTED_ROOT = (
    REPO_ROOT / "artifacts.local/downloads/ag-st-icl-boundary-r0/selected12"
)
DEFAULT_IDENTITY_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-icl-mesh-support-identity-r2/result.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-icl-pixel-boundary-r0/result.json"
)
DEPTH_SCALE = 5000.0
DOWNSAMPLE = 4
SAMPLE_OFFSET = 2
MESH_ASSOCIATION_MAX_M = 0.05
EXACT_GEOMETRY_GAP_M = 0.06


def canonical_camera_to_world(camera_to_icl_world: np.ndarray) -> np.ndarray:
    # ICL mesh uses +Y up.  Canonical AG geometry uses +Z up while leaving the
    # camera frame unchanged: x_c=x_i, y_c=-z_i, z_c=y_i.
    icl_to_canonical = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = icl_to_canonical @ camera_to_icl_world[:3, :3]
    result[:3, 3] = icl_to_canonical @ camera_to_icl_world[:3, 3]
    return result


def load_mesh_surface_samples(
    path: Path,
    area_per_sample_m2: float = 0.001,
) -> dict[str, Any]:
    vertices: list[list[float]] = []
    current_object = "unnamed"
    current_surface = current_object
    points: list[np.ndarray] = []
    surface_ids: list[np.ndarray] = []
    surface_names: list[str] = []
    surface_to_id: dict[str, int] = {}
    triangle_count = 0
    total_area = 0.0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("o "):
            current_object = line[2:].strip()
            current_surface = current_object
        elif line.startswith("usemtl "):
            current_surface = line[7:].strip()
        elif line.startswith("v "):
            values = [float(value) for value in line.split()[1:4]]
            require(len(values) == 3, "ICL OBJ vertex invalid")
            vertices.append(values)
        elif line.startswith("f "):
            indices = [int(token.split("/", 1)[0]) for token in line.split()[1:]]
            require(len(indices) >= 3 and all(index > 0 for index in indices), "ICL OBJ face invalid")
            surface_id = surface_to_id.setdefault(current_surface, len(surface_to_id))
            if surface_id == len(surface_names):
                surface_names.append(current_surface)
            for offset in range(1, len(indices) - 1):
                triangle = np.asarray(
                    [
                        vertices[indices[0] - 1],
                        vertices[indices[offset] - 1],
                        vertices[indices[offset + 1] - 1],
                    ],
                    dtype=np.float64,
                )
                area = float(
                    np.linalg.norm(
                        np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                    )
                    / 2.0
                )
                if area <= 1e-10:
                    continue
                samples = _triangle_samples(triangle, area, area_per_sample_m2)
                points.append(samples)
                surface_ids.append(np.full(len(samples), surface_id, dtype=np.int32))
                triangle_count += 1
                total_area += area
    require(points and surface_names, "ICL mesh surface samples missing")
    return {
        "points_world": np.concatenate(points, axis=0),
        "surface_ids": np.concatenate(surface_ids, axis=0),
        "surface_names": surface_names,
        "vertex_count": len(vertices),
        "triangle_count": triangle_count,
        "surface_count": len(surface_names),
        "surface_area_m2": total_area,
    }


def downsample_exact_depth(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
    require(raw.shape == (480, 640) and raw.dtype == np.uint16, "ICL depth payload drift")
    # The published ICL calibration has fy=-480.  Flip the raster vertically so
    # the AG factor builder can use its equivalent positive-focal convention
    # without reflecting the reconstructed camera coordinates.
    raw = np.flipud(raw)
    depth = raw[SAMPLE_OFFSET::DOWNSAMPLE, SAMPLE_OFFSET::DOWNSAMPLE].astype(np.float32) / DEPTH_SCALE
    valid = depth > 0.0
    intrinsics = np.asarray(INTRINSICS, dtype=np.float64).copy()
    intrinsics[1, 1] = abs(intrinsics[1, 1])
    intrinsics[1, 2] = (raw.shape[0] - 1) - intrinsics[1, 2]
    intrinsics[0, 0] /= DOWNSAMPLE
    intrinsics[1, 1] /= DOWNSAMPLE
    intrinsics[0, 2] = (intrinsics[0, 2] - SAMPLE_OFFSET) / DOWNSAMPLE
    intrinsics[1, 2] = (intrinsics[1, 2] - SAMPLE_OFFSET) / DOWNSAMPLE
    return depth, valid, intrinsics


def mesh_surface_labels(
    depth: np.ndarray,
    valid: np.ndarray,
    intrinsics: np.ndarray,
    camera_to_icl_world: np.ndarray,
    tree: cKDTree,
    sample_surface_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_camera = backproject_depth_grid(depth, intrinsics).astype(np.float64)
    points_world = (
        np.einsum("...j,ij->...i", points_camera, camera_to_icl_world[:3, :3])
        + camera_to_icl_world[:3, 3]
    )
    labels = np.full(depth.shape, -1, dtype=np.int32)
    distance = np.full(depth.shape, np.inf, dtype=np.float32)
    flat = np.flatnonzero(valid)
    distances, indices = tree.query(points_world.reshape(-1, 3)[flat], k=1, workers=-1)
    known = distances <= MESH_ASSOCIATION_MAX_M
    labels.reshape(-1)[flat[known]] = sample_surface_ids[indices[known]]
    distance.reshape(-1)[flat] = distances.astype(np.float32)
    return labels, distance, points_world


def exact_mesh_boundary_target(
    labels: np.ndarray,
    points_world: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.zeros(labels.shape, dtype=np.bool_)
    evaluable = np.zeros(labels.shape, dtype=np.bool_)
    for first, second in (
        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
    ):
        pair_known = valid[first] & valid[second] & (labels[first] >= 0) & (labels[second] >= 0)
        gap = np.linalg.norm(points_world[first] - points_world[second], axis=-1)
        transition = pair_known & (labels[first] != labels[second]) & (gap >= EXACT_GEOMETRY_GAP_M)
        evaluable[first] |= pair_known
        evaluable[second] |= pair_known
        target[first] |= transition
        target[second] |= transition
    return target, evaluable


def _boundary_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    target_near = maximum_filter(target.astype(np.uint8), size=5) > 0
    predicted_near = maximum_filter(predicted.astype(np.uint8), size=5) > 0
    return {
        "predicted_seed_pixels": int(np.sum(predicted)),
        "exact_target_pixels": int(np.sum(target)),
        "precision_within_2px": int(np.sum(predicted & target_near)) / max(1, int(np.sum(predicted))),
        "recall_within_2px": int(np.sum(target & predicted_near)) / max(1, int(np.sum(target))),
    }


def run(
    rgbd_archive: Path,
    selected_root: Path,
    mesh_path: Path,
    mesh_archive: Path,
    pose_path: Path,
    identity_result_path: Path,
) -> dict[str, Any]:
    for path in (rgbd_archive, mesh_path, mesh_archive, pose_path, identity_result_path):
        require(path.is_file(), f"ICL boundary source missing: {path}")
    require(
        sha256_file(rgbd_archive)
        == "4ECA8C2E9F77C1BD7436C746D22EA6144B8C01FE9BC29A84E734186823F1F1AD",
        "ICL RGB-D archive hash drift",
    )
    identity_result = json.loads(identity_result_path.read_text(encoding="utf-8"))
    require(identity_result.get("status") == "ICL_EXACT_MESH_SUPPORT_IDENTITY_PASS", "ICL identity result invalid")
    selected_indices = [int(value) for value in identity_result["selection"]["selected_pose_indices"]]
    frame_identity = {int(row["pose_index"]): row for row in identity_result["frames"]}
    poses = parse_global_pose_text(pose_path.read_text(encoding="utf-8"))
    mesh = load_mesh_surface_samples(mesh_path)
    tree = cKDTree(mesh["points_world"])
    exact_floor_height = float(identity_result["mesh"]["exact_floor_world_height_m"])

    frames: list[dict[str, Any]] = []
    evaluable_metrics: list[dict[str, Any]] = []
    for index in selected_indices:
        depth_path = selected_root / f"depth/{index}.png"
        rgb_path = selected_root / f"rgb/{index}.png"
        require(depth_path.is_file() and rgb_path.is_file(), f"ICL selected RGB-D missing: {index}")
        depth, valid, intrinsics = downsample_exact_depth(depth_path)
        pose_icl = poses[index]
        labels, distance, points_world = mesh_surface_labels(
            depth,
            valid,
            intrinsics,
            pose_icl,
            tree,
            mesh["surface_ids"],
        )
        target, target_evaluable = exact_mesh_boundary_target(labels, points_world, valid)
        known_coverage = float(np.mean(labels >= 0))
        identity_evaluable = bool(frame_identity[index]["support_identity_evaluable"])
        frame_evaluable = bool(
            identity_evaluable
            and known_coverage >= 0.40
            and int(np.sum(target)) >= 20
        )
        metrics = None
        predicted_count = 0
        if identity_evaluable:
            camera_height = float(pose_icl[1, 3]) - exact_floor_height
            pose_canonical = canonical_camera_to_world(pose_icl)
            factors = compute_geometric_factors(
                depth,
                valid,
                intrinsics,
                pose_canonical,
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
            predicted_count = int(np.sum(predicted))
            if frame_evaluable:
                metrics = _boundary_metrics(predicted, target)
                evaluable_metrics.append(metrics)
        frames.append(
            {
                "pose_index": index,
                "identity_evaluable": identity_evaluable,
                "frame_evaluable": frame_evaluable,
                "mesh_known_coverage": known_coverage,
                "mesh_distance_median_m": float(np.median(distance[np.isfinite(distance)])),
                "mesh_distance_p95_m": float(np.quantile(distance[np.isfinite(distance)], 0.95)),
                "target_evaluable_coverage": float(np.mean(target_evaluable)),
                "exact_target_pixels": int(np.sum(target)),
                "predicted_seed_pixels": predicted_count,
                "metrics": metrics,
                "depth_path": str(depth_path.resolve()),
                "depth_sha256": sha256_file(depth_path),
                "rgb_path": str(rgb_path.resolve()),
                "rgb_sha256": sha256_file(rgb_path),
            }
        )
    require(evaluable_metrics, "ICL boundary has no evaluable frames")
    macro_precision = float(np.mean([row["precision_within_2px"] for row in evaluable_metrics]))
    macro_recall = float(np.mean([row["recall_within_2px"] for row in evaluable_metrics]))
    exact_pixels = int(sum(row["exact_target_pixels"] for row in evaluable_metrics))
    gates = {
        "evaluable_frame_count_ge_6": len(evaluable_metrics) >= 6,
        "exact_target_pixel_count_ge_100": exact_pixels >= 100,
        "macro_precision_within_2px_ge_0p50": macro_precision >= 0.50,
        "macro_recall_within_2px_ge_0p60": macro_recall >= 0.60,
    }
    passed = all(gates.values())
    return {
        "schema": "blindassist_ag_st_icl_pixel_boundary_canary_v1",
        "status": "ICL_PIXEL_EXACT_BOUNDARY_PASS" if passed else "ICL_PIXEL_EXACT_BOUNDARY_FAIL",
        "source": {
            "dataset": "ICL-NUIM living room lr kt0 exact TUM-compatible RGB-D plus exact OBJ",
            "rgbd_archive": str(rgbd_archive.resolve()),
            "rgbd_archive_bytes": rgbd_archive.stat().st_size,
            "rgbd_archive_sha256": sha256_file(rgbd_archive),
            "mesh_archive": str(mesh_archive.resolve()),
            "mesh_archive_sha256": sha256_file(mesh_archive),
            "global_pose_sha256": sha256_file(pose_path),
            "identity_result_sha256": sha256_file(identity_result_path),
        },
        "selection": {
            "pose_indices": selected_indices,
            "selection_reused_from_identity_before_pixel_outputs": True,
            "downsample": DOWNSAMPLE,
            "output_resolution_hw": [120, 160],
        },
        "mesh_association": {
            "maximum_distance_m": MESH_ASSOCIATION_MAX_M,
            "sample_count": len(mesh["points_world"]),
            "surface_count": mesh["surface_count"],
            "triangle_count": mesh["triangle_count"],
            "surface_area_m2": mesh["surface_area_m2"],
        },
        "target": {
            "definition": "adjacent exact-depth pixels map within 0.05 m to different exact-mesh material surfaces and have at least 0.06 m 3D gap",
            "unknown": "mesh-unassociated, identity-ineligible, or insufficient-target frames",
        },
        "evaluable_frame_count": len(evaluable_metrics),
        "exact_target_pixel_count": exact_pixels,
        "macro_precision_within_2px": macro_precision,
        "macro_recall_within_2px": macro_recall,
        "gates": gates,
        "frames": frames,
        "decision": {
            "boundary_external_pixel_exact_supported": passed,
            "boundary_training_authorized": passed,
            "complete_truth_required": False,
            "if_failed": "Keep boundary diagnostic and materialize only exact-mesh/geometric-consensus positives; do not tune on these 12 views.",
        },
        "claim_boundary": (
            "External synthetic exact RGB-D/mesh pixel boundary evidence on one ICL scene. "
            "The mesh target is selective and UNKNOWN outside associated surfaces; no real-world, task, product, or safety claim."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgbd-archive", type=Path, default=DEFAULT_RGBD_ARCHIVE)
    parser.add_argument("--selected-root", type=Path, default=DEFAULT_SELECTED_ROOT)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--mesh-archive", type=Path, default=DEFAULT_MESH_ARCHIVE)
    parser.add_argument("--global-poses", type=Path, default=DEFAULT_GLOBAL_POSES)
    parser.add_argument("--identity-result", type=Path, default=DEFAULT_IDENTITY_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), f"ICL pixel boundary output exists: {args.output}")
    result = run(
        args.rgbd_archive,
        args.selected_root,
        args.mesh,
        args.mesh_archive,
        args.global_poses,
        args.identity_result,
    )
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
