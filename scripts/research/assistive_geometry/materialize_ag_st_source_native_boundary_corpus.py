#!/usr/bin/env python3
"""Build a compact multi-source boundary corpus without Teacher-filled labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import minimum_filter

from build_ag_st_factor_labels import (
    PROVENANCE_SOURCE_NATIVE,
    TIER_A_SOURCE,
    _pairwise_point_to_plane_edges,
    backproject_depth_grid,
    compute_dense_normals,
)
from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARKIT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-superteacher-factor-labels-multiteacher-train16-identity-r1"
)
DEFAULT_TUM_DIR = REPO_ROOT / "artifacts.local/experiments/ag-st-tum-support-identity-factors-r0"
DEFAULT_ICL_DIR = REPO_ROOT / "artifacts.local/experiments/ag-st-icl-fresh-depth-boundary-r2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts.local/experiments/ag-st-source-native-boundary-corpus-r0"


def conservative_source_boundary(
    depth_m: np.ndarray,
    source_valid: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    source = np.asarray(source_valid, dtype=np.bool_)
    source_neighborhood = minimum_filter(
        source.astype(np.uint8), size=3, mode="constant", cval=0
    ) > 0
    normals, normal_valid = compute_dense_normals(depth, source, intrinsics)
    points = backproject_depth_grid(depth, intrinsics)
    point_plane_edge, _, neighbor_count = _pairwise_point_to_plane_edges(
        points,
        normals,
        normal_valid,
        source,
    )
    probability = np.clip((point_plane_edge - 0.15) / 0.30, 0.0, 1.0).astype(np.float32)
    valid = source_neighborhood & normal_valid & (neighbor_count > 0)
    probability[~valid] = 0.0
    return probability, valid


def build_payload(
    probability: np.ndarray,
    valid: np.ndarray,
    uncertainty_px: np.ndarray,
) -> dict[str, np.ndarray]:
    score = np.asarray(probability, dtype=np.float32)
    mask = np.asarray(valid, dtype=np.bool_)
    uncertainty = np.asarray(uncertainty_px, dtype=np.float32)
    require(score.shape == mask.shape == uncertainty.shape, "boundary corpus shape mismatch")
    require(np.all(np.isfinite(score)) and np.all((score >= 0) & (score <= 1)), "boundary probability invalid")
    require(np.all(np.isfinite(uncertainty[mask])) and np.all(uncertainty[mask] >= 0), "boundary uncertainty invalid")
    tier = np.where(mask, TIER_A_SOURCE, 0).astype(np.uint8)
    provenance = np.where(mask, PROVENANCE_SOURCE_NATIVE, 0).astype(np.uint8)
    output_uncertainty = np.where(mask, uncertainty, np.inf).astype(np.float32)
    return {
        "boundary_probability_hw": score,
        "boundary_truth_valid_hw": mask.astype(np.uint8),
        "boundary_unknown_hw": (~mask).astype(np.uint8),
        "boundary_uncertainty_px_hw": output_uncertainty,
        "boundary_quality_tier_hw": tier,
        "boundary_provenance_hw": provenance,
    }


def _factor_parent(source_name: str, path: Path) -> str:
    if source_name == "arkitscenes":
        return path.stem.split("_", 1)[0]
    parts = path.stem.split("__")
    require(len(parts) >= 3, f"unexpected TUM factor identity: {path.name}")
    return parts[1]


def _materialize_factor_source(
    source_name: str,
    source_dir: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(source_dir.glob("*.npz"))
    require(paths, f"source boundary directory empty: {source_dir}")
    for path in paths:
        with np.load(path) as values:
            depth = np.asarray(values["metric_depth_m_hw"], dtype=np.float32)
            source_valid = np.asarray(values["source_native_valid_hw"], dtype=np.bool_)
            intrinsics = np.asarray(values["intrinsics_output"], dtype=np.float64)
            depth_uncertainty = np.asarray(values["depth_uncertainty_proxy_m_hw"], dtype=np.float32)
            probability, valid = conservative_source_boundary(depth, source_valid, intrinsics)
            uncertainty_px = 0.5 + 2.0 * np.clip(
                np.nan_to_num(depth_uncertainty, nan=1.0, posinf=1.0) / 0.10,
                0.0,
                1.0,
            )
        payload = build_payload(probability, valid, uncertainty_px)
        output_path = output_dir / f"{source_name}__{path.stem}.npz"
        np.savez_compressed(output_path, **payload)
        rows.append(
            {
                "source": source_name,
                "parent_id": _factor_parent(source_name, path),
                "frame_id": path.stem,
                "source_factor_path": str(path.resolve()),
                "source_factor_sha256": sha256_file(path),
                "output": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "pixels": int(probability.size),
                "valid_pixels": int(np.sum(valid)),
                "positive_pixels": int(np.sum(valid & (probability >= 0.5))),
            }
        )
    return rows


def _materialize_icl(source_dir: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = sorted(source_dir.glob("*.npz"))
    require(len(paths) == 12, "ICL exact boundary frame count drift")
    for path in paths:
        with np.load(path) as values:
            probability = np.asarray(values["boundary_probability_truth_hw"], dtype=np.float32)
            valid = np.asarray(values["boundary_truth_valid_hw"], dtype=np.bool_)
        payload = build_payload(
            probability,
            valid,
            np.full(probability.shape, 0.25, dtype=np.float32),
        )
        output_path = output_dir / f"icl_exact__{path.stem}.npz"
        np.savez_compressed(output_path, **payload)
        rows.append(
            {
                "source": "icl_exact",
                "parent_id": "icl_living_room_kt1",
                "frame_id": path.stem,
                "source_factor_path": str(path.resolve()),
                "source_factor_sha256": sha256_file(path),
                "output": str(output_path.resolve()),
                "output_sha256": sha256_file(output_path),
                "pixels": int(probability.size),
                "valid_pixels": int(np.sum(valid)),
                "positive_pixels": int(np.sum(valid & (probability >= 0.5))),
            }
        )
    return rows


def run(arkit_dir: Path, tum_dir: Path, icl_dir: Path, output_dir: Path) -> dict[str, Any]:
    for path in (arkit_dir, tum_dir, icl_dir):
        require(path.is_dir(), f"boundary corpus input missing: {path}")
    require(not output_dir.exists(), f"boundary corpus output exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    rows = [
        *_materialize_factor_source("arkitscenes", arkit_dir, output_dir),
        *_materialize_factor_source("tum_rgbd", tum_dir, output_dir),
        *_materialize_icl(icl_dir, output_dir),
    ]
    by_source: dict[str, dict[str, Any]] = {}
    for source in sorted({row["source"] for row in rows}):
        selected = [row for row in rows if row["source"] == source]
        by_source[source] = {
            "frame_count": len(selected),
            "parent_count": len({row["parent_id"] for row in selected}),
            "valid_pixels": sum(row["valid_pixels"] for row in selected),
            "positive_pixels": sum(row["positive_pixels"] for row in selected),
            "positive_parent_count": len(
                {row["parent_id"] for row in selected if row["positive_pixels"] > 0}
            ),
        }
    parent_count = len({(row["source"], row["parent_id"]) for row in rows})
    positive_parent_count = len(
        {(row["source"], row["parent_id"]) for row in rows if row["positive_pixels"] > 0}
    )
    gates = {
        "source_count_ge_3": len(by_source) >= 3,
        "parent_count_ge_20": parent_count >= 20,
        "frame_count_ge_75": len(rows) >= 75,
        "positive_parent_count_ge_16": positive_parent_count >= 16,
        "every_source_has_positive_pixels": all(value["positive_pixels"] > 0 for value in by_source.values()),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_source_native_boundary_corpus_v1",
        "status": "SOURCE_NATIVE_BOUNDARY_CORPUS_PASS" if passed else "SOURCE_NATIVE_BOUNDARY_CORPUS_DENOMINATOR_FAIL",
        "question": "Can source-native/synthetic-exact depth supply a multi-parent boundary corpus while all Teacher-filled boundary regions remain UNKNOWN?",
        "label_contract": {
            "arkitscenes_tum": "source-only 3x3 neighbourhood; camera-space point-to-plane probability clip((residual-0.15)/0.30,0,1)",
            "icl_exact": "source-exact 4-neighbour 3D gap >=0.06 m",
            "teacher_filled_pixels": "UNKNOWN and excluded",
            "negative": "operator-valid source neighbourhood with boundary probability below 0.5",
            "provenance": "source-native sensor-derived or source-exact synthetic; never Teacher consensus",
        },
        "source_count": len(by_source),
        "parent_count": parent_count,
        "frame_count": len(rows),
        "positive_parent_count": positive_parent_count,
        "by_source": by_source,
        "gates": gates,
        "frames": rows,
        "decision": {
            "masked_source_boundary_training_authorized": passed,
            "teacher_filled_boundary_training_authorized": False,
            "complete_truth_required": False,
            "next_execution": "Bind exact RGB locators and run a source-balanced boundary-only masked-student canary; retain Teacher-filled pixels as UNKNOWN.",
        },
        "claim_boundary": "Multi-source WILD_LAB source-derived boundary labels only; no formal F1, task, real-world safety, deployment, or product claim.",
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arkit-dir", type=Path, default=DEFAULT_ARKIT_DIR)
    parser.add_argument("--tum-dir", type=Path, default=DEFAULT_TUM_DIR)
    parser.add_argument("--icl-dir", type=Path, default=DEFAULT_ICL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    result = run(args.arkit_dir, args.tum_dir, args.icl_dir, args.output_dir)
    print(json.dumps({key: result[key] for key in ("status", "source_count", "parent_count", "frame_count", "positive_parent_count", "by_source", "gates")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
