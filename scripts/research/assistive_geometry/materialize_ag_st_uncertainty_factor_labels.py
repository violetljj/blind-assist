#!/usr/bin/env python3
"""Materialize metric, support, and camera-angular uncertainty factor labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from download_b0_arkitscenes_assets import require, sha256_file


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARKIT_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-unified-factor-labels-train16-r0/result.json"
)
DEFAULT_TUM_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-unified-factor-labels-tum7-r0/result.json"
)
DEFAULT_BONN_RESULT = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-bonn-fit-angular-factor-labels-r0/result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-uncertainty-factor-labels-trisource-r3"
)
EXPECTED_SOURCE_FRAME_COUNTS = {
    "arkitscenes": 48,
    "tum_rgbd": 21,
    "bonn_rgbd_fit": 24,
}


def _camera_rays(
    x: np.ndarray,
    y: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    k = np.asarray(intrinsics, dtype=np.float64)
    rays = np.stack(
        (
            (x - k[0, 2]) / k[0, 0],
            (y - k[1, 2]) / k[1, 1],
            np.ones_like(x, dtype=np.float64),
        ),
        axis=-1,
    )
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def pixel_radius_to_angular_uncertainty(
    uncertainty_px_hw: np.ndarray,
    valid_hw: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    radius = np.asarray(uncertainty_px_hw, dtype=np.float64)
    valid = np.asarray(valid_hw, dtype=np.bool_)
    k = np.asarray(intrinsics, dtype=np.float64)
    require(radius.ndim == 2 and radius.shape == valid.shape, "angular uncertainty shape drift")
    require(
        k.shape == (3, 3)
        and np.isfinite(k).all()
        and k[0, 0] > 0.0
        and k[1, 1] > 0.0,
        "angular uncertainty intrinsics invalid",
    )
    require(
        np.isfinite(radius[valid]).all() and np.all(radius[valid] >= 0.0),
        "valid pixel uncertainty invalid",
    )
    safe_radius = np.where(valid, radius, 0.0)
    y, x = np.indices(radius.shape, dtype=np.float64)
    center = _camera_rays(x, y, k)
    candidates = []
    for offset_x, offset_y in (
        (safe_radius, 0.0),
        (-safe_radius, 0.0),
        (0.0, safe_radius),
        (0.0, -safe_radius),
    ):
        shifted = _camera_rays(x + offset_x, y + offset_y, k)
        cosine = np.clip(np.sum(center * shifted, axis=-1), -1.0, 1.0)
        candidates.append(np.arccos(cosine))
    output = np.max(np.stack(candidates, axis=0), axis=0).astype(np.float32)
    output[~valid] = np.nan
    return output


def support_uncertainty_proxy(
    support_probability_hw: np.ndarray,
    support_valid_hw: np.ndarray,
    depth_m_hw: np.ndarray,
    depth_uncertainty_m_hw: np.ndarray,
    plane_residual_m: float,
) -> np.ndarray:
    probability = np.asarray(support_probability_hw, dtype=np.float32)
    valid = np.asarray(support_valid_hw, dtype=np.bool_)
    depth = np.asarray(depth_m_hw, dtype=np.float32)
    depth_uncertainty = np.asarray(depth_uncertainty_m_hw, dtype=np.float32)
    require(
        probability.shape == valid.shape == depth.shape == depth_uncertainty.shape,
        "support uncertainty shape drift",
    )
    require(
        np.isfinite(probability[valid]).all()
        and np.all((probability[valid] >= 0.0) & (probability[valid] <= 1.0)),
        "support probability invalid",
    )
    if not np.any(valid):
        return np.full(probability.shape, np.nan, dtype=np.float32)
    require(math.isfinite(plane_residual_m) and plane_residual_m >= 0.0, "support plane residual invalid")
    margin_uncertainty = 1.0 - np.abs(2.0 * probability - 1.0)
    metric_tolerance = 0.03 + 0.05 * np.maximum(depth, 0.0)
    depth_ratio = np.clip(
        depth_uncertainty / np.maximum(metric_tolerance, 1e-4),
        0.0,
        1.0,
    )
    plane_ratio = float(np.clip(plane_residual_m / 0.10, 0.0, 1.0))
    combined = 1.0 - (
        (1.0 - np.clip(margin_uncertainty, 0.0, 1.0))
        * (1.0 - depth_ratio)
        * (1.0 - plane_ratio)
    )
    combined = np.clip(combined, 0.0, 1.0).astype(np.float32)
    combined[~valid] = np.nan
    return combined


def _source_for(result_index: int, descriptor: dict[str, Any]) -> str:
    if result_index == 0:
        return "arkitscenes"
    if result_index == 1:
        require(descriptor.get("source") == "tum_rgbd", "TUM source identity drift")
        return "tum_rgbd"
    require(descriptor.get("source") == "bonn_rgbd_fit", "Bonn source identity drift")
    return "bonn_rgbd_fit"


def _provenance(values: Any, primary: str, fallback: str) -> np.ndarray:
    key = primary if primary in values else fallback
    return np.asarray(values[key], dtype=np.uint8)


def build_payload(values: Any) -> dict[str, np.ndarray]:
    depth = np.asarray(values["metric_depth_m_hw"], dtype=np.float32)
    depth_valid = np.asarray(values["metric_depth_valid_hw"], dtype=np.bool_)
    depth_uncertainty = np.asarray(values["depth_uncertainty_proxy_m_hw"], dtype=np.float32)
    depth_tier = np.asarray(values["quality_tier_hw"], dtype=np.uint8)
    depth_provenance = np.asarray(values["provenance_code_hw"], dtype=np.uint8)
    intrinsics = np.asarray(values["intrinsics_output"], dtype=np.float64)
    depth_factor_valid = (
        depth_valid
        & (depth_tier > 0)
        & np.isfinite(depth_uncertainty)
        & (depth_uncertainty >= 0.0)
    )

    support_probability = np.asarray(values["support_probability_pseudo_hw"], dtype=np.float32) if "support_probability_pseudo_hw" in values else np.asarray(values["support_truth_hw"], dtype=np.float32)
    support_valid = np.asarray(values["support_truth_valid_hw"], dtype=np.bool_)
    support_tier = np.asarray(values["support_quality_tier_hw"], dtype=np.uint8)
    support_provenance = _provenance(
        values,
        "support_provenance_code_hw",
        "support_provenance_hw",
    )
    plane_valid = bool(np.asarray(values["support_plane_valid"]).item()) if "support_plane_valid" in values else False
    plane_residual = float(np.asarray(values["support_plane_fit_residual_diagnostic_m"]).item()) if plane_valid else math.nan
    support_factor_valid = support_valid & (support_tier > 0) & plane_valid
    support_uncertainty = support_uncertainty_proxy(
        support_probability,
        support_factor_valid,
        depth,
        depth_uncertainty,
        plane_residual,
    )

    boundary_valid = np.asarray(values["boundary_truth_valid_hw"], dtype=np.bool_)
    boundary_tier = np.asarray(values["boundary_quality_tier_hw"], dtype=np.uint8)
    boundary_provenance = np.asarray(values["boundary_provenance_hw"], dtype=np.uint8)
    boundary_px = np.asarray(values["boundary_uncertainty_px_hw"], dtype=np.float32)
    boundary_factor_valid = (
        boundary_valid
        & (boundary_tier > 0)
        & np.isfinite(boundary_px)
        & (boundary_px >= 0.0)
    )
    boundary_angular = pixel_radius_to_angular_uncertainty(
        boundary_px,
        boundary_factor_valid,
        intrinsics,
    )

    def masked_tier(tier: np.ndarray, valid: np.ndarray) -> np.ndarray:
        return np.where(valid, tier, 0).astype(np.uint8)

    def masked_provenance(provenance: np.ndarray, valid: np.ndarray) -> np.ndarray:
        return np.where(valid, provenance, 0).astype(np.uint8)

    return {
        "depth_uncertainty_m_hw": np.where(depth_factor_valid, depth_uncertainty, np.nan).astype(np.float32),
        "depth_uncertainty_valid_hw": depth_factor_valid.astype(np.uint8),
        "depth_uncertainty_unknown_hw": (~depth_factor_valid).astype(np.uint8),
        "depth_uncertainty_quality_tier_hw": masked_tier(depth_tier, depth_factor_valid),
        "depth_uncertainty_provenance_hw": masked_provenance(depth_provenance, depth_factor_valid),
        "support_uncertainty_probability_hw": support_uncertainty,
        "support_uncertainty_valid_hw": support_factor_valid.astype(np.uint8),
        "support_uncertainty_unknown_hw": (~support_factor_valid).astype(np.uint8),
        "support_uncertainty_quality_tier_hw": masked_tier(support_tier, support_factor_valid),
        "support_uncertainty_provenance_hw": masked_provenance(support_provenance, support_factor_valid),
        "boundary_angular_uncertainty_rad_hw": boundary_angular,
        "boundary_uncertainty_valid_hw": boundary_factor_valid.astype(np.uint8),
        "boundary_uncertainty_unknown_hw": (~boundary_factor_valid).astype(np.uint8),
        "boundary_uncertainty_quality_tier_hw": masked_tier(boundary_tier, boundary_factor_valid),
        "boundary_uncertainty_provenance_hw": masked_provenance(boundary_provenance, boundary_factor_valid),
        "intrinsics_output": intrinsics,
    }


def validate_payload(payload: dict[str, np.ndarray]) -> None:
    factors = (
        ("depth", "depth_uncertainty_m_hw", None),
        ("support", "support_uncertainty_probability_hw", (0.0, 1.0)),
        ("boundary", "boundary_angular_uncertainty_rad_hw", (0.0, math.pi)),
    )
    reference_shape = payload["depth_uncertainty_valid_hw"].shape
    for prefix, value_key, bounds in factors:
        value = np.asarray(payload[value_key], dtype=np.float32)
        valid = np.asarray(payload[f"{prefix}_uncertainty_valid_hw"], dtype=np.bool_)
        unknown = np.asarray(payload[f"{prefix}_uncertainty_unknown_hw"], dtype=np.bool_)
        tier = np.asarray(payload[f"{prefix}_uncertainty_quality_tier_hw"], dtype=np.uint8)
        provenance = np.asarray(payload[f"{prefix}_uncertainty_provenance_hw"], dtype=np.uint8)
        require(
            value.shape == valid.shape == unknown.shape == tier.shape == provenance.shape == reference_shape,
            f"{prefix} uncertainty payload shape drift",
        )
        require(np.array_equal(unknown, ~valid), f"{prefix} UNKNOWN complement drift")
        require(np.isfinite(value[valid]).all(), f"{prefix} valid uncertainty nonfinite")
        require(np.isnan(value[~valid]).all(), f"{prefix} invalid uncertainty not NaN")
        require(
            np.all(tier[valid] > 0)
            and np.all(provenance[valid] > 0)
            and np.all(tier[~valid] == 0)
            and np.all(provenance[~valid] == 0),
            f"{prefix} tier/provenance closure drift",
        )
        if bounds is not None:
            require(
                np.all(value[valid] >= bounds[0]) and np.all(value[valid] <= bounds[1]),
                f"{prefix} uncertainty bounds drift",
            )


def run(result_paths: list[Path], output_dir: Path) -> dict[str, Any]:
    require(len(result_paths) == 3, "uncertainty factor input count drift")
    require(not output_dir.exists(), f"uncertainty factor output exists: {output_dir}")
    output_dir.mkdir(parents=True)
    inputs = []
    rows: list[dict[str, Any]] = []
    payload_invariant_count = 0
    coverage: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "frames": 0,
            "parents": 0,
            "depth_valid_pixels": 0,
            "support_valid_pixels": 0,
            "boundary_valid_pixels": 0,
        }
    )
    parents: dict[str, set[str]] = defaultdict(set)
    for result_index, result_path in enumerate(result_paths):
        require(result_path.is_file(), f"uncertainty input result missing: {result_path}")
        document = json.loads(result_path.read_text(encoding="utf-8"))
        inputs.append(
            {
                "path": str(result_path),
                "sha256": sha256_file(result_path),
                "schema": document.get("schema"),
                "status": document.get("status"),
            }
        )
        for descriptor in document["frames"]:
            source = _source_for(result_index, descriptor)
            source_path = Path(descriptor["output"])
            require(source_path.is_file(), f"uncertainty source payload missing: {source_path}")
            with np.load(source_path, allow_pickle=False) as values:
                payload = build_payload(values)
            validate_payload(payload)
            payload_invariant_count += 1
            output_path = output_dir / f"{source}__{descriptor['frame_id']}.npz"
            np.savez_compressed(output_path, **payload)
            depth_count = int(np.sum(payload["depth_uncertainty_valid_hw"]))
            support_count = int(np.sum(payload["support_uncertainty_valid_hw"]))
            boundary_count = int(np.sum(payload["boundary_uncertainty_valid_hw"]))
            parent_id = str(descriptor["parent_id"])
            parents[source].add(parent_id)
            coverage[source]["frames"] += 1
            coverage[source]["depth_valid_pixels"] += depth_count
            coverage[source]["support_valid_pixels"] += support_count
            coverage[source]["boundary_valid_pixels"] += boundary_count
            rows.append(
                {
                    "source": source,
                    "parent_id": parent_id,
                    "frame_id": str(descriptor["frame_id"]),
                    "source_path": str(source_path),
                    "source_sha256": descriptor["output_sha256"],
                    "output": str(output_path.resolve()),
                    "output_sha256": sha256_file(output_path),
                    "shape_hw": list(payload["depth_uncertainty_valid_hw"].shape),
                    "depth_uncertainty_valid_pixels": depth_count,
                    "support_uncertainty_valid_pixels": support_count,
                    "boundary_uncertainty_valid_pixels": boundary_count,
                }
            )
    for source in coverage:
        coverage[source]["parents"] = len(parents[source])
    coverage_output = {source: coverage[source] for source in sorted(coverage)}
    gates = {
        "source_frame_counts_exact": {
            source: coverage_output[source]["frames"] for source in sorted(coverage_output)
        }
        == EXPECTED_SOURCE_FRAME_COUNTS,
        "all_93_frames_materialized": len(rows) == 93,
        "all_sources_have_depth_uncertainty": all(
            coverage_output[source]["depth_valid_pixels"] > 0 for source in coverage_output
        ),
        "all_sources_have_boundary_angular_uncertainty": all(
            coverage_output[source]["boundary_valid_pixels"] > 0 for source in coverage_output
        ),
        "arkit_and_tum_have_support_uncertainty": all(
            coverage_output[source]["support_valid_pixels"] > 0
            for source in ("arkitscenes", "tum_rgbd")
        ),
        "bonn_support_remains_unknown": coverage_output["bonn_rgbd_fit"]["support_valid_pixels"] == 0,
        "all_output_hashes_present": all(len(row["output_sha256"]) == 64 for row in rows),
        "all_payload_invariants_pass": payload_invariant_count == 93,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_st_uncertainty_factor_labels_trisource_v1",
        "status": "TRISOURCE_UNCERTAINTY_FACTOR_LABELS_PASS" if passed else "TRISOURCE_UNCERTAINTY_FACTOR_LABELS_FAIL",
        "question": "Can existing source-native/anchored evidence be converted into factor-specific uncertainty supervision without complete truth or opening external outcomes?",
        "training_performed": False,
        "contract": {
            "depth": "Existing metric uncertainty proxy preserved with its own validity, tier, provenance and UNKNOWN closure.",
            "support": "Union of support decision-margin ambiguity, metric-depth uncertainty ratio and support-plane fit residual; valid only where support truth and plane evidence are valid.",
            "boundary": "Maximum camera-ray angular displacement induced by the existing pixel localization uncertainty in plus/minus x/y directions.",
            "bonn_support": "UNKNOWN because Bonn FIT has no gravity/support-plane evidence.",
        },
        "inputs": inputs,
        "frame_count": len(rows),
        "parent_count": len({(row["source"], row["parent_id"]) for row in rows}),
        "coverage": coverage_output,
        "gates": gates,
        "frames": rows,
        "decision": {
            "uncertainty_supervision_corpus_materialized": passed,
            "complete_truth_required": False,
            "student_training_authorized_by_this_result": False,
            "next_execution": "Run a masked uncertainty-head learnability canary over FIT parents only; retain support UNKNOWN on Bonn.",
        },
        "claim_boundary": "Source-anchored/proxy uncertainty label materialization only. No calibrated probabilistic truth, student learnability, task utility, formal F1, safety, deployment, or product claim.",
    }
    result_path = output_dir / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arkit-result", type=Path, default=DEFAULT_ARKIT_RESULT)
    parser.add_argument("--tum-result", type=Path, default=DEFAULT_TUM_RESULT)
    parser.add_argument("--bonn-result", type=Path, default=DEFAULT_BONN_RESULT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        [args.arkit_result.resolve(), args.tum_result.resolve(), args.bonn_result.resolve()],
        args.output_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "coverage": result["coverage"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )
    return 0 if all(result["gates"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
