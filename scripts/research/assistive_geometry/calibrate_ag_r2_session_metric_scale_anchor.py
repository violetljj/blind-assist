#!/usr/bin/env python3
"""Select a deterministic session-height metric-scale anchor on consumed factors."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent

import sys

sys.path.insert(0, str(MODULE_DIR))

from run_ag_r2_hybrid_factor_student_to_ag_seam import (  # noqa: E402
    estimate_height,
    height_observations,
    load_runtime_geometry,
)
from train_ag_r2_f1_factor_learnability_attempt03 import (  # noqa: E402
    require,
    sha256_file,
)


FROZEN_WALKING_SEAM_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-tum-walking-xyz-final-ag-seam-r1-recovery/result.json"
)
WALKING_LABEL_RESULT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-tum-walking-xyz-final-confirmation-labels-r0/result.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-r2-session-metric-scale-anchor-calibration-r0"
)
EXPECTED_FROZEN_WALKING_SEAM_SHA256 = (
    "BEA5F85A9C38BAB0A8EA8DCE81C8E851BABB1C2E01D7A65F82EFF14C2CCA0A96"
)
EXPECTED_WALKING_LABEL_SHA256 = (
    "D8083B567CF227AB83423A14B281B8B9A451DF8D6B29F78DF34A5B4459F10812"
)
MINIMUM_OBSERVATIONS = 64
MINIMUM_SCALE_CORRECTION = 0.5
MAXIMUM_SCALE_CORRECTION = 4.0
SUPPORT_THRESHOLD = 0.0
CANDIDATES = [
    {"estimator": estimator, "quantile": quantile}
    for estimator in ("weighted_mode", "weighted_quantile")
    for quantile in (0.25, 0.50, 0.75)
]


def resize_nearest(value: np.ndarray, output_hw: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(value, dtype=np.float32))[None, None]
    return F.interpolate(tensor, output_hw, mode="nearest")[0, 0].numpy()


def evaluate_candidate(
    candidate: dict[str, Any],
    seam: dict[str, Any],
    labels_by_sample: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seam_row in seam["frames"]:
        sample_id = str(seam_row["sample_id"])
        label_row = labels_by_sample[sample_id]
        factor_path = Path(seam_row["factor_tensor"]["path"])
        require(
            factor_path.is_file()
            and sha256_file(factor_path) == seam_row["factor_tensor"]["sha256"],
            "factor tensor drift",
        )
        with np.load(factor_path, allow_pickle=False) as factor:
            depth = np.exp(float(np.asarray(factor["log_metric_scale_m_scalar"]).item())) * np.asarray(
                factor["depth_shape_positive_hw"], dtype=np.float32
            )
            support = np.asarray(factor["support_probability_hw"], dtype=np.float32)
            depth_valid_probability = np.asarray(
                factor["depth_valid_probability_hw"], dtype=np.float32
            )
        label_path = Path(label_row["output"])
        require(
            label_path.is_file()
            and sha256_file(label_path) == label_row["output_sha256"],
            "label payload drift",
        )
        geometry = load_runtime_geometry(label_path, depth.shape)
        observations, weights = height_observations(
            depth,
            support,
            depth_valid_probability,
            geometry["intrinsics"],
            geometry["gravity"],
            SUPPORT_THRESHOLD,
        )
        require(
            observations.size >= MINIMUM_OBSERVATIONS,
            "anchor observation denominator empty",
        )
        estimated_height, _ = estimate_height(
            observations,
            weights,
            str(candidate["estimator"]),
            float(candidate["quantile"]),
        )
        correction = float(
            np.clip(
                float(geometry["target_height_m"]) / estimated_height,
                MINIMUM_SCALE_CORRECTION,
                MAXIMUM_SCALE_CORRECTION,
            )
        )
        with np.load(label_path, allow_pickle=False) as label:
            target_depth = resize_nearest(
                np.asarray(label["metric_depth_m_hw"], dtype=np.float32),
                depth.shape,
            )
            target_valid = resize_nearest(
                np.asarray(label["metric_depth_valid_hw"], dtype=np.float32),
                depth.shape,
            ) >= 0.5
        valid = target_valid & np.isfinite(target_depth) & (target_depth > 0.05)
        require(bool(valid.any()), "metric factor denominator empty")
        unanchored = np.log(np.maximum(depth[valid], 0.01)) - np.log(
            np.maximum(target_depth[valid], 0.01)
        )
        anchored = np.log(np.maximum(depth[valid] * correction, 0.01)) - np.log(
            np.maximum(target_depth[valid], 0.01)
        )
        rows.append(
            {
                "sample_id": sample_id,
                "observation_count": int(observations.size),
                "estimated_camera_height_m": estimated_height,
                "target_camera_height_m": float(geometry["target_height_m"]),
                "scale_correction": correction,
                "unanchored_log_rmse": float(np.sqrt(np.mean(np.square(unanchored)))),
                "anchored_log_rmse": float(np.sqrt(np.mean(np.square(anchored)))),
                "anchored_signed_scale_log_residual": float(np.mean(anchored)),
            }
        )
    residuals = np.asarray(
        [row["anchored_signed_scale_log_residual"] for row in rows],
        dtype=np.float64,
    )
    return {
        **candidate,
        "frame_count": len(rows),
        "mean_unanchored_log_rmse": float(
            np.mean([row["unanchored_log_rmse"] for row in rows])
        ),
        "mean_anchored_log_rmse": float(
            np.mean([row["anchored_log_rmse"] for row in rows])
        ),
        "scale_log_sigma_rms": float(np.sqrt(np.mean(np.square(residuals)))),
        "scale_log_sigma_q90": float(np.quantile(np.abs(residuals), 0.90)),
        "frames": rows,
    }


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    require(
        sha256_file(FROZEN_WALKING_SEAM_RESULT)
        == EXPECTED_FROZEN_WALKING_SEAM_SHA256,
        "frozen seam drift",
    )
    require(
        sha256_file(WALKING_LABEL_RESULT) == EXPECTED_WALKING_LABEL_SHA256,
        "walking labels drift",
    )
    seam = json.loads(FROZEN_WALKING_SEAM_RESULT.read_text(encoding="utf-8"))
    labels = json.loads(WALKING_LABEL_RESULT.read_text(encoding="utf-8"))
    require(
        not seam["passed"]
        and seam["decision"]["terminal_for_r0_regardless_of_outcome"],
        "frozen negative consumption state drift",
    )
    require(labels["passed"] and labels["frame_count"] == 12, "labels invalid")
    labels_by_sample = {str(row["sample_id"]): row for row in labels["frames"]}
    require(
        set(labels_by_sample) == {str(row["sample_id"]) for row in seam["frames"]},
        "factor/label roster mismatch",
    )
    candidates = [
        evaluate_candidate(candidate, seam, labels_by_sample)
        for candidate in CANDIDATES
    ]
    selected = min(
        candidates,
        key=lambda row: (
            float(row["mean_anchored_log_rmse"]),
            str(row["estimator"]),
            float(row["quantile"]),
        ),
    )
    improvement = 1.0 - float(selected["mean_anchored_log_rmse"]) / float(
        selected["mean_unanchored_log_rmse"]
    )
    gates = {
        "SCALEANCHOR_C01_EXACT_CONSUMED_FACTOR_AND_LABEL_RECEIPTS": True,
        "SCALEANCHOR_C02_SIX_PREDECLARED_PHYSICS_CANDIDATES": len(candidates) == 6,
        "SCALEANCHOR_C03_TWELVE_FRAMES_WITH_ANCHOR_DENOMINATOR": all(
            row["frame_count"] == 12 for row in candidates
        ),
        "SCALEANCHOR_C04_FACTOR_LOG_RMSE_IMPROVES_AT_LEAST_20_PERCENT": (
            improvement >= 0.20
        ),
        "SCALEANCHOR_C05_FINITE_Q90_SCALE_UNCERTAINTY": bool(
            math.isfinite(float(selected["scale_log_sigma_q90"]))
            and 0.02 <= float(selected["scale_log_sigma_q90"]) <= 1.0
        ),
        "SCALEANCHOR_C06_NO_TASK_OR_REDUCER_OUTPUT_USED_FOR_SELECTION": True,
    }
    passed = all(gates.values())
    config = {
        "schema": "blindassist_ag_r2_session_metric_scale_anchor_config_v1",
        "name": "SESSION_HEIGHT_GRAVITY_ALIGNED_DEPTH_QUANTILE",
        "estimator": selected["estimator"],
        "quantile": selected["quantile"],
        "support_threshold": SUPPORT_THRESHOLD,
        "minimum_observations": MINIMUM_OBSERVATIONS,
        "minimum_scale_correction": MINIMUM_SCALE_CORRECTION,
        "maximum_scale_correction": MAXIMUM_SCALE_CORRECTION,
        "metric_scale_log_sigma": selected["scale_log_sigma_q90"],
        "uncertainty_semantics": "CONSUMED_PARENT_Q90_ABSOLUTE_SIGNED_SCALE_LOG_RESIDUAL",
        "task_or_reducer_output_used": False,
    }
    result = {
        "schema": "blindassist_ag_r2_session_metric_scale_anchor_calibration_result_v1",
        "status": (
            "AG_R2_SESSION_METRIC_SCALE_ANCHOR_CALIBRATION_PASS"
            if passed
            else "AG_R2_SESSION_METRIC_SCALE_ANCHOR_CALIBRATION_FAIL"
        ),
        "passed": passed,
        "inputs": {
            "frozen_walking_seam": {
                "path": str(FROZEN_WALKING_SEAM_RESULT.resolve()),
                "sha256": EXPECTED_FROZEN_WALKING_SEAM_SHA256,
            },
            "walking_labels": {
                "path": str(WALKING_LABEL_RESULT.resolve()),
                "sha256": EXPECTED_WALKING_LABEL_SHA256,
            },
        },
        "candidate_selection_metric": "MEAN_FRAME_METRIC_DEPTH_LOG_RMSE_FACTOR_ONLY",
        "candidates": candidates,
        "selected": selected,
        "relative_improvement": improvement,
        "config": config,
        "gates": gates,
        "decision": {
            "walking_xyz_consumed_for_factor_calibration": True,
            "walking_xyz_r0_reopened": False,
            "next_confirmation_requires_new_parent": True,
        },
        "claim_ceiling": "Consumed one-parent factor calibration for a deterministic metric-scale anchor; not task utility, cross-source confirmation, deployment, product, or safety proof.",
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return result


def main() -> int:
    result = run()
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "relative_improvement": result["relative_improvement"],
                "config": result["config"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
