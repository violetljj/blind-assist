#!/usr/bin/env python3
"""Validate the AG R2 F1 factor-learnability execution lock before optimizer use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_LEARNABILITY_EXECUTION_LOCK_2026-08-11.json"
EXPECTED_PRIMARY = [
    "depth_shape_abs_log_error",
    "depth_scale_abs_log_error",
    "depth_nll",
    "support_brier",
    "support_plane_angular_error_rad",
    "camera_height_abs_log_error",
    "support_nll",
    "obstacle_brier",
    "boundary_distance_abs_error_px",
    "boundary_nll",
]
EXPECTED_LOSSES = {
    "depth_shape_log_charbonnier",
    "metric_scale_log_huber",
    "depth_heteroscedastic_nll",
    "depth_validity_brier",
    "support_probability_brier",
    "support_plane_angular",
    "camera_height_log_huber",
    "support_residual_heteroscedastic_nll",
    "support_validity_brier",
    "obstacle_evidence_brier",
    "boundary_probability_brier",
    "boundary_localization_heteroscedastic_nll",
    "evidence_validity_brier",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    gates: dict[str, bool] = {}
    gates["identity_and_status"] = (
        lock.get("lock_id")
        == "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_LEARNABILITY_EXECUTION_LOCK_2026-08-11"
        and lock.get("status") == "F1_FACTOR_LEARNABILITY_EXECUTION_AUTHORIZED"
    )
    bindings_exact = True
    for binding in lock["bindings"].values():
        target = REPO_ROOT / binding["path"]
        bindings_exact &= target.is_file() and sha256_file(target) == binding["sha256"]
    runtime = lock["runtime"]
    checkpoint = Path(runtime["depthart_checkpoint"])
    extension = Path(runtime["depthart_cuda_extension"])
    bindings_exact &= checkpoint.is_file() and sha256_file(checkpoint) == runtime["depthart_checkpoint_sha256"]
    bindings_exact &= extension.is_file() and sha256_file(extension) == runtime["depthart_cuda_extension_sha256"]
    gates["all_bindings_exact"] = bindings_exact
    baseline_path = REPO_ROOT / lock["bindings"]["fit_only_baseline_result"]["path"]
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    gates["baseline_fit_only_complete"] = (
        baseline.get("status") == "F1_FIT_ONLY_BASELINES_AND_NORMALIZATION_FROZEN_BEFORE_MODEL_INITIALIZATION"
        and baseline.get("fit_frame_count") == 27
        and baseline.get("fit_parent_count") == 9
        and set(baseline.get("optimizer_normalization", {})) == EXPECTED_LOSSES
        and all(float(value) > 0.0 for value in baseline["optimizer_normalization"].values())
    )
    roles = lock["data_roles"]
    gates["roles_sealed"] = (
        roles["FIT"] == {"parents": 9, "frames": 27, "optimizer_access": True}
        and roles["CHECKPOINT_SELECTION"]["parents"] == 2
        and roles["CHECKPOINT_SELECTION"]["frames"] == 6
        and roles["CHECKPOINT_SELECTION"]["optimizer_access"] is False
        and roles["TRAIN_CANARY"]["parents"] == 2
        and roles["TRAIN_CANARY"]["frames"] == 6
        and roles["TRAIN_CANARY"]["optimizer_access"] is False
        and roles["parent_disjoint"] is True
    )
    model = lock["model"]
    gates["factor_only_model"] = (
        model["input"] == ["rgb_u8_hwc", "intrinsics_output"]
        and model["prediction_fields"] == 14
        and model["head_parameters"] == 307215
        and model["hidden_channels"] == 96
        and model["feature_channels"] == 192
        and model["final_task_head_present"] is False
        and model["reducer_in_graph"] is False
    )
    training = lock["training"]
    gates["training_budget_exact"] = (
        training["optimizer"] == "AdamW"
        and training["optimizer_steps"] == 2400
        and training["warmup_steps"] == 100
        and training["seeds"] == [17, 29, 43]
        and training["checkpoint_steps"] == [0, 300, 600, 1200, 2400]
        and training["total_optimizer_steps"] == 7200
        and training["best_seed_selection"] is False
    )
    selection = lock["checkpoint_selection"]
    gates["selection_factor_only"] = (
        selection["role"] == "CHECKPOINT_SELECTION"
        and selection["primary_metrics"] == EXPECTED_PRIMARY
        and selection["aggregate_training_loss_forbidden"] is True
        and selection["reducer_or_task_metric_forbidden"] is True
    )
    canary = lock["canary_success"]
    gates["canary_conjunctive"] = (
        canary["role"] == "TRAIN_CANARY"
        and canary["all_seeds_required"] is True
        and canary["all_primary_metrics_required"] is True
        and canary["uncertainty_families"] == ["depth", "support", "boundary"]
        and canary["aggregate_compensation"] is False
    )
    authority = lock["execution_authority"]
    gates["authority_bounded"] = (
        authority["optimizer_step"] is True
        and authority["checkpoint_creation"] is True
        and authority["selection_read"] is True
        and authority["canary_read_after_selection"] is True
        and authority["factor_tensor_adapter_real_seam"] is False
        and authority["reducer_execution"] is False
        and authority["task_evaluation"] is False
        and authority["f2_development"] is False
        and authority["mobile_or_default_app"] is False
    )
    gates["successor_unique"] = lock["unique_successor"] == "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_LEARNABILITY_RESULT"
    return {
        "schema": "blindassist_assistive_geometry_r2_f1_factor_execution_lock_validation_v1",
        "status": "F1_FACTOR_EXECUTION_LOCK_VALID" if all(gates.values()) else "F1_FACTOR_EXECUTION_LOCK_INVALID",
        "passed": all(gates.values()),
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    result = validate(args.lock.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
