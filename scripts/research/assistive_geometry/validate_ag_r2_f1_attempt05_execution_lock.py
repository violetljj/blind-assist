#!/usr/bin/env python3
"""Validate the Attempt-05 learned-uncertainty execution lock."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT05_UNCERTAINTY_CALIBRATION_AND_FRESH_CANARY_LOCK_2026-08-11.json"
)


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve(relative: str) -> Path:
    return (REPO_ROOT / relative).resolve()


def validate(lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    gates: dict[str, bool] = {}
    gates["A5_L01_SCHEMA_STATUS"] = (
        lock["schema"] == "blindassist_assistive_geometry_r2_f1_attempt05_uncertainty_calibration_and_fresh_canary_lock_v1"
        and lock["status"] == "ATTEMPT05_UNCERTAINTY_CALIBRATION_AND_FRESH_CANARY_EXECUTION_AUTHORIZED"
    )
    binding_exact = True
    for row in lock["bindings"].values():
        if not isinstance(row, dict) or "path" not in row or "sha256" not in row:
            continue
        path = resolve(row["path"])
        binding_exact = binding_exact and path.is_file() and sha256_file(path) == row["sha256"]
    gates["A5_L02_ALL_BINDINGS_EXACT"] = binding_exact

    fresh_binding = lock["bindings"]["fresh_label_result"]
    fresh = json.loads(resolve(fresh_binding["path"]).read_text(encoding="utf-8"))
    role_counts = {
        role: len([row for row in fresh["frames"] if row["role"] == role])
        for role in ("CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    gates["A5_L03_FRESH_LABELS_AND_ROLES"] = (
        bool(fresh["passed"])
        and role_counts == {"CHECKPOINT_SELECTION": 6, "TRAIN_CANARY": 6}
        and fresh["decision"]["selection_or_canary_ag_f1_model_metrics_opened"] is False
        and set(fresh["parent_joint"].values()) == {True}
    )

    calibration_binding = lock["bindings"]["calibration_result"]
    calibration = json.loads(resolve(calibration_binding["path"]).read_text(encoding="utf-8"))
    gates["A5_L04_CALIBRATION_INTERNAL_PASS"] = (
        bool(calibration["passed"])
        and calibration["depth"]["internal_validation"]["nondecreasing"]
        and calibration["boundary"]["internal_validation"]["nondecreasing"]
        and calibration["decision"]["point_factor_parameters_changed"] is False
        and calibration["decision"]["support_uncertainty_changed"] is False
    )
    gates["A5_L05_HELD_FIREWALL"] = (
        calibration["attempt05_held_label_result_excluded"]["sha256"] == fresh_binding["sha256"]
        and calibration["decision"]["fresh_attempt05_selection_or_canary_metrics_opened"] is False
        and set(calibration["fit_parents"]).isdisjoint(set(lock["roles"]["CHECKPOINT_SELECTION"]["parents"]))
        and set(calibration["fit_parents"]).isdisjoint(set(lock["roles"]["TRAIN_CANARY"]["parents"]))
        and set(calibration["internal_validation_parents"]).isdisjoint(set(lock["roles"]["CHECKPOINT_SELECTION"]["parents"]))
        and set(calibration["internal_validation_parents"]).isdisjoint(set(lock["roles"]["TRAIN_CANARY"]["parents"]))
    )

    checkpoint_binding = lock["bindings"]["calibrator_checkpoint"]
    checkpoint = torch.load(resolve(checkpoint_binding["path"]), map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    gates["A5_L06_CHECKPOINT_CONTENT"] = (
        checkpoint["schema"] == "blindassist_ag_r2_f1_attempt05_uncertainty_calibrators_v1"
        and metadata["attempt05_held_label_result_excluded"]["sha256"] == fresh_binding["sha256"]
        and checkpoint["models"]["depth"]["feature_names"] == lock["candidate"]["depth_feature_names"]
        and checkpoint["models"]["boundary"]["feature_names"] == lock["candidate"]["boundary_feature_names"]
    )
    gates["A5_L07_EXECUTION_ORDER_FAIL_CLOSED"] = (
        lock["authority"]["read_checkpoint_selection_metrics"] is True
        and lock["authority"]["read_train_canary_metrics_only_after_complete_selection_pass"] is True
        and lock["authority"]["serialize_factor_tensors_only_after_complete_canary_pass"] is True
        and lock["authority"]["optimizer_step"] is False
        and lock["authority"]["factor_tensor_adapter"] is False
        and lock["authority"]["reducer_or_task_evaluation"] is False
    )
    gates["A5_L08_SUCCESS_GATE_COMPLETE"] = (
        lock["success_gate"]["all_ten_primary_metrics_required"] is True
        and lock["success_gate"]["all_three_uncertainty_families_required"] is True
        and lock["success_gate"]["aggregate_compensation"] is False
        and lock["success_gate"]["reducer_or_task_rescue"] is False
    )
    return {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt05_execution_lock_validation_v1",
        "lock": str(lock_path.resolve()),
        "lock_sha256": sha256_file(lock_path),
        "gates": gates,
        "passed": all(gates.values()),
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
