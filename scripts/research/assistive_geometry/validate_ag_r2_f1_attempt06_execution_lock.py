#!/usr/bin/env python3
"""Validate the Attempt-06 recalibration and fresh-selection execution lock."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / (
    "docs/research/assistive-geometry/"
    "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT06_UNCERTAINTY_RECALIBRATION_AND_FRESH_SELECTION_LOCK_2026-08-11.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def resolve(relative: str) -> Path:
    return (REPO_ROOT / relative).resolve()


def validate(lock_path: Path) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    gates = {}
    gates["A6_L01_SCHEMA_STATUS"] = (
        lock["schema"] == "blindassist_assistive_geometry_r2_f1_attempt06_uncertainty_recalibration_and_fresh_selection_lock_v1"
        and lock["status"] == "ATTEMPT06_UNCERTAINTY_RECALIBRATION_AND_FRESH_SELECTION_EXECUTION_AUTHORIZED"
    )
    binding_exact = True
    for row in lock["bindings"].values():
        if not isinstance(row, dict) or "path" not in row or "sha256" not in row:
            continue
        path = resolve(row["path"])
        binding_exact = binding_exact and path.is_file() and sha256_file(path) == row["sha256"]
    gates["A6_L02_ALL_BINDINGS_EXACT"] = binding_exact

    selection = json.loads(resolve(lock["bindings"]["selection_label_result"]["path"]).read_text(encoding="utf-8"))
    gates["A6_L03_SELECTION_LABELS_FRESH"] = (
        bool(selection["passed"])
        and selection["frame_count"] == 6
        and selection["parent_count"] == 2
        and selection["decision"]["selection_model_or_baseline_metrics_opened"] is False
        and set(selection["parent_joint"].values()) == {True}
        and {row["parent_id"] for row in selection["frames"]} == set(lock["roles"]["CHECKPOINT_SELECTION"]["parents"])
    )
    preserved = json.loads(resolve(lock["bindings"]["preserved_label_result"]["path"]).read_text(encoding="utf-8"))
    attempt05 = json.loads(resolve(lock["bindings"]["attempt05_runtime_result"]["path"]).read_text(encoding="utf-8"))
    preserved_canary = [row for row in preserved["frames"] if row["role"] == "TRAIN_CANARY"]
    gates["A6_L04_CANARY_PRESERVED_UNOPENED"] = (
        bool(preserved["passed"])
        and len(preserved_canary) == 6
        and {row["parent_id"] for row in preserved_canary} == set(lock["roles"]["TRAIN_CANARY"]["parents"])
        and attempt05["selection_passed_before_canary_open"] is False
        and attempt05["canary_gate"] is None
        and attempt05["canary_evaluation"] is None
    )
    calibration = json.loads(resolve(lock["bindings"]["calibration_result"]["path"]).read_text(encoding="utf-8"))
    gates["A6_L05_CALIBRATION_INTERNAL_PASS"] = (
        bool(calibration["passed"])
        and calibration["boundary"]["internal_validation"]["nondecreasing"]
        and calibration["support"]["internal_validation"]["nondecreasing"]
        and calibration["support"]["internal_validation"]["proper_score_gain"] > 0.0
        and calibration["decision"]["point_factor_parameters_changed"] is False
        and calibration["decision"]["depth_calibrator_changed"] is False
    )
    gates["A6_L06_HELD_FIREWALL"] = (
        calibration["data_receipt"]["attempt06_selection_excluded"]["sha256"]
        == lock["bindings"]["selection_label_result"]["sha256"]
        and calibration["data_receipt"]["preserved_canary_excluded"]["sha256"]
        == lock["bindings"]["preserved_label_result"]["sha256"]
        and calibration["decision"]["attempt06_selection_metrics_opened"] is False
        and calibration["decision"]["preserved_canary_metrics_opened"] is False
    )
    checkpoint = torch.load(resolve(lock["bindings"]["calibrator_checkpoint"]["path"]), map_location="cpu", weights_only=True)
    gates["A6_L07_CHECKPOINT_CONTENT"] = (
        checkpoint["schema"] == "blindassist_ag_r2_f1_attempt06_uncertainty_calibrators_v1"
        and set(checkpoint["models"]) == {"depth", "boundary", "support"}
        and checkpoint["metadata"]["attempt05_depth_calibrator_preserved_sha256"]
        == lock["candidate"]["attempt05_depth_calibrator_sha256"]
        and checkpoint["models"]["support"]["feature_names"] == lock["candidate"]["support_feature_names"]
    )
    gates["A6_L08_EXECUTION_ORDER_AND_SUCCESS"] = (
        lock["authority"]["read_checkpoint_selection_metrics"] is True
        and lock["authority"]["read_preserved_canary_only_after_complete_selection_pass"] is True
        and lock["authority"]["serialize_factor_tensors_only_after_complete_canary_pass"] is True
        and lock["authority"]["optimizer_step"] is False
        and lock["authority"]["factor_tensor_adapter"] is False
        and lock["authority"]["reducer_or_task_evaluation"] is False
        and lock["success_gate"]["all_ten_primary_metrics_required"] is True
        and lock["success_gate"]["all_three_uncertainty_families_required"] is True
        and lock["success_gate"]["aggregate_compensation"] is False
    )
    return {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt06_execution_lock_validation_v1",
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
