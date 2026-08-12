#!/usr/bin/env python3
"""Validate the exact Attempt-03 gravity/support-geometry execution lock."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT03_GRAVITY_AND_SUPPORT_GEOMETRY_EXECUTION_LOCK_R1_2026-08-11.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate(lock_path: Path = DEFAULT_LOCK) -> dict[str, Any]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    bindings = lock["bindings"]
    bound_files = {
        key: REPO_ROOT / value["path"]
        for key, value in bindings.items()
        if isinstance(value, dict) and "path" in value and "sha256" in value
    }
    exact = {
        key: path.is_file() and sha256_file(path) == bindings[key]["sha256"]
        for key, path in bound_files.items()
    }
    fresh_result = json.loads(bound_files["fresh_label_result"].read_text(encoding="utf-8"))
    runner_text = bound_files["runner"].read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_text)
    calls = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(runner_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    role_counts = {
        role: sum(row["role"] == role for row in fresh_result["frames"])
        for role in ("CHECKPOINT_SELECTION", "TRAIN_CANARY")
    }
    gates = {
        "A3_E01_BINDINGS_EXACT": all(exact.values()),
        "A3_E02_FRESH_LABELS_EXACT": fresh_result["passed"] is True and role_counts == {"CHECKPOINT_SELECTION": 6, "TRAIN_CANARY": 6},
        "A3_E03_NO_OPTIMIZER": "Adam" not in calls and "AdamW" not in calls and "optimizer" not in runner_text.lower(),
        "A3_E04_IMU_GRAVITY_INPUT_EXPLICIT": "gravity_up_camera_xyz" in runner_text and "support_plane_normal_camera_xyz" in runner_text,
        "A3_E05_DETERMINISTIC_HEIGHT_EXPLICIT": all(token in runner_text for token in ("intrinsics_output", "camera_to_world_output", "parent_vio_world_plane", "weighted_quantile", "per_pixel_height", "camera_height_m")),
        "A3_E06_CANARY_DELAYED": "if all_selection_eligible" in runner_text and "canary_after_all_selection_eligible" in runner_text,
        "A3_E07_TASK_FIREWALL": "load_native_targets" in runner_text and "reducer_or_task_outcome_read\": False" in runner_text,
    }
    return {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt03_execution_lock_validation_v1",
        "passed": all(gates.values()),
        "lock": str(lock_path.resolve()),
        "lock_sha256": sha256_file(lock_path),
        "binding_checks": exact,
        "role_frame_counts": role_counts,
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
