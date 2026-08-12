#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK = REPO_ROOT / "docs/research/assistive-geometry/BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT02_FACTOR_SPLIT_EXECUTION_LOCK_2026-08-11.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate(path: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    gates = {}
    gates["identity"] = (
        lock.get("lock_id")
        == "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_ATTEMPT02_FACTOR_SPLIT_EXECUTION_LOCK_2026-08-11"
        and lock.get("status") == "ATTEMPT02_FACTOR_SPLIT_EXECUTION_AUTHORIZED"
    )
    exact = True
    for binding in lock["bindings"].values():
        if not isinstance(binding, dict) or "path" not in binding:
            continue
        target = REPO_ROOT / binding["path"]
        exact &= target.is_file() and sha256_file(target) == binding["sha256"]
    gates["bindings"] = exact
    roles = lock["roles"]
    gates["roles"] = (
        roles["FIT"]["parents"] == 9
        and roles["FIT"]["frames"] == 27
        and roles["CHECKPOINT_SELECTION"]["parents"]
        == ["rgbd_dataset_freiburg1_teddy", "rgbd_dataset_freiburg2_dishes"]
        and roles["TRAIN_CANARY"]["parents"]
        == ["rgbd_dataset_freiburg2_coke", "rgbd_dataset_freiburg2_flowerbouquet"]
        and set(roles["forbidden_consumed_attempt01_canary"])
        == {"rgbd_dataset_freiburg2_large_with_loop", "rgbd_dataset_freiburg2_pioneer_slam3"}
    )
    model = lock["model"]
    gates["model"] = (
        model["trainable_parameters"] == 88566
        and model["component_parameter_sharing"] is False
        and model["prediction_fields"] == 14
        and model["final_task_head"] is False
        and model["reducer_in_graph"] is False
    )
    training = lock["training"]
    gates["training"] = (
        training["seeds"] == [17, 29, 43]
        and training["optimizer_steps"] == 2400
        and training["checkpoint_steps"] == [0, 100, 200, 300, 600, 1200, 2400]
        and training["total_optimizer_steps"] == 7200
    )
    authority = lock["authority"]
    gates["authority"] = (
        authority["optimizer_step"] is True
        and authority["canary_read_after_composite_seal"] is True
        and authority["old_attempt01_canary_read"] is False
        and authority["factor_tensor_adapter"] is False
        and authority["reducer_or_task_evaluation"] is False
        and authority["f2_or_mobile"] is False
    )
    gates["successor"] = (
        lock["unique_successor"]
        == "BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_LEARNABILITY_ATTEMPT02_RESULT"
    )
    return {
        "schema": "blindassist_assistive_geometry_r2_f1_attempt02_execution_lock_validation_v1",
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
