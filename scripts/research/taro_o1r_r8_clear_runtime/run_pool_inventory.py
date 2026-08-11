#!/usr/bin/env python3
"""Seal CRC-validated container inventory and exact frame plan for R8 pool."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r8_clear_runtime import pool_cohort


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r8_clear_pool_inventory_execution_lock.v1"
LOCK_ID = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_POOL_INVENTORY_ONE_SHOT_EXECUTION_LOCK"
SOURCE_ROOT = "artifacts.local/datasets/taro/o1r-r8-clear-pool-source-r0"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r8-clear-pool-inventory-r0"
PASS_TERMINAL = "TARO_O1R_R8_CLEAR_POOL_INVENTORY_AND_FRAME_PLAN_PASS"
EXPECTED_BINDINGS = {
    "R8_PROTOCOL": "docs/research/taro/TARO_O1R_R8_SOURCE_ONLY_CLEAR_NEGATIVE_CONTROL_COHORT_ENRICHMENT_PROTOCOL_LOCK_2026-08-12.json",
    "R8_DOWNLOAD_RECEIPTS": "artifacts.local/evidence/taro/o1r-r8-clear-pool-source-r0/download-receipts.json",
    "R8_DOWNLOAD_RESULT": "artifacts.local/evidence/taro/o1r-r8-clear-pool-source-r0/result.json",
    "R8_DOWNLOAD_MANIFEST": "artifacts.local/evidence/taro/o1r-r8-clear-pool-source-r0/manifest.json",
    "CONTAINER_RUNTIME": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R8_INVENTORY_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_pool_inventory.py",
}
EXPECTED_AUTHORITY = {
    "container_inventory": True,
    "zip_crc_validation": True,
    "trajectory_parse": True,
    "exact_frame_plan": True,
    "pixel_array_decode": False,
    "model_execution": False,
    "faro_read": False,
    "truth_scoring": False,
    "training": False,
}


class PoolInventoryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PoolInventoryError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R8_INVENTORY_OUTPUT_COLLISION", f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.canonical_json_bytes(dict(value)) + b"\n"
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "R8_INVENTORY_PARTIAL_COLLISION", f"partial exists: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": len(payload), "sha256": materializer.sha256_bytes(payload)}


def build_inventory(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    pool = pool_cohort.build_pool(root)
    source = materializer.safe_join(root, SOURCE_ROOT)
    parents = []
    total_materialized = 0
    for parent in pool["pool"]:
        video = parent["video_id"]
        upsampling_path = materializer.safe_join(source, f"upsampling/Training/{video}.zip")
        intrinsics_path = materializer.safe_join(source, f"raw/Training/{video}/lowres_wide_intrinsics.zip")
        trajectory_path = materializer.safe_join(source, f"raw/Training/{video}/lowres_wide.traj")
        upsampling_bytes = materializer.zip_uncompressed_bytes(upsampling_path)
        intrinsics_bytes = materializer.zip_uncompressed_bytes(intrinsics_path)
        materialized_bytes = upsampling_bytes + intrinsics_bytes + trajectory_path.stat().st_size
        total_materialized += materialized_bytes
        upsampling = materializer.index_upsampling_archive(upsampling_path, video, maximum_uncompressed_bytes=upsampling_bytes)
        intrinsics = materializer.index_intrinsics_archive(intrinsics_path, video, maximum_uncompressed_bytes=intrinsics_bytes)
        trajectory = materializer.parse_trajectory_payload(trajectory_path.read_bytes())
        frame_plan = materializer.exact_frame_plan(video, upsampling, intrinsics, trajectory)
        parents.append({
            "visit_id": parent["visit_id"], "video_id": video, "official_fold": parent["official_fold"], "pool_rank_sha256": parent["pool_rank_sha256"],
            "container_bindings": {
                "upsampling": {"path": upsampling_path.relative_to(root).as_posix(), "bytes": upsampling_path.stat().st_size, "sha256": materializer.sha256_file(upsampling_path), "uncompressed_bytes": upsampling_bytes},
                "intrinsics": {"path": intrinsics_path.relative_to(root).as_posix(), "bytes": intrinsics_path.stat().st_size, "sha256": materializer.sha256_file(intrinsics_path), "uncompressed_bytes": intrinsics_bytes},
                "trajectory": {"path": trajectory_path.relative_to(root).as_posix(), "bytes": trajectory_path.stat().st_size, "sha256": materializer.sha256_file(trajectory_path), "row_count": len(trajectory)},
            },
            "modality_member_counts": {role: len(rows) for role, rows in upsampling.items()},
            "intrinsics_member_count": len(intrinsics), "frame_plan": frame_plan, "materialized_bytes": materialized_bytes,
        })
    result = {
        "schema": "blindassist.taro.o1r.r8_clear_pool_inventory.v1",
        "request_plan_sha256": pool["request_plan"]["expanded_requests_sha256"],
        "parent_count": len(parents),
        "exact_pose_bounded_frame_count": sum(row["frame_plan"]["exact_pose_bounded_frame_count"] for row in parents),
        "materialized_bytes": total_materialized,
        "parents": parents,
        "zip_crc_validated": True, "pixel_arrays_decoded": False, "faro_values_interpreted": False, "truth_values_interpreted": False, "model_outputs_read": False, "training": False,
    }
    result["content_sha256"] = adapter.canonical_sha256(result)
    return validate_inventory(result)


def validate_inventory(
    value: Mapping[str, Any],
    *,
    expected_frame_counts: Sequence[int] | None = None,
    expected_materialized_bytes: int | None = None,
) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value)))
    content = result.pop("content_sha256", None)
    require(isinstance(content, str) and adapter.canonical_sha256(result) == content, "R8_INVENTORY_HASH_DRIFT", "inventory hash drift")
    result["content_sha256"] = content
    parents = result.get("parents")
    require(isinstance(parents, list) and len(parents) == result.get("parent_count") == 24, "R8_INVENTORY_PARENT_COUNT", "inventory parent count drift")
    expected = [(visit, video, rank) for visit, video, rank in pool_cohort.EXPECTED_POOL]
    observed = [(row.get("visit_id"), row.get("video_id"), row.get("pool_rank_sha256")) for row in parents]
    require(observed == expected, "R8_INVENTORY_ROSTER_DRIFT", "inventory roster drift")
    counts = [row["frame_plan"]["exact_pose_bounded_frame_count"] for row in parents]
    require(all(isinstance(count, int) and count > 0 for count in counts) and sum(counts) == result.get("exact_pose_bounded_frame_count"), "R8_INVENTORY_FRAME_COUNT_DRIFT", "inventory exact frame counts drift")
    require(all(len(row["frame_plan"]["exact_timestamp_tokens"]) == row["frame_plan"]["exact_pose_bounded_frame_count"] for row in parents), "R8_INVENTORY_TOKEN_COUNT_DRIFT", "inventory timestamp token count drift")
    require(result.get("materialized_bytes") == sum(row["materialized_bytes"] for row in parents), "R8_INVENTORY_BYTE_DRIFT", "inventory materialized byte count drift")
    if expected_frame_counts is not None:
        require(counts == list(expected_frame_counts), "R8_INVENTORY_FROZEN_FRAME_COUNT_DRIFT", "inventory differs from frozen per-parent counts")
    if expected_materialized_bytes is not None:
        require(result["materialized_bytes"] == expected_materialized_bytes, "R8_INVENTORY_FROZEN_BYTE_DRIFT", "inventory differs from frozen byte count")
    require(result.get("zip_crc_validated") is True and result.get("pixel_arrays_decoded") is False and result.get("faro_values_interpreted") is False and result.get("truth_values_interpreted") is False and result.get("model_outputs_read") is False and result.get("training") is False, "R8_INVENTORY_SCOPE_DRIFT", "inventory scope drift")
    return result


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R8_INVENTORY_LOCK_IDENTITY", "inventory lock identity/authority drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv and lock.get("source_root") == SOURCE_ROOT and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R8_INVENTORY_LOCK_POLICY", "inventory argv/root policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R8_INVENTORY_BINDINGS", "inventory binding count drift")
    verified = {}
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in verified and EXPECTED_BINDINGS.get(role) == relative, "R8_INVENTORY_BINDING_ROW", "inventory binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R8_INVENTORY_BINDING_HASH", f"inventory binding drift: {relative}")
        verified[role] = row
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R8_INVENTORY_AUTHORITY_DRIFT", "inventory authority drift")
    frozen = lock.get("frozen_expectations", {})
    require(isinstance(frozen.get("per_parent_frame_counts"), list) and len(frozen["per_parent_frame_counts"]) == 24 and all(isinstance(value, int) and value > 0 for value in frozen["per_parent_frame_counts"]) and frozen.get("exact_pose_bounded_frame_count") == sum(frozen["per_parent_frame_counts"]) and isinstance(frozen.get("materialized_bytes"), int) and frozen["materialized_bytes"] > 0, "R8_INVENTORY_EXPECTATIONS_INVALID", "inventory frozen expectations invalid")
    require(lock.get("resource_budget") == {"maximum_materialized_bytes": frozen["materialized_bytes"], "maximum_evidence_bytes": 33554432}, "R8_INVENTORY_BUDGET_DRIFT", "inventory budget drift")
    require(not _repo_path(OUTPUT_ROOT).exists(), "R8_INVENTORY_ROOT_COLLISION", "inventory output root exists")
    lock["_lock_path"], lock["_frozen"] = lock_path, frozen
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    output = _repo_path(OUTPUT_ROOT)
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    files["start-receipt.json"] = _write_exclusive(output / "start-receipt.json", {"schema": "blindassist.taro.o1r.r8_clear_pool_inventory_start.v1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "one_shot_consumed_on_root_creation": True})
    inventory = build_inventory(REPO_ROOT)
    validate_inventory(inventory, expected_frame_counts=lock["_frozen"]["per_parent_frame_counts"], expected_materialized_bytes=lock["_frozen"]["materialized_bytes"])
    files["exact-frame-plan.json"] = _write_exclusive(output / "exact-frame-plan.json", inventory)
    result = {"schema": "blindassist.taro.o1r.r8_clear_pool_inventory_result.v1", "execution_valid": True, "terminal": PASS_TERMINAL, "passed": True, "parent_count": 24, "exact_pose_bounded_frame_count": inventory["exact_pose_bounded_frame_count"], "per_parent_frame_counts": [row["frame_plan"]["exact_pose_bounded_frame_count"] for row in inventory["parents"]], "materialized_bytes": inventory["materialized_bytes"], "inventory_content_sha256": inventory["content_sha256"], "zip_crc_validated": True, "pixel_arrays_decoded": False, "faro_values_interpreted": False, "truth_values_interpreted": False, "model_outputs_read": False, "training": False, "one_shot_consumed": True}
    files["result.json"] = _write_exclusive(output / "result.json", result)
    _write_exclusive(output / "manifest.json", {"schema": "blindassist.taro.o1r.r8_clear_pool_inventory_manifest.v1", "files": files, "one_shot_consumed": True})
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"execution_valid": False, "error_code": getattr(error, "code", type(error).__name__), "message": str(error), "one_shot_consumed": _repo_path(OUTPUT_ROOT).exists()}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
