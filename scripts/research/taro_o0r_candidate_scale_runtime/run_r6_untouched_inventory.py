#!/usr/bin/env python3
"""Seal label-blind container inventory and exact frame identities for TARO R6."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.taro_o0r_candidate_scale_runtime import r6_untouched_cohort as cohort
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o0r.r6_untouched_inventory_execution_lock.v1"
LOCK_ID = "TARO_O0R_R6_UNTOUCHED_SOURCE_INVENTORY_ONE_SHOT_EXECUTION_LOCK"
SOURCE_ROOT = "artifacts.local/datasets/taro/o0r-r6-untouched-source-r0"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o0r-r6-untouched-inventory-r0"
EXPECTED_BINDINGS = {
    "R6_DATA_LOCK": "docs/research/taro/TARO_O0R_R6_UNTOUCHED_COHORT_AND_DATA_USE_LOCK_2026-08-11.json",
    "R6_DOWNLOAD_RECEIPTS": "artifacts.local/evidence/taro/o0r-r6-untouched-source-r0/download-receipts.json",
    "R6_DOWNLOAD_RESULT": "artifacts.local/evidence/taro/o0r-r6-untouched-source-r0/result.json",
    "R6_DOWNLOAD_MANIFEST": "artifacts.local/evidence/taro/o0r-r6-untouched-source-r0/manifest.json",
    "CONTAINER_RUNTIME": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R6_INVENTORY_RUNNER": "scripts/research/taro_o0r_candidate_scale_runtime/run_r6_untouched_inventory.py",
}


class R6InventoryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise R6InventoryError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _write(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R6_INVENTORY_OUTPUT_COLLISION", f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = adapter.canonical_json_bytes(dict(value)) + b"\n"
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return {"path": path.relative_to(REPO_ROOT).as_posix(), "bytes": len(payload), "sha256": materializer.sha256_bytes(payload)}


def build_inventory(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    plan = cohort.build_plan(repo_root)
    source = materializer.safe_join(repo_root, SOURCE_ROOT)
    parents = []
    total_materialized = 0
    for parent in plan["selection"]["roster"]:
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
        parents.append(
            {
                "visit_id": parent["visit_id"],
                "video_id": video,
                "official_fold": parent["official_fold"],
                "container_bindings": {
                    "upsampling": {"path": upsampling_path.relative_to(repo_root).as_posix(), "bytes": upsampling_path.stat().st_size, "sha256": materializer.sha256_file(upsampling_path), "uncompressed_bytes": upsampling_bytes},
                    "intrinsics": {"path": intrinsics_path.relative_to(repo_root).as_posix(), "bytes": intrinsics_path.stat().st_size, "sha256": materializer.sha256_file(intrinsics_path), "uncompressed_bytes": intrinsics_bytes},
                    "trajectory": {"path": trajectory_path.relative_to(repo_root).as_posix(), "bytes": trajectory_path.stat().st_size, "sha256": materializer.sha256_file(trajectory_path), "row_count": len(trajectory)},
                },
                "modality_member_counts": {role: len(rows) for role, rows in upsampling.items()},
                "intrinsics_member_count": len(intrinsics),
                "frame_plan": frame_plan,
                "materialized_bytes": materialized_bytes,
            }
        )
    result = {
        "schema": "blindassist.taro.o0r.r6_untouched_source_inventory.v1",
        "request_plan_sha256": plan["request_plan"]["expanded_requests_sha256"],
        "parent_count": len(parents),
        "exact_pose_bounded_frame_count": sum(row["frame_plan"]["exact_pose_bounded_frame_count"] for row in parents),
        "materialized_bytes": total_materialized,
        "parents": parents,
        "zip_crc_validated": True,
        "pixel_arrays_decoded": False,
        "truth_values_interpreted": False,
        "model_outputs_read": False,
        "training": False,
    }
    result["content_sha256"] = adapter.canonical_sha256(result)
    return validate_inventory(result)


def validate_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(dict(value)))
    content = result.pop("content_sha256", None)
    require(isinstance(content, str) and adapter.canonical_sha256(result) == content, "R6_INVENTORY_HASH_DRIFT", "inventory hash drift")
    result["content_sha256"] = content
    parents = result.get("parents")
    require(isinstance(parents, list) and len(parents) == result.get("parent_count") == 8, "R6_INVENTORY_PARENT_COUNT", "inventory parent count drift")
    expected = [(visit, video) for visit, video, _ in cohort.EXPECTED_ROSTER]
    require([(row.get("visit_id"), row.get("video_id")) for row in parents] == expected, "R6_INVENTORY_ROSTER_DRIFT", "inventory roster drift")
    counts = [row["frame_plan"]["exact_pose_bounded_frame_count"] for row in parents]
    require(counts == [16, 14, 8, 13, 11, 24, 5, 29] and sum(counts) == result.get("exact_pose_bounded_frame_count") == 120, "R6_INVENTORY_FRAME_COUNT_DRIFT", "inventory exact frame counts drift")
    require(all(len(row["frame_plan"]["exact_timestamp_tokens"]) == row["frame_plan"]["exact_pose_bounded_frame_count"] for row in parents), "R6_INVENTORY_TOKEN_COUNT_DRIFT", "inventory timestamp token count drift")
    require(result.get("materialized_bytes") == sum(row["materialized_bytes"] for row in parents) == 390499454, "R6_INVENTORY_BYTE_DRIFT", "inventory materialized bytes drift")
    require(result.get("zip_crc_validated") is True and result.get("pixel_arrays_decoded") is False and result.get("truth_values_interpreted") is False and result.get("model_outputs_read") is False and result.get("training") is False, "R6_INVENTORY_SCOPE_DRIFT", "inventory scope drift")
    return result


def validate_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R6_INVENTORY_LOCK_IDENTITY", "inventory lock identity drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv, "R6_INVENTORY_ARGV_DRIFT", "inventory argv drift")
    require(lock.get("source_root") == SOURCE_ROOT and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R6_INVENTORY_ROOT_DRIFT", "inventory root drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R6_INVENTORY_BINDINGS", "inventory binding count drift")
    verified = {}
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in verified and EXPECTED_BINDINGS.get(role) == relative, "R6_INVENTORY_BINDING_ROW", "inventory binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R6_INVENTORY_BINDING_HASH", f"inventory binding drift: {relative}")
        verified[role] = row
    authority = lock.get("execution_authority", {})
    require(authority == {"container_inventory": True, "zip_crc_validation": True, "trajectory_parse": True, "exact_frame_plan": True, "pixel_array_decode": False, "model_execution": False, "truth_scoring": False, "training": False}, "R6_INVENTORY_AUTHORITY_DRIFT", "inventory authority drift")
    lock["_lock_path"] = lock_path
    return lock


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_lock(lock_path)
    output = _repo_path(OUTPUT_ROOT)
    require(not output.exists(), "R6_INVENTORY_ROOT_COLLISION", "inventory output root exists")
    output.mkdir(parents=True, exist_ok=False)
    start = {"schema": "blindassist.taro.o0r.r6_untouched_inventory_start.v1", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "one_shot_consumed": True}
    files = {"start-receipt.json": _write(output / "start-receipt.json", start)}
    inventory = build_inventory(REPO_ROOT)
    files["exact-frame-plan.json"] = _write(output / "exact-frame-plan.json", inventory)
    result = {"schema": "blindassist.taro.o0r.r6_untouched_inventory_result.v1", "execution_valid": True, "terminal": "TARO_O0R_R6_UNTOUCHED_SOURCE_INVENTORY_PASS", "passed": True, "parent_count": 8, "exact_pose_bounded_frame_count": 120, "materialized_bytes": inventory["materialized_bytes"], "inventory_content_sha256": inventory["content_sha256"], "pixel_arrays_decoded": False, "truth_values_interpreted": False, "model_outputs_read": False, "one_shot_consumed": True}
    files["result.json"] = _write(output / "result.json", result)
    _write(output / "manifest.json", {"schema": "blindassist.taro.o0r.r6_untouched_inventory_manifest.v1", "files": files, "one_shot_consumed": True})
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
