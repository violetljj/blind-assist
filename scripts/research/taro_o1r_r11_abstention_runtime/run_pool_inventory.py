#!/usr/bin/env python3
"""Seal CRC-validated container inventory and exact frame plan for TARO R11."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_download
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_head


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r11_fresh_pool_inventory_execution_lock.v1"
LOCK_ID = "TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_ONE_SHOT_EXECUTION_LOCK"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_INVENTORY_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
SOURCE_ROOT = run_pool_download.SOURCE_ROOT
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0"
PASS_TERMINAL = "TARO_O1R_R11_FRESH_POOL_INVENTORY_AND_FRAME_PLAN_PASS"
NOT_READY_TERMINAL = "TARO_O1R_R11_FRESH_POOL_INVENTORY_NOT_PHASE_A_READY_NO_REPLACEMENT"
INVALID_TERMINAL = "TARO_O1R_R11_FRESH_POOL_INVENTORY_EXECUTION_INVALID"
DOWNLOAD_FORMAL_RESULT = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_BOUNDED_SOURCE_DOWNLOAD_RESULT_2026-08-12.json"
DOWNLOAD_FORMAL_CONTENT_SHA256 = "817E3D7586C261DAD6F11D78A341959A13561C0E4CDB033CB6A208E8ED137D63"
DOWNLOAD_MANIFEST_CONTENT_SHA256 = "FCE3E06DFEC0C2769EFCC3DF5172DF251B9BF7754D077C92A4B3CC3ACA3B7A21"
PROTOCOL_CONTENT_SHA256 = "2A2854364E41CE2E94FE2D1DBF1F5EF068E18335DE1171A8486BC41CBAECF756"
AUTHORIZATION_CONTENT_SHA256 = "CF7814D52532FAB6A5EE8A4CA8EA29E9A7EF1017E075CF8FE597EEBE0834FF5F"
POOL_CONTENT_SHA256 = "9F1EE94980C9B2EB0C8D7A6503A25E11587760247C5A30F656DB28E60A27FFAF"
REQUEST_PLAN_SHA256 = "FE3578E4F8403F9F57DA767B21DC5EFBCAF6BBF6514DF776A7B3124B966BD521"
MAXIMUM_MATERIALIZED_BYTES = 32212254720
MAXIMUM_EVIDENCE_BYTES = 67108864
MAXIMUM_WALL_SECONDS = 7200
EXPECTED_ARGV = [
    "-m",
    "scripts.research.taro_o1r_r11_abstention_runtime.run_pool_inventory",
    "--execution-lock",
    LOCK_RELATIVE,
]
EXPECTED_BINDINGS = {
    "R11_PROTOCOL": run_pool_head.PROTOCOL_RELATIVE,
    "R11_DATA_USE_AUTHORIZATION": run_pool_head.AUTHORIZATION_RELATIVE,
    "R11_POOL_PLANNER": "scripts/research/taro_o1r_r11_abstention_runtime/fresh_pool.py",
    "R11_DOWNLOAD_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_download.py",
    "R11_DOWNLOAD_RECEIPTS": f"{run_pool_download.EVIDENCE_ROOT}/download-receipts.json",
    "R11_DOWNLOAD_RESULT": f"{run_pool_download.EVIDENCE_ROOT}/result.json",
    "R11_DOWNLOAD_MANIFEST": f"{run_pool_download.EVIDENCE_ROOT}/manifest.json",
    "R11_DOWNLOAD_FORMAL_RESULT": DOWNLOAD_FORMAL_RESULT,
    "CONTAINER_RUNTIME": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "R11_INVENTORY_RUNNER": "scripts/research/taro_o1r_r11_abstention_runtime/run_pool_inventory.py",
    "R11_INVENTORY_TEST": "scripts/research/taro_o1r_r11_abstention_runtime/test_run_pool_inventory.py",
}
ARTIFACT_BINDING_ROLES = {"R11_DOWNLOAD_RECEIPTS", "R11_DOWNLOAD_RESULT", "R11_DOWNLOAD_MANIFEST"}
EXPECTED_AUTHORITY = {
    "source_container_read": True,
    "zip_central_directory_inventory": True,
    "zip_declared_crc_index": True,
    "zip_member_payload_read": False,
    "zip_member_payload_crc_validation": False,
    "trajectory_parse": True,
    "exact_frame_plan": True,
    "pixel_array_decode": False,
    "source_frame_materialization": False,
    "model_execution": False,
    "faro_read": False,
    "truth_scoring": False,
    "training": False,
    "device": False,
    "deployment": False,
    "product": False,
    "safety": False,
}
EXPECTED_RESOURCE_BUDGET = {
    "maximum_declared_materialized_bytes": MAXIMUM_MATERIALIZED_BYTES,
    "maximum_evidence_bytes": MAXIMUM_EVIDENCE_BYTES,
    "maximum_wall_seconds": MAXIMUM_WALL_SECONDS,
    "network_requests": 0,
}
EXPECTED_INVENTORY_POLICY = {
    "zip_index_mode": "CENTRAL_DIRECTORY_METADATA_ONLY",
    "highres_depth_metadata_allowed": True,
    "zip_member_payload_reads_allowed_roles": [],
    "highres_depth_member_payload_reads": 0,
    "member_crc_status": "DECLARED_ONLY_NOT_PAYLOAD_VALIDATED",
    "trajectory_payload_reads_per_parent": 1,
    "pixel_arrays_decoded": False,
}


class PoolInventoryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PoolInventoryError(code, message)


@dataclass(frozen=True)
class DeclaredMemberBinding:
    """ZIP central-directory binding; member payload and CRC are not read."""

    role: str
    timestamp_token: str
    source_member_path: str
    canonical_member_path: str
    bytes: int
    declared_crc32: str


def _validate_zip_info_metadata_only(info: zipfile.ZipInfo) -> None:
    name = info.filename
    require("\\" not in name, "R11_INVENTORY_ZIP_MEMBER_PATH", "ZIP member uses backslash")
    pure = PurePosixPath(name)
    require(
        not pure.is_absolute() and all(part not in ("", ".", "..") for part in pure.parts),
        "R11_INVENTORY_ZIP_MEMBER_PATH",
        "unsafe ZIP member path",
    )
    require(
        not (info.external_attr >> 16) & 0o170000 == 0o120000,
        "R11_INVENTORY_ZIP_SYMLINK",
        "ZIP symlink forbidden",
    )
    require(info.flag_bits & 0x1 == 0, "R11_INVENTORY_ZIP_ENCRYPTED", "encrypted ZIP member forbidden")
    require(
        info.compress_type in materializer.SUPPORTED_ZIP_COMPRESSION,
        "R11_INVENTORY_ZIP_COMPRESSION",
        "ZIP compression method is not frozen",
    )
    require(
        info.file_size >= 0 and info.compress_size >= 0,
        "R11_INVENTORY_ZIP_MEMBER_SIZE",
        "ZIP member size is invalid",
    )


def _timestamp_token_from_member(video_id: str, name: str, suffix: str) -> str:
    pure = PurePosixPath(name)
    require(pure.suffix.lower() == suffix, "R11_INVENTORY_MEMBER_SUFFIX", "member suffix drift")
    prefix = f"{video_id}_"
    require(
        pure.stem.startswith(prefix),
        "R11_INVENTORY_MEMBER_VIDEO_PREFIX",
        "member is not bound to video identity",
    )
    token = pure.stem[len(prefix) :]
    try:
        adapter.decimal_timestamp_ns(token)
    except adapter.AdapterError as error:
        raise PoolInventoryError("R11_INVENTORY_FRAME_TOKEN", str(error)) from error
    return token


def _metadata_zip_rows(
    path: Path, *, maximum_declared_uncompressed_bytes: int
) -> tuple[list[zipfile.ZipInfo], int]:
    require(
        isinstance(maximum_declared_uncompressed_bytes, int)
        and not isinstance(maximum_declared_uncompressed_bytes, bool)
        and maximum_declared_uncompressed_bytes > 0,
        "R11_INVENTORY_MATERIALIZED_BUDGET",
        "declared materialized byte budget invalid",
    )
    try:
        with zipfile.ZipFile(path) as bundle:
            rows = bundle.infolist()
            seen_names: set[str] = set()
            total = 0
            for info in rows:
                _validate_zip_info_metadata_only(info)
                require(
                    info.filename not in seen_names,
                    "R11_INVENTORY_ZIP_MEMBER_DUPLICATE",
                    "duplicate ZIP member path",
                )
                seen_names.add(info.filename)
                total += int(info.file_size)
                require(
                    total <= maximum_declared_uncompressed_bytes,
                    "R11_INVENTORY_MATERIALIZED_BUDGET",
                    "ZIP declared uncompressed bytes exceed budget",
                )
            return rows, total
    except zipfile.BadZipFile as error:
        raise PoolInventoryError("R11_INVENTORY_ZIP_CONTAINER", "source archive is not a valid ZIP") from error


def index_upsampling_archive_metadata_only(
    path: Path, video_id: str, *, maximum_declared_uncompressed_bytes: int
) -> tuple[dict[str, dict[str, DeclaredMemberBinding]], int]:
    """Index source roles from the central directory without opening members."""

    rows, declared_uncompressed_bytes = _metadata_zip_rows(
        path, maximum_declared_uncompressed_bytes=maximum_declared_uncompressed_bytes
    )
    result: dict[str, dict[str, DeclaredMemberBinding]] = {
        role: {} for role in materializer.UPSAMPLING_DIRECTORY_TO_ROLE.values()
    }
    for info in rows:
        if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != ".png":
            continue
        pure = PurePosixPath(info.filename)
        if len(pure.parts) < 2 or pure.parts[-2] not in materializer.UPSAMPLING_DIRECTORY_TO_ROLE:
            continue
        role = materializer.UPSAMPLING_DIRECTORY_TO_ROLE[pure.parts[-2]]
        token = _timestamp_token_from_member(video_id, info.filename, ".png")
        require(token not in result[role], "R11_INVENTORY_FRAME_MEMBER_DUPLICATE", "duplicate modality timestamp")
        require(info.file_size > 0, "R11_INVENTORY_ZIP_MEMBER_SIZE", "bound source member must be non-empty")
        result[role][token] = DeclaredMemberBinding(
            role=role,
            timestamp_token=token,
            source_member_path=info.filename,
            canonical_member_path=f"{role}/{token}.png",
            bytes=int(info.file_size),
            declared_crc32=f"{info.CRC:08X}",
        )
    require(all(result.values()), "R11_INVENTORY_MODALITY_MISSING", "one or more source modalities are empty")
    return result, declared_uncompressed_bytes


def index_intrinsics_archive_metadata_only(
    path: Path, video_id: str, *, maximum_declared_uncompressed_bytes: int
) -> tuple[dict[str, DeclaredMemberBinding], int]:
    """Index intrinsics from the central directory without opening members."""

    rows, declared_uncompressed_bytes = _metadata_zip_rows(
        path, maximum_declared_uncompressed_bytes=maximum_declared_uncompressed_bytes
    )
    result: dict[str, DeclaredMemberBinding] = {}
    for info in rows:
        if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != ".pincam":
            continue
        token = _timestamp_token_from_member(video_id, info.filename, ".pincam")
        require(token not in result, "R11_INVENTORY_INTRINSICS_MEMBER_DUPLICATE", "duplicate intrinsics timestamp")
        require(info.file_size > 0, "R11_INVENTORY_ZIP_MEMBER_SIZE", "bound intrinsics member must be non-empty")
        result[token] = DeclaredMemberBinding(
            role="intrinsics",
            timestamp_token=token,
            source_member_path=info.filename,
            canonical_member_path=f"intrinsics/{token}.pincam",
            bytes=int(info.file_size),
            declared_crc32=f"{info.CRC:08X}",
        )
    require(bool(result), "R11_INVENTORY_INTRINSICS_MISSING", "intrinsics archive has no pincam members")
    return result, declared_uncompressed_bytes


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "R11_INVENTORY_JSON_OBJECT", f"JSON object required: {path}")
    return value


def _validate_content_seal(value: Mapping[str, Any], code: str) -> dict[str, Any]:
    record = json.loads(json.dumps(dict(value)))
    claimed = record.pop("content_sha256", None)
    require(isinstance(claimed, str) and claimed == adapter.canonical_sha256(record), code, "content seal drift")
    record["content_sha256"] = claimed
    return record


def _sealed_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = dict(value)
    require("content_sha256" not in record, "R11_INVENTORY_INTERNAL", "record already sealed")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _canonical_json_line(value: Mapping[str, Any]) -> bytes:
    return adapter.canonical_json_bytes(dict(value)) + b"\n"


def _write_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    require(not path.exists(), "R11_INVENTORY_OUTPUT_COLLISION", f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_json_line(value)
    partial = path.with_name(path.name + ".partial")
    require(not partial.exists(), "R11_INVENTORY_PARTIAL_COLLISION", f"partial exists: {partial}")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "bytes": len(payload),
        "sha256": materializer.sha256_bytes(payload),
    }


def _git_bytes(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(completed.returncode == 0, "R11_INVENTORY_IMPLEMENTATION_COMMIT", f"binding absent from implementation commit: {relative}")
    return completed.stdout


def _validate_implementation_ancestor(commit: Any) -> str:
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit) is not None, "R11_INVENTORY_IMPLEMENTATION_COMMIT", "implementation commit must be a lowercase full SHA")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(completed.returncode == 0, "R11_INVENTORY_IMPLEMENTATION_COMMIT", "implementation commit is not an ancestor of HEAD")
    return commit


def _validate_download_receipt_record(
    row: Mapping[str, str], head: Mapping[str, Any], value: Mapping[str, Any]
) -> dict[str, Any]:
    receipt = _validate_content_seal(value, "R11_INVENTORY_DOWNLOAD_RECEIPT_HASH")
    require(
        receipt.get("schema") == "blindassist.taro.o1r.r11_source_asset_download_receipt.v1"
        and all(receipt.get(field) == row[field] for field in ("visit_id", "video_id", "asset", "url", "relative_path")),
        "R11_INVENTORY_DOWNLOAD_RECEIPT_IDENTITY",
        "download receipt identity drift",
    )
    require(
        receipt.get("bytes") == head.get("content_length_bytes") == receipt.get("head_content_length_bytes")
        and receipt.get("head_etag") == head.get("etag")
        and receipt.get("head_last_modified") == head.get("last_modified")
        and receipt.get("redirect_chain") == []
        and isinstance(receipt.get("attempt_count"), int)
        and not isinstance(receipt.get("attempt_count"), bool)
        and 1 <= receipt["attempt_count"] <= run_pool_download.EXPECTED_BUDGET["maximum_attempts_per_asset"]
        and isinstance(receipt.get("prior_transport_errors"), list)
        and all(isinstance(item, str) for item in receipt["prior_transport_errors"])
        and len(receipt["prior_transport_errors"]) == receipt["attempt_count"] - 1,
        "R11_INVENTORY_DOWNLOAD_RECEIPT_HEAD_BINDING",
        "download receipt/HEAD binding drift",
    )
    require(
        isinstance(receipt.get("sha256"), str)
        and re.fullmatch(r"[0-9A-F]{64}", receipt["sha256"]) is not None
        and isinstance(receipt.get("crc32"), str)
        and re.fullmatch(r"[0-9A-F]{8}", receipt["crc32"]) is not None,
        "R11_INVENTORY_DOWNLOAD_RECEIPT_DIGEST",
        "download receipt digest format drift",
    )
    return receipt


def verify_download_evidence(*, verify_source_files: bool = True) -> dict[str, Any]:
    download_lock = run_pool_download.validate_execution_lock(
        _repo_path(run_pool_download.LOCK_RELATIVE), require_roots_absent=False
    )
    evidence = _repo_path(run_pool_download.EVIDENCE_ROOT)
    manifest = _validate_content_seal(
        _read_json(_repo_path(EXPECTED_BINDINGS["R11_DOWNLOAD_MANIFEST"])),
        "R11_INVENTORY_DOWNLOAD_MANIFEST_SEAL",
    )
    expected_files = {"start-receipt.json", "download-receipts.json", "result.json"} | {
        f"receipts/{index:03d}.json" for index in range(1, run_pool_download.ASSET_COUNT + 1)
    }
    files = manifest.get("files")
    require(
        manifest.get("schema") == "blindassist.taro.o1r.r11_fresh_pool_download_manifest.v1"
        and manifest.get("one_shot_consumed") is True
        and manifest.get("content_sha256") == DOWNLOAD_MANIFEST_CONTENT_SHA256
        and isinstance(files, dict)
        and set(files) == expected_files,
        "R11_INVENTORY_DOWNLOAD_MANIFEST",
        "download manifest identity/file set drift",
    )
    for relative, binding in files.items():
        target = materializer.safe_join(evidence, relative)
        require(
            target.is_file()
            and target.stat().st_size == binding.get("bytes")
            and materializer.sha256_file(target) == binding.get("sha256"),
            "R11_INVENTORY_DOWNLOAD_FILE",
            f"download artifact drift: {relative}",
        )

    plan = fresh_pool.build_pool(REPO_ROOT)
    rows = run_pool_download.expanded_download_plan(plan)
    receipts_doc = _read_json(_repo_path(EXPECTED_BINDINGS["R11_DOWNLOAD_RECEIPTS"]))
    receipts = receipts_doc.get("receipts")
    require(
        receipts_doc.get("schema") == "blindassist.taro.o1r.r11_fresh_pool_download_receipts.v1"
        and isinstance(receipts, list)
        and len(receipts) == run_pool_download.ASSET_COUNT,
        "R11_INVENTORY_DOWNLOAD_RECEIPTS",
        "download receipt identity/count drift",
    )
    source = _repo_path(SOURCE_ROOT)
    head_lookup = {row["url"]: row for row in download_lock["_head"]["assets"]}
    total = 0
    checked_receipts: list[dict[str, Any]] = []
    for index, (expected, observed) in enumerate(zip(rows, receipts, strict=True), start=1):
        individual = _read_json(evidence / f"receipts/{index:03d}.json")
        require(individual == observed, "R11_INVENTORY_DOWNLOAD_RECEIPT_ROW", "aggregate/individual receipt drift")
        checked = _validate_download_receipt_record(expected, head_lookup[expected["url"]], observed)
        if verify_source_files:
            checked = run_pool_download.validate_download_receipt(
                expected, head_lookup[expected["url"]], observed, source_root=source
            )
        require(checked["attempt_count"] == 1 and checked["prior_transport_errors"] == [], "R11_INVENTORY_DOWNLOAD_RETRY", "download retry drift")
        total += checked["bytes"]
        checked_receipts.append(checked)
    require(total == run_pool_download.HEAD_TOTAL_BYTES, "R11_INVENTORY_SOURCE_TREE", "source byte total drift")
    if verify_source_files:
        actual_source_files = [path for path in source.rglob("*") if path.is_file()]
        require(
            len(actual_source_files) == run_pool_download.ASSET_COUNT
            and {path.relative_to(source).as_posix() for path in actual_source_files} == {row["relative_path"] for row in rows},
            "R11_INVENTORY_SOURCE_TREE",
            "source tree identity/count drift",
        )
    result = _read_json(_repo_path(EXPECTED_BINDINGS["R11_DOWNLOAD_RESULT"]))
    require(
        result.get("schema") == "blindassist.taro.o1r.r11_fresh_pool_download_result.v1"
        and result.get("execution_valid") is True
        and result.get("passed") is True
        and result.get("terminal") == run_pool_download.PASS_TERMINAL
        and result.get("asset_count") == run_pool_download.ASSET_COUNT
        and result.get("source_bytes") == total
        and result.get("network_get_requests") == run_pool_download.ASSET_COUNT
        and result.get("recovered_asset_count") == 0
        and all(result.get(key) is False for key in ("archive_decode", "source_frame_decode", "model_execution", "faro_read", "truth_scoring", "training")),
        "R11_INVENTORY_DOWNLOAD_RESULT",
        "download result not admitted",
    )
    formal = _validate_content_seal(
        _read_json(_repo_path(DOWNLOAD_FORMAL_RESULT)), "R11_INVENTORY_DOWNLOAD_FORMAL_RESULT"
    )
    require(
        formal["content_sha256"] == DOWNLOAD_FORMAL_CONTENT_SHA256
        and formal.get("passed") is True
        and formal.get("status") == run_pool_download.PASS_TERMINAL
        and formal.get("source_integrity", {}).get("source_bytes") == total
        and formal.get("phase_firewall", {}).get("archive_decode") is False,
        "R11_INVENTORY_DOWNLOAD_FORMAL_RESULT",
        "formal download result drift",
    )
    return {
        "plan": plan,
        "rows": rows,
        "source_bytes": total,
        "manifest": manifest,
        "receipts": checked_receipts,
        "receipt_by_path": {row["relative_path"]: row for row in checked_receipts},
        "source_files_verified": verify_source_files,
    }


def _declared_member_index_sha256(value: Mapping[str, Any]) -> str:
    rows: list[dict[str, Any]] = []
    for role, members in sorted(value.items()):
        for token, binding in sorted(members.items()):
            rows.append(
                {
                    "role": role,
                    "timestamp_token": token,
                    "source_member_path": binding.source_member_path,
                    "canonical_member_path": binding.canonical_member_path,
                    "bytes": binding.bytes,
                    "declared_crc32": binding.declared_crc32,
                }
            )
    return adapter.canonical_sha256(rows)


def _declared_intrinsics_index_sha256(value: Mapping[str, Any]) -> str:
    return adapter.canonical_sha256(
        [
            {
                "role": binding.role,
                "timestamp_token": token,
                "source_member_path": binding.source_member_path,
                "canonical_member_path": binding.canonical_member_path,
                "bytes": binding.bytes,
                "declared_crc32": binding.declared_crc32,
            }
            for token, binding in sorted(value.items())
        ]
    )


def build_inventory(
    repo_root: Path,
    *,
    maximum_declared_materialized_bytes: int = MAXIMUM_MATERIALIZED_BYTES,
    deadline: float | None = None,
    monotonic_fn: Callable[[], float] = time.monotonic,
    progress_fn: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    root = repo_root.resolve()
    pool = fresh_pool.build_pool(root)
    source = materializer.safe_join(root, SOURCE_ROOT)
    parents: list[dict[str, Any]] = []
    total_declared_materialized = 0
    for index, parent in enumerate(pool["pool"], start=1):
        if deadline is not None:
            require(monotonic_fn() <= deadline, "R11_INVENTORY_TIMEOUT", "inventory wall budget exceeded")
        video = parent["video_id"]
        upsampling_path = materializer.safe_join(source, f"upsampling/Training/{video}.zip")
        intrinsics_path = materializer.safe_join(source, f"raw/Training/{video}/lowres_wide_intrinsics.zip")
        trajectory_path = materializer.safe_join(source, f"raw/Training/{video}/lowres_wide.traj")
        require(
            all(path.is_file() for path in (upsampling_path, intrinsics_path, trajectory_path)),
            "R11_INVENTORY_SOURCE_MISSING",
            f"source container missing: {video}",
        )
        remaining = maximum_declared_materialized_bytes - total_declared_materialized
        require(remaining > 0, "R11_INVENTORY_MATERIALIZED_BUDGET", "declared materialized byte budget exhausted")
        upsampling, upsampling_bytes = index_upsampling_archive_metadata_only(
            upsampling_path,
            video,
            maximum_declared_uncompressed_bytes=remaining,
        )
        remaining -= upsampling_bytes
        intrinsics, intrinsics_bytes = index_intrinsics_archive_metadata_only(
            intrinsics_path,
            video,
            maximum_declared_uncompressed_bytes=remaining,
        )
        trajectory_payload = trajectory_path.read_bytes()
        trajectory_bytes = len(trajectory_payload)
        declared_materialized_bytes = upsampling_bytes + intrinsics_bytes + trajectory_bytes
        total_declared_materialized += declared_materialized_bytes
        require(
            total_declared_materialized <= maximum_declared_materialized_bytes,
            "R11_INVENTORY_MATERIALIZED_BUDGET",
            "declared materialized byte budget exceeded",
        )
        trajectory = materializer.parse_trajectory_payload(trajectory_payload)
        frame_plan = materializer.exact_frame_plan(video, upsampling, intrinsics, trajectory)
        parents.append(
            {
                "visit_id": parent["visit_id"],
                "video_id": video,
                "official_fold": parent["official_fold"],
                "pool_rank_sha256": parent["pool_rank_sha256"],
                "container_bindings": {
                    "upsampling": {
                        "path": upsampling_path.relative_to(root).as_posix(),
                        "bytes": upsampling_path.stat().st_size,
                        "sha256": materializer.sha256_file(upsampling_path),
                        "declared_uncompressed_bytes": upsampling_bytes,
                        "recognized_member_index_sha256": _declared_member_index_sha256(upsampling),
                    },
                    "intrinsics": {
                        "path": intrinsics_path.relative_to(root).as_posix(),
                        "bytes": intrinsics_path.stat().st_size,
                        "sha256": materializer.sha256_file(intrinsics_path),
                        "declared_uncompressed_bytes": intrinsics_bytes,
                        "recognized_member_index_sha256": _declared_intrinsics_index_sha256(intrinsics),
                    },
                    "trajectory": {
                        "path": trajectory_path.relative_to(root).as_posix(),
                        "bytes": trajectory_bytes,
                        "sha256": materializer.sha256_bytes(trajectory_payload),
                        "row_count": len(trajectory),
                    },
                },
                "modality_member_counts": {role: len(members) for role, members in upsampling.items()},
                "intrinsics_member_count": len(intrinsics),
                "frame_plan": frame_plan,
                "declared_materialized_bytes": declared_materialized_bytes,
            }
        )
        if deadline is not None:
            require(monotonic_fn() <= deadline, "R11_INVENTORY_TIMEOUT", "inventory wall budget exceeded")
        if progress_fn is not None:
            progress_fn(
                {
                    "inventoried_parent": index,
                    "parent_count": len(pool["pool"]),
                    "video_id": video,
                    "exact_pose_bounded_frame_count": frame_plan["exact_pose_bounded_frame_count"],
                    "declared_materialized_bytes": declared_materialized_bytes,
                }
            )
    result = {
        "schema": "blindassist.taro.o1r.r11_fresh_pool_inventory.v1",
        "protocol_content_sha256": PROTOCOL_CONTENT_SHA256,
        "authorization_receipt_content_sha256": AUTHORIZATION_CONTENT_SHA256,
        "download_formal_result_content_sha256": DOWNLOAD_FORMAL_CONTENT_SHA256,
        "download_manifest_content_sha256": DOWNLOAD_MANIFEST_CONTENT_SHA256,
        "pool_content_sha256": pool["pool_content_sha256"],
        "request_plan_sha256": pool["request_plan"]["expanded_requests_sha256"],
        "parent_count": len(parents),
        "asset_count": run_pool_download.ASSET_COUNT,
        "compressed_source_bytes": run_pool_download.HEAD_TOTAL_BYTES,
        "exact_pose_bounded_frame_count": sum(
            row["frame_plan"]["exact_pose_bounded_frame_count"] for row in parents
        ),
        "declared_materialized_bytes": total_declared_materialized,
        "parents": parents,
        "inventory_policy": dict(EXPECTED_INVENTORY_POLICY),
        "read_accounting": {
            "zip_central_directory_metadata_read_operations": len(parents) * 2,
            "trajectory_payload_reads": len(parents),
            "zip_member_payload_reads": 0,
            "highres_depth_member_payload_reads": 0,
            "pixel_arrays_decoded": 0,
            "model_executions": 0,
            "network_requests": 0,
        },
        "zip_declared_crc_indexed": True,
        "zip_member_payload_crc_validated": False,
        "pixel_arrays_decoded": False,
        "source_frames_materialized": False,
        "faro_values_interpreted": False,
        "truth_values_interpreted": False,
        "model_outputs_read": False,
        "training": False,
    }
    result["content_sha256"] = adapter.canonical_sha256(result)
    return validate_inventory(result)


def validate_inventory(value: Mapping[str, Any]) -> dict[str, Any]:
    result = _validate_content_seal(value, "R11_INVENTORY_HASH_DRIFT")
    require(result.get("schema") == "blindassist.taro.o1r.r11_fresh_pool_inventory.v1", "R11_INVENTORY_SCHEMA", "inventory schema drift")
    parents = result.get("parents")
    parent_count = result.get("parent_count")
    require(
        isinstance(parent_count, int)
        and not isinstance(parent_count, bool)
        and isinstance(parents, list)
        and len(parents) == parent_count == 48,
        "R11_INVENTORY_PARENT_COUNT",
        "inventory parent count drift",
    )
    expected_roster = list(fresh_pool.EXPECTED_POOL)
    require(all(isinstance(row, dict) for row in parents), "R11_INVENTORY_PARENT_ROW", "inventory parent row invalid")
    observed_roster = [(row.get("visit_id"), row.get("video_id"), row.get("pool_rank_sha256")) for row in parents]
    require(observed_roster == expected_roster, "R11_INVENTORY_ROSTER_DRIFT", "inventory roster drift")
    counts: list[int] = []
    parent_bytes: list[int] = []
    for row in parents:
        require(
            set(row)
            == {
                "visit_id",
                "video_id",
                "official_fold",
                "pool_rank_sha256",
                "container_bindings",
                "modality_member_counts",
                "intrinsics_member_count",
                "frame_plan",
                "declared_materialized_bytes",
            }
            and row.get("official_fold") == "Training",
            "R11_INVENTORY_PARENT_ROW",
            "inventory parent fields/fold drift",
        )
        video = row["video_id"]
        bindings = row.get("container_bindings")
        require(
            isinstance(bindings, dict) and set(bindings) == {"upsampling", "intrinsics", "trajectory"},
            "R11_INVENTORY_CONTAINER_BINDING",
            "container binding roles drift",
        )
        expected_paths = {
            "upsampling": f"{SOURCE_ROOT}/upsampling/Training/{video}.zip",
            "intrinsics": f"{SOURCE_ROOT}/raw/Training/{video}/lowres_wide_intrinsics.zip",
            "trajectory": f"{SOURCE_ROOT}/raw/Training/{video}/lowres_wide.traj",
        }
        for role in ("upsampling", "intrinsics"):
            binding = bindings[role]
            require(
                isinstance(binding, dict)
                and set(binding)
                == {
                    "path",
                    "bytes",
                    "sha256",
                    "declared_uncompressed_bytes",
                    "recognized_member_index_sha256",
                }
                and binding.get("path") == expected_paths[role]
                and isinstance(binding.get("bytes"), int)
                and not isinstance(binding.get("bytes"), bool)
                and binding["bytes"] > 0
                and isinstance(binding.get("declared_uncompressed_bytes"), int)
                and not isinstance(binding.get("declared_uncompressed_bytes"), bool)
                and binding["declared_uncompressed_bytes"] > 0
                and isinstance(binding.get("sha256"), str)
                and re.fullmatch(r"[0-9A-F]{64}", binding["sha256"]) is not None
                and isinstance(binding.get("recognized_member_index_sha256"), str)
                and re.fullmatch(r"[0-9A-F]{64}", binding["recognized_member_index_sha256"])
                is not None,
                "R11_INVENTORY_CONTAINER_BINDING",
                f"{role} container binding drift",
            )
        trajectory_binding = bindings["trajectory"]
        require(
            isinstance(trajectory_binding, dict)
            and set(trajectory_binding) == {"path", "bytes", "sha256", "row_count"}
            and trajectory_binding.get("path") == expected_paths["trajectory"]
            and isinstance(trajectory_binding.get("bytes"), int)
            and not isinstance(trajectory_binding.get("bytes"), bool)
            and trajectory_binding["bytes"] > 0
            and isinstance(trajectory_binding.get("row_count"), int)
            and not isinstance(trajectory_binding.get("row_count"), bool)
            and trajectory_binding["row_count"] >= 2
            and isinstance(trajectory_binding.get("sha256"), str)
            and re.fullmatch(r"[0-9A-F]{64}", trajectory_binding["sha256"]) is not None,
            "R11_INVENTORY_CONTAINER_BINDING",
            "trajectory container binding drift",
        )
        frame_plan = row.get("frame_plan")
        require(isinstance(frame_plan, dict), "R11_INVENTORY_FRAME_PLAN", "frame plan missing")
        count = frame_plan.get("exact_pose_bounded_frame_count")
        tokens = frame_plan.get("exact_timestamp_tokens")
        source_common_count = frame_plan.get("source_common_frame_count")
        exact_intrinsics_count = frame_plan.get("exact_intrinsics_common_frame_count")
        pose_rejected = frame_plan.get("pose_rejected")
        require(
            set(frame_plan)
            == {
                "video_id",
                "source_common_frame_count",
                "exact_intrinsics_common_frame_count",
                "exact_pose_bounded_frame_count",
                "exact_timestamp_tokens",
                "pose_rejected",
                "selection_rule",
                "truth_or_model_fields_read",
            }
            and frame_plan.get("video_id") == video
            and isinstance(source_common_count, int)
            and not isinstance(source_common_count, bool)
            and source_common_count >= 0
            and isinstance(exact_intrinsics_count, int)
            and not isinstance(exact_intrinsics_count, bool)
            and source_common_count >= exact_intrinsics_count >= 0
            and isinstance(count, int)
            and not isinstance(count, bool)
            and exact_intrinsics_count >= count >= 0
            and isinstance(tokens, list)
            and all(isinstance(token, str) for token in tokens)
            and len(tokens) == len(set(tokens)) == count
            and isinstance(pose_rejected, list)
            and len(pose_rejected) == exact_intrinsics_count - count
            and frame_plan.get("truth_or_model_fields_read") is False
            and frame_plan.get("selection_rule") == "ALL_SAME_STEM_MODALITIES_EXACT_INTRINSICS_WITH_BOUNDED_POSE_SORTED_BY_EXACT_NS_THEN_TOKEN",
            "R11_INVENTORY_FRAME_COUNT_DRIFT",
            "frame plan count/token/scope drift",
        )
        try:
            token_keys = [(adapter.decimal_timestamp_ns(token), token) for token in tokens]
        except adapter.AdapterError as error:
            raise PoolInventoryError("R11_INVENTORY_FRAME_TOKEN", str(error)) from error
        require(
            token_keys == sorted(token_keys) and len({key[0] for key in token_keys}) == count,
            "R11_INVENTORY_FRAME_TOKEN",
            "admitted frame tokens are not exact-ns unique and sorted",
        )
        rejected_tokens: list[str] = []
        for rejection in pose_rejected:
            require(
                isinstance(rejection, dict)
                and set(rejection) == {"timestamp_token", "reason_code"}
                and isinstance(rejection.get("timestamp_token"), str)
                and isinstance(rejection.get("reason_code"), str)
                and bool(re.fullmatch(r"[A-Z0-9_]+", rejection["reason_code"])),
                "R11_INVENTORY_POSE_REJECTION",
                "pose rejection row drift",
            )
            try:
                adapter.decimal_timestamp_ns(rejection["timestamp_token"])
            except adapter.AdapterError as error:
                raise PoolInventoryError("R11_INVENTORY_POSE_REJECTION", str(error)) from error
            rejected_tokens.append(rejection["timestamp_token"])
        require(
            len(rejected_tokens) == len(set(rejected_tokens))
            and not (set(rejected_tokens) & set(tokens)),
            "R11_INVENTORY_POSE_REJECTION",
            "pose rejection token duplicates/admitted overlap",
        )
        modality_counts = row.get("modality_member_counts")
        require(
            isinstance(modality_counts, dict)
            and set(modality_counts) == {"color", "highres_depth", "lowres_depth", "confidence"}
            and all(
                isinstance(value, int)
                and not isinstance(value, bool)
                and value >= source_common_count
                for value in modality_counts.values()
            )
            and isinstance(row.get("intrinsics_member_count"), int)
            and not isinstance(row.get("intrinsics_member_count"), bool)
            and row["intrinsics_member_count"] >= exact_intrinsics_count,
            "R11_INVENTORY_MODALITY_COUNTS",
            "modality/intrinsics member count drift",
        )
        declared_materialized_bytes = row.get("declared_materialized_bytes")
        bound_declared_materialized_bytes = (
            bindings["upsampling"]["declared_uncompressed_bytes"]
            + bindings["intrinsics"]["declared_uncompressed_bytes"]
            + bindings["trajectory"]["bytes"]
        )
        require(
            isinstance(declared_materialized_bytes, int)
            and not isinstance(declared_materialized_bytes, bool)
            and declared_materialized_bytes > 0
            and declared_materialized_bytes == bound_declared_materialized_bytes,
            "R11_INVENTORY_BYTE_DRIFT",
            "parent declared materialized bytes invalid",
        )
        counts.append(count)
        parent_bytes.append(declared_materialized_bytes)
    exact_frame_count = result.get("exact_pose_bounded_frame_count")
    declared_materialized_bytes = result.get("declared_materialized_bytes")
    require(
        isinstance(exact_frame_count, int)
        and not isinstance(exact_frame_count, bool)
        and exact_frame_count >= 0
        and sum(counts) == exact_frame_count,
        "R11_INVENTORY_FRAME_COUNT_DRIFT",
        "inventory frame count sum drift",
    )
    require(
        isinstance(declared_materialized_bytes, int)
        and not isinstance(declared_materialized_bytes, bool)
        and 0 < declared_materialized_bytes <= MAXIMUM_MATERIALIZED_BYTES
        and sum(parent_bytes) == declared_materialized_bytes,
        "R11_INVENTORY_BYTE_DRIFT",
        "inventory declared materialized byte sum drift",
    )
    read_accounting = result.get("read_accounting")
    require(
        result.get("protocol_content_sha256") == PROTOCOL_CONTENT_SHA256
        and result.get("authorization_receipt_content_sha256") == AUTHORIZATION_CONTENT_SHA256
        and result.get("download_formal_result_content_sha256") == DOWNLOAD_FORMAL_CONTENT_SHA256
        and result.get("download_manifest_content_sha256") == DOWNLOAD_MANIFEST_CONTENT_SHA256
        and result.get("pool_content_sha256") == POOL_CONTENT_SHA256
        and result.get("request_plan_sha256") == REQUEST_PLAN_SHA256
        and result.get("asset_count") == run_pool_download.ASSET_COUNT
        and result.get("compressed_source_bytes") == run_pool_download.HEAD_TOTAL_BYTES
        and result.get("inventory_policy") == EXPECTED_INVENTORY_POLICY
        and isinstance(read_accounting, dict)
        and read_accounting
        == {
            "zip_central_directory_metadata_read_operations": 48 * 2,
            "trajectory_payload_reads": 48,
            "zip_member_payload_reads": 0,
            "highres_depth_member_payload_reads": 0,
            "pixel_arrays_decoded": 0,
            "model_executions": 0,
            "network_requests": 0,
        }
        and result.get("zip_declared_crc_indexed") is True
        and result.get("zip_member_payload_crc_validated") is False
        and result.get("pixel_arrays_decoded") is False
        and result.get("source_frames_materialized") is False
        and result.get("faro_values_interpreted") is False
        and result.get("truth_values_interpreted") is False
        and result.get("model_outputs_read") is False
        and result.get("training") is False,
        "R11_INVENTORY_SCOPE_DRIFT",
        "inventory scope/request drift",
    )
    return result


def validate_execution_lock(path: Path, *, require_output_absent: bool = True) -> dict[str, Any]:
    lock_path = path.resolve()
    require(lock_path == _repo_path(LOCK_RELATIVE), "R11_INVENTORY_LOCK_PATH", "inventory lock path drift")
    lock = _validate_content_seal(_read_json(lock_path), "R11_INVENTORY_LOCK_HASH")
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R11_INVENTORY_LOCK_IDENTITY", "inventory lock identity drift")
    require(lock.get("argv") == EXPECTED_ARGV and lock.get("source_root") == SOURCE_ROOT and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R11_INVENTORY_LOCK_POLICY", "inventory argv/root policy drift")
    implementation_commit = _validate_implementation_ancestor(lock.get("implementation_commit"))
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R11_INVENTORY_BINDINGS", "inventory binding count drift")
    verified: dict[str, dict[str, Any]] = {}
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in verified and EXPECTED_BINDINGS.get(role) == relative, "R11_INVENTORY_BINDING_ROW", "inventory binding row drift")
        target = _repo_path(relative)
        payload = target.read_bytes() if target.is_file() else b""
        require(len(payload) == row.get("bytes") and materializer.sha256_bytes(payload) == row.get("sha256"), "R11_INVENTORY_BINDING_HASH", f"inventory binding drift: {relative}")
        if role not in ARTIFACT_BINDING_ROLES:
            require(payload == _git_bytes(implementation_commit, relative), "R11_INVENTORY_BINDING_HASH", f"implementation-commit binding drift: {relative}")
        verified[role] = dict(row)
    download = verify_download_evidence(verify_source_files=False)
    run_pool_head.validate_authorization_receipt(_read_json(_repo_path(run_pool_head.AUTHORIZATION_RELATIVE)))
    require(lock.get("request_plan_sha256") == download["plan"]["request_plan"]["expanded_requests_sha256"], "R11_INVENTORY_PLAN_DRIFT", "request plan drift")
    require(
        lock.get("protocol_content_sha256") == PROTOCOL_CONTENT_SHA256
        and lock.get("authorization_receipt_content_sha256") == AUTHORIZATION_CONTENT_SHA256
        and lock.get("pool_content_sha256") == POOL_CONTENT_SHA256
        and lock.get("download_manifest_content_sha256") == DOWNLOAD_MANIFEST_CONTENT_SHA256
        and lock.get("download_formal_result_content_sha256") == DOWNLOAD_FORMAL_CONTENT_SHA256,
        "R11_INVENTORY_DOWNLOAD_FORMAL_RESULT",
        "protocol/pool/download binding drift",
    )
    require(
        lock.get("execution_authority") == EXPECTED_AUTHORITY
        and lock.get("inventory_policy") == EXPECTED_INVENTORY_POLICY
        and lock.get("resource_budget") == EXPECTED_RESOURCE_BUDGET,
        "R11_INVENTORY_AUTHORITY_BUDGET",
        "inventory authority/policy/budget drift",
    )
    user = lock.get("user_authority")
    require(isinstance(user, dict) and user.get("confirmed_by") == "user" and user.get("confirmed_at") == "2026-08-12" and user.get("confirmation_verbatim") == "授权" and user.get("scope") == run_pool_head.EXPECTED_USER_SCOPE, "R11_INVENTORY_USER_AUTHORITY", "inventory user authority drift")
    require(_repo_path(SOURCE_ROOT).is_dir(), "R11_INVENTORY_SOURCE_MISSING", "source root missing")
    if require_output_absent:
        require(not _repo_path(OUTPUT_ROOT).exists(), "R11_INVENTORY_ROOT_COLLISION", "inventory output root exists")
    lock["_lock_path"] = lock_path
    lock["_verified_bindings"] = verified
    lock["_download"] = download
    return lock


def _validate_inventory_download_binding(
    inventory: Mapping[str, Any], download: Mapping[str, Any]
) -> None:
    require(download.get("source_files_verified") is True, "R11_INVENTORY_SOURCE_TREE", "source files were not verified after execution reservation")
    receipts = download.get("receipt_by_path")
    require(isinstance(receipts, dict) and len(receipts) == run_pool_download.ASSET_COUNT, "R11_INVENTORY_DOWNLOAD_RECEIPTS", "download receipt lookup drift")
    observed_paths: set[str] = set()
    for parent in inventory["parents"]:
        for binding in parent["container_bindings"].values():
            path = binding["path"]
            receipt = receipts.get(path.removeprefix(f"{SOURCE_ROOT}/"))
            require(
                isinstance(receipt, dict)
                and binding["bytes"] == receipt.get("bytes")
                and binding["sha256"] == receipt.get("sha256"),
                "R11_INVENTORY_CONTAINER_DOWNLOAD_BINDING",
                f"inventory container does not match sealed download receipt: {path}",
            )
            observed_paths.add(path.removeprefix(f"{SOURCE_ROOT}/"))
    require(
        observed_paths == set(receipts),
        "R11_INVENTORY_CONTAINER_DOWNLOAD_BINDING",
        "inventory/download container path set drift",
    )


def _write_failure(output: Path, error: BaseException, files: dict[str, dict[str, Any]]) -> None:
    files["failure.json"] = _write_exclusive(
        output / "failure.json",
        _sealed_record(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_inventory_failure.v1",
                "execution_valid": False,
                "terminal": INVALID_TERMINAL,
                "failure_code": str(getattr(error, "code", type(error).__name__)),
                "message": str(error),
                "one_shot_consumed": True,
            }
        ),
    )
    _write_exclusive(
        output / "manifest.json",
        _sealed_record(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_inventory_failure_manifest.v1",
                "terminal": INVALID_TERMINAL,
                "files": files,
                "one_shot_consumed": True,
            }
        ),
    )


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    output = _repo_path(OUTPUT_ROOT)
    output.mkdir(parents=True, exist_ok=False)
    files: dict[str, dict[str, Any]] = {}
    deadline = time.monotonic() + lock["resource_budget"]["maximum_wall_seconds"]
    try:
        files["start-receipt.json"] = _write_exclusive(
            output / "start-receipt.json",
            _sealed_record(
                {
                    "schema": "blindassist.taro.o1r.r11_fresh_pool_inventory_start.v1",
                    "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]),
                    "download_formal_result_content_sha256": DOWNLOAD_FORMAL_CONTENT_SHA256,
                    "one_shot_consumed_on_root_creation": True,
                }
            ),
        )
        download = verify_download_evidence(verify_source_files=True)
        inventory = build_inventory(
            REPO_ROOT,
            maximum_declared_materialized_bytes=lock["resource_budget"]["maximum_declared_materialized_bytes"],
            deadline=deadline,
            progress_fn=lambda value: print(json.dumps(value, sort_keys=True), flush=True),
        )
        validate_inventory(inventory)
        _validate_inventory_download_binding(inventory, download)
        files["exact-frame-plan.json"] = _write_exclusive(output / "exact-frame-plan.json", inventory)
        per_parent_frame_counts = [
            row["frame_plan"]["exact_pose_bounded_frame_count"] for row in inventory["parents"]
        ]
        ready = all(count > 0 for count in per_parent_frame_counts)
        terminal = PASS_TERMINAL if ready else NOT_READY_TERMINAL
        result = _sealed_record(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_inventory_result.v1",
                "execution_valid": True,
                "terminal": terminal,
                "passed": ready,
                "phase_a_ready": ready,
                "parent_replacement_attempted": False,
                "parent_count": 48,
                "asset_count": run_pool_download.ASSET_COUNT,
                "compressed_source_bytes": run_pool_download.HEAD_TOTAL_BYTES,
                "exact_pose_bounded_frame_count": inventory["exact_pose_bounded_frame_count"],
                "per_parent_frame_counts": per_parent_frame_counts,
                "per_parent_declared_materialized_bytes": [
                    row["declared_materialized_bytes"] for row in inventory["parents"]
                ],
                "declared_materialized_bytes": inventory["declared_materialized_bytes"],
                "inventory_content_sha256": inventory["content_sha256"],
                "source_container_integrity_verified_after_root_creation": True,
                "zip_declared_crc_indexed": True,
                "zip_member_payload_crc_validated": False,
                "zip_member_payload_reads": 0,
                "highres_depth_member_payload_reads": 0,
                "pixel_arrays_decoded": False,
                "source_frames_materialized": False,
                "faro_values_interpreted": False,
                "truth_values_interpreted": False,
                "model_outputs_read": False,
                "training": False,
                "one_shot_consumed": True,
            }
        )
        files["result.json"] = _write_exclusive(output / "result.json", result)
        manifest = _sealed_record(
            {
                "schema": "blindassist.taro.o1r.r11_fresh_pool_inventory_manifest.v1",
                "terminal": terminal,
                "files": files,
                "one_shot_consumed": True,
            }
        )
        evidence_bytes = sum(row["bytes"] for row in files.values()) + len(_canonical_json_line(manifest))
        require(evidence_bytes <= lock["resource_budget"]["maximum_evidence_bytes"], "R11_INVENTORY_EVIDENCE_BUDGET", "inventory evidence budget exceeded")
        require(time.monotonic() <= deadline, "R11_INVENTORY_TIMEOUT", "inventory wall budget exceeded before success seal")
        _write_exclusive(output / "manifest.json", manifest)
        return result
    except Exception as error:
        try:
            _write_failure(output, error, files)
        except Exception:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"execution_valid": False, "terminal": INVALID_TERMINAL, "error_code": getattr(error, "code", type(error).__name__), "message": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
