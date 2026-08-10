#!/usr/bin/env python3
"""Run a future hash-bound TARO O0R source/truth-only one-shot execution."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import psutil

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime.materializer import (
    EXECUTION_LOCK_SCHEMA,
    MATERIALIZER_MANIFEST_SCHEMA,
    TRUTH_RESULT_SCHEMA,
    AtomicEvidenceWriter,
    MaterializerError,
    MemberBinding,
    adapter_fit_source_frame,
    build_eval_truth_record,
    canonical_sha256,
    decode_source_frame,
    download_bound_asset,
    evaluate_truth_only_gates,
    exact_frame_plan,
    expanded_asset_plan,
    index_intrinsics_archive,
    index_upsampling_archive,
    load_json,
    parse_trajectory_payload,
    require,
    safe_join,
    sha256_file,
    uncertainty_model_receipt,
    uncertainty_model_artifact,
    validate_authorization,
    validate_head_receipt,
    verify_bound_container,
    zip_uncompressed_bytes,
)


EXPECTED_TRUTH_BINDINGS = {
    "TRUTH_RECOVERY_R2_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_RECOVERY_R2_LOCK_2026-08-10.json",
    "TRUTH_ONLY_PREFLIGHT_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK_2026-08-10.json",
    "DATA_USE_AUTHORIZATION": "docs/research/taro/TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_R1_RECEIPT_2026-08-10.json",
    "MATERIALIZER_IMPLEMENTATION_LOCK": "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_MATERIALIZER_R2_IMPLEMENTATION_LOCK_2026-08-10.json",
    "SOURCE_ADAPTER": "scripts/research/taro_o0r_source_adapter_runtime/source_adapter.py",
    "MATERIALIZER": "scripts/research/taro_o0r_truth_materializer_runtime/materializer.py",
    "TRUTH_RUNNER": "scripts/research/taro_o0r_truth_materializer_runtime/run_truth_only.py",
    "HEAD_RECEIPT": "artifacts.local/evidence/taro/o0r-arkitscenes-head-r1/head-receipt.json",
    "SOURCE_CACHE_DOWNLOAD_RECEIPTS": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r1/download-receipts.json.gz",
}
EXPECTED_ROOTS = {
    "SOURCE": "artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r2",
    "WORK": "artifacts.local/work/taro/o0r-arkitscenes-source-adapter-r2",
    "TRUTH_EVIDENCE": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r2",
    "O0R_EVIDENCE_SEALED": "artifacts.local/evidence/taro/o0r-arkitscenes-factor-headroom-r2",
}
EXPECTED_SOURCE_CACHE = {
    "mode": "VERIFIED_HARDLINK_REUSE_NO_NETWORK",
    "source_root": "artifacts.local/datasets/taro/o0r-arkitscenes-source-adapter-r1",
    "download_receipts_role": "SOURCE_CACHE_DOWNLOAD_RECEIPTS",
    "network_requests_allowed": 0,
}
EXPECTED_AUTHORITY = {
    "truth_only_execution": True,
    "head_or_head_receipt_mutation": False,
    "source_download": False,
    "source_cache_reuse": True,
    "truth_materialization": True,
    "depthart_inference": False,
    "factorial_execution": False,
    "training": False,
    "device": False,
    "product": False,
    "safety": False,
}


@dataclass(frozen=True)
class PreparedParent:
    parent: dict[str, str]
    upsampling_archive: Path
    intrinsics_archive: Path
    trajectory_path: Path
    upsampling_inventory: dict[str, dict[str, MemberBinding]]
    intrinsics_inventory: dict[str, MemberBinding]
    trajectory_rows: list[dict[str, Any]]
    container_receipts: dict[str, dict[str, Any]]
    frame_plan: dict[str, Any]
    materialized_bytes: int


def _repo_path(repo_root: Path, relative: str) -> Path:
    return safe_join(repo_root, relative)


def _load_cached_download_receipts(path: Path) -> dict[str, dict[str, Any]]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            payload = json.load(stream)
    except Exception as error:
        raise MaterializerError("SOURCE_CACHE_RECEIPT_INVALID", "source cache receipt cannot be decoded") from error
    require(isinstance(payload, list) and len(payload) == 72, "SOURCE_CACHE_RECEIPT_INVALID", "source cache must bind exactly 72 download receipts")
    lookup: dict[str, dict[str, Any]] = {}
    for receipt in payload:
        require(isinstance(receipt, dict) and isinstance(receipt.get("url"), str), "SOURCE_CACHE_RECEIPT_INVALID", "source cache receipt row is invalid")
        require(receipt["url"] not in lookup, "SOURCE_CACHE_RECEIPT_INVALID", "source cache receipt URL is duplicated", url=receipt["url"])
        lookup[receipt["url"]] = dict(receipt)
    return lookup


def _reuse_cached_asset(
    row: Mapping[str, str],
    head_row: Mapping[str, Any],
    *,
    source_root: Path,
    cache_root: Path,
    cache_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    receipt = dict(cache_lookup.get(str(row["url"]), {}))
    require(
        receipt.get("asset") == row["asset"]
        and receipt.get("relative_path") == row["relative_path"]
        and receipt.get("url") == row["url"]
        and receipt.get("bytes") == head_row.get("content_length_bytes")
        and receipt.get("head_content_length_bytes") == head_row.get("content_length_bytes")
        and receipt.get("head_etag") == head_row.get("etag")
        and receipt.get("head_last_modified") == head_row.get("last_modified"),
        "SOURCE_CACHE_IDENTITY_DRIFT",
        "cached source receipt differs from the bound R1 HEAD identity",
        url=row["url"],
    )
    cached = safe_join(cache_root, row["relative_path"])
    verify_bound_container(cached, receipt)
    destination = safe_join(source_root, row["relative_path"])
    require(not destination.exists(), "SOURCE_CACHE_DESTINATION_COLLISION", "R2 source destination already exists", path=str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(cached, destination)
    except OSError as error:
        raise MaterializerError("SOURCE_CACHE_HARDLINK_FAILED", "verified R1 source cache could not be hard-linked into R2", source=str(cached), destination=str(destination)) from error
    verify_bound_container(destination, receipt)
    return receipt


def validate_execution_lock(path: Path, repo_root: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    lock_path = path.resolve()
    lock = load_json(lock_path)
    require(lock.get("schema") == EXECUTION_LOCK_SCHEMA, "EXECUTION_LOCK_SCHEMA_DRIFT", "truth-only execution lock schema drift")
    require(lock.get("lock_id") == "TARO_O0R_ARKITSCENES_TRUTH_ONLY_R2_EXECUTION_LOCK", "EXECUTION_LOCK_IDENTITY_DRIFT", "truth-only execution lock id drift")
    require(lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "ONE_SHOT_ALREADY_CONSUMED", "truth-only execution is not authorized and unconsumed")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "EXECUTION_AUTHORITY_OVERCLAIM", "truth-only execution authority drift")
    require(lock.get("roots") == EXPECTED_ROOTS, "EXECUTION_ROOT_DRIFT", "truth-only execution roots drift")
    require(lock.get("source_cache") == EXPECTED_SOURCE_CACHE, "SOURCE_CACHE_LOCK_DRIFT", "R2 source-cache binding drift")
    require(lock.get("overwrite") is False and lock.get("rerun") is False, "ONE_SHOT_POLICY_DRIFT", "truth-only execution must forbid overwrite/rerun")
    required_environment = lock.get("required_environment")
    require(isinstance(required_environment, dict), "EXECUTION_ENVIRONMENT_MISSING", "required environment missing")
    for key, expected in required_environment.items():
        require(os.environ.get(key) == str(expected), "EXECUTION_ENVIRONMENT_DRIFT", "required environment mismatch", key=key)
    expected_argv = lock.get("argv")
    actual_argv = [
        Path(sys.argv[0]).resolve().relative_to(root).as_posix(),
        "--execution-lock",
        lock_path.relative_to(root).as_posix(),
    ]
    require(expected_argv == actual_argv, "EXECUTION_ARGV_DRIFT", "truth-only argv drift", actual=actual_argv)
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_TRUTH_BINDINGS), "EXECUTION_BINDINGS_MISSING", "truth-only binding cardinality drift")
    seen: set[str] = set()
    verified: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        require(isinstance(binding, dict) and set(binding) == {"role", "path", "bytes", "sha256"}, "EXECUTION_BINDING_FIELDS", "execution binding fields drift")
        role = str(binding["role"])
        relative = str(binding["path"])
        require(role not in seen and EXPECTED_TRUTH_BINDINGS.get(role) == relative, "EXECUTION_BINDING_PATH", "execution binding role/path drift", role=role)
        seen.add(role)
        bound_path = _repo_path(root, relative)
        require(bound_path.is_file(), "BOUND_FILE_MISSING", "execution binding missing", path=relative)
        require(bound_path.stat().st_size == binding["bytes"] and sha256_file(bound_path) == binding["sha256"], "BOUND_HASH_DRIFT", "execution binding drift", path=relative)
        verified[role] = dict(binding)
    require(seen == set(EXPECTED_TRUTH_BINDINGS), "EXECUTION_BINDING_ROLE_SET", "execution binding role set drift")
    lock["_verified_bindings"] = verified
    return lock


def _decode_prepared_frame(
    prepared: PreparedParent,
    timestamp_token: str,
    decode_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    return decode_fn(
        parent=prepared.parent,
        timestamp_token=timestamp_token,
        upsampling_archive=prepared.upsampling_archive,
        intrinsics_archive=prepared.intrinsics_archive,
        trajectory_path=prepared.trajectory_path,
        upsampling_inventory=prepared.upsampling_inventory,
        intrinsics_inventory=prepared.intrinsics_inventory,
        trajectory_rows=prepared.trajectory_rows,
        container_receipts=prepared.container_receipts,
    )


class _FitFrameSequence(Sequence[dict[str, Any]]):
    def __init__(
        self,
        entries: Sequence[tuple[PreparedParent, str]],
        *,
        decode_fn: Callable[..., dict[str, Any]],
        on_decode: Callable[[PreparedParent, str, dict[str, Any]], None],
        resource_guard: Callable[[], None],
    ) -> None:
        self._entries = list(entries)
        self._decode_fn = decode_fn
        self._on_decode = on_decode
        self._resource_guard = resource_guard
        self.decoded_indices: list[int] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[item] for item in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        require(index not in self.decoded_indices, "ADAPTER_FIT_FRAME_REUSED", "adapter-fit frame decoded more than once", index=index)
        self._resource_guard()
        prepared, token = self._entries[index]
        frame = _decode_prepared_frame(prepared, token, self._decode_fn)
        self._on_decode(prepared, token, frame)
        self.decoded_indices.append(index)
        return adapter_fit_source_frame(frame)


def _prepare_downloaded_parents(
    preflight: Mapping[str, Any],
    plan_rows: Sequence[Mapping[str, str]],
    download_receipts: Sequence[Mapping[str, Any]],
    *,
    source_root: Path,
) -> list[PreparedParent]:
    receipt_lookup = {(str(row["video_id"]), str(row["asset"])): dict(receipt) for row, receipt in zip(plan_rows, download_receipts, strict=True)}
    row_lookup = {(str(row["video_id"]), str(row["asset"])): dict(row) for row in plan_rows}
    maximum_materialized = int(preflight["resource_budget"]["maximum_materialized_source_bytes"])
    materialized_total = 0
    output: list[PreparedParent] = []
    for raw_parent in preflight["asset_plan"]["selected_parents"]:
        parent = {key: str(raw_parent[key]) for key in ("role", "visit_id", "video_id", "official_fold")}
        video_id = parent["video_id"]
        rows = {asset: row_lookup[(video_id, asset)] for asset in ("upsampling.zip", "lowres_wide_intrinsics.zip", "lowres_wide.traj")}
        receipts = {asset: receipt_lookup[(video_id, asset)] for asset in rows}
        upsampling_path = _repo_path(source_root, rows["upsampling.zip"]["relative_path"])
        intrinsics_path = _repo_path(source_root, rows["lowres_wide_intrinsics.zip"]["relative_path"])
        trajectory_path = _repo_path(source_root, rows["lowres_wide.traj"]["relative_path"])
        upsampling_bytes = zip_uncompressed_bytes(upsampling_path, maximum_materialized - materialized_total)
        materialized_total += upsampling_bytes
        intrinsics_bytes = zip_uncompressed_bytes(intrinsics_path, maximum_materialized - materialized_total)
        materialized_total += intrinsics_bytes
        trajectory_bytes = trajectory_path.stat().st_size
        materialized_total += trajectory_bytes
        require(materialized_total <= maximum_materialized, "MATERIALIZED_SOURCE_BUDGET_EXCEEDED", "materialized source exceeds frozen budget")
        upsampling = index_upsampling_archive(upsampling_path, video_id, maximum_uncompressed_bytes=upsampling_bytes)
        intrinsics = index_intrinsics_archive(intrinsics_path, video_id, maximum_uncompressed_bytes=intrinsics_bytes)
        trajectory_rows = parse_trajectory_payload(trajectory_path.read_bytes())
        frame_plan = exact_frame_plan(video_id, upsampling, intrinsics, trajectory_rows)
        output.append(
            PreparedParent(
                parent=parent,
                upsampling_archive=upsampling_path,
                intrinsics_archive=intrinsics_path,
                trajectory_path=trajectory_path,
                upsampling_inventory=upsampling,
                intrinsics_inventory=intrinsics,
                trajectory_rows=trajectory_rows,
                container_receipts=receipts,
                frame_plan=frame_plan,
                materialized_bytes=upsampling_bytes + intrinsics_bytes + trajectory_bytes,
            )
        )
    require(len(output) == 24 and materialized_total <= maximum_materialized, "PARENT_PLAN_CARDINALITY", "exact 24-parent materialization plan required")
    return output


def _validate_download_receipts(
    plan_rows: Sequence[Mapping[str, str]],
    receipts: Sequence[Mapping[str, Any]],
    head_lookup: Mapping[str, Mapping[str, Any]],
    source_root: Path,
) -> None:
    require(len(receipts) == len(plan_rows) == 72, "DOWNLOAD_RECEIPT_CARDINALITY", "exactly 72 download receipts are required")
    for row, receipt in zip(plan_rows, receipts, strict=True):
        require(
            receipt.get("asset") == row["asset"]
            and receipt.get("url") == row["url"]
            and receipt.get("relative_path") == row["relative_path"]
            and receipt.get("bytes") == head_lookup[row["url"]]["content_length_bytes"]
            and receipt.get("head_content_length_bytes") == head_lookup[row["url"]]["content_length_bytes"]
            and receipt.get("head_etag") == head_lookup[row["url"]].get("etag")
            and receipt.get("head_last_modified") == head_lookup[row["url"]].get("last_modified")
            and receipt.get("redirect_chain") == []
            and receipt.get("attempt_count") == 1,
            "DOWNLOAD_RECEIPT_IDENTITY_DRIFT",
            "download receipt does not reproduce the bound plan/HEAD identity",
            url=row["url"],
        )
        destination = safe_join(source_root, row["relative_path"])
        verify_bound_container(destination, receipt)


def materialize_prepared_parents(
    prepared_parents: Sequence[PreparedParent],
    *,
    writer: AtomicEvidenceWriter,
    started: float,
    wall_seconds: float,
    peak_rss_bytes: int,
    decode_fn: Callable[..., dict[str, Any]] = decode_source_frame,
    fit_fn: Callable[[Sequence[dict[str, Any]]], Any] = adapter.fit_uncertainty_model,
    geometry_fn: Callable[..., Any] = adapter.derive_faro_geometry,
    eval_fn: Callable[..., dict[str, Any]] = build_eval_truth_record,
) -> tuple[dict[str, Any], dict[str, Any]]:
    process = psutil.Process()

    def resource_guard() -> None:
        require(time.monotonic() - started <= wall_seconds, "TRUTH_MATERIALIZATION_TIMEOUT", "truth-only wall-time budget exceeded")
        require(process.memory_info().rss <= peak_rss_bytes, "TRUTH_MATERIALIZATION_RSS_EXCEEDED", "truth-only RSS budget exceeded")

    fit_entries = [
        (parent, token)
        for parent in prepared_parents
        if parent.parent["role"] == "ADAPTER_FIT"
        for token in parent.frame_plan["exact_timestamp_tokens"]
    ]
    eval_decode_count = 0

    def record_fit_decode(prepared: PreparedParent, token: str, frame: dict[str, Any]) -> None:
        writer.write_json_gzip(
            f"source-frames/adapter-fit/{prepared.parent['visit_id']}/{prepared.parent['video_id']}/{token}.json.gz",
            {
                "source_frame_receipt": frame["source_frame_receipt"],
                "bound_source_frame_envelope": frame["bound_source_frame_envelope"],
                "model_outputs_absent": True,
            },
        )

    fit_sequence = _FitFrameSequence(
        fit_entries,
        decode_fn=decode_fn,
        on_decode=record_fit_decode,
        resource_guard=resource_guard,
    )
    require(len(fit_sequence) > 0 and eval_decode_count == 0, "ADAPTER_FIT_PHASE_INVALID", "adapter-fit phase requires frames and zero eval decodes")
    uncertainty_model = fit_fn(fit_sequence)
    require(fit_sequence.decoded_indices == list(range(len(fit_sequence))) and eval_decode_count == 0, "ADAPTER_FIT_PHASE_INCOMPLETE", "all fit frames must decode exactly once before eval")
    model_receipt = uncertainty_model_receipt(uncertainty_model)
    writer.write_json("uncertainty-model-receipt.json", model_receipt)
    writer.write_content_addressed_artifact(
        "uncertainty-model-artifact.json.gz",
        uncertainty_model_artifact(uncertainty_model),
    )

    parent_summaries: list[dict[str, Any]] = []
    frame_failures: list[dict[str, Any]] = []
    eval_parents = [parent for parent in prepared_parents if parent.parent["role"] == "O0R_EVAL_CANDIDATE"]
    for prepared in eval_parents:
        exact_count = len(prepared.frame_plan["exact_timestamp_tokens"])
        source_eligible = 0
        admitted = 0
        clear_frames = 0
        occupied_frames = 0
        for token in prepared.frame_plan["exact_timestamp_tokens"]:
            resource_guard()
            eval_decode_count += 1
            try:
                frame = _decode_prepared_frame(prepared, token, decode_fn)
            except (MaterializerError, adapter.AdapterError) as error:
                frame_failures.append({"parent_id": prepared.parent["visit_id"], "video_id": prepared.parent["video_id"], "timestamp_token": token, "phase": "SOURCE_RECEIPT", "error_code": error.code})
                continue
            receipt = frame["source_frame_receipt"]
            highres = np.asarray(frame["highres_faro_depth_mm"])
            matrix = np.asarray(receipt["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
            try:
                geometry = geometry_fn(highres, matrix, receipt["gravity_up_camera_xyz"], receipt)
            except adapter.AdapterError as error:
                frame_failures.append({"parent_id": prepared.parent["visit_id"], "video_id": prepared.parent["video_id"], "timestamp_token": token, "phase": "SOURCE_ELIGIBILITY", "error_code": error.code})
                continue
            source_eligible += 1
            try:
                truth_record = eval_fn(frame, uncertainty_model, geometry=geometry)
            except (MaterializerError, adapter.AdapterError) as error:
                frame_failures.append({"parent_id": prepared.parent["visit_id"], "video_id": prepared.parent["video_id"], "timestamp_token": token, "phase": "QUERY_TRUTH", "error_code": error.code})
                continue
            artifact_receipt = writer.write_content_addressed_artifact(
                f"truth-frames/{prepared.parent['visit_id']}/{prepared.parent['video_id']}/{token}.json.gz",
                truth_record,
            )
            bundle = truth_record["query_bundle"]
            if bundle["complete_factor_query_truth"] is True:
                admitted += 1
                clear_frames += int(bundle["state_counts"]["CLEAR_OBSERVED"] >= 1)
                occupied_frames += int(bundle["state_counts"]["OCCUPIED_OBSERVED"] >= 1)
            else:
                frame_failures.append({"parent_id": prepared.parent["visit_id"], "video_id": prepared.parent["video_id"], "timestamp_token": token, "phase": "QUERY_TRUTH", "error_code": "QUERY_BUNDLE_INCOMPLETE", "artifact_canonical_sha256": artifact_receipt["artifact_canonical_sha256"]})
        parent_summaries.append(
            {
                "parent_id": prepared.parent["visit_id"],
                "video_id": prepared.parent["video_id"],
                "exact_timestamp_frames": exact_count,
                "source_eligible_frames": source_eligible,
                "admitted_frames": admitted,
                "clear_frames": clear_frames,
                "occupied_frames": occupied_frames,
            }
        )
    require(eval_decode_count == sum(len(parent.frame_plan["exact_timestamp_tokens"]) for parent in eval_parents), "EVAL_PHASE_INCOMPLETE", "every exact eval frame must be decoded once")
    fit_parent_ids = [parent.parent["visit_id"] for parent in prepared_parents if parent.parent["role"] == "ADAPTER_FIT"]
    gates = evaluate_truth_only_gates(fit_parent_ids, parent_summaries)
    phase_receipt = {
        "fit_frame_count": len(fit_sequence),
        "fit_frame_decode_count": len(fit_sequence.decoded_indices),
        "eval_decode_count_before_uncertainty_seal": 0,
        "eval_frame_decode_count": eval_decode_count,
        "uncertainty_model_sha256": model_receipt["content_sha256"],
        "model_outputs_during_truth_only": 0,
    }
    writer.write_json_gzip("frame-failures.json.gz", frame_failures)
    return gates, phase_receipt


def execute_truth_only(
    lock_path: Path,
    *,
    download_fn: Callable[..., dict[str, Any]] | None = None,
    decode_fn: Callable[..., dict[str, Any]] = decode_source_frame,
    fit_fn: Callable[[Sequence[dict[str, Any]]], Any] = adapter.fit_uncertainty_model,
    eval_fn: Callable[..., dict[str, Any]] = build_eval_truth_record,
    root_mkdir_fn: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    repo_root = REPO_ROOT.resolve()
    lock = validate_execution_lock(lock_path, repo_root)
    bindings = lock["_verified_bindings"]
    preflight_path = _repo_path(repo_root, bindings["TRUTH_ONLY_PREFLIGHT_LOCK"]["path"])
    authorization_path = _repo_path(repo_root, bindings["DATA_USE_AUTHORIZATION"]["path"])
    head_path = _repo_path(repo_root, bindings["HEAD_RECEIPT"]["path"])
    preflight = load_json(preflight_path)
    authorization = load_json(authorization_path)
    head_receipt = load_json(head_path)
    require(lock.get("required_environment") == preflight.get("required_environment"), "EXECUTION_ENVIRONMENT_DRIFT", "truth-only execution environment differs from preflight lock")
    plan_rows = validate_authorization(preflight, authorization, preflight_sha256=sha256_file(preflight_path))
    head_lookup = validate_head_receipt(preflight, sha256_file(authorization_path), head_receipt)
    require(plan_rows == expanded_asset_plan(preflight), "ASSET_PLAN_DRIFT", "authorization-expanded asset plan drift")
    roots = {role: _repo_path(repo_root, relative) for role, relative in EXPECTED_ROOTS.items()}
    require(all(not path.exists() for path in roots.values()), "ROOT_COLLISION", "one or more frozen execution roots already exist")
    budget = preflight["resource_budget"]
    require(lock.get("resource_budget") == budget, "EXECUTION_BUDGET_DRIFT", "truth-only execution budget drift")
    started = time.monotonic()
    writer = AtomicEvidenceWriter(roots["TRUTH_EVIDENCE"], int(budget["maximum_scientific_evidence_bytes"]))
    start_receipt = {
        "schema": "blindassist.taro.o0r.truth_only_execution_start_receipt.v1",
        "execution_lock_sha256": sha256_file(lock_path),
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "argv": lock["argv"],
        "verified_binding_sha256s": {role: row["sha256"] for role, row in sorted(bindings.items())},
        "truth_root_created_consumes_one_shot": True,
        "model_outputs_absent": True,
    }
    mkdir_exclusive = root_mkdir_fn or (lambda path: path.mkdir(parents=True, exist_ok=False))
    try:
        writer.activate(start_receipt)
        mkdir_exclusive(roots["SOURCE"])
        mkdir_exclusive(roots["WORK"])
        if download_fn is None:
            cache_config = lock["source_cache"]
            cache_root = _repo_path(repo_root, cache_config["source_root"])
            cache_receipt_path = _repo_path(repo_root, bindings[cache_config["download_receipts_role"]]["path"])
            cache_lookup = _load_cached_download_receipts(cache_receipt_path)

            def active_download_fn(row: Mapping[str, str], head_row: Mapping[str, Any], *, source_root: Path) -> dict[str, Any]:
                return _reuse_cached_asset(
                    row,
                    head_row,
                    source_root=source_root,
                    cache_root=cache_root,
                    cache_lookup=cache_lookup,
                )
        else:
            active_download_fn = download_fn
        download_receipts: list[dict[str, Any]] = []
        for row in plan_rows:
            require(time.monotonic() - started <= float(budget["truth_only_materialization_wall_seconds"]), "TRUTH_MATERIALIZATION_TIMEOUT", "truth-only wall-time budget exceeded")
            download_receipts.append(active_download_fn(row, head_lookup[row["url"]], source_root=roots["SOURCE"]))
        _validate_download_receipts(plan_rows, download_receipts, head_lookup, roots["SOURCE"])
        require(sum(row["bytes"] for row in download_receipts) == head_receipt["total_content_length_bytes"], "DOWNLOAD_TOTAL_BYTES_DRIFT", "download byte total differs from bound HEAD total")
        writer.write_json_gzip("download-receipts.json.gz", download_receipts)
        prepared = _prepare_downloaded_parents(preflight, plan_rows, download_receipts, source_root=roots["SOURCE"])
        frame_plan_receipt = [
            {
                "parent": parent.parent,
                "frame_plan": parent.frame_plan,
                "materialized_bytes": parent.materialized_bytes,
                "container_receipts": parent.container_receipts,
            }
            for parent in prepared
        ]
        writer.write_json_gzip("exact-frame-plan.json.gz", frame_plan_receipt)
        gates, phase_receipt = materialize_prepared_parents(
            prepared,
            writer=writer,
            started=started,
            wall_seconds=float(budget["truth_only_materialization_wall_seconds"]),
            peak_rss_bytes=int(budget["truth_only_peak_rss_bytes"]),
            decode_fn=decode_fn,
            fit_fn=fit_fn,
            eval_fn=eval_fn,
        )
        require(not roots["O0R_EVIDENCE_SEALED"].exists(), "FACTORIAL_ROOT_CREATED", "truth-only run created the forbidden O0R factor-headroom root")
        elapsed = time.monotonic() - started
        result = {
            "schema": TRUTH_RESULT_SCHEMA,
            "terminal": gates["terminal"],
            "passed": gates["passed"],
            "scientific_status": "TRUTH_ONLY_ADMISSION_PASS" if gates["passed"] else "NOT_EVALUABLE",
            "gates": gates,
            "phase_receipt": phase_receipt,
            "resource_receipt": {
                "elapsed_seconds": elapsed,
                "peak_rss_bytes_observed_at_completion": psutil.Process().memory_info().rss,
                "evidence_bytes_before_result": writer.bytes_written,
            },
            "model_outputs_absent": True,
            "depthart_inference_count": 0,
            "factorial_execution_count": 0,
            "claim_ceiling": "Source/truth-only ARKitScenes WILD_LAB admission only; no factor headroom, DepthART, device, product or safety claim.",
        }
        writer.write_json("result.json", result)
        completion = {
            "schema": "blindassist.taro.o0r.truth_only_execution_completion_receipt.v1",
            "terminal": result["terminal"],
            "passed": result["passed"],
            "one_shot_consumed": True,
            "elapsed_seconds": elapsed,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "psutil": psutil.__version__,
            "evidence_bytes_before_completion": writer.bytes_written,
        }
        writer.write_json("completion-receipt.json", completion)
        manifest = {
            "schema": MATERIALIZER_MANIFEST_SCHEMA,
            "files": {key: value for key, value in sorted(writer.file_receipts.items())},
            "file_count_before_manifest": len(writer.file_receipts),
            "bytes_before_manifest": writer.bytes_written,
            "truth_root_consumed": True,
        }
        writer.write_json("manifest.json", manifest)
        return result
    except Exception as error:
        one_shot_consumed = roots["TRUTH_EVIDENCE"].exists()
        if not one_shot_consumed:
            raise
        code = error.code if isinstance(error, (MaterializerError, adapter.AdapterError)) else type(error).__name__
        failure = {
            "schema": TRUTH_RESULT_SCHEMA,
            "terminal": "TARO_O0R_NOT_EVALUABLE_SOURCE_TRUTH_OR_INTERFACE",
            "passed": False,
            "scientific_status": "NOT_EVALUABLE",
            "failure_code": code,
            "message": str(error),
            "one_shot_consumed": one_shot_consumed,
            "model_outputs_absent": True,
        }
        try:
            writer.write_json("failure.json", failure)
            writer.write_json(
                "manifest.json",
                {
                    "schema": MATERIALIZER_MANIFEST_SCHEMA,
                    "files": {key: value for key, value in sorted(writer.file_receipts.items())},
                    "file_count_before_manifest": len(writer.file_receipts),
                    "bytes_before_manifest": writer.bytes_written,
                    "truth_root_consumed": one_shot_consumed,
                },
            )
        except Exception:
            pass
        return failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = execute_truth_only(args.execution_lock.resolve())
    except Exception as error:
        code = error.code if isinstance(error, (MaterializerError, adapter.AdapterError)) else type(error).__name__
        truth_root = _repo_path(REPO_ROOT.resolve(), EXPECTED_ROOTS["TRUTH_EVIDENCE"])
        consumed = truth_root.exists()
        status = "EXECUTION_ONE_SHOT_CONSUMED" if consumed else "EXECUTION_NOT_STARTED"
        print(json.dumps({"status": status, "error_code": code, "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"]}, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
