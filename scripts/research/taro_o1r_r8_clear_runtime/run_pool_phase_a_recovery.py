#!/usr/bin/env python3
"""Recover R8 Phase A from the fully sealed R0 candidates without inference."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
import time
import zipfile
from collections import Counter
from itertools import groupby
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import psutil

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation_io as r6io
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as r6_reducer
from scripts.research.taro_o1r_reducer_integration_runtime.locked_uncertainty import load_locked_uncertainty_model
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_phase_a as shared
from scripts.research.taro_o1r_r8_clear_runtime import run_pool_phase_a


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_execution_lock.v1"
LOCK_ID = "TARO_O1R_R8_CLEAR_NEGATIVE_CONTROL_POOL_PHASE_A_SOURCE_ONLY_RECOVERY_R1_ONE_SHOT_EXECUTION_LOCK"
INVENTORY_PATH = run_pool_phase_a.INVENTORY_PATH
CANDIDATE_ROOT = run_pool_phase_a.OUTPUT_ROOT
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r8-clear-pool-phase-a-r1"
PASS_TERMINAL = "TARO_O1R_R8_CLEAR_POOL_PHASE_A_SOURCE_ONLY_RECOVERY_SEALED_PASS_R1"
FAIL_TERMINAL = "TARO_O1R_R8_CLEAR_POOL_PHASE_A_SOURCE_ONLY_RECOVERY_EXECUTION_INVALID_R1"
PARENT_COUNT = run_pool_phase_a.PARENT_COUNT
FRAME_COUNT = run_pool_phase_a.FRAME_COUNT
QUERY_COUNT = run_pool_phase_a.QUERY_COUNT

EXPECTED_BINDINGS = {
    "R8_PROTOCOL": "docs/research/taro/TARO_O1R_R8_SOURCE_ONLY_CLEAR_NEGATIVE_CONTROL_COHORT_ENRICHMENT_PROTOCOL_LOCK_2026-08-12.json",
    "R8_INVENTORY_PLAN": INVENTORY_PATH,
    "R8_INVENTORY_RESULT": "artifacts.local/evidence/taro/o1r-r8-clear-pool-inventory-r0/result.json",
    "R0_EXECUTION_RECEIPT": f"{CANDIDATE_ROOT}/execution-receipt.json",
    "R0_CANDIDATE_COMPLETION": f"{CANDIDATE_ROOT}/candidate-completion.json",
    "DEPTHART_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/depthart_runner.py",
    "PROSPECTIVE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py",
    "R6_REDUCER_RUNTIME": "scripts/research/taro_o1r_reducer_integration_runtime/reducer_integration.py",
    "LOCKED_UNCERTAINTY_LOADER": "scripts/research/taro_o1r_reducer_integration_runtime/locked_uncertainty.py",
    "LOCKED_UNCERTAINTY_ARTIFACT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-artifact.json.gz",
    "LOCKED_UNCERTAINTY_RECEIPT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-receipt.json",
    "R7_CANARY_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "SHARED_PHASE_A_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_phase_a.py",
    "R8_PHASE_A_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_pool_phase_a.py",
    "R8_RECOVERY_RUNNER": "scripts/research/taro_o1r_r8_clear_runtime/run_pool_phase_a_recovery.py",
}
EXPECTED_AUTHORITY = {
    "sealed_candidate_adoption": True,
    "candidate_inference": False,
    "candidate_inference_count": 0,
    "source_frame_decode": True,
    "source_only_phase_a": True,
    "faro_payload_read": False,
    "truth_scoring": False,
    "threshold_fit": False,
    "training": False,
    "network": False,
    "device": False,
    "product": False,
    "safety": False,
}
EXPECTED_USER_AUTHORITY = {
    "confirmed_by": "user",
    "confirmed_at": "2026-08-12",
    "confirmation_verbatim": "推进taro成功落地",
    "scope": "Resume after the user-requested computer-restart pause: validate and adopt the exact 402 sealed R0 DepthART candidates, then rerun only the source-only feature phase in a fresh R1 root; no candidate inference, FARO, truth scoring, or training.",
}


class PoolPhaseARecoveryError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PoolPhaseARecoveryError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R8_RECOVERY_SEAL_COLLISION", "recovery caller supplied a seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Mapping[str, Any], schema: str) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    observed = record.pop("content_sha256", None)
    require(record.get("schema") == schema and isinstance(observed, str) and adapter.canonical_sha256(record) == observed, "R8_RECOVERY_SEAL_DRIFT", "recovery seal/schema drift")
    record["content_sha256"] = observed
    return record


def validate_candidate_completion(value: Mapping[str, Any]) -> dict[str, Any]:
    completion = shared._validate_seal(dict(value), "blindassist.taro.o1r.r7_fresh_candidate_completion.v1")
    require(
        completion.get("frame_count") == FRAME_COUNT
        and completion.get("payload_reads") == {"color": FRAME_COUNT, "intrinsics": FRAME_COUNT}
        and completion.get("faro_reads") == 0
        and completion.get("all_candidates_sealed_before_source_features") is True,
        "R8_RECOVERY_CANDIDATE_COMPLETION_INVALID",
        "R0 candidate completion is not adoptable",
    )
    return completion


def verify_candidate_set(frames: Sequence[r6io.R6FrameRef], root: Path) -> dict[str, Any]:
    completion = validate_candidate_completion(_read_json(root / "candidate-completion.json"))
    input_hashes: list[str] = []
    record_hashes: list[str] = []
    for index, frame in enumerate(frames, 1):
        candidate_input, record, _native, _highres = shared._load_candidate(root, frame)
        input_hashes.append(candidate_input["content_sha256"])
        record_hashes.append(record["content_sha256"])
        if index % 50 == 0 or index == FRAME_COUNT:
            print(json.dumps({"phase": "R8_RECOVERY_CANDIDATE_VALIDATION", "completed": index, "total": FRAME_COUNT}, sort_keys=True), flush=True)
    require(adapter.canonical_sha256(input_hashes) == completion["candidate_input_hash_sequence_sha256"], "R8_RECOVERY_CANDIDATE_INPUT_SEQUENCE", "R0 candidate input sequence drift")
    require(adapter.canonical_sha256(record_hashes) == completion["candidate_record_hash_sequence_sha256"], "R8_RECOVERY_CANDIDATE_RECORD_SEQUENCE", "R0 candidate record sequence drift")
    return _seal({
        "schema": "blindassist.taro.o1r.r8_clear_pool_candidate_adoption.v1",
        "candidate_root": CANDIDATE_ROOT,
        "frame_count": FRAME_COUNT,
        "candidate_completion_sha256": completion["content_sha256"],
        "candidate_input_hash_sequence_sha256": completion["candidate_input_hash_sequence_sha256"],
        "candidate_record_hash_sequence_sha256": completion["candidate_record_hash_sequence_sha256"],
        "candidate_inference_count": 0,
        "all_candidates_revalidated_before_source_features": True,
        "faro_reads": 0,
        "truth_reads": 0,
    })


def validate_completion(value: Mapping[str, Any]) -> dict[str, Any]:
    completion = _validate_seal(value, "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_completion.v1")
    require(completion.get("parent_count") == PARENT_COUNT and completion.get("frame_count") == FRAME_COUNT and completion.get("query_count") == QUERY_COUNT, "R8_RECOVERY_COMPLETION_COUNT", "recovery completion count drift")
    require(completion.get("faro_reads") == completion.get("truth_reads") == completion.get("candidate_inference_count") == completion.get("training_steps") == completion.get("network_requests") == 0 and completion.get("clear_output_allowed") is False, "R8_RECOVERY_COMPLETION_FIREWALL", "recovery completion firewall drift")
    return completion


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = _read_json(lock_path)
    require(lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False, "R8_RECOVERY_LOCK_IDENTITY", "recovery lock identity drift")
    require(lock.get("user_authority") == EXPECTED_USER_AUTHORITY, "R8_RECOVERY_USER_AUTHORITY", "recovery user authority drift")
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv and lock.get("inventory_path") == INVENTORY_PATH and lock.get("candidate_root") == CANDIDATE_ROOT and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R8_RECOVERY_LOCK_POLICY", "recovery argv/root policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R8_RECOVERY_BINDINGS", "recovery binding count drift")
    verified: dict[str, dict[str, Any]] = {}
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in verified and EXPECTED_BINDINGS.get(role) == relative, "R8_RECOVERY_BINDING_ROW", "recovery binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R8_RECOVERY_BINDING_HASH", f"recovery binding drift: {relative}")
        verified[role] = row
    candidate_completion = validate_candidate_completion(_read_json(_repo_path(EXPECTED_BINDINGS["R0_CANDIDATE_COMPLETION"])))
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R8_RECOVERY_AUTHORITY", "recovery authority drift")
    require(lock.get("resource_budget") == {"maximum_wall_seconds": 14400, "maximum_peak_rss_bytes": 17179869184, "maximum_evidence_bytes": 2147483648}, "R8_RECOVERY_BUDGET", "recovery budget drift")
    require(lock.get("exact_cohort") == {"parent_count": PARENT_COUNT, "physical_frame_count": FRAME_COUNT, "query_count": QUERY_COUNT, "per_parent_frame_counts": run_pool_phase_a.FROZEN_FRAME_COUNTS}, "R8_RECOVERY_COHORT", "recovery cohort drift")
    require(lock.get("candidate_admission") == {"frame_count": FRAME_COUNT, "candidate_completion_content_sha256": candidate_completion["content_sha256"], "candidate_input_hash_sequence_sha256": candidate_completion["candidate_input_hash_sequence_sha256"], "candidate_record_hash_sequence_sha256": candidate_completion["candidate_record_hash_sequence_sha256"], "all_402_candidates_replayed_and_hash_verified_before_lock": True, "candidate_inference_count_in_recovery": 0}, "R8_RECOVERY_CANDIDATE_ADMISSION", "recovery candidate admission drift")
    require(lock.get("phase_firewall") == {"adopt_all_candidates_before_source_features": True, "source_feature_inputs": ["SEALED_CANDIDATE_DEPTH", "APPLE_DEPTH", "CONFIDENCE", "INTRINSICS", "TRAJECTORY"], "old_partial_source_records_adopted": False, "faro_reads_before_recovery_completion": 0, "clear_output_allowed": False}, "R8_RECOVERY_FIREWALL", "recovery phase firewall drift")
    require(not _repo_path(OUTPUT_ROOT).exists(), "R8_RECOVERY_ROOT_COLLISION", "recovery output root exists")
    lock["_lock_path"] = lock_path
    return lock


def _write_failure(writer: FactorEvidenceWriter, error: Exception) -> None:
    try:
        writer.write_json("failure.json", {"schema": "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_failure.v1", "terminal": FAIL_TERMINAL, "execution_valid": False, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "candidate_inference_count": 0, "faro_reads": 0, "one_shot_consumed": True})
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_manifest.v1", "terminal": FAIL_TERMINAL, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
    except Exception:
        pass


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    budget = lock["resource_budget"]
    started = time.monotonic()
    process = psutil.Process()

    def guard() -> None:
        require(time.monotonic() - started <= float(budget["maximum_wall_seconds"]), "R8_RECOVERY_TIMEOUT", "recovery wall budget exceeded")
        require(process.memory_info().rss <= int(budget["maximum_peak_rss_bytes"]), "R8_RECOVERY_RSS", "recovery RSS budget exceeded")

    frames = run_pool_phase_a.load_frames(_repo_path(INVENTORY_PATH))
    adoption = verify_candidate_set(frames, _repo_path(CANDIDATE_ROOT))
    guard()
    writer = FactorEvidenceWriter(_repo_path(OUTPUT_ROOT), int(budget["maximum_evidence_bytes"]))
    writer.activate({"schema": "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_execution_receipt.v1", "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]), "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "expected_parent_count": PARENT_COUNT, "expected_frame_count": FRAME_COUNT, "expected_query_count": QUERY_COUNT, "adopted_candidate_completion_sha256": adoption["candidate_completion_sha256"], "candidate_inference_count": 0, "faro_payload_read": False, "training_steps": 0, "network_requests": 0, "one_shot_consumed_on_root_creation": True})
    try:
        writer.write_json("candidate-adoption.json", adoption)
        uncertainty_model = load_locked_uncertainty_model()
        source_reads: Counter[str] = Counter()
        source_hashes: list[str] = []
        prospective_hashes: list[str] = []
        reducer_hashes: list[str] = []
        state_counts: Counter[str] = Counter()
        parent_state_counts: dict[str, Counter[str]] = {parent: Counter() for parent, *_ in run_pool_phase_a.pool_cohort.EXPECTED_POOL}

        def observed_source(role: str, _: str) -> None:
            require(role in {"lowres_depth", "confidence"}, "R8_RECOVERY_SOURCE_FIREWALL", "source recovery read a forbidden payload")
            source_reads[role] += 1

        completed = 0
        for _parent_key, parent_frames_iter in groupby(frames, key=lambda row: (row.parent_id, row.video_id)):
            parent_frames = list(parent_frames_iter)
            with zipfile.ZipFile(parent_frames[0].upsampling_archive) as up_bundle:
                for frame in parent_frames:
                    candidate_input, candidate, _native, highres = shared._load_candidate(_repo_path(CANDIDATE_ROOT), frame)
                    apple_payload, apple_binding = r6io._read_member(up_bundle, frame.members["lowres_depth"], observer=observed_source)
                    conf_payload, conf_binding = r6io._read_member(up_bundle, frame.members["confidence"], observer=observed_source)
                    apple = np.ascontiguousarray(materializer._decode_png(apple_payload, "lowres_depth"))
                    confidence = np.ascontiguousarray(materializer._decode_png(conf_payload, "confidence"))
                    source_receipt = shared._seal({
                        "schema": "blindassist.taro.o1r.r7_fresh_source_frame_receipt.v1",
                        "analysis_role": "FRESH_CONFIRMATION",
                        "parent_id": frame.parent_id,
                        "video_id": frame.video_id,
                        "timestamp_token": frame.timestamp_token,
                        "physical_frame_id": frame.physical_frame_id,
                        "candidate_input_sha256": candidate_input["content_sha256"],
                        "candidate_frame_record_sha256": candidate["content_sha256"],
                        "apple_depth_binding": shared._member_binding(frame, "lowres_depth", apple_binding),
                        "confidence_binding": shared._member_binding(frame, "confidence", conf_binding),
                        "apple_depth_sha256": adapter.canonical_sha256(apple),
                        "confidence_sha256": adapter.canonical_sha256(confidence),
                        "intrinsics_highres": candidate_input["intrinsics_highres"],
                        "lowres_intrinsics": candidate_input["lowres_intrinsics"],
                        "gravity_up_camera_xyz": candidate_input["gravity_up_camera_xyz"],
                        "max_source_timestamp_ns": candidate_input["max_source_timestamp_ns"],
                        "faro_payload_read": False,
                        "truth_payload_read": False,
                    })
                    writer.write_json(shared._source_receipt_relative(frame), source_receipt)
                    low = candidate_input["lowres_intrinsics"]
                    low_matrix = [[float(low["fx"]), 0.0, float(low["cx"])], [0.0, float(low["fy"]), float(low["cy"])], [0.0, 0.0, 1.0]]
                    bundle = prospective.build_prospective_factor_bundle(parent_id=frame.parent_id, video_id=frame.video_id, timestamp_token=frame.timestamp_token, source_frame_receipt_sha256=source_receipt["content_sha256"], candidate_frame_record_sha256=candidate["content_sha256"], max_source_timestamp_ns=int(source_receipt["max_source_timestamp_ns"]), candidate_highres_depth_m=highres, apple_depth_mm=apple, confidence=confidence, intrinsics_highres_3x3=source_receipt["intrinsics_highres"]["matrix_3x3"], intrinsics_apple_3x3=low_matrix, gravity_up_camera_xyz=source_receipt["gravity_up_camera_xyz"])
                    prior = r6_reducer._integrate_with_validated_model(prospective_bundle=bundle, candidate_highres_depth_m=highres, confidence=confidence, intrinsics_apple_3x3=low_matrix, uncertainty_model=uncertainty_model)
                    source_features = r7_canary.build_source_frame_record(bundle, highres, apple, confidence, low_matrix, source_receipt["intrinsics_highres"]["matrix_3x3"], prior)
                    writer.write_json_gzip(shared._lineage_relative(frame), {"prospective_bundle": bundle, "r6_reducer_bundle": prior, "r7_source_frame_record": source_features})
                    source_hashes.append(source_features["content_sha256"])
                    prospective_hashes.append(bundle["content_sha256"])
                    reducer_hashes.append(prior["content_sha256"])
                    for feature in source_features["query_features"]:
                        state = shared._positive_state(feature)
                        state_counts[state] += 1
                        parent_state_counts[frame.parent_id][state] += 1
                    completed += 1
                    guard()
                    if completed % 10 == 0 or completed == FRAME_COUNT:
                        print(json.dumps({"phase": "R8_RECOVERY_SOURCE_FEATURES", "completed": completed, "total": FRAME_COUNT, "physical_frame_id": frame.physical_frame_id}, sort_keys=True), flush=True)
        require(source_reads == Counter({"lowres_depth": FRAME_COUNT, "confidence": FRAME_COUNT}), "R8_RECOVERY_SOURCE_READ_COUNT", "source recovery read counts drift")
        require(sum(state_counts.values()) == QUERY_COUNT and state_counts["CLEAR_OBSERVED"] == 0, "R8_RECOVERY_SOURCE_STATE", "source recovery emitted CLEAR or lost queries")
        completion = validate_completion(_seal({
            "schema": "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_completion.v1",
            "parent_count": PARENT_COUNT,
            "frame_count": FRAME_COUNT,
            "query_count": QUERY_COUNT,
            "candidate_adoption_sha256": adoption["content_sha256"],
            "source_frame_hash_sequence_sha256": adapter.canonical_sha256(source_hashes),
            "prospective_bundle_hash_sequence_sha256": adapter.canonical_sha256(prospective_hashes),
            "r6_reducer_hash_sequence_sha256": adapter.canonical_sha256(reducer_hashes),
            "frozen_positive_candidate_state_counts": {state: int(state_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
            "per_parent_state_counts": {parent: {state: int(counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")} for parent, counts in parent_state_counts.items()},
            "source_payload_reads": dict(sorted(source_reads.items())),
            "candidate_inference_count": 0,
            "faro_reads": 0,
            "truth_reads": 0,
            "clear_output_allowed": False,
            "all_source_records_sealed_before_faro": True,
            "training_steps": 0,
            "network_requests": 0,
        }))
        writer.write_json("phase-a-completion.json", completion)
        result = {"schema": "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_result.v1", "terminal": PASS_TERMINAL, "passed": True, "execution_valid": True, "parent_count": PARENT_COUNT, "frame_count": FRAME_COUNT, "query_count": QUERY_COUNT, "candidate_inference_count": 0, "adopted_candidate_count": FRAME_COUNT, "frozen_positive_candidate_state_counts": completion["frozen_positive_candidate_state_counts"], "phase_a_completion_sha256": completion["content_sha256"], "faro_reads": 0, "truth_scoring": False, "clear_output_allowed": False, "training_steps": 0, "network_requests": 0, "elapsed_seconds": round(time.monotonic() - started, 6), "one_shot_consumed": True, "claim_ceiling": "Sealed R8 source-only positive-occupancy features using revalidated candidates; no FARO label, effectiveness, deployment, product, or safety evidence."}
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r8_clear_pool_phase_a_recovery_manifest.v1", "terminal": PASS_TERMINAL, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
        return result
    except Exception as error:
        _write_failure(writer, error)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-lock", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = execute(args.execution_lock)
    except Exception as error:
        print(json.dumps({"terminal": FAIL_TERMINAL, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
