#!/usr/bin/env python3
"""Independent validator for a completed TARO R11 source-only Phase A root."""

from __future__ import annotations

import argparse
import copy
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import prospective_factor_runtime as prospective
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as reducer
from scripts.research.taro_o1r_r7_canary_runtime import positive_occupancy_factor as r7_positive
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r11_abstention_runtime import abstention_candidate
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_inventory


REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-phase-a-r0"
LOCK_RELATIVE = "docs/research/taro/TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK_2026-08-12.json"
INVENTORY_PATH = "artifacts.local/evidence/taro/o1r-r11-fresh-pool-inventory-r0/exact-frame-plan.json"
PASS_TERMINAL = "TARO_O1R_R11_FRESH_POOL_PHASE_A_SOURCE_ONLY_SEALED_PASS"
PARENT_COUNT = 48
FRAME_COUNT = 1043
QUERY_COUNT = 9387
PRE_MANIFEST_FILE_COUNT = 5219
FROZEN_FRAME_COUNTS = [
    20, 14, 23, 24, 29, 7, 12, 14, 10, 21, 28, 15, 11, 28, 29, 72,
    36, 14, 18, 4, 54, 32, 83, 17, 15, 16, 29, 10, 12, 34, 7, 14,
    11, 6, 9, 1, 46, 6, 27, 26, 50, 9, 11, 27, 12, 9, 28, 13,
]
INVENTORY_CONTENT_SHA256 = "35156C2901A4CBEEDB6D611A56ABE3D711CEB68EF932480C21428BA4FF741600"


class PhaseAValidationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise PhaseAValidationError(code, message)


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhaseAValidationError("R11_PHASE_A_VALIDATION_JSON", f"JSON record cannot be read: {path}") from error
    require(isinstance(value, dict), "R11_PHASE_A_VALIDATION_JSON", f"JSON object required: {path}")
    return value


def _load_json_gzip(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, EOFError, gzip.BadGzipFile, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseAValidationError("R11_PHASE_A_VALIDATION_JSON", f"gzip JSON record cannot be read: {path}") from error
    require(isinstance(value, dict), "R11_PHASE_A_VALIDATION_JSON", f"gzip JSON object required: {path}")
    return value


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R11_PHASE_A_VALIDATION_RECORD", "sealed record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema
        and isinstance(observed, str)
        and adapter.canonical_sha256(record) == observed,
        "R11_PHASE_A_VALIDATION_SEAL",
        f"record seal/schema drift: {schema}",
    )
    record["content_sha256"] = observed
    return record


def _relative_paths(parent_id: str, video_id: str, token: str) -> dict[str, str]:
    return {
        "candidate_input": f"candidate-inputs/{parent_id}/{video_id}/{token}.json",
        "candidate_blob": f"candidates/{parent_id}/{video_id}/{token}.depth.npy.gz",
        "candidate_record": f"candidates/{parent_id}/{video_id}/{token}.json",
        "source_receipt": f"phase-a-sources/{parent_id}/{video_id}/{token}.json",
        "lineage": f"phase-a-lineage/{parent_id}/{video_id}/{token}.json.gz",
    }


def _frame_rows(inventory: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    validated = run_pool_inventory.validate_inventory(inventory)
    rows = [
        (str(parent["visit_id"]), str(parent["video_id"]), str(token))
        for parent in validated["parents"]
        for token in parent["frame_plan"]["exact_timestamp_tokens"]
    ]
    require(len(rows) == len(set(rows)) == FRAME_COUNT, "R11_PHASE_A_VALIDATION_COHORT", "frame cohort drift")
    return rows


def _validate_factor_pair(base: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[Counter[str], Counter[str], int]:
    r7 = r7_positive.validate_positive_occupancy_factor(dict(base))
    r11 = abstention_candidate.validate_abstention_bundle(dict(candidate))
    require(
        r7["physical_frame_id"] == r11["physical_frame_id"]
        and r7["source_frame_record_sha256"] == r11["source_frame_record_sha256"],
        "R11_PHASE_A_VALIDATION_FACTOR_LINEAGE",
        "R7/R11 factor lineage drift",
    )
    r7_counts: Counter[str] = Counter()
    r11_counts: Counter[str] = Counter()
    for left, right in zip(r7["query_results"], r11["query_results"], strict=True):
        require(
            left["query_id"] == right["query_id"]
            and left["grid_index"] == right["grid_index"]
            and (right["state"] != "OCCUPIED_OBSERVED" or left["state"] == "OCCUPIED_OBSERVED"),
            "R11_PHASE_A_VALIDATION_FACTOR_SUBSET",
            "R11 positive is not an ordered subset of R7 positive",
        )
        r7_counts[left["state"]] += 1
        r11_counts[right["state"]] += 1
    abstained = r7_counts["OCCUPIED_OBSERVED"] - r11_counts["OCCUPIED_OBSERVED"]
    require(
        sum(r7_counts.values()) == sum(r11_counts.values()) == 9
        and r7_counts["CLEAR_OBSERVED"] == r11_counts["CLEAR_OBSERVED"] == 0
        and abstained >= 0
        and r11["base_positive_count"] == r7_counts["OCCUPIED_OBSERVED"]
        and r11["candidate_positive_count"] == r11_counts["OCCUPIED_OBSERVED"]
        and r11["abstained_base_positive_count"] == abstained,
        "R11_PHASE_A_VALIDATION_FACTOR_COUNTS",
        "R7/R11 factor count identity drift",
    )
    return r7_counts, r11_counts, abstained


def _validate_manifest(root: Path, manifest: Mapping[str, Any], expected_files: set[str]) -> dict[str, dict[str, Any]]:
    sealed = _validate_seal(manifest, "blindassist.taro.o1r.r11_fresh_pool_phase_a_manifest.v1")
    files = sealed.get("files")
    require(
        sealed.get("terminal") == PASS_TERMINAL
        and sealed.get("one_shot_consumed") is True
        and isinstance(files, dict)
        and set(files) == expected_files
        and sealed.get("file_count_before_manifest") == len(files) == PRE_MANIFEST_FILE_COUNT,
        "R11_PHASE_A_VALIDATION_MANIFEST",
        "manifest identity/file set drift",
    )
    total = 0
    for relative, receipt in files.items():
        target = materializer.safe_join(root, relative)
        require(
            isinstance(receipt, dict)
            and receipt.get("path") == relative
            and target.is_file()
            and target.stat().st_size == receipt.get("bytes")
            and materializer.sha256_file(target) == receipt.get("sha256"),
            "R11_PHASE_A_VALIDATION_MANIFEST_FILE",
            f"manifest file binding drift: {relative}",
        )
        total += int(receipt["bytes"])
    require(total == sealed.get("bytes_before_manifest"), "R11_PHASE_A_VALIDATION_MANIFEST", "manifest byte sum drift")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    require(actual == expected_files | {"manifest.json"}, "R11_PHASE_A_VALIDATION_ROOT_SET", "evidence root file set drift")
    return {str(key): dict(value) for key, value in files.items()}


def validate_evidence(root: Path | None = None, lock_path: Path | None = None) -> dict[str, Any]:
    evidence_root = (root or _repo_path(EVIDENCE_ROOT)).resolve()
    require(evidence_root.is_dir(), "R11_PHASE_A_VALIDATION_ROOT", "Phase A evidence root missing")
    inventory = run_pool_inventory.validate_inventory(_load_json(_repo_path(INVENTORY_PATH)))
    require(
        inventory["content_sha256"] == INVENTORY_CONTENT_SHA256
        and inventory["parent_count"] == PARENT_COUNT
        and inventory["exact_pose_bounded_frame_count"] == FRAME_COUNT,
        "R11_PHASE_A_VALIDATION_INVENTORY",
        "sealed inventory drift",
    )
    rows = _frame_rows(inventory)
    expected_files = {
        "execution-receipt.json",
        "candidate-completion.json",
        "phase-a-completion.json",
        "result.json",
    }
    for parent_id, video_id, token in rows:
        expected_files.update(_relative_paths(parent_id, video_id, token).values())
    manifest = _load_json(evidence_root / "manifest.json")
    _validate_manifest(evidence_root, manifest, expected_files)

    execution = _validate_seal(
        _load_json(evidence_root / "execution-receipt.json"),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_execution_receipt.v1",
    )
    candidate_completion = _validate_seal(
        _load_json(evidence_root / "candidate-completion.json"),
        "blindassist.taro.o1r.r11_fresh_pool_candidate_completion.v1",
    )
    completion = _validate_seal(
        _load_json(evidence_root / "phase-a-completion.json"),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_completion.v1",
    )
    result = _validate_seal(
        _load_json(evidence_root / "result.json"),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_result.v1",
    )
    execution_lock_path = (lock_path or _repo_path(LOCK_RELATIVE)).resolve()
    execution_lock = _validate_seal(
        _load_json(execution_lock_path),
        "blindassist.taro.o1r.r11_fresh_pool_phase_a_execution_lock.v1",
    )
    require(
        execution.get("execution_lock_sha256") == materializer.sha256_file(execution_lock_path)
        and execution.get("execution_lock_content_sha256") == execution_lock["content_sha256"]
        and execution_lock.get("lock_id") == "TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_ONE_SHOT_EXECUTION_LOCK"
        and execution_lock.get("status") == "AUTHORIZED_UNCONSUMED"
        and execution_lock.get("consumed") is False
        and execution_lock.get("inventory_path") == INVENTORY_PATH
        and execution_lock.get("output_root") == EVIDENCE_ROOT
        and execution_lock.get("overwrite") is False
        and execution_lock.get("rerun") is False
        and execution_lock.get("inventory_content_sha256") == INVENTORY_CONTENT_SHA256
        and execution_lock.get("execution_authority", {}).get("candidate_inference_count") == FRAME_COUNT
        and execution_lock.get("execution_authority", {}).get("highres_depth_member_payload_read") is False
        and execution_lock.get("execution_authority", {}).get("faro_payload_read") is False
        and execution_lock.get("execution_authority", {}).get("truth_scoring") is False
        and execution_lock.get("execution_authority", {}).get("training") is False
        and execution_lock.get("execution_authority", {}).get("network") is False
        and execution_lock.get("resource_budget", {}).get("maximum_wall_seconds") == 57_600
        and execution_lock.get("resource_budget", {}).get("maximum_peak_rss_bytes") == 17_179_869_184
        and execution_lock.get("resource_budget", {}).get("maximum_cuda_allocated_bytes") == 12_884_901_888
        and execution_lock.get("resource_budget", {}).get("maximum_evidence_bytes") == 2_147_483_648,
        "R11_PHASE_A_VALIDATION_EXECUTION_LOCK",
        "execution receipt/lock binding drift",
    )

    candidate_input_hashes: list[str] = []
    candidate_record_hashes: list[str] = []
    source_receipt_hashes: list[str] = []
    source_frame_hashes: list[str] = []
    prospective_hashes: list[str] = []
    reducer_hashes: list[str] = []
    r7_hashes: list[str] = []
    r11_hashes: list[str] = []
    r7_counts: Counter[str] = Counter()
    r11_counts: Counter[str] = Counter()
    abstained_total = 0
    runtime_identity_hashes: set[str] = set()
    parent_counts: dict[tuple[str, str], dict[str, Any]] = {
        identity: {"r7": Counter(), "r11": Counter(), "abstained": 0}
        for identity in [(visit, video) for visit, video, _rank in fresh_pool.EXPECTED_POOL]
    }

    for parent_id, video_id, token in rows:
        relative = _relative_paths(parent_id, video_id, token)
        physical_frame_id = f"{video_id}:{token}"
        candidate_input = _validate_seal(
            _load_json(evidence_root / relative["candidate_input"]),
            "blindassist.taro.o1r.r11_fresh_pool_candidate_input.v1",
        )
        candidate_record = _validate_seal(
            _load_json(evidence_root / relative["candidate_record"]),
            "blindassist.taro.o1r.r11_fresh_pool_candidate_frame.v1",
        )
        require(
            candidate_input["parent_id"] == candidate_record["parent_id"] == parent_id
            and candidate_input["video_id"] == candidate_record["video_id"] == video_id
            and candidate_input["timestamp_token"] == candidate_record["timestamp_token"] == token
            and candidate_input["physical_frame_id"] == candidate_record["physical_frame_id"] == physical_frame_id
            and candidate_record["candidate_input_sha256"] == candidate_input["content_sha256"]
            and candidate_input.get("allowed_model_inputs") == ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"]
            and candidate_input.get("highres_depth_member_payload_read") is False
            and candidate_record.get("highres_depth_member_payload_read") is False
            and candidate_record.get("faro_payload_read") is False,
            "R11_PHASE_A_VALIDATION_CANDIDATE_LINEAGE",
            "candidate identity/authority drift",
        )
        inference = _validate_seal(
            candidate_record["inference_receipt"],
            "blindassist.taro.o1r.r11_fresh_pool_depthart_inference.v1",
        )
        require(
            inference.get("model_id") == adapter.BASELINE_MODEL_ID
            and inference.get("checkpoint_sha256") == adapter.BASELINE_CHECKPOINT_SHA256
            and inference.get("preprocess_id") == depthart_runner.PREPROCESS_ID
            and inference.get("postprocess_id") == depthart_runner.POSTPROCESS_ID
            and inference.get("candidate_input_sha256") == candidate_input["content_sha256"]
            and inference.get("physical_frame_id") == physical_frame_id
            and inference.get("truth_alignment_used") is False
            and inference.get("highres_depth_member_payload_read") is False
            and inference.get("faro_payload_read") is False
            and isinstance(inference.get("runtime_identity"), dict),
            "R11_PHASE_A_VALIDATION_INFERENCE",
            "DepthART inference identity/firewall drift",
        )
        runtime_identity_hashes.add(adapter.canonical_sha256(inference["runtime_identity"]))
        blob = candidate_record["native_depth_blob"]
        blob_path = evidence_root / relative["candidate_blob"]
        payload = blob_path.read_bytes()
        require(
            blob.get("path") == relative["candidate_blob"]
            and len(payload) == blob.get("bytes")
            and materializer.sha256_bytes(payload) == blob.get("sha256"),
            "R11_PHASE_A_VALIDATION_CANDIDATE_BLOB",
            "candidate blob binding drift",
        )
        native = np.ascontiguousarray(depthart_runner.decode_npy_gzip_bytes(payload), dtype=np.float32)
        require(
            native.shape == depthart_runner.NATIVE_SHAPE_HW
            and bool(np.all(np.isfinite(native)))
            and adapter.canonical_sha256(native) == blob.get("array_sha256"),
            "R11_PHASE_A_VALIDATION_CANDIDATE_ARRAY",
            "candidate native array drift",
        )
        require(
            blob.get("shape_hw") == list(depthart_runner.NATIVE_SHAPE_HW)
            and blob.get("dtype") == "float32"
            and blob.get("encoding") == "DETERMINISTIC_GZIP_NPY_MTIME_0"
            and inference.get("native_depth_sha256") == adapter.canonical_sha256(native),
            "R11_PHASE_A_VALIDATION_CANDIDATE_ARRAY",
            "candidate blob metadata/inference hash drift",
        )
        candidate_depth_highres_m = depthart_runner.upsample_native_depth(native)
        require(
            adapter.canonical_sha256(candidate_depth_highres_m)
            == candidate_record["inference_receipt"].get("candidate_depth_highres_sha256"),
            "R11_PHASE_A_VALIDATION_CANDIDATE_ARRAY",
            "candidate high-resolution replay drift",
        )

        source = _validate_seal(
            _load_json(evidence_root / relative["source_receipt"]),
            "blindassist.taro.o1r.r11_fresh_pool_source_frame_receipt.v1",
        )
        lineage = _validate_seal(
            _load_json_gzip(evidence_root / relative["lineage"]),
            "blindassist.taro.o1r.r11_fresh_pool_phase_a_lineage.v1",
        )
        require(
            source["physical_frame_id"] == lineage["physical_frame_id"] == physical_frame_id
            and source["candidate_input_sha256"] == candidate_input["content_sha256"]
            and source["candidate_frame_record_sha256"] == candidate_record["content_sha256"]
            and lineage["source_frame_receipt_sha256"] == source["content_sha256"]
            and source.get("highres_depth_member_payload_read") is False
            and source.get("faro_payload_read") is False
            and source.get("truth_payload_read") is False
            and lineage.get("highres_depth_member_payload_read") is False
            and lineage.get("faro_payload_read") is False
            and lineage.get("truth_inputs") == 0,
            "R11_PHASE_A_VALIDATION_SOURCE_LINEAGE",
            "source/lineage identity or firewall drift",
        )
        prospective_bundle = prospective.validate_prospective_factor_bundle(
            lineage["prospective_bundle"], candidate_highres_depth_m=candidate_depth_highres_m
        )
        reducer_bundle = reducer.validate_reducer_bundle(lineage["r6_reducer_bundle"])
        source_frame = r7_canary.validate_source_frame_record(lineage["r7_source_frame_record"])
        require(
            prospective_bundle["source_frame_receipt_sha256"] == source["content_sha256"]
            and prospective_bundle["candidate_frame_record_sha256"] == candidate_record["content_sha256"]
            and reducer_bundle["prospective_bundle_sha256"] == prospective_bundle["content_sha256"]
            and source_frame["physical_frame_id"] == physical_frame_id,
            "R11_PHASE_A_VALIDATION_NESTED_LINEAGE",
            "nested prospective/reducer/source lineage drift",
        )
        frame_r7, frame_r11, abstained = _validate_factor_pair(
            lineage["r7_positive_factor_bundle"], lineage["r11_abstention_bundle"]
        )
        require(
            lineage["r7_positive_factor_bundle"]["source_frame_record_sha256"] == source_frame["content_sha256"]
            and lineage["r11_abstention_bundle"]["source_frame_record_sha256"] == source_frame["content_sha256"],
            "R11_PHASE_A_VALIDATION_NESTED_LINEAGE",
            "nested factor/source hash drift",
        )
        candidate_input_hashes.append(candidate_input["content_sha256"])
        candidate_record_hashes.append(candidate_record["content_sha256"])
        source_receipt_hashes.append(source["content_sha256"])
        source_frame_hashes.append(source_frame["content_sha256"])
        prospective_hashes.append(prospective_bundle["content_sha256"])
        reducer_hashes.append(reducer_bundle["content_sha256"])
        r7_hashes.append(lineage["r7_positive_factor_bundle"]["content_sha256"])
        r11_hashes.append(lineage["r11_abstention_bundle"]["content_sha256"])
        r7_counts.update(frame_r7)
        r11_counts.update(frame_r11)
        abstained_total += abstained
        parent = parent_counts[(parent_id, video_id)]
        parent["r7"].update(frame_r7)
        parent["r11"].update(frame_r11)
        parent["abstained"] += abstained

    states = ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")
    expected_r7 = {state: int(r7_counts[state]) for state in states}
    expected_r11 = {state: int(r11_counts[state]) for state in states}
    expected_parent_counts = [
        {
            "visit_id": identity[0],
            "video_id": identity[1],
            "frame_count": FROZEN_FRAME_COUNTS[index],
            "query_count": FROZEN_FRAME_COUNTS[index] * 9,
            "r7_state_counts": {state: int(parent_counts[identity]["r7"][state]) for state in states},
            "r11_state_counts": {state: int(parent_counts[identity]["r11"][state]) for state in states},
            "r11_abstained_base_positive_count": int(parent_counts[identity]["abstained"]),
        }
        for index, identity in enumerate(parent_counts)
    ]
    expected_ledger = {
        "attempts_by_role": {
            "color": FRAME_COUNT,
            "intrinsics": FRAME_COUNT,
            "lowres_depth": FRAME_COUNT,
            "confidence": FRAME_COUNT,
            "highres_depth": 0,
        },
        "completed_by_role": {
            "color": FRAME_COUNT,
            "intrinsics": FRAME_COUNT,
            "lowres_depth": FRAME_COUNT,
            "confidence": FRAME_COUNT,
            "highres_depth": 0,
        },
    }
    ledger = completion.get("source_payload_read_accounting")
    expected_roles = {"color", "intrinsics", "lowres_depth", "confidence", "highres_depth"}
    require(
        isinstance(ledger, dict)
        and ledger.get("attempts_by_role") == expected_ledger["attempts_by_role"]
        and ledger.get("completed_by_role") == expected_ledger["completed_by_role"]
        and isinstance(ledger.get("bytes_by_role"), dict)
        and set(ledger["bytes_by_role"]) == expected_roles
        and all(int(ledger["bytes_by_role"][role]) > 0 for role in expected_roles - {"highres_depth"})
        and ledger["bytes_by_role"]["highres_depth"] == 0
        and ledger.get("total_zip_member_payload_reads") == 4 * FRAME_COUNT
        and ledger.get("trajectory_payload_reads") == PARENT_COUNT
        and ledger.get("candidate_blob_reloads") == FRAME_COUNT
        and ledger.get("depthart_inferences") == FRAME_COUNT
        and all(ledger.get(key) == 0 for key in ("faro_values_interpreted", "truth_reads", "label_reads", "outcome_reads", "network_requests", "training_steps")),
        "R11_PHASE_A_VALIDATION_READ_LEDGER",
        "source payload read ledger drift",
    )
    sequence_fields = {
        "candidate_input_hash_sequence_sha256": candidate_input_hashes,
        "candidate_record_hash_sequence_sha256": candidate_record_hashes,
        "source_receipt_hash_sequence_sha256": source_receipt_hashes,
        "source_frame_hash_sequence_sha256": source_frame_hashes,
        "prospective_bundle_hash_sequence_sha256": prospective_hashes,
        "r6_reducer_hash_sequence_sha256": reducer_hashes,
        "r7_factor_hash_sequence_sha256": r7_hashes,
        "r11_factor_hash_sequence_sha256": r11_hashes,
    }
    require(
        all(completion.get(field) == adapter.canonical_sha256(values) for field, values in sequence_fields.items())
        and completion.get("parent_count") == PARENT_COUNT
        and completion.get("frame_count") == FRAME_COUNT
        and completion.get("query_count") == QUERY_COUNT
        and completion.get("inventory_content_sha256") == INVENTORY_CONTENT_SHA256
        and completion.get("r7_base_state_counts") == expected_r7
        and completion.get("r11_candidate_state_counts") == expected_r11
        and completion.get("r11_abstained_base_positive_count") == abstained_total
        and completion.get("per_parent_factor_counts") == expected_parent_counts
        and completion.get("highres_depth_member_payload_reads") == 0
        and completion.get("faro_reads") == completion.get("truth_reads") == 0
        and completion.get("clear_output_allowed") is False
        and completion.get("unknown_is_negative") is False
        and completion.get("all_candidate_records_sealed_before_source_features") is True
        and completion.get("all_r7_and_r11_records_sealed_before_parent_scoring") is True
        and completion.get("all_source_records_sealed_before_faro") is True
        and completion.get("r9_parent_scoring_performed") is False
        and completion.get("top24_selection_performed") is False
        and completion.get("training_steps") == completion.get("network_requests") == 0,
        "R11_PHASE_A_VALIDATION_COMPLETION",
        "Phase A completion aggregation/barrier drift",
    )
    require(
        candidate_completion.get("frame_count") == FRAME_COUNT
        and candidate_completion.get("candidate_input_hash_sequence_sha256") == adapter.canonical_sha256(candidate_input_hashes)
        and candidate_completion.get("candidate_record_hash_sequence_sha256") == adapter.canonical_sha256(candidate_record_hashes)
        and candidate_completion.get("payload_read_attempts") == {"color": FRAME_COUNT, "intrinsics": FRAME_COUNT}
        and candidate_completion.get("payload_reads_completed") == {"color": FRAME_COUNT, "intrinsics": FRAME_COUNT}
        and candidate_completion.get("candidate_inference_count") == FRAME_COUNT
        and candidate_completion.get("highres_depth_member_payload_reads") == 0
        and candidate_completion.get("faro_reads") == candidate_completion.get("truth_reads") == 0
        and candidate_completion.get("all_candidates_sealed_before_source_features") is True,
        "R11_PHASE_A_VALIDATION_CANDIDATE_COMPLETION",
        "candidate completion drift",
    )
    require(
        execution.get("expected_parent_count") == PARENT_COUNT
        and execution.get("expected_frame_count") == FRAME_COUNT
        and execution.get("expected_query_count") == QUERY_COUNT
        and execution.get("source_integrity_verification_begins_after_root_creation") is True
        and execution.get("highres_depth_member_payload_read") is False
        and execution.get("faro_payload_read") is False
        and execution.get("truth_scoring") is False
        and execution.get("parent_scoring") is False
        and execution.get("top24_selection") is False
        and execution.get("training_steps") == execution.get("network_requests") == 0
        and execution.get("one_shot_consumed_on_root_creation") is True,
        "R11_PHASE_A_VALIDATION_EXECUTION_RECEIPT",
        "execution receipt drift",
    )
    require(
        result.get("terminal") == PASS_TERMINAL
        and result.get("passed") is True
        and result.get("execution_valid") is True
        and result.get("parent_count") == PARENT_COUNT
        and result.get("frame_count") == FRAME_COUNT
        and result.get("query_count") == QUERY_COUNT
        and result.get("candidate_inference_count") == FRAME_COUNT
        and result.get("r7_base_state_counts") == expected_r7
        and result.get("r11_candidate_state_counts") == expected_r11
        and result.get("r11_abstained_base_positive_count") == abstained_total
        and result.get("phase_a_completion_sha256") == completion["content_sha256"]
        and len(runtime_identity_hashes) == 1
        and result.get("runtime_identity_sha256") == next(iter(runtime_identity_hashes))
        and result.get("highres_depth_member_payload_reads") == result.get("faro_reads") == 0
        and result.get("truth_scoring") is False
        and result.get("r9_parent_scoring_performed") is False
        and result.get("top24_selection_performed") is False
        and result.get("clear_output_allowed") is False
        and result.get("unknown_is_negative") is False
        and result.get("training_steps") == result.get("network_requests") == 0
        and result.get("one_shot_consumed") is True,
        "R11_PHASE_A_VALIDATION_RESULT",
        "Phase A result drift",
    )
    return {
        "passed": True,
        "terminal": PASS_TERMINAL,
        "producer_module_imported": False,
        "parent_count": PARENT_COUNT,
        "frame_count": FRAME_COUNT,
        "query_count": QUERY_COUNT,
        "root_file_count": PRE_MANIFEST_FILE_COUNT + 1,
        "r7_base_state_counts": expected_r7,
        "r11_candidate_state_counts": expected_r11,
        "r11_abstained_base_positive_count": abstained_total,
        "highres_depth_member_payload_reads": 0,
        "faro_reads": 0,
        "truth_reads": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=None)
    parser.add_argument("--execution-lock", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        result = validate_evidence(args.evidence_root, args.execution_lock)
    except Exception as error:
        print(
            json.dumps(
                {
                    "passed": False,
                    "failure_code": str(getattr(error, "code", type(error).__name__)),
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
