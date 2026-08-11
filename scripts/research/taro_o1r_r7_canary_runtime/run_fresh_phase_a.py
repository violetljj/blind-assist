#!/usr/bin/env python3
"""Run sealed TARO R7 fresh candidate inference and label-blind Phase A."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import subprocess
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
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_factor_headroom_runtime import candidate_inputs
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer
from scripts.research.taro_o1r_reducer_integration_runtime import reducer_integration as r6_reducer
from scripts.research.taro_o1r_reducer_integration_runtime.locked_uncertainty import load_locked_uncertainty_model
from scripts.research.taro_o1r_r7_canary_runtime import fresh_confirmation_cohort as cohort
from scripts.research.taro_o1r_r7_canary_runtime import r7_canary
from scripts.research.taro_o1r_r7_canary_runtime import run_fresh_inventory
from scripts.research.taro_o1r_r7_canary_runtime import validate_fresh_confirmation_protocol


REPO_ROOT = Path(__file__).resolve().parents[3]
LOCK_SCHEMA = "blindassist.taro.o1r.r7_fresh_phase_a_execution_lock.v1"
LOCK_ID = "TARO_O1R_R7_FRESH_CONFIRMATION_PHASE_A_SOURCE_AND_MODEL_ONE_SHOT_EXECUTION_LOCK"
INVENTORY_PATH = "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-inventory-r0/exact-frame-plan.json"
OUTPUT_ROOT = "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-phase-a-r0"
PASS_TERMINAL = "TARO_O1R_R7_FRESH_CONFIRMATION_PHASE_A_SEALED_PASS"
FAIL_TERMINAL = "TARO_O1R_R7_FRESH_CONFIRMATION_PHASE_A_EXECUTION_INVALID"
FRAME_COUNT = 170
QUERY_COUNT = FRAME_COUNT * 9
FROZEN_POSITIVE_INDEX = (0, 0, 2)

EXPECTED_BINDINGS = {
    "R7_FRESH_PROTOCOL": "docs/research/taro/TARO_O1R_R7_FRESH_PARENT_DISJOINT_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json",
    "R7_DATA_LOCK": "docs/research/taro/TARO_O1R_R7_FRESH_CONFIRMATION_COHORT_AND_DATA_USE_LOCK_2026-08-12.json",
    "R7_DOWNLOAD_RECEIPTS": "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-source-r0/download-receipts.json",
    "R7_DOWNLOAD_RESULT": "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-source-r0/result.json",
    "R7_INVENTORY_PLAN": INVENTORY_PATH,
    "R7_INVENTORY_RESULT": "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-inventory-r0/result.json",
    "R7_INVENTORY_MANIFEST": "artifacts.local/evidence/taro/o1r-r7-fresh-confirmation-inventory-r0/manifest.json",
    "R7_INVENTORY_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_inventory.py",
    "DEPTHART_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/depthart_runner.py",
    "CANDIDATE_INPUT_RUNTIME": "scripts/research/taro_o0r_factor_headroom_runtime/candidate_inputs.py",
    "PROSPECTIVE_RUNTIME": "scripts/research/taro_o0r_candidate_scale_runtime/prospective_factor_runtime.py",
    "R6_REDUCER_RUNTIME": "scripts/research/taro_o1r_reducer_integration_runtime/reducer_integration.py",
    "LOCKED_UNCERTAINTY_LOADER": "scripts/research/taro_o1r_reducer_integration_runtime/locked_uncertainty.py",
    "LOCKED_UNCERTAINTY_ARTIFACT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-artifact.json.gz",
    "LOCKED_UNCERTAINTY_RECEIPT": "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-receipt.json",
    "R7_CANARY_RUNTIME": "scripts/research/taro_o1r_r7_canary_runtime/r7_canary.py",
    "EVIDENCE_WRITER": "scripts/research/taro_o0r_factor_headroom_runtime/evidence.py",
    "PHASE_A_RUNNER": "scripts/research/taro_o1r_r7_canary_runtime/run_fresh_phase_a.py",
}
EXPECTED_AUTHORITY = {
    "source_frame_decode": True,
    "depthart_inference": True,
    "candidate_inference_count": FRAME_COUNT,
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


class FreshPhaseAError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FreshPhaseAError(code, message, **context)


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(adapter.canonical_json_bytes(dict(value)).decode("utf-8"))
    require("content_sha256" not in record, "R7_PHASE_A_SEAL_COLLISION", "caller supplied a content seal")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R7_PHASE_A_RECORD_INVALID", "sealed record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(
        record.get("schema") == schema and isinstance(observed, str) and adapter.canonical_sha256(record) == observed,
        "R7_PHASE_A_RECORD_HASH_DRIFT",
        "sealed record hash/schema drift",
        schema=schema,
    )
    record["content_sha256"] = observed
    return record


def _repo_path(relative: str) -> Path:
    return materializer.safe_join(REPO_ROOT, relative)


def _candidate_input_relative(frame: r6io.R6FrameRef) -> str:
    return f"candidate-inputs/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def _candidate_blob_relative(frame: r6io.R6FrameRef) -> str:
    return f"candidates/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.depth.npy.gz"


def _candidate_record_relative(frame: r6io.R6FrameRef) -> str:
    return f"candidates/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def _source_receipt_relative(frame: r6io.R6FrameRef) -> str:
    return f"phase-a-sources/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def _lineage_relative(frame: r6io.R6FrameRef) -> str:
    return f"phase-a-lineage/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def _load_frames(inventory_path: Path) -> list[r6io.R6FrameRef]:
    inventory = run_fresh_inventory.validate_inventory(json.loads(inventory_path.read_text(encoding="utf-8")))
    frames: list[r6io.R6FrameRef] = []
    for parent, expected, expected_count in zip(
        inventory["parents"], cohort.EXPECTED_ROSTER, run_fresh_inventory.EXPECTED_FRAME_COUNTS, strict=True
    ):
        identity = (str(parent["visit_id"]), str(parent["video_id"]))
        require(identity == expected[:2], "R7_PHASE_A_ROSTER_DRIFT", "fresh inventory roster drift")
        bindings = parent["container_bindings"]
        up_path = _repo_path(bindings["upsampling"]["path"])
        intr_path = _repo_path(bindings["intrinsics"]["path"])
        traj_path = _repo_path(bindings["trajectory"]["path"])
        r6io._verify_container(up_path, bindings["upsampling"])
        r6io._verify_container(intr_path, bindings["intrinsics"])
        r6io._verify_container(traj_path, bindings["trajectory"])
        up_index = r6io._index_upsampling(up_path, identity[1])
        intr_index = r6io._index_intrinsics(intr_path, identity[1])
        trajectory = tuple(materializer.parse_trajectory_payload(traj_path.read_bytes()))
        tokens = parent["frame_plan"]["exact_timestamp_tokens"]
        require(len(tokens) == expected_count, "R7_PHASE_A_FRAME_COUNT_DRIFT", "fresh parent frame count drift")
        for token in tokens:
            require(
                all(token in up_index[role] for role in up_index) and token in intr_index,
                "R7_PHASE_A_MEMBER_MISSING",
                "fresh exact member is absent",
                token=token,
            )
            members = {role: up_index[role][token] for role in up_index}
            members["intrinsics"] = intr_index[token]
            frames.append(
                r6io.R6FrameRef(
                    identity[0], identity[1], token, f"{identity[1]}:{token}", up_path, intr_path, traj_path,
                    {key: dict(value) for key, value in bindings.items()}, trajectory, members,
                )
            )
    require(len(frames) == FRAME_COUNT, "R7_PHASE_A_COHORT_DRIFT", "fresh cohort is not exact 8/170")
    require(len({(row.parent_id, row.video_id, row.timestamp_token) for row in frames}) == FRAME_COUNT, "R7_PHASE_A_KEY_DUPLICATE", "fresh frame key duplicated")
    return frames


def _member_binding(frame: r6io.R6FrameRef, role: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    container_role = "intrinsics" if role == "intrinsics" else "upsampling"
    return {"container_sha256": frame.container_bindings[container_role]["sha256"], **dict(binding)}


def _read_candidate_input(
    frame: r6io.R6FrameRef,
    up_bundle: zipfile.ZipFile,
    intr_bundle: zipfile.ZipFile,
    reads: Counter[str],
) -> tuple[dict[str, Any], np.ndarray]:
    def observed(role: str, _: str) -> None:
        require(role in {"color", "intrinsics"}, "R7_CANDIDATE_PAYLOAD_FIREWALL", "candidate read a forbidden payload", role=role)
        reads[role] += 1

    color_payload, color_binding = r6io._read_member(up_bundle, frame.members["color"], observer=observed)
    intr_payload, intr_binding = r6io._read_member(intr_bundle, frame.members["intrinsics"], observer=observed)
    color = np.ascontiguousarray(candidate_inputs._decode_color(color_payload))
    low = materializer.parse_pincam_payload(intr_payload)
    high = adapter.scale_lowres_intrinsics(low)
    transform, pose = adapter.interpolate_camera_to_world_exact(frame.trajectory_rows, frame.timestamp_token)
    gravity = adapter._normalize_vector(transform[2, :3], "R7_FRESH_GRAVITY_INVALID")
    record = _seal(
        {
            "schema": "blindassist.taro.o1r.r7_fresh_candidate_input.v1",
            "analysis_role": "FRESH_CONFIRMATION",
            "parent_id": frame.parent_id,
            "video_id": frame.video_id,
            "timestamp_token": frame.timestamp_token,
            "physical_frame_id": frame.physical_frame_id,
            "color_binding": _member_binding(frame, "color", color_binding),
            "intrinsics_binding": _member_binding(frame, "intrinsics", intr_binding),
            "trajectory_binding": dict(frame.container_bindings["trajectory"]),
            "color_decoded_sha256": adapter.canonical_sha256(color),
            "lowres_intrinsics": low,
            "intrinsics_highres": high,
            "camera_to_world_4x4": transform.tolist(),
            "gravity_up_camera_xyz": gravity.tolist(),
            "sensor_timestamp_ns": pose["frame_timestamp_ns"],
            "max_source_timestamp_ns": pose["max_source_timestamp_ns"],
            "allowed_model_inputs": ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"],
            "faro_payload_read": False,
            "truth_payload_read": False,
            "prior_outcome_read": False,
        }
    )
    return record, color


def _run_candidate(
    model: Any,
    runtime_identity: Mapping[str, Any],
    candidate_input: Mapping[str, Any],
    color: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    matrix = np.asarray(candidate_input["intrinsics_highres"]["matrix_3x3"], dtype=np.float32)
    tensor, resized_k = depthart_runner.preprocess_depthart_input(color, matrix)
    with torch.inference_mode():
        prediction = model(torch.from_numpy(tensor).to("cuda"), torch.from_numpy(resized_k).to("cuda"))
    native_batch = prediction.detach().float().cpu().numpy()
    require(native_batch.shape == (1, *depthart_runner.NATIVE_SHAPE_HW), "R7_NATIVE_DEPTH_INVALID", "DepthART native shape drift")
    native = np.ascontiguousarray(native_batch[0], dtype=np.float32)
    require(bool(np.all(np.isfinite(native))), "R7_NATIVE_DEPTH_INVALID", "DepthART output contains non-finite values")
    highres = depthart_runner.upsample_native_depth(native)
    inference = _seal(
        {
            "schema": "blindassist.taro.o1r.r7_fresh_depthart_inference.v1",
            "analysis_role": "FRESH_CONFIRMATION",
            "model_id": adapter.BASELINE_MODEL_ID,
            "checkpoint_sha256": adapter.BASELINE_CHECKPOINT_SHA256,
            "preprocess_id": depthart_runner.PREPROCESS_ID,
            "postprocess_id": depthart_runner.POSTPROCESS_ID,
            "candidate_input_sha256": candidate_input["content_sha256"],
            "parent_id": candidate_input["parent_id"],
            "video_id": candidate_input["video_id"],
            "timestamp_token": candidate_input["timestamp_token"],
            "physical_frame_id": candidate_input["physical_frame_id"],
            "input_tensor_sha256": adapter.canonical_sha256(tensor),
            "resized_intrinsics_sha256": adapter.canonical_sha256(resized_k),
            "native_depth_sha256": adapter.canonical_sha256(native),
            "highres_depth_sha256": adapter.canonical_sha256(highres),
            "runtime_identity": dict(runtime_identity),
            "truth_alignment_used": False,
            "faro_payload_read": False,
        }
    )
    return native, highres, inference


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_candidate(root: Path, frame: r6io.R6FrameRef) -> tuple[dict[str, Any], dict[str, Any], np.ndarray, np.ndarray]:
    candidate_input = _validate_seal(_load_json(root / _candidate_input_relative(frame)), "blindassist.taro.o1r.r7_fresh_candidate_input.v1")
    record = _validate_seal(_load_json(root / _candidate_record_relative(frame)), "blindassist.taro.o1r.r7_fresh_candidate_frame.v1")
    require(record["candidate_input_sha256"] == candidate_input["content_sha256"] and record["physical_frame_id"] == frame.physical_frame_id, "R7_CANDIDATE_LINEAGE_DRIFT", "candidate lineage drift")
    blob = record["native_depth_blob"]
    payload = (root / blob["path"]).read_bytes()
    require(len(payload) == blob["bytes"] and materializer.sha256_bytes(payload) == blob["sha256"], "R7_CANDIDATE_BLOB_DRIFT", "candidate blob hash drift")
    native = np.ascontiguousarray(depthart_runner.decode_npy_gzip_bytes(payload), dtype=np.float32)
    require(native.shape == depthart_runner.NATIVE_SHAPE_HW and adapter.canonical_sha256(native) == blob["array_sha256"], "R7_CANDIDATE_ARRAY_DRIFT", "candidate native array drift")
    highres = depthart_runner.upsample_native_depth(native)
    require(adapter.canonical_sha256(highres) == record["inference_receipt"]["highres_depth_sha256"], "R7_CANDIDATE_HIGHRES_DRIFT", "candidate highres replay drift")
    return candidate_input, record, native, highres


def _positive_state(feature: Mapping[str, Any]) -> str:
    if feature["r6_state"] == "OCCUPIED_OBSERVED":
        return "OCCUPIED_OBSERVED"
    if feature["query_receipt"] is None:
        return "UNKNOWN"
    i, j, k = FROZEN_POSITIVE_INDEX
    return "OCCUPIED_OBSERVED" if bool(feature["occupied_hits"][i][j][k]) else "UNKNOWN"


def validate_execution_lock(path: Path) -> dict[str, Any]:
    lock_path = path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(
        lock.get("schema") == LOCK_SCHEMA and lock.get("lock_id") == LOCK_ID
        and lock.get("status") == "AUTHORIZED_UNCONSUMED" and lock.get("consumed") is False,
        "R7_PHASE_A_LOCK_IDENTITY", "Phase-A execution lock identity drift",
    )
    actual_argv = [Path(sys.argv[0]).resolve().relative_to(REPO_ROOT).as_posix(), "--execution-lock", lock_path.relative_to(REPO_ROOT).as_posix()]
    require(lock.get("argv") == actual_argv, "R7_PHASE_A_ARGV_DRIFT", "Phase-A argv drift")
    require(lock.get("inventory_path") == INVENTORY_PATH and lock.get("output_root") == OUTPUT_ROOT and lock.get("overwrite") is False and lock.get("rerun") is False, "R7_PHASE_A_ROOT_DRIFT", "Phase-A root policy drift")
    bindings = lock.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R7_PHASE_A_BINDINGS", "Phase-A binding count drift")
    verified: dict[str, dict[str, Any]] = {}
    for row in bindings:
        role, relative = row.get("role"), row.get("path")
        require(set(row) == {"role", "path", "bytes", "sha256"} and role not in verified and EXPECTED_BINDINGS.get(role) == relative, "R7_PHASE_A_BINDING_ROW", "Phase-A binding row drift")
        target = _repo_path(relative)
        require(target.is_file() and target.stat().st_size == row["bytes"] and materializer.sha256_file(target) == row["sha256"], "R7_PHASE_A_BINDING_HASH", f"Phase-A binding drift: {relative}")
        verified[role] = row
    require(validate_fresh_confirmation_protocol.validate(_repo_path(EXPECTED_BINDINGS["R7_FRESH_PROTOCOL"]))["passed"], "R7_PHASE_A_PROTOCOL_INVALID", "fresh confirmation protocol is invalid")
    inventory = run_fresh_inventory.validate_inventory(_load_json(_repo_path(INVENTORY_PATH)))
    require(inventory["exact_pose_bounded_frame_count"] == FRAME_COUNT, "R7_PHASE_A_INVENTORY_INVALID", "fresh inventory count drift")
    require(lock.get("execution_authority") == EXPECTED_AUTHORITY, "R7_PHASE_A_AUTHORITY_DRIFT", "Phase-A authority drift")
    require(lock.get("resource_budget") == {"maximum_wall_seconds": 28800, "maximum_peak_rss_bytes": 17179869184, "maximum_cuda_allocated_bytes": 8500000000, "maximum_evidence_bytes": 2147483648}, "R7_PHASE_A_BUDGET_DRIFT", "Phase-A resource budget drift")
    identity = lock.get("candidate_identity", {})
    source = Path(identity.get("source_root", "")).resolve()
    checkpoint = Path(identity.get("checkpoint_path", "")).resolve()
    require(identity.get("model_id") == adapter.BASELINE_MODEL_ID and identity.get("source_commit") == depthart_runner.EXPECTED_SOURCE_GIT_COMMIT and identity.get("checkpoint_sha256") == adapter.BASELINE_CHECKPOINT_SHA256 and identity.get("preprocess_id") == depthart_runner.PREPROCESS_ID and identity.get("postprocess_id") == depthart_runner.POSTPROCESS_ID, "R7_PHASE_A_CANDIDATE_IDENTITY", "candidate identity drift")
    require(source.is_dir() and checkpoint.is_file() and checkpoint.stat().st_size == identity.get("checkpoint_bytes") and materializer.sha256_file(checkpoint) == identity.get("checkpoint_sha256"), "R7_PHASE_A_CANDIDATE_ASSET", "candidate source/checkpoint drift")
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(source), "status", "--short"], capture_output=True, text=True, check=True).stdout.strip()
    require(commit == identity["source_commit"] and not dirty, "R7_PHASE_A_CANDIDATE_SOURCE_DRIFT", "candidate source tree drift")
    require(not _repo_path(OUTPUT_ROOT).exists(), "R7_PHASE_A_ROOT_COLLISION", "Phase-A evidence root exists")
    lock["_lock_path"], lock["_source_root"], lock["_checkpoint_path"] = lock_path, source, checkpoint
    return lock


def _write_failure(writer: FactorEvidenceWriter, error: Exception) -> None:
    try:
        writer.write_json("failure.json", {"schema": "blindassist.taro.o1r.r7_fresh_phase_a_failure.v1", "terminal": FAIL_TERMINAL, "execution_valid": False, "failure_code": str(getattr(error, "code", type(error).__name__)), "message": str(error), "one_shot_consumed": True})
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r7_fresh_phase_a_manifest.v1", "terminal": FAIL_TERMINAL, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
    except Exception:
        pass


def execute(lock_path: Path) -> dict[str, Any]:
    lock = validate_execution_lock(lock_path)
    output = _repo_path(OUTPUT_ROOT)
    budget = lock["resource_budget"]
    writer = FactorEvidenceWriter(output, int(budget["maximum_evidence_bytes"]))
    started = time.monotonic()
    process = psutil.Process()

    def guard() -> None:
        require(time.monotonic() - started <= float(budget["maximum_wall_seconds"]), "R7_PHASE_A_TIMEOUT", "Phase-A wall budget exceeded")
        require(process.memory_info().rss <= int(budget["maximum_peak_rss_bytes"]), "R7_PHASE_A_RSS_EXCEEDED", "Phase-A RSS budget exceeded")

    writer.activate(
        {
            "schema": "blindassist.taro.o1r.r7_fresh_phase_a_execution_receipt.v1",
            "execution_lock_sha256": materializer.sha256_file(lock["_lock_path"]),
            "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "analysis_role": "FRESH_CONFIRMATION",
            "expected_parent_count": 8,
            "expected_frame_count": FRAME_COUNT,
            "expected_query_count": QUERY_COUNT,
            "faro_payload_read": False,
            "training_steps": 0,
            "network_requests": 0,
            "one_shot_consumed_on_root_creation": True,
        }
    )
    try:
        frames = _load_frames(_repo_path(INVENTORY_PATH))
        model, runtime_identity = depthart_runner.load_official_depthart(lock["_source_root"], lock["_checkpoint_path"], device="cuda", seed=0)
        candidate_reads: Counter[str] = Counter()
        candidate_hashes: list[str] = []
        candidate_input_hashes: list[str] = []
        completed = 0
        for parent_key, parent_frames_iter in groupby(frames, key=lambda row: (row.parent_id, row.video_id)):
            parent_frames = list(parent_frames_iter)
            with zipfile.ZipFile(parent_frames[0].upsampling_archive) as up_bundle, zipfile.ZipFile(parent_frames[0].intrinsics_archive) as intr_bundle:
                for frame in parent_frames:
                    candidate_input, color = _read_candidate_input(frame, up_bundle, intr_bundle, candidate_reads)
                    writer.write_json(_candidate_input_relative(frame), candidate_input)
                    native, _highres, inference = _run_candidate(model, runtime_identity, candidate_input, color)
                    blob_payload = depthart_runner.deterministic_npy_gzip_bytes(native)
                    blob_receipt = writer.write_bytes(_candidate_blob_relative(frame), blob_payload)
                    blob = {**blob_receipt, "array_sha256": adapter.canonical_sha256(native), "shape_hw": list(native.shape), "dtype": "float32", "encoding": "DETERMINISTIC_GZIP_NPY_MTIME_0"}
                    candidate = _seal(
                        {
                            "schema": "blindassist.taro.o1r.r7_fresh_candidate_frame.v1",
                            "analysis_role": "FRESH_CONFIRMATION",
                            "parent_id": frame.parent_id,
                            "video_id": frame.video_id,
                            "timestamp_token": frame.timestamp_token,
                            "physical_frame_id": frame.physical_frame_id,
                            "candidate_input_sha256": candidate_input["content_sha256"],
                            "inference_receipt": inference,
                            "native_depth_blob": blob,
                            "faro_payload_read": False,
                            "truth_alignment_used": False,
                        }
                    )
                    writer.write_json(_candidate_record_relative(frame), candidate)
                    candidate_input_hashes.append(candidate_input["content_sha256"])
                    candidate_hashes.append(candidate["content_sha256"])
                    completed += 1
                    guard()
                    if completed % 10 == 0 or completed == FRAME_COUNT:
                        print(json.dumps({"phase": "R7_FRESH_CANDIDATE", "completed": completed, "total": FRAME_COUNT, "physical_frame_id": frame.physical_frame_id}, sort_keys=True), flush=True)
        require(candidate_reads == Counter({"color": FRAME_COUNT, "intrinsics": FRAME_COUNT}), "R7_CANDIDATE_READ_COUNT_DRIFT", "candidate read counts drift", reads=dict(candidate_reads))
        writer.write_json(
            "candidate-completion.json",
            _seal(
                {
                    "schema": "blindassist.taro.o1r.r7_fresh_candidate_completion.v1",
                    "frame_count": FRAME_COUNT,
                    "candidate_input_hash_sequence_sha256": adapter.canonical_sha256(candidate_input_hashes),
                    "candidate_record_hash_sequence_sha256": adapter.canonical_sha256(candidate_hashes),
                    "payload_reads": dict(sorted(candidate_reads.items())),
                    "faro_reads": 0,
                    "all_candidates_sealed_before_source_features": True,
                }
            ),
        )
        try:
            import torch
            cuda_peak = int(torch.cuda.max_memory_allocated())
            del model
            torch.cuda.empty_cache()
        except Exception:
            cuda_peak = 0
        require(cuda_peak <= int(budget["maximum_cuda_allocated_bytes"]), "R7_PHASE_A_CUDA_EXCEEDED", "candidate CUDA budget exceeded")

        uncertainty_model = load_locked_uncertainty_model()
        source_reads: Counter[str] = Counter()
        source_hashes: list[str] = []
        prospective_hashes: list[str] = []
        reducer_hashes: list[str] = []
        state_counts: Counter[str] = Counter()
        parent_state_counts: dict[str, Counter[str]] = {parent: Counter() for parent, _, _ in cohort.EXPECTED_ROSTER}
        completed = 0

        def observed_source(role: str, _: str) -> None:
            require(role in {"lowres_depth", "confidence"}, "R7_SOURCE_PAYLOAD_FIREWALL", "source feature phase read a forbidden payload", role=role)
            source_reads[role] += 1

        for parent_key, parent_frames_iter in groupby(frames, key=lambda row: (row.parent_id, row.video_id)):
            parent_frames = list(parent_frames_iter)
            with zipfile.ZipFile(parent_frames[0].upsampling_archive) as up_bundle:
                for frame in parent_frames:
                    candidate_input, candidate, _native, highres = _load_candidate(output, frame)
                    apple_payload, apple_binding = r6io._read_member(up_bundle, frame.members["lowres_depth"], observer=observed_source)
                    conf_payload, conf_binding = r6io._read_member(up_bundle, frame.members["confidence"], observer=observed_source)
                    apple = np.ascontiguousarray(materializer._decode_png(apple_payload, "lowres_depth"))
                    confidence = np.ascontiguousarray(materializer._decode_png(conf_payload, "confidence"))
                    source = _seal(
                        {
                            "schema": "blindassist.taro.o1r.r7_fresh_source_frame_receipt.v1",
                            "analysis_role": "FRESH_CONFIRMATION",
                            "parent_id": frame.parent_id,
                            "video_id": frame.video_id,
                            "timestamp_token": frame.timestamp_token,
                            "physical_frame_id": frame.physical_frame_id,
                            "candidate_input_sha256": candidate_input["content_sha256"],
                            "candidate_frame_record_sha256": candidate["content_sha256"],
                            "apple_depth_binding": _member_binding(frame, "lowres_depth", apple_binding),
                            "confidence_binding": _member_binding(frame, "confidence", conf_binding),
                            "apple_depth_sha256": adapter.canonical_sha256(apple),
                            "confidence_sha256": adapter.canonical_sha256(confidence),
                            "intrinsics_highres": candidate_input["intrinsics_highres"],
                            "lowres_intrinsics": candidate_input["lowres_intrinsics"],
                            "gravity_up_camera_xyz": candidate_input["gravity_up_camera_xyz"],
                            "max_source_timestamp_ns": candidate_input["max_source_timestamp_ns"],
                            "faro_payload_read": False,
                            "truth_payload_read": False,
                        }
                    )
                    writer.write_json(_source_receipt_relative(frame), source)
                    low = candidate_input["lowres_intrinsics"]
                    low_matrix = [[float(low["fx"]), 0.0, float(low["cx"])], [0.0, float(low["fy"]), float(low["cy"])], [0.0, 0.0, 1.0]]
                    bundle = prospective.build_prospective_factor_bundle(
                        parent_id=frame.parent_id,
                        video_id=frame.video_id,
                        timestamp_token=frame.timestamp_token,
                        source_frame_receipt_sha256=source["content_sha256"],
                        candidate_frame_record_sha256=candidate["content_sha256"],
                        max_source_timestamp_ns=int(source["max_source_timestamp_ns"]),
                        candidate_highres_depth_m=highres,
                        apple_depth_mm=apple,
                        confidence=confidence,
                        intrinsics_highres_3x3=source["intrinsics_highres"]["matrix_3x3"],
                        intrinsics_apple_3x3=low_matrix,
                        gravity_up_camera_xyz=source["gravity_up_camera_xyz"],
                    )
                    prior = r6_reducer._integrate_with_validated_model(
                        prospective_bundle=bundle,
                        candidate_highres_depth_m=highres,
                        confidence=confidence,
                        intrinsics_apple_3x3=low_matrix,
                        uncertainty_model=uncertainty_model,
                    )
                    source_features = r7_canary.build_source_frame_record(
                        bundle,
                        highres,
                        apple,
                        confidence,
                        low_matrix,
                        source["intrinsics_highres"]["matrix_3x3"],
                        prior,
                    )
                    writer.write_json_gzip(
                        _lineage_relative(frame),
                        {"prospective_bundle": bundle, "r6_reducer_bundle": prior, "r7_source_frame_record": source_features},
                    )
                    source_hashes.append(source_features["content_sha256"])
                    prospective_hashes.append(bundle["content_sha256"])
                    reducer_hashes.append(prior["content_sha256"])
                    for feature in source_features["query_features"]:
                        state = _positive_state(feature)
                        state_counts[state] += 1
                        parent_state_counts[frame.parent_id][state] += 1
                    completed += 1
                    guard()
                    if completed % 10 == 0 or completed == FRAME_COUNT:
                        print(json.dumps({"phase": "R7_FRESH_SOURCE_FEATURES", "completed": completed, "total": FRAME_COUNT, "physical_frame_id": frame.physical_frame_id}, sort_keys=True), flush=True)
        require(source_reads == Counter({"lowres_depth": FRAME_COUNT, "confidence": FRAME_COUNT}), "R7_SOURCE_READ_COUNT_DRIFT", "source feature read counts drift", reads=dict(source_reads))
        require(sum(state_counts.values()) == QUERY_COUNT and state_counts["CLEAR_OBSERVED"] == 0, "R7_SOURCE_STATE_DRIFT", "fresh source candidate emitted CLEAR or lost queries")
        completion = _seal(
            {
                "schema": "blindassist.taro.o1r.r7_fresh_phase_a_completion.v1",
                "parent_count": 8,
                "frame_count": FRAME_COUNT,
                "query_count": QUERY_COUNT,
                "source_frame_hash_sequence_sha256": adapter.canonical_sha256(source_hashes),
                "prospective_bundle_hash_sequence_sha256": adapter.canonical_sha256(prospective_hashes),
                "r6_reducer_hash_sequence_sha256": adapter.canonical_sha256(reducer_hashes),
                "frozen_positive_candidate_state_counts": {state: int(state_counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")},
                "per_parent_state_counts": {parent: {state: int(counts[state]) for state in ("CLEAR_OBSERVED", "OCCUPIED_OBSERVED", "UNKNOWN")} for parent, counts in parent_state_counts.items()},
                "candidate_payload_reads": dict(sorted(candidate_reads.items())),
                "source_payload_reads": dict(sorted(source_reads.items())),
                "faro_reads": 0,
                "truth_reads": 0,
                "clear_output_allowed": False,
                "all_source_records_sealed_before_faro": True,
                "training_steps": 0,
                "network_requests": 0,
            }
        )
        writer.write_json("phase-a-completion.json", completion)
        reloaded = _validate_seal(_load_json(output / "phase-a-completion.json"), "blindassist.taro.o1r.r7_fresh_phase_a_completion.v1")
        require(reloaded == completion and reloaded["faro_reads"] == 0, "R7_PHASE_A_COMPLETION_RELOAD_DRIFT", "Phase-A completion reload drift")
        result = {
            "schema": "blindassist.taro.o1r.r7_fresh_phase_a_result.v1",
            "terminal": PASS_TERMINAL,
            "passed": True,
            "execution_valid": True,
            "parent_count": 8,
            "frame_count": FRAME_COUNT,
            "query_count": QUERY_COUNT,
            "candidate_inference_count": FRAME_COUNT,
            "frozen_positive_candidate_state_counts": completion["frozen_positive_candidate_state_counts"],
            "phase_a_completion_sha256": completion["content_sha256"],
            "runtime_identity_sha256": adapter.canonical_sha256(runtime_identity),
            "faro_reads": 0,
            "truth_scoring": False,
            "clear_output_allowed": False,
            "training_steps": 0,
            "network_requests": 0,
            "cuda_peak_allocated_bytes": cuda_peak,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "one_shot_consumed": True,
            "claim_ceiling": "Sealed fresh source-only positive-occupancy features; no FARO label, effectiveness, deployment, product, or safety evidence.",
        }
        writer.write_json("result.json", result)
        writer.write_json("manifest.json", {"schema": "blindassist.taro.o1r.r7_fresh_phase_a_manifest.v1", "terminal": PASS_TERMINAL, "files": dict(sorted(writer.file_receipts.items())), "file_count_before_manifest": len(writer.file_receipts), "bytes_before_manifest": writer.bytes_written})
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
    print(json.dumps({"terminal": result["terminal"], "passed": result["passed"], "execution_valid": result["execution_valid"], "frame_count": result["frame_count"], "query_count": result["query_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
