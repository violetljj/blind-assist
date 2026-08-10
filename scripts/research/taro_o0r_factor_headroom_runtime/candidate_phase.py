#!/usr/bin/env python3
"""Seal every exact eval DepthART candidate before any per-frame truth is read."""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_factor_headroom_runtime.candidate_inputs import iter_candidate_inputs
from scripts.research.taro_o0r_factor_headroom_runtime.depthart_runner import (
    NATIVE_SHAPE_HW,
    deterministic_npy_gzip_bytes,
    decode_npy_gzip_bytes,
    infer_depthart_candidate,
    validate_candidate_input_receipt,
    validate_depthart_inference_receipt,
    sha256_file,
)
from scripts.research.taro_o0r_factor_headroom_runtime.evidence import FactorEvidenceWriter
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


CANDIDATE_FRAME_RECORD_SCHEMA = "blindassist.taro.o0r.depthart_candidate_frame_record.v1"
CANDIDATE_PHASE_COMPLETION_SCHEMA = "blindassist.taro.o0r.depthart_candidate_phase_completion.v1"


class CandidatePhaseError(RuntimeError):
    """Stable candidate-phase error."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise CandidatePhaseError(code, message, **context)


def _canonical_copy(value: Any) -> Any:
    return json.loads(adapter.canonical_json_bytes(value).decode("utf-8"))


def seal_record(value: Mapping[str, Any]) -> dict[str, Any]:
    record = _canonical_copy(dict(value))
    require("content_sha256" not in record, "CANDIDATE_PHASE_SEAL_COLLISION", "caller supplied a content hash")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return _canonical_copy(record)


def validate_candidate_phase_completion(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "CANDIDATE_PHASE_COMPLETION_INVALID", "candidate phase completion must be an object")
    completion = _canonical_copy(value)
    observed = completion.pop("content_sha256", None)
    require(isinstance(observed, str) and adapter.canonical_sha256(completion) == observed, "CANDIDATE_PHASE_COMPLETION_INVALID", "candidate phase completion seal drift")
    completion["content_sha256"] = observed
    expected = {
        "schema",
        "candidate_frame_count",
        "candidate_frame_sequence_sha256",
        "candidate_frame_record_hashes_sha256",
        "parent_frame_counts",
        "runtime_identity",
        "runtime_identity_sha256",
        "truth_frame_packages_opened_before_completion",
        "truth_payload_read_by_candidate_phase",
        "truth_alignment_used_by_candidate_phase",
        "candidate_outputs_sealed_before_truth_join",
        "content_sha256",
    }
    require(set(completion) == expected and completion["schema"] == CANDIDATE_PHASE_COMPLETION_SCHEMA, "CANDIDATE_PHASE_COMPLETION_INVALID", "candidate phase completion fields/schema drift")
    require(isinstance(completion["candidate_frame_count"], int) and not isinstance(completion["candidate_frame_count"], bool) and completion["candidate_frame_count"] > 0, "CANDIDATE_PHASE_COMPLETION_INVALID", "candidate frame count is invalid")
    require(completion["runtime_identity_sha256"] == adapter.canonical_sha256(completion["runtime_identity"]), "CANDIDATE_PHASE_COMPLETION_INVALID", "candidate runtime identity hash drift")
    require(
        completion["truth_frame_packages_opened_before_completion"] == 0
        and completion["truth_payload_read_by_candidate_phase"] is False
        and completion["truth_alignment_used_by_candidate_phase"] is False
        and completion["candidate_outputs_sealed_before_truth_join"] is True,
        "CANDIDATE_PHASE_TRUTH_FIREWALL_BREACH",
        "candidate outputs were not sealed behind the truth firewall",
    )
    return completion


def validate_candidate_frame_record(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "CANDIDATE_FRAME_RECORD_INVALID", "candidate frame record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(isinstance(observed, str) and adapter.canonical_sha256(record) == observed, "CANDIDATE_FRAME_RECORD_INVALID", "candidate frame record seal drift")
    record["content_sha256"] = observed
    expected = {
        "schema",
        "candidate_input_receipt",
        "inference_receipt",
        "native_depth_blob",
        "candidate_phase_truth_payload_read",
        "candidate_phase_truth_alignment_used",
        "content_sha256",
    }
    require(set(record) == expected and record["schema"] == CANDIDATE_FRAME_RECORD_SCHEMA, "CANDIDATE_FRAME_RECORD_INVALID", "candidate frame record fields/schema drift")
    candidate_input = validate_candidate_input_receipt(record["candidate_input_receipt"])
    inference = validate_depthart_inference_receipt(record["inference_receipt"])
    require(
        inference["candidate_input_receipt_sha256"] == candidate_input["content_sha256"]
        and (inference["parent_id"], inference["video_id"], inference["timestamp_token"])
        == (candidate_input["parent_id"], candidate_input["video_id"], candidate_input["timestamp_token"]),
        "CANDIDATE_FRAME_RECORD_INVALID",
        "candidate frame input/inference binding drift",
    )
    blob = record["native_depth_blob"]
    expected_blob = {"path", "bytes", "sha256", "array_sha256", "shape_hw", "dtype", "encoding"}
    require(isinstance(blob, dict) and set(blob) == expected_blob, "CANDIDATE_FRAME_RECORD_INVALID", "candidate native blob binding fields drift")
    require(blob["path"] == _record_relative(candidate_input["parent_id"], candidate_input["video_id"], candidate_input["timestamp_token"], "depth.npy.gz"), "CANDIDATE_FRAME_RECORD_INVALID", "candidate native blob path drift")
    require(blob["array_sha256"] == inference["native_output_array_sha256"] and blob["shape_hw"] == list(NATIVE_SHAPE_HW) and blob["dtype"] == "float32" and blob["encoding"] == "DETERMINISTIC_GZIP_NPY_MTIME_0", "CANDIDATE_FRAME_RECORD_INVALID", "candidate native blob array binding drift")
    require(record["candidate_phase_truth_payload_read"] is False and record["candidate_phase_truth_alignment_used"] is False, "CANDIDATE_FRAME_TRUTH_FIREWALL_BREACH", "candidate frame record crossed the truth firewall")
    return record


def load_sealed_candidate_frame(factor_root: Path, parent_id: str, video_id: str, token: str) -> dict[str, Any]:
    """Reload and verify one candidate record/blob after phase completion."""

    root = factor_root.resolve()
    record_relative = _record_relative(parent_id, video_id, token, "record.json")
    record_path = (root / Path(*record_relative.split("/"))).resolve()
    require(root in record_path.parents and record_path.is_file(), "CANDIDATE_FRAME_RECORD_MISSING", "sealed candidate frame record is missing", path=record_relative)
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except Exception as error:
        raise CandidatePhaseError("CANDIDATE_FRAME_RECORD_INVALID", "candidate frame record cannot be decoded", path=record_relative) from error
    validated = validate_candidate_frame_record(record)
    blob = validated["native_depth_blob"]
    blob_path = (root / Path(*str(blob["path"]).split("/"))).resolve()
    require(root in blob_path.parents and blob_path.is_file(), "CANDIDATE_NATIVE_BLOB_MISSING", "sealed candidate native blob is missing", path=blob["path"])
    require(blob_path.stat().st_size == blob["bytes"] and sha256_file(blob_path) == blob["sha256"], "CANDIDATE_NATIVE_BLOB_HASH_MISMATCH", "sealed candidate native blob differs from its binding", path=blob["path"])
    native = np.ascontiguousarray(decode_npy_gzip_bytes(blob_path.read_bytes()), dtype=np.float32)
    require(native.shape == NATIVE_SHAPE_HW and bool(np.all(np.isfinite(native))) and adapter.canonical_sha256(native) == blob["array_sha256"], "CANDIDATE_NATIVE_BLOB_ARRAY_MISMATCH", "sealed candidate native array differs from its binding")
    return {"candidate_frame_record": validated, "native_depth_m": native}


def expected_candidate_keys(frame_plan_receipt: Sequence[Mapping[str, Any]]) -> list[tuple[str, str, str]]:
    """Return the immutable eval candidate key sequence from the 24-parent R3 plan."""

    require(isinstance(frame_plan_receipt, Sequence) and len(frame_plan_receipt) == 24, "CANDIDATE_FRAME_PLAN_INVALID", "R3 frame plan must contain 24 parents")
    keys: list[tuple[str, str, str]] = []
    observed_roster: list[tuple[str, str]] = []
    for row in frame_plan_receipt:
        require(isinstance(row, Mapping), "CANDIDATE_FRAME_PLAN_INVALID", "R3 frame-plan row must be an object")
        parent = row.get("parent")
        plan = row.get("frame_plan")
        require(isinstance(parent, Mapping) and isinstance(plan, Mapping), "CANDIDATE_FRAME_PLAN_INVALID", "R3 frame-plan parent/plan is malformed")
        if parent.get("role") != "O0R_EVAL_CANDIDATE":
            continue
        parent_id, video_id = str(parent.get("visit_id")), str(parent.get("video_id"))
        adapter._validate_roster_identity("O0R_EVAL_CANDIDATE", parent_id, video_id)
        observed_roster.append((parent_id, video_id))
        tokens = plan.get("exact_timestamp_tokens")
        require(isinstance(tokens, list) and bool(tokens), "CANDIDATE_FRAME_PLAN_INVALID", "eval candidate plan has no exact frames", parent_id=parent_id)
        normalized = [str(token) for token in tokens]
        require(normalized == sorted(normalized, key=lambda token: (adapter.decimal_timestamp_ns(token), token)) and len(set(normalized)) == len(normalized), "CANDIDATE_FRAME_PLAN_INVALID", "eval candidate timestamps are duplicated or unordered", parent_id=parent_id)
        keys.extend((parent_id, video_id, token) for token in normalized)
    require(observed_roster == list(adapter.O0R_EVAL_CANDIDATE_ROSTER), "CANDIDATE_FRAME_PLAN_ROSTER_DRIFT", "eval candidate roster/order drift")
    require(bool(keys) and len(keys) == len(set(keys)), "CANDIDATE_FRAME_PLAN_INVALID", "candidate key sequence is empty or duplicated")
    return keys


def _record_relative(parent_id: str, video_id: str, token: str, suffix: str) -> str:
    require(all(value and "/" not in value and "\\" not in value and value not in (".", "..") for value in (parent_id, video_id, token)), "CANDIDATE_IDENTITY_INVALID", "candidate identity is unsafe for evidence paths")
    return f"candidates/{parent_id}/{video_id}/{token}.{suffix}"


def run_candidate_phase(
    frame_plan_receipt: Sequence[Mapping[str, Any]],
    source_root: Path,
    *,
    writer: FactorEvidenceWriter,
    model: Any,
    runtime_identity: Mapping[str, Any],
    candidate_iterator_fn: Callable[..., Iterable[dict[str, Any]]] = iter_candidate_inputs,
    inference_fn: Callable[..., dict[str, Any]] = infer_depthart_candidate,
    guard_fn: Callable[[], None] | None = None,
    progress_fn: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run the complete truth-blind candidate phase and persist its seal."""

    require(writer.activated, "WRITER_NOT_ACTIVATED", "factor evidence root must be activated before the candidate phase")
    expected = expected_candidate_keys(frame_plan_receipt)
    observed: list[tuple[str, str, str]] = []
    frame_record_hashes: list[str] = []
    parent_counts: Counter[str] = Counter()
    guard = guard_fn or (lambda: None)

    for item in candidate_iterator_fn(frame_plan_receipt, source_root):
        guard()
        require(isinstance(item, dict) and set(item) == {"candidate_input_receipt", "color_rgb_u8"}, "CANDIDATE_ITERATOR_OUTPUT_INVALID", "candidate iterator returned unexpected fields")
        candidate_input = validate_candidate_input_receipt(item["candidate_input_receipt"])
        color = np.asarray(item["color_rgb_u8"])
        key = (candidate_input["parent_id"], candidate_input["video_id"], candidate_input["timestamp_token"])
        require(len(observed) < len(expected) and key == expected[len(observed)], "CANDIDATE_SEQUENCE_DRIFT", "candidate iterator order/identity differs from the frozen plan", expected=expected[len(observed)] if len(observed) < len(expected) else None, actual=key)
        result = inference_fn(
            model,
            candidate_input_receipt=candidate_input,
            color_rgb_u8=color,
            runtime_identity=runtime_identity,
            device="cuda",
        )
        guard()
        require(isinstance(result, dict) and set(result) >= {"native_depth_m", "inference_receipt"}, "CANDIDATE_INFERENCE_OUTPUT_INVALID", "candidate inference returned unexpected fields")
        native = np.ascontiguousarray(np.asarray(result["native_depth_m"], dtype=np.float32))
        inference = validate_depthart_inference_receipt(result["inference_receipt"])
        require(native.shape == NATIVE_SHAPE_HW and bool(np.all(np.isfinite(native))), "CANDIDATE_NATIVE_INVALID", "candidate native output must be finite 448x608 float32")
        require(adapter.canonical_sha256(native) == inference["native_output_array_sha256"], "CANDIDATE_NATIVE_RECEIPT_MISMATCH", "candidate native output differs from its inference receipt")
        require(
            (inference["parent_id"], inference["video_id"], inference["timestamp_token"]) == key
            and inference["candidate_input_receipt_sha256"] == candidate_input["content_sha256"],
            "CANDIDATE_INFERENCE_IDENTITY_MISMATCH",
            "candidate inference receipt does not bind its exact input",
        )

        blob_relative = _record_relative(*key, "depth.npy.gz")
        blob_receipt = writer.write_bytes(blob_relative, deterministic_npy_gzip_bytes(native))
        native_binding = dict(blob_receipt) | {
            "array_sha256": inference["native_output_array_sha256"],
            "shape_hw": list(NATIVE_SHAPE_HW),
            "dtype": "float32",
            "encoding": "DETERMINISTIC_GZIP_NPY_MTIME_0",
        }
        frame_record = seal_record(
            {
                "schema": CANDIDATE_FRAME_RECORD_SCHEMA,
                "candidate_input_receipt": candidate_input,
                "inference_receipt": inference,
                "native_depth_blob": native_binding,
                "candidate_phase_truth_payload_read": False,
                "candidate_phase_truth_alignment_used": False,
            }
        )
        validate_candidate_frame_record(frame_record)
        writer.write_json(_record_relative(*key, "record.json"), frame_record)
        observed.append(key)
        frame_record_hashes.append(frame_record["content_sha256"])
        parent_counts[key[0]] += 1
        if progress_fn is not None:
            progress_fn({"phase": "CANDIDATE_INFERENCE", "completed": len(observed), "total": len(expected), "physical_frame_id": inference["physical_frame_id"]})

    guard()
    require(observed == expected, "CANDIDATE_PHASE_INCOMPLETE", "candidate phase did not seal every frozen exact eval frame", expected_count=len(expected), observed_count=len(observed))
    completion = seal_record(
        {
            "schema": CANDIDATE_PHASE_COMPLETION_SCHEMA,
            "candidate_frame_count": len(observed),
            "candidate_frame_sequence_sha256": adapter.canonical_sha256([list(key) for key in observed]),
            "candidate_frame_record_hashes_sha256": adapter.canonical_sha256(frame_record_hashes),
            "parent_frame_counts": dict(sorted(parent_counts.items())),
            "runtime_identity": copy.deepcopy(dict(runtime_identity)),
            "runtime_identity_sha256": adapter.canonical_sha256(runtime_identity),
            "truth_frame_packages_opened_before_completion": 0,
            "truth_payload_read_by_candidate_phase": False,
            "truth_alignment_used_by_candidate_phase": False,
            "candidate_outputs_sealed_before_truth_join": True,
        }
    )
    writer.write_json("candidate-phase-completion.json", completion)
    return validate_candidate_phase_completion(completion)


__all__ = [
    "CANDIDATE_FRAME_RECORD_SCHEMA",
    "CANDIDATE_PHASE_COMPLETION_SCHEMA",
    "CandidatePhaseError",
    "expected_candidate_keys",
    "run_candidate_phase",
    "seal_record",
    "load_sealed_candidate_frame",
    "validate_candidate_frame_record",
    "validate_candidate_phase_completion",
]
