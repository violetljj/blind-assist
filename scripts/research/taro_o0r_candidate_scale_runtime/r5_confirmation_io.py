#!/usr/bin/env python3
"""Exact cohort and phase-scoped payload I/O for TARO R5 confirmation."""

from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_factor_headroom_runtime import candidate_inputs
from scripts.research.taro_o0r_factor_headroom_runtime import depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


class R5ConfirmationIOError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R5ConfirmationIOError(code, message, **context)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R5ConfirmationIOError("R5_JSON_READ_FAILED", "R5 JSON artifact cannot be read", path=str(path)) from error


def _load_json_gzip(path: Path) -> Any:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise R5ConfirmationIOError("R5_GZIP_JSON_READ_FAILED", "R5 gzip JSON artifact cannot be read", path=str(path)) from error


@dataclass(frozen=True)
class R5FrameRef:
    parent_id: str
    video_id: str
    timestamp_token: str
    physical_frame_id: str
    plan_row: dict[str, Any]
    source_record_path: Path
    source_record_sha256: str
    source_frame_receipt: dict[str, Any]
    bound_source_frame_envelope: dict[str, Any]
    upsampling_archive: Path
    upsampling_container_receipt: dict[str, Any]


def expected_keys(frames: Sequence[R5FrameRef]) -> list[tuple[str, str, str]]:
    return [(frame.parent_id, frame.video_id, frame.timestamp_token) for frame in frames]


def load_exact_cohort(
    frame_plan_path: Path,
    r3_evidence_root: Path,
    source_root: Path,
    *,
    verify_containers: bool = True,
) -> list[R5FrameRef]:
    plan = _load_json_gzip(frame_plan_path)
    require(isinstance(plan, list) and len(plan) == 24, "R5_FRAME_PLAN_INVALID", "R3 exact frame plan must contain 24 parents")
    fit_rows = [row for row in plan if row.get("parent", {}).get("role") == "ADAPTER_FIT"]
    require(len(fit_rows) == len(r5.R5_ROSTER), "R5_FRAME_PLAN_INVALID", "R3 plan does not contain the exact eight fit parents")
    frames: list[R5FrameRef] = []
    for row, expected_identity, expected_count in zip(fit_rows, r5.R5_ROSTER, r5.EXPECTED_PARENT_FRAME_COUNTS):
        parent = row.get("parent", {})
        plan_block = row.get("frame_plan", {})
        containers = row.get("container_receipts", {})
        identity = (str(parent.get("visit_id")), str(parent.get("video_id")))
        require(identity == expected_identity, "R5_FRAME_PLAN_ROSTER_DRIFT", "R5 fit parent order drift", expected=expected_identity, actual=identity)
        tokens = plan_block.get("exact_timestamp_tokens")
        require(
            isinstance(tokens, list)
            and len(tokens) == expected_count
            and tokens == sorted(tokens, key=lambda token: (adapter.decimal_timestamp_ns(token), token))
            and len(tokens) == len(set(tokens)),
            "R5_FRAME_PLAN_TOKEN_DRIFT",
            "R5 exact timestamp sequence drift",
            identity=identity,
        )
        container = containers.get("upsampling.zip")
        require(isinstance(container, dict) and isinstance(container.get("relative_path"), str), "R5_CONTAINER_RECEIPT_INVALID", "R5 upsampling receipt is absent")
        archive = materializer.safe_join(source_root, container["relative_path"])
        if verify_containers:
            materializer.verify_bound_container(archive, container)
        for token in tokens:
            relative = Path("source-frames") / "adapter-fit" / identity[0] / identity[1] / f"{token}.json.gz"
            record_path = (r3_evidence_root / relative).resolve()
            require(r3_evidence_root.resolve() in record_path.parents and record_path.is_file(), "R5_SOURCE_RECORD_MISSING", "R5 predecessor source record is missing", path=relative.as_posix())
            record = _load_json_gzip(record_path)
            require(isinstance(record, dict) and record.get("model_outputs_absent") is True, "R5_SOURCE_RECORD_INVALID", "predecessor fit record carries model output")
            source = r5.validate_r5_source_receipt(record.get("source_frame_receipt"))
            try:
                envelope = materializer.validate_bound_source_frame_envelope(record.get("bound_source_frame_envelope"), source)
            except materializer.MaterializerError as error:
                raise R5ConfirmationIOError(error.code, str(error), **error.context) from error
            require(
                (source["parent_id"], source["session_id"], source["sensor_timestamp"]["decimal_token"], source["physical_frame_id"])
                == (identity[0], identity[1], token, f"{identity[1]}:{token}"),
                "R5_SOURCE_RECORD_IDENTITY_DRIFT",
                "R5 predecessor source record identity drift",
            )
            frames.append(
                R5FrameRef(
                    parent_id=identity[0],
                    video_id=identity[1],
                    timestamp_token=token,
                    physical_frame_id=f"{identity[1]}:{token}",
                    plan_row=dict(row),
                    source_record_path=record_path,
                    source_record_sha256=_sha256(record_path.read_bytes()),
                    source_frame_receipt=source,
                    bound_source_frame_envelope=envelope,
                    upsampling_archive=archive,
                    upsampling_container_receipt=dict(container),
                )
            )
    r5._validate_expected_keys(expected_keys(frames))
    return frames


def read_bound_payload(
    frame: R5FrameRef,
    bundle: zipfile.ZipFile,
    role: str,
    *,
    read_observer: Callable[[str, str], None] | None = None,
) -> np.ndarray:
    require(role in {"color", "lowres_depth", "confidence", "highres_depth"}, "R5_PAYLOAD_ROLE_INVALID", "R5 runner requested a disallowed payload role", role=role)
    member = frame.bound_source_frame_envelope["members"][role]
    require(member["source_container_sha256"] == frame.upsampling_container_receipt["sha256"], "R5_MEMBER_CONTAINER_DRIFT", "R5 member and plan container differ", role=role)
    try:
        payload, info = candidate_inputs._read_exact_member(
            bundle,
            member["source_member_path"],
            role=role,
            read_observer=read_observer,
        )
    except candidate_inputs.CandidateInputError as error:
        raise R5ConfirmationIOError(error.code, str(error), **error.context) from error
    require(
        len(payload) == member["source_member_bytes"]
        and _sha256(payload) == member["source_member_sha256"]
        and f"{info.CRC:08X}" == member["source_member_crc32"],
        "R5_MEMBER_HASH_DRIFT",
        "R5 bound member bytes drift",
        role=role,
        physical_frame_id=frame.physical_frame_id,
    )
    if role == "color":
        value = candidate_inputs._decode_color(payload)
    else:
        try:
            value = materializer._decode_png(payload, role)
        except materializer.MaterializerError as error:
            raise R5ConfirmationIOError(error.code, str(error), **error.context) from error
    require(
        adapter.canonical_sha256(value) == member["decoded_content_sha256"]
        == frame.source_frame_receipt["decoded_payload_bindings"][role]["decoded_content_sha256"],
        "R5_DECODED_PAYLOAD_HASH_DRIFT",
        "R5 decoded payload differs from predecessor binding",
        role=role,
        physical_frame_id=frame.physical_frame_id,
    )
    return np.ascontiguousarray(value)


def candidate_blob_relative(frame: R5FrameRef) -> str:
    return f"candidates/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.depth.npy.gz"


def candidate_record_relative(frame: R5FrameRef) -> str:
    return f"candidates/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def source_decision_relative(frame: R5FrameRef) -> str:
    return f"phase-a-decisions/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def query_records_relative(frame: R5FrameRef) -> str:
    return f"phase-b-query-records/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def _safe_artifact(root: Path, relative: str) -> Path:
    path = materializer.safe_join(root, relative).resolve()
    require(root.resolve() in path.parents, "R5_ARTIFACT_PATH_INVALID", "R5 artifact path escapes evidence root", path=relative)
    return path


def load_candidate_frame(root: Path, frame: R5FrameRef) -> tuple[dict[str, Any], np.ndarray]:
    record_path = _safe_artifact(root, candidate_record_relative(frame))
    require(record_path.is_file(), "R5_CANDIDATE_RECORD_MISSING", "R5 sealed candidate record is missing", path=str(record_path))
    record = r5.validate_candidate_frame_record(_load_json(record_path))
    blob = record["native_depth_blob"]
    require(blob["path"] == candidate_blob_relative(frame), "R5_CANDIDATE_BLOB_PATH_DRIFT", "R5 candidate blob path drift")
    blob_path = _safe_artifact(root, blob["path"])
    require(blob_path.is_file(), "R5_CANDIDATE_BLOB_MISSING", "R5 sealed candidate blob is missing", path=str(blob_path))
    payload = blob_path.read_bytes()
    require(len(payload) == blob["bytes"] and _sha256(payload) == blob["sha256"], "R5_CANDIDATE_BLOB_HASH_DRIFT", "R5 candidate blob bytes drift")
    native = depthart_runner.decode_npy_gzip_bytes(payload)
    require(native.shape == depthart_runner.NATIVE_SHAPE_HW and native.dtype == np.float32 and adapter.canonical_sha256(native) == blob["array_sha256"], "R5_CANDIDATE_BLOB_ARRAY_DRIFT", "R5 candidate blob array drift")
    return record, native


def load_source_decision(root: Path, frame: R5FrameRef) -> dict[str, Any]:
    path = _safe_artifact(root, source_decision_relative(frame))
    require(path.is_file(), "R5_SOURCE_DECISION_MISSING", "R5 source decision is missing", path=str(path))
    decision = r5.validate_source_decision(_load_json(path))
    require(decision["physical_frame_id"] == frame.physical_frame_id, "R5_SOURCE_DECISION_IDENTITY_DRIFT", "R5 source decision frame drift")
    return decision


__all__ = [
    "R5ConfirmationIOError", "R5FrameRef", "candidate_blob_relative", "candidate_record_relative",
    "expected_keys", "load_candidate_frame", "load_exact_cohort", "load_source_decision",
    "query_records_relative", "read_bound_payload", "source_decision_relative",
]
