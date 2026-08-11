#!/usr/bin/env python3
"""Hash-bound local I/O for the 24-parent TARO formation replay cohort.

This module joins the predecessor source receipts and truth-blind DepthART
candidates without opening FARO.  FARO remains an explicit, separately
observable Phase-B read through :func:`read_bound_payload`.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation_io as r5io
from scripts.research.taro_o0r_factor_headroom_runtime import candidate_inputs, candidate_phase, depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


EXPECTED_PARENT_COUNT = 24
EXPECTED_FRAME_COUNT = 450
EXPECTED_ROLE_FRAME_COUNTS = {"ADAPTER_FIT": 211, "O0R_EVAL_CANDIDATE": 239}
FORMATION_ROSTER = tuple(adapter.ADAPTER_FIT_ROSTER) + tuple(adapter.O0R_EVAL_CANDIDATE_ROSTER)


class FormationReplayIOError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise FormationReplayIOError(code, message, **context)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FormationReplayIOError("FORMATION_JSON_READ_FAILED", "formation JSON cannot be read", path=str(path)) from error


def _load_json_gzip(path: Path) -> Any:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise FormationReplayIOError("FORMATION_GZIP_JSON_READ_FAILED", "formation gzip JSON cannot be read", path=str(path)) from error


@dataclass(frozen=True)
class FormationFrameRef:
    source_role: str
    parent_id: str
    video_id: str
    timestamp_token: str
    physical_frame_id: str
    plan_row: dict[str, Any]
    plan_row_sha256: str
    predecessor_record_path: Path
    predecessor_record_sha256: str
    source_frame_receipt: dict[str, Any]
    bound_source_frame_envelope: dict[str, Any]
    upsampling_archive: Path
    upsampling_container_receipt: dict[str, Any]


def expected_keys(frames: Sequence[FormationFrameRef]) -> list[tuple[str, str, str, str]]:
    keys = [(frame.source_role, frame.parent_id, frame.video_id, frame.timestamp_token) for frame in frames]
    require(len(keys) == EXPECTED_FRAME_COUNT and len(keys) == len(set(keys)), "FORMATION_KEY_SEQUENCE_INVALID", "formation frame sequence is not the exact unique 450")
    counts = Counter(role for role, _, _, _ in keys)
    require(dict(counts) == EXPECTED_ROLE_FRAME_COUNTS, "FORMATION_ROLE_COUNTS_DRIFT", "formation role frame counts drift", counts=dict(counts))
    return keys


def _load_predecessor_record(
    evidence_root: Path,
    source_role: str,
    parent_id: str,
    video_id: str,
    token: str,
) -> tuple[Path, bytes, dict[str, Any]]:
    uniform = evidence_root / "source-frames" / "all" / parent_id / video_id / f"{token}.json.gz"
    if uniform.is_file():
        path = uniform.resolve()
        payload = path.read_bytes()
        record = _load_json_gzip(path)
        require(
            isinstance(record, dict)
            and record.get("source_role") == source_role
            and record.get("model_outputs_absent") is True,
            "FORMATION_SOURCE_RECORD_INVALID",
            "uniform formation source record is invalid",
            path=str(path),
        )
        return path, payload, record
    if source_role == "ADAPTER_FIT":
        relative = Path("source-frames") / "adapter-fit" / parent_id / video_id / f"{token}.json.gz"
    else:
        relative = Path("truth-frames") / parent_id / video_id / f"{token}.json.gz"
    path = (evidence_root / relative).resolve()
    require(evidence_root.resolve() in path.parents and path.is_file(), "FORMATION_PREDECESSOR_MISSING", "formation predecessor record is missing", path=relative.as_posix())
    payload = path.read_bytes()
    encoded = _load_json_gzip(path)
    if source_role == "ADAPTER_FIT":
        require(isinstance(encoded, dict) and encoded.get("model_outputs_absent") is True, "FORMATION_FIT_SOURCE_INVALID", "fit predecessor source record is invalid")
        record = encoded
    else:
        try:
            hydrated = materializer.hydrate_content_addressed_artifact(
                encoded,
                lambda blob_path: (_ for _ in ()).throw(
                    FormationReplayIOError("FORMATION_UNEXPECTED_ARRAY_BLOB", "compact predecessor unexpectedly requested an array blob", path=blob_path)
                ),
            )
            record = materializer.validate_eval_truth_commitment_record(hydrated)
        except materializer.MaterializerError as error:
            raise FormationReplayIOError(error.code, str(error), **error.context) from error
    return path, payload, record


def load_exact_cohort(
    frame_plan_path: Path,
    predecessor_evidence_root: Path,
    source_root: Path,
    *,
    verify_containers: bool = True,
) -> list[FormationFrameRef]:
    plan_path = frame_plan_path.resolve()
    evidence_root = predecessor_evidence_root.resolve()
    data_root = source_root.resolve()
    plan = _load_json_gzip(plan_path)
    require(isinstance(plan, list) and len(plan) == EXPECTED_PARENT_COUNT, "FORMATION_PLAN_INVALID", "formation exact plan must contain 24 parents")
    observed_roster: list[tuple[str, str]] = []
    parent_count = Counter()
    frames: list[FormationFrameRef] = []
    for row in plan:
        require(isinstance(row, dict), "FORMATION_PLAN_INVALID", "formation plan row must be an object")
        parent = row.get("parent")
        block = row.get("frame_plan")
        containers = row.get("container_receipts")
        require(isinstance(parent, dict) and isinstance(block, dict) and isinstance(containers, dict), "FORMATION_PLAN_INVALID", "formation plan parent/frame/container block is malformed")
        role = str(parent.get("role"))
        identity = (str(parent.get("visit_id")), str(parent.get("video_id")))
        require(role in EXPECTED_ROLE_FRAME_COUNTS, "FORMATION_ROLE_INVALID", "formation source role is not admitted", role=role)
        adapter._validate_roster_identity(role, *identity)
        observed_roster.append(identity)
        tokens = block.get("exact_timestamp_tokens")
        require(
            isinstance(tokens, list)
            and bool(tokens)
            and tokens == sorted(tokens, key=lambda item: (adapter.decimal_timestamp_ns(item), item))
            and len(tokens) == len(set(tokens)),
            "FORMATION_TOKEN_SEQUENCE_INVALID",
            "formation exact timestamp sequence is empty, duplicated, or unordered",
            identity=identity,
        )
        parent_count[role] += len(tokens)
        upsampling = containers.get("upsampling.zip")
        require(isinstance(upsampling, dict) and isinstance(upsampling.get("relative_path"), str), "FORMATION_CONTAINER_RECEIPT_INVALID", "formation upsampling receipt is missing")
        archive = materializer.safe_join(data_root, upsampling["relative_path"])
        if verify_containers:
            try:
                materializer.verify_bound_container(archive, upsampling)
            except materializer.MaterializerError as error:
                raise FormationReplayIOError(error.code, str(error), **error.context) from error
        row_sha = adapter.canonical_sha256(row)
        for token in tokens:
            path, encoded, predecessor = _load_predecessor_record(evidence_root, role, identity[0], identity[1], token)
            try:
                source = adapter._validate_base_receipt(dict(predecessor["source_frame_receipt"]))
                envelope = materializer.validate_bound_source_frame_envelope(predecessor["bound_source_frame_envelope"], source)
            except (KeyError, TypeError, adapter.AdapterError, materializer.MaterializerError) as error:
                code = getattr(error, "code", "FORMATION_SOURCE_RECEIPT_INVALID")
                context = getattr(error, "context", {})
                raise FormationReplayIOError(code, str(error), **context) from error
            require(
                source["source_role"] == role
                and (source["parent_id"], source["session_id"], source["sensor_timestamp"]["decimal_token"], source["physical_frame_id"])
                == (identity[0], identity[1], token, f"{identity[1]}:{token}"),
                "FORMATION_SOURCE_IDENTITY_DRIFT",
                "formation predecessor source identity/role drift",
            )
            require(envelope["source_frame_receipt_sha256"] == source["content_sha256"], "FORMATION_SOURCE_ENVELOPE_DRIFT", "formation envelope is not bound to its source receipt")
            frames.append(
                FormationFrameRef(
                    source_role=role,
                    parent_id=identity[0],
                    video_id=identity[1],
                    timestamp_token=token,
                    physical_frame_id=f"{identity[1]}:{token}",
                    plan_row=dict(row),
                    plan_row_sha256=row_sha,
                    predecessor_record_path=path,
                    predecessor_record_sha256=_sha(encoded),
                    source_frame_receipt=source,
                    bound_source_frame_envelope=envelope,
                    upsampling_archive=archive,
                    upsampling_container_receipt=dict(upsampling),
                )
            )
    require(observed_roster == list(FORMATION_ROSTER), "FORMATION_ROSTER_ORDER_DRIFT", "formation roster/order differs from the frozen 8+16 source rosters")
    require(dict(parent_count) == EXPECTED_ROLE_FRAME_COUNTS, "FORMATION_ROLE_COUNTS_DRIFT", "formation exact role counts differ from 211+239", counts=dict(parent_count))
    expected_keys(frames)
    return frames


def read_bound_payload(
    frame: FormationFrameRef,
    bundle: zipfile.ZipFile,
    role: str,
    *,
    read_observer: Callable[[str, str], None] | None = None,
) -> np.ndarray:
    require(role in {"color", "lowres_depth", "confidence", "highres_depth"}, "FORMATION_PAYLOAD_ROLE_INVALID", "formation replay requested a disallowed payload role", role=role)
    member = frame.bound_source_frame_envelope["members"][role]
    require(member["source_container_sha256"] == frame.upsampling_container_receipt["sha256"], "FORMATION_MEMBER_CONTAINER_DRIFT", "formation source member container drift", role=role)
    try:
        payload, info = candidate_inputs._read_exact_member(bundle, member["source_member_path"], role=role, read_observer=read_observer)
    except candidate_inputs.CandidateInputError as error:
        raise FormationReplayIOError(error.code, str(error), **error.context) from error
    require(
        len(payload) == member["source_member_bytes"]
        and _sha(payload) == member["source_member_sha256"]
        and f"{info.CRC:08X}" == member["source_member_crc32"],
        "FORMATION_MEMBER_HASH_DRIFT",
        "formation source member bytes differ from predecessor binding",
        role=role,
        physical_frame_id=frame.physical_frame_id,
    )
    if role == "color":
        value = candidate_inputs._decode_color(payload)
    else:
        try:
            value = materializer._decode_png(payload, role)
        except materializer.MaterializerError as error:
            raise FormationReplayIOError(error.code, str(error), **error.context) from error
    require(
        adapter.canonical_sha256(value) == member["decoded_content_sha256"]
        == frame.source_frame_receipt["decoded_payload_bindings"][role]["decoded_content_sha256"],
        "FORMATION_DECODED_PAYLOAD_HASH_DRIFT",
        "formation decoded source payload differs from predecessor binding",
        role=role,
        physical_frame_id=frame.physical_frame_id,
    )
    return np.ascontiguousarray(value)


def load_candidate_frame(
    fit_candidate_root: Path,
    eval_candidate_root: Path,
    frame: FormationFrameRef,
) -> tuple[dict[str, Any], np.ndarray]:
    if frame.source_role == "ADAPTER_FIT":
        try:
            record, native = r5io.load_candidate_frame(fit_candidate_root.resolve(), frame)
        except (r5.R5ConfirmationError, r5io.R5ConfirmationIOError) as error:
            raise FormationReplayIOError(error.code, str(error), **error.context) from error
        candidate_input = record["candidate_input_receipt"]
        require(candidate_input["source_frame_receipt_sha256"] == frame.source_frame_receipt["content_sha256"], "FORMATION_CANDIDATE_SOURCE_DRIFT", "fit candidate is not bound to the predecessor source receipt")
    else:
        try:
            loaded = candidate_phase.load_sealed_candidate_frame(eval_candidate_root.resolve(), frame.parent_id, frame.video_id, frame.timestamp_token)
        except candidate_phase.CandidatePhaseError as error:
            raise FormationReplayIOError(error.code, str(error), **error.context) from error
        record = loaded["candidate_frame_record"]
        native = loaded["native_depth_m"]
        candidate_input = record["candidate_input_receipt"]
        require(candidate_input["source_role"] == frame.source_role, "FORMATION_CANDIDATE_ROLE_DRIFT", "eval candidate source role drift")
    require(
        (candidate_input["parent_id"], candidate_input["video_id"], candidate_input["timestamp_token"], candidate_input["physical_frame_id"])
        == (frame.parent_id, frame.video_id, frame.timestamp_token, frame.physical_frame_id),
        "FORMATION_CANDIDATE_IDENTITY_DRIFT",
        "formation candidate identity differs from exact source frame",
    )
    source = frame.source_frame_receipt
    require(
        candidate_input["color_decoded_content_sha256"] == source["decoded_payload_bindings"]["color"]["decoded_content_sha256"]
        and candidate_input["effective_intrinsics_sha256"] == adapter.canonical_sha256(source["intrinsics_highres"]),
        "FORMATION_CANDIDATE_INPUT_LINEAGE_DRIFT",
        "formation candidate RGB/K lineage differs from source receipt",
    )
    inference = record["inference_receipt"]
    require(
        inference["model_id"] == adapter.BASELINE_MODEL_ID
        and inference["checkpoint_sha256"] == adapter.BASELINE_CHECKPOINT_SHA256
        and inference["preprocess_id"] == depthart_runner.PREPROCESS_ID
        and inference["postprocess_id"] == depthart_runner.POSTPROCESS_ID,
        "FORMATION_CANDIDATE_RUNTIME_DRIFT",
        "formation candidate model/transform identity drift",
    )
    return record, np.ascontiguousarray(native, dtype=np.float32)


def preflight_inventory(
    frames: Sequence[FormationFrameRef],
    fit_candidate_root: Path,
    eval_candidate_root: Path,
) -> dict[str, Any]:
    keys = expected_keys(frames)
    candidate_hashes: list[str] = []
    native_hashes: list[str] = []
    source_hashes: list[str] = []
    parent_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    for frame in frames:
        candidate, native = load_candidate_frame(fit_candidate_root, eval_candidate_root, frame)
        candidate_hashes.append(candidate["content_sha256"])
        native_hashes.append(adapter.canonical_sha256(native))
        source_hashes.append(frame.source_frame_receipt["content_sha256"])
        parent_counts[frame.parent_id] += 1
        role_counts[frame.source_role] += 1
    return {
        "schema": "blindassist.taro.o0r.r6_formation_replay_preflight.v1",
        "parent_count": len(parent_counts),
        "frame_count": len(frames),
        "role_frame_counts": dict(sorted(role_counts.items())),
        "parent_frame_counts": dict(sorted(parent_counts.items())),
        "frame_key_sequence_sha256": adapter.canonical_sha256([list(row) for row in keys]),
        "source_receipt_hash_sequence_sha256": adapter.canonical_sha256(source_hashes),
        "candidate_record_hash_sequence_sha256": adapter.canonical_sha256(candidate_hashes),
        "candidate_native_hash_sequence_sha256": adapter.canonical_sha256(native_hashes),
        "all_source_receipts_valid": True,
        "all_source_containers_bound": True,
        "all_candidates_truth_blind_and_bound": True,
        "faro_payload_reads": 0,
        "r6_untouched_parent_overlap": sorted({frame.parent_id for frame in frames} & set(__import__("scripts.research.taro_o0r_candidate_scale_runtime.prospective_factor_runtime", fromlist=["FORBIDDEN_R6_UNTOUCHED_PARENTS"]).FORBIDDEN_R6_UNTOUCHED_PARENTS)),
    }


__all__ = [
    "EXPECTED_FRAME_COUNT", "EXPECTED_PARENT_COUNT", "EXPECTED_ROLE_FRAME_COUNTS", "FORMATION_ROSTER",
    "FormationFrameRef", "FormationReplayIOError", "expected_keys", "load_candidate_frame", "load_exact_cohort",
    "preflight_inventory", "read_bound_payload",
]
