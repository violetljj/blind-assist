#!/usr/bin/env python3
"""Phase-scoped local I/O for the exact TARO R6 untouched cohort."""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import r6_confirmation as r6
from scripts.research.taro_o0r_candidate_scale_runtime import run_r6_untouched_inventory
from scripts.research.taro_o0r_factor_headroom_runtime import candidate_inputs, depthart_runner
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


class R6ConfirmationIOError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R6ConfirmationIOError(code, message, **context)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise R6ConfirmationIOError("R6_JSON_READ_FAILED", "R6 JSON artifact cannot be read", path=str(path)) from error


@dataclass(frozen=True)
class MemberRef:
    role: str
    path: str
    bytes: int
    crc32: str


@dataclass(frozen=True)
class R6FrameRef:
    parent_id: str
    video_id: str
    timestamp_token: str
    physical_frame_id: str
    upsampling_archive: Path
    intrinsics_archive: Path
    trajectory_path: Path
    container_bindings: dict[str, Any]
    trajectory_rows: tuple[dict[str, Any], ...]
    members: dict[str, MemberRef]


def expected_keys(frames: Sequence[R6FrameRef]) -> list[tuple[str, str, str]]:
    return r6.validate_expected_keys([(row.parent_id, row.video_id, row.timestamp_token) for row in frames])


def _verify_container(path: Path, binding: Mapping[str, Any]) -> None:
    require(path.is_file() and path.stat().st_size == int(binding["bytes"]) and materializer.sha256_file(path) == binding["sha256"], "R6_CONTAINER_BINDING_DRIFT", "R6 source container differs from inventory", path=str(path))


def _index_upsampling(path: Path, video_id: str) -> dict[str, dict[str, MemberRef]]:
    result = {role: {} for role in ("color", "highres_depth", "lowres_depth", "confidence")}
    with zipfile.ZipFile(path) as bundle:
        for info in bundle.infolist():
            pure = PurePosixPath(info.filename)
            if info.is_dir() or pure.suffix.lower() != ".png" or len(pure.parts) < 2 or pure.parts[-2] not in materializer.UPSAMPLING_DIRECTORY_TO_ROLE:
                continue
            role = materializer.UPSAMPLING_DIRECTORY_TO_ROLE[pure.parts[-2]]
            token = materializer._timestamp_token_from_member(video_id, info.filename, ".png")
            require(token not in result[role], "R6_MEMBER_DUPLICATE", "R6 upsampling member token is duplicated", role=role, token=token)
            result[role][token] = MemberRef(role, info.filename, int(info.file_size), f"{info.CRC:08X}")
    require(all(result.values()), "R6_MODALITY_MISSING", "R6 upsampling archive lacks a required modality")
    return result


def _index_intrinsics(path: Path, video_id: str) -> dict[str, MemberRef]:
    result: dict[str, MemberRef] = {}
    with zipfile.ZipFile(path) as bundle:
        for info in bundle.infolist():
            if info.is_dir() or PurePosixPath(info.filename).suffix.lower() != ".pincam":
                continue
            token = materializer._timestamp_token_from_member(video_id, info.filename, ".pincam")
            require(token not in result, "R6_MEMBER_DUPLICATE", "R6 intrinsics member token is duplicated", token=token)
            result[token] = MemberRef("intrinsics", info.filename, int(info.file_size), f"{info.CRC:08X}")
    require(bool(result), "R6_INTRINSICS_MISSING", "R6 intrinsics archive has no pincam members")
    return result


def load_exact_cohort(inventory_path: Path, repo_root: Path) -> list[R6FrameRef]:
    inventory = run_r6_untouched_inventory.validate_inventory(_load_json(inventory_path.resolve()))
    root = repo_root.resolve()
    frames: list[R6FrameRef] = []
    for parent_row, expected_identity, expected_count in zip(inventory["parents"], r6.ROSTER, r6.EXPECTED_PARENT_FRAME_COUNTS):
        identity = (str(parent_row["visit_id"]), str(parent_row["video_id"]))
        require(identity == expected_identity, "R6_COHORT_ROSTER_DRIFT", "R6 inventory roster order drift")
        bindings = parent_row["container_bindings"]
        up_path = materializer.safe_join(root, bindings["upsampling"]["path"])
        intr_path = materializer.safe_join(root, bindings["intrinsics"]["path"])
        traj_path = materializer.safe_join(root, bindings["trajectory"]["path"])
        _verify_container(up_path, bindings["upsampling"])
        _verify_container(intr_path, bindings["intrinsics"])
        _verify_container(traj_path, bindings["trajectory"])
        up_index = _index_upsampling(up_path, identity[1])
        intr_index = _index_intrinsics(intr_path, identity[1])
        trajectory = tuple(materializer.parse_trajectory_payload(traj_path.read_bytes()))
        tokens = parent_row["frame_plan"]["exact_timestamp_tokens"]
        require(isinstance(tokens, list) and len(tokens) == expected_count, "R6_COHORT_TOKEN_DRIFT", "R6 exact frame count drift", identity=identity)
        for token in tokens:
            require(all(token in up_index[role] for role in up_index) and token in intr_index, "R6_COHORT_MEMBER_MISSING", "R6 exact frame member is absent", token=token)
            members = {role: up_index[role][token] for role in up_index}
            members["intrinsics"] = intr_index[token]
            frames.append(R6FrameRef(identity[0], identity[1], token, f"{identity[1]}:{token}", up_path, intr_path, traj_path, {key: dict(value) for key, value in bindings.items()}, trajectory, members))
    expected_keys(frames)
    return frames


def _read_member(bundle: zipfile.ZipFile, member: MemberRef, *, observer: Callable[[str, str], None] | None = None) -> tuple[bytes, dict[str, Any]]:
    try:
        payload, info = candidate_inputs._read_exact_member(bundle, member.path, role=member.role, read_observer=observer)
    except candidate_inputs.CandidateInputError as error:
        raise R6ConfirmationIOError(error.code, str(error), **error.context) from error
    require(len(payload) == member.bytes and f"{info.CRC:08X}" == member.crc32, "R6_MEMBER_BINDING_DRIFT", "R6 member bytes/CRC differ from inventory", role=member.role)
    return payload, {"member_path": member.path, "bytes": len(payload), "sha256": _sha(payload), "crc32": member.crc32}


def read_phase_a_frame(frame: R6FrameRef, up_bundle: zipfile.ZipFile, intr_bundle: zipfile.ZipFile, *, observer: Callable[[str, str], None] | None = None) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    asset_bindings: dict[str, Any] = {}
    for role in ("color", "lowres_depth", "confidence"):
        payload, binding = _read_member(up_bundle, frame.members[role], observer=observer)
        decoded[role] = candidate_inputs._decode_color(payload) if role == "color" else materializer._decode_png(payload, role)
        asset_bindings[role] = {"container_sha256": frame.container_bindings["upsampling"]["sha256"], **binding}
    intr_payload, intr_binding = _read_member(intr_bundle, frame.members["intrinsics"], observer=observer)
    decoded["intrinsics"] = materializer.parse_pincam_payload(intr_payload)
    asset_bindings["intrinsics"] = {"container_sha256": frame.container_bindings["intrinsics"]["sha256"], **intr_binding}
    trajectory_payload = frame.trajectory_path.read_bytes()
    if observer is not None:
        observer("trajectory", "lowres_wide.traj")
    decoded["trajectory"] = list(frame.trajectory_rows)
    asset_bindings["trajectory"] = {"container_sha256": frame.container_bindings["trajectory"]["sha256"], "member_path": "lowres_wide.traj", "bytes": len(trajectory_payload), "sha256": _sha(trajectory_payload), "crc32": materializer.crc32_bytes(trajectory_payload)}
    source = r6.build_phase_a_source_receipt(
        parent_id=frame.parent_id, video_id=frame.video_id, timestamp_token=frame.timestamp_token,
        lowres_intrinsics=decoded["intrinsics"], trajectory_rows=decoded["trajectory"], container_bindings=frame.container_bindings,
        asset_bindings=asset_bindings, decoded_payload_hashes={role: adapter.canonical_sha256(decoded[role]) for role in r6.PHASE_A_ASSET_ROLES},
    )
    return {"source_receipt": source, "color_rgb_u8": np.ascontiguousarray(decoded["color"]), "apple_depth_mm": np.ascontiguousarray(decoded["lowres_depth"]), "confidence": np.ascontiguousarray(decoded["confidence"])}


def read_faro_frame(frame: R6FrameRef, up_bundle: zipfile.ZipFile, *, observer: Callable[[str, str], None] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    payload, binding = _read_member(up_bundle, frame.members["highres_depth"], observer=observer)
    raw = np.ascontiguousarray(materializer._decode_png(payload, "highres_depth"))
    return raw, {"container_sha256": frame.container_bindings["upsampling"]["sha256"], **binding}


def source_receipt_relative(frame: R6FrameRef) -> str:
    return f"phase-a-sources/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def candidate_blob_relative(frame: R6FrameRef) -> str:
    return f"candidates/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.depth.npy.gz"


def candidate_record_relative(frame: R6FrameRef) -> str:
    return f"candidates/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def source_decision_relative(frame: R6FrameRef) -> str:
    return f"phase-a-decisions/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def query_pairs_relative(frame: R6FrameRef) -> str:
    return f"phase-b-query-pairs/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json.gz"


def truth_binding_relative(frame: R6FrameRef) -> str:
    return f"phase-b-truth-bindings/{frame.parent_id}/{frame.video_id}/{frame.timestamp_token}.json"


def _safe(root: Path, relative: str) -> Path:
    path = materializer.safe_join(root.resolve(), relative).resolve()
    require(root.resolve() in path.parents, "R6_ARTIFACT_PATH_INVALID", "R6 artifact path escapes evidence root")
    return path


def load_source_receipt(root: Path, frame: R6FrameRef) -> dict[str, Any]:
    path = _safe(root, source_receipt_relative(frame))
    require(path.is_file(), "R6_SOURCE_RECEIPT_MISSING", "R6 sealed source receipt is missing")
    source = r6.validate_phase_a_source_receipt(_load_json(path))
    require(source["physical_frame_id"] == frame.physical_frame_id, "R6_SOURCE_RECEIPT_IDENTITY_DRIFT", "R6 source receipt frame drift")
    return source


def load_candidate_frame(root: Path, frame: R6FrameRef) -> tuple[dict[str, Any], np.ndarray]:
    path = _safe(root, candidate_record_relative(frame))
    require(path.is_file(), "R6_CANDIDATE_RECORD_MISSING", "R6 sealed candidate record is missing")
    record = r6.validate_candidate_frame(_load_json(path))
    blob = record["native_depth_blob"]
    require(blob["path"] == candidate_blob_relative(frame), "R6_CANDIDATE_BLOB_PATH_DRIFT", "R6 candidate blob path drift")
    blob_path = _safe(root, blob["path"])
    payload = blob_path.read_bytes()
    require(len(payload) == blob["bytes"] and _sha(payload) == blob["sha256"], "R6_CANDIDATE_BLOB_HASH_DRIFT", "R6 candidate blob bytes drift")
    native = depthart_runner.decode_npy_gzip_bytes(payload)
    require(native.shape == depthart_runner.NATIVE_SHAPE_HW and native.dtype == np.float32 and adapter.canonical_sha256(native) == blob["array_sha256"], "R6_CANDIDATE_BLOB_ARRAY_DRIFT", "R6 candidate blob array drift")
    return record, native


def load_source_decision(root: Path, frame: R6FrameRef) -> dict[str, Any]:
    path = _safe(root, source_decision_relative(frame))
    require(path.is_file(), "R6_SOURCE_DECISION_MISSING", "R6 sealed source decision is missing")
    decision = r6.validate_source_decision(_load_json(path))
    require(decision["physical_frame_id"] == frame.physical_frame_id, "R6_SOURCE_DECISION_IDENTITY_DRIFT", "R6 source decision frame drift")
    return decision


__all__ = [
    "MemberRef", "R6ConfirmationIOError", "R6FrameRef", "candidate_blob_relative", "candidate_record_relative", "expected_keys", "load_candidate_frame",
    "load_exact_cohort", "load_source_decision", "load_source_receipt", "query_pairs_relative", "read_faro_frame", "read_phase_a_frame", "source_decision_relative", "source_receipt_relative", "truth_binding_relative",
]
