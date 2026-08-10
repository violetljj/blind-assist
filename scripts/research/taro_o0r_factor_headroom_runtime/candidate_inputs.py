#!/usr/bin/env python3
"""Read only RGB and intrinsics for the truth-blind TARO candidate phase."""

from __future__ import annotations

import binascii
import io
import math
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.taro_o0r_factor_headroom_runtime.depthart_runner import (
    build_candidate_input_receipt,
)
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_truth_materializer_runtime import materializer


class CandidateInputError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise CandidateInputError(code, message, **context)


def _validate_zip_info(info: zipfile.ZipInfo, expected_path: str) -> None:
    require(info.filename == expected_path and "\\" not in info.filename, "CANDIDATE_ZIP_MEMBER_PATH_INVALID", "candidate member path drift")
    pure = PurePosixPath(info.filename)
    require(not pure.is_absolute() and all(part not in ("", ".", "..") for part in pure.parts), "CANDIDATE_ZIP_MEMBER_PATH_INVALID", "candidate member path is unsafe")
    require(not (info.external_attr >> 16) & 0o170000 == 0o120000 and info.flag_bits & 0x1 == 0, "CANDIDATE_ZIP_MEMBER_UNSAFE", "candidate member is symlinked or encrypted")
    require(info.compress_type in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA} and info.file_size > 0, "CANDIDATE_ZIP_MEMBER_INVALID", "candidate member compression/size is invalid")


def _read_exact_member(
    bundle: zipfile.ZipFile,
    expected_path: str,
    *,
    role: str,
    read_observer: Callable[[str, str], None] | None = None,
) -> tuple[bytes, zipfile.ZipInfo]:
    try:
        info = bundle.getinfo(expected_path)
    except KeyError as error:
        raise CandidateInputError("CANDIDATE_MEMBER_MISSING", "candidate RGB/K member is missing", role=role, member=expected_path) from error
    _validate_zip_info(info, expected_path)
    if read_observer is not None:
        read_observer(role, expected_path)
    try:
        payload = bundle.read(info)
    except Exception as error:
        raise CandidateInputError("CANDIDATE_MEMBER_READ_FAILED", "candidate RGB/K member cannot be read", role=role, member=expected_path) from error
    require(len(payload) == info.file_size and f"{binascii.crc32(payload) & 0xFFFFFFFF:08X}" == f"{info.CRC:08X}", "CANDIDATE_MEMBER_INTEGRITY_DRIFT", "candidate RGB/K member length or CRC drift", role=role)
    return payload, info


def _decode_color(payload: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            color = np.asarray(image.convert("RGB")).copy()
    except Exception as error:
        raise CandidateInputError("CANDIDATE_COLOR_DECODE_FAILED", "candidate RGB cannot be decoded") from error
    require(color.shape == (*adapter.HIGHRES_SHAPE_HW, 3) and color.dtype == np.uint8, "CANDIDATE_COLOR_DECODE_INVALID", "candidate RGB must decode to uint8 1440x1920x3")
    return np.ascontiguousarray(color)


def _parse_pincam(payload: bytes) -> dict[str, Any]:
    try:
        fields = payload.decode("utf-8").split()
        values = [float(value) for value in fields]
    except (UnicodeDecodeError, ValueError) as error:
        raise CandidateInputError("CANDIDATE_INTRINSICS_DECODE_FAILED", "candidate pincam cannot be decoded") from error
    require(len(values) == 6 and all(math.isfinite(value) for value in values), "CANDIDATE_INTRINSICS_INVALID", "candidate pincam must contain six finite fields")
    width, height = int(values[0]), int(values[1])
    require(values[0] == width and values[1] == height, "CANDIDATE_INTRINSICS_INVALID", "candidate pincam dimensions must be integral")
    result = {"width": width, "height": height, "fx": values[2], "fy": values[3], "cx": values[4], "cy": values[5]}
    adapter.scale_lowres_intrinsics(result)
    return result


def _container_identity(path: Path, receipt: Mapping[str, Any], role: str) -> str:
    require(path.is_file(), "CANDIDATE_CONTAINER_MISSING", "candidate source container is missing", role=role, path=str(path))
    require(isinstance(receipt, Mapping) and isinstance(receipt.get("sha256"), str) and re.fullmatch(r"[0-9A-Fa-f]{64}", str(receipt["sha256"])), "CANDIDATE_CONTAINER_RECEIPT_INVALID", "candidate source container receipt is invalid", role=role)
    try:
        materializer.verify_bound_container(path, receipt)
    except materializer.MaterializerError as error:
        raise CandidateInputError("CANDIDATE_CONTAINER_INTEGRITY_DRIFT", "candidate source container differs from R3 receipt", role=role) from error
    return str(receipt["sha256"]).upper()


def _member_binding(
    *,
    container_sha256: str,
    canonical_member_path: str,
    payload: bytes,
    info: zipfile.ZipInfo,
) -> dict[str, Any]:
    import hashlib

    return {
        "container_id": f"sha256:{container_sha256}",
        "member_path": canonical_member_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
        "crc32": f"{info.CRC:08X}",
    }


def iter_candidate_inputs(
    frame_plan_receipt: Sequence[Mapping[str, Any]],
    source_root: Path,
    *,
    read_observer: Callable[[str, str], None] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield all exact eval RGB/K inputs without opening any truth modality."""

    require(isinstance(frame_plan_receipt, Sequence) and len(frame_plan_receipt) == 24, "CANDIDATE_FRAME_PLAN_INVALID", "R3 frame plan must contain exactly 24 parents")
    eval_rows = [row for row in frame_plan_receipt if row.get("parent", {}).get("role") == "O0R_EVAL_CANDIDATE"]
    require(len(eval_rows) == len(adapter.O0R_EVAL_CANDIDATE_ROSTER), "CANDIDATE_FRAME_PLAN_INVALID", "R3 frame plan must contain exactly 16 eval parents")
    observed_roster: list[tuple[str, str]] = []
    for row in eval_rows:
        parent = row.get("parent")
        plan = row.get("frame_plan")
        containers = row.get("container_receipts")
        require(isinstance(parent, Mapping) and isinstance(plan, Mapping) and isinstance(containers, Mapping), "CANDIDATE_FRAME_PLAN_INVALID", "candidate parent plan is malformed")
        visit_id, video_id = str(parent.get("visit_id")), str(parent.get("video_id"))
        adapter._validate_roster_identity("O0R_EVAL_CANDIDATE", visit_id, video_id)
        observed_roster.append((visit_id, video_id))
        tokens = plan.get("exact_timestamp_tokens")
        require(isinstance(tokens, list) and bool(tokens) and tokens == sorted(tokens, key=lambda token: (adapter.decimal_timestamp_ns(token), token)) and len(set(tokens)) == len(tokens), "CANDIDATE_FRAME_PLAN_INVALID", "candidate exact timestamp plan is empty, duplicated, or unordered")
        upsampling = source_root / "upsampling" / "Training" / f"{video_id}.zip"
        intrinsics = source_root / "raw" / "Training" / video_id / "lowres_wide_intrinsics.zip"
        up_sha = _container_identity(upsampling, containers.get("upsampling.zip", {}), "upsampling.zip")
        intr_sha = _container_identity(intrinsics, containers.get("lowres_wide_intrinsics.zip", {}), "lowres_wide_intrinsics.zip")
        with zipfile.ZipFile(upsampling) as up_bundle, zipfile.ZipFile(intrinsics) as intr_bundle:
            for token in tokens:
                color_source_path = f"{video_id}/wide/{video_id}_{token}.png"
                intrinsics_source_path = f"lowres_wide_intrinsics/{video_id}_{token}.pincam"
                color_payload, color_info = _read_exact_member(up_bundle, color_source_path, role="color", read_observer=read_observer)
                intrinsics_payload, intrinsics_info = _read_exact_member(intr_bundle, intrinsics_source_path, role="intrinsics", read_observer=read_observer)
                color = _decode_color(color_payload)
                lowres = _parse_pincam(intrinsics_payload)
                receipt = build_candidate_input_receipt(
                    visit_id=visit_id,
                    video_id=video_id,
                    timestamp_token=token,
                    color_member_binding=_member_binding(container_sha256=up_sha, canonical_member_path=f"color/{token}.png", payload=color_payload, info=color_info),
                    intrinsics_member_binding=_member_binding(container_sha256=intr_sha, canonical_member_path=f"intrinsics/{token}.pincam", payload=intrinsics_payload, info=intrinsics_info),
                    color_rgb_u8=color,
                    lowres_intrinsics=lowres,
                )
                yield {"candidate_input_receipt": receipt, "color_rgb_u8": color}
    require(observed_roster == list(adapter.O0R_EVAL_CANDIDATE_ROSTER), "CANDIDATE_FRAME_PLAN_ROSTER_DRIFT", "candidate frame-plan eval roster/order drift")


__all__ = ["CandidateInputError", "iter_candidate_inputs"]
