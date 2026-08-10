#!/usr/bin/env python3
"""Deterministic, truth-blind DepthART candidate inference for TARO O0R.

The candidate phase accepts only a frozen O0R source receipt, its registered
RGB payload, and the bound effective intrinsics.  FARO, AppleDepth, confidence,
query truth, and task outcomes are deliberately absent from the interface.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


CANDIDATE_INPUT_RECEIPT_SCHEMA = "blindassist.taro.o0r.depthart_candidate_input_receipt.v1"
INFERENCE_RECEIPT_SCHEMA = "blindassist.taro.o0r.depthart_inference_receipt.v1"
PREPROCESS_ID = "DEPTHART_OFFICIAL_LOWER_BOUND_448_RGB_CUBIC_IMAGENET_V1"
POSTPROCESS_ID = "TARO_TORCH_CPU_BILINEAR_ALIGN_CORNERS_TRUE_FLOAT32_448X608_TO_1440X1920_V1"
NATIVE_SHAPE_HW = (448, 608)
HIGHRES_SHAPE_HW = adapter.HIGHRES_SHAPE_HW
TARGET_WIDTH = 448
TARGET_HEIGHT = 448
EXPECTED_SOURCE_GIT_COMMIT = "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c"
_SHA256 = re.compile(r"^[0-9A-Fa-f]{64}$")


class DepthARTRuntimeError(RuntimeError):
    """Fail-closed candidate-runtime error with a stable machine code."""

    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise DepthARTRuntimeError(code, message, **context)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_copy(value: Any) -> Any:
    return json.loads(adapter.canonical_json_bytes(value).decode("utf-8"))


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = _canonical_copy(dict(value))
    require("content_sha256" not in payload, "SEAL_INPUT_INVALID", "content hash must not be supplied by the caller")
    payload["content_sha256"] = adapter.canonical_sha256(payload)
    return _canonical_copy(payload)


def _validate_seal(value: Any, code: str) -> dict[str, Any]:
    require(isinstance(value, dict), code, "sealed value must be an object")
    payload = _canonical_copy(value)
    observed = payload.pop("content_sha256", None)
    require(isinstance(observed, str) and bool(_SHA256.fullmatch(observed)), code, "sealed value hash is malformed")
    require(adapter.canonical_sha256(payload) == observed.upper(), code, "sealed value hash mismatch")
    payload["content_sha256"] = observed.upper()
    return payload


def lower_bound_size(
    width: int,
    height: int,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    multiple: int = 32,
) -> tuple[int, int]:
    """Exact local copy of DepthART metric/common.py lower-bound sizing."""

    require(all(isinstance(item, int) and item > 0 for item in (width, height, target_width, target_height, multiple)), "PREPROCESS_SIZE_INVALID", "preprocess sizes must be positive integers")
    scale = max(target_width / width, target_height / height)

    def constrain(value: float, minimum: int) -> int:
        result = int(np.round(value / multiple) * multiple)
        return result if result >= minimum else int(np.ceil(value / multiple) * multiple)

    return constrain(scale * width, target_width), constrain(scale * height, target_height)


def _validate_source_member_binding(value: Any, role: str, timestamp_token: str) -> dict[str, Any]:
    require(isinstance(value, dict), "CANDIDATE_INPUT_MEMBER_INVALID", "source member binding must be an object", role=role)
    keys = {"container_id", "member_path", "bytes", "sha256", "crc32"}
    require(set(value) == keys, "CANDIDATE_INPUT_MEMBER_INVALID", "source member binding key set drift", role=role)
    require(isinstance(value["container_id"], str) and bool(value["container_id"]), "CANDIDATE_INPUT_MEMBER_INVALID", "source container identity is required", role=role)
    member_path = value["member_path"]
    require(isinstance(member_path, str) and bool(member_path) and "\\" not in member_path and not member_path.startswith("/") and all(part not in ("", ".", "..") for part in member_path.split("/")), "CANDIDATE_INPUT_MEMBER_INVALID", "source member path is invalid", role=role)
    require(member_path.rsplit("/", 1)[-1].rsplit(".", 1)[0] == timestamp_token, "CANDIDATE_INPUT_MEMBER_INVALID", "source member timestamp stem drift", role=role)
    require(isinstance(value["bytes"], int) and not isinstance(value["bytes"], bool) and value["bytes"] > 0, "CANDIDATE_INPUT_MEMBER_INVALID", "source member byte count is invalid", role=role)
    require(isinstance(value["sha256"], str) and bool(_SHA256.fullmatch(value["sha256"])), "CANDIDATE_INPUT_MEMBER_INVALID", "source member SHA-256 is invalid", role=role)
    require(isinstance(value["crc32"], str) and bool(re.fullmatch(r"[0-9A-Fa-f]{8}", value["crc32"])), "CANDIDATE_INPUT_MEMBER_INVALID", "source member CRC32 is invalid", role=role)
    return _canonical_copy(value)


def validate_candidate_input_receipt(value: Any) -> dict[str, Any]:
    receipt = _validate_seal(value, "CANDIDATE_INPUT_RECEIPT_HASH_MISMATCH")
    keys = {
        "schema",
        "source_id",
        "source_role",
        "parent_id",
        "video_id",
        "physical_frame_id",
        "timestamp_token",
        "color_member_binding",
        "intrinsics_member_binding",
        "color_decoded_content_sha256",
        "lowres_intrinsics",
        "intrinsics_highres",
        "effective_intrinsics_sha256",
        "allowed_model_inputs",
        "truth_payloads_opened",
        "content_sha256",
    }
    require(set(receipt) == keys and receipt["schema"] == CANDIDATE_INPUT_RECEIPT_SCHEMA and receipt["source_id"] == adapter.SOURCE_ID, "CANDIDATE_INPUT_RECEIPT_KEY_SET", "candidate input receipt key/schema drift")
    require(receipt["source_role"] == "O0R_EVAL_CANDIDATE", "CANDIDATE_SOURCE_ROLE_INVALID", "candidate input must be roster-bound eval data")
    adapter._validate_roster_identity(receipt["source_role"], receipt["parent_id"], receipt["video_id"])
    require(receipt["physical_frame_id"] == f"{receipt['video_id']}:{receipt['timestamp_token']}" and adapter.decimal_timestamp_ns(receipt["timestamp_token"]) >= 0, "CANDIDATE_INPUT_IDENTITY_INVALID", "candidate frame identity/timestamp drift")
    _validate_source_member_binding(receipt["color_member_binding"], "color", receipt["timestamp_token"])
    _validate_source_member_binding(receipt["intrinsics_member_binding"], "intrinsics", receipt["timestamp_token"])
    require(isinstance(receipt["color_decoded_content_sha256"], str) and bool(_SHA256.fullmatch(receipt["color_decoded_content_sha256"])), "CANDIDATE_INPUT_HASH_INVALID", "decoded RGB hash is invalid")
    expected_highres = adapter.scale_lowres_intrinsics(receipt["lowres_intrinsics"])
    require(adapter.canonical_sha256(expected_highres) == adapter.canonical_sha256(receipt["intrinsics_highres"]), "CANDIDATE_INPUT_INTRINSICS_INVALID", "effective K is not the frozen scaling of source lowres K")
    require(receipt["effective_intrinsics_sha256"] == adapter.canonical_sha256(receipt["intrinsics_highres"]), "CANDIDATE_INPUT_INTRINSICS_INVALID", "effective K hash drift")
    require(receipt["allowed_model_inputs"] == ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"] and receipt["truth_payloads_opened"] is False, "CANDIDATE_INPUT_TRUTH_LEAK", "candidate input must contain only RGB and K")
    return receipt


def build_candidate_input_receipt(
    *,
    visit_id: str,
    video_id: str,
    timestamp_token: str,
    color_member_binding: Mapping[str, Any],
    intrinsics_member_binding: Mapping[str, Any],
    color_rgb_u8: np.ndarray,
    lowres_intrinsics: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the only two payloads authorized for candidate inference: RGB/K."""

    adapter._validate_roster_identity("O0R_EVAL_CANDIDATE", visit_id, video_id)
    adapter.decimal_timestamp_ns(timestamp_token)
    color = np.asarray(color_rgb_u8)
    require(color.shape == (*HIGHRES_SHAPE_HW, 3) and color.dtype == np.uint8, "COLOR_INPUT_INVALID", "candidate RGB must be uint8 1440x1920x3")
    color_binding = _validate_source_member_binding(dict(color_member_binding), "color", timestamp_token)
    intrinsics_binding = _validate_source_member_binding(dict(intrinsics_member_binding), "intrinsics", timestamp_token)
    highres = adapter.scale_lowres_intrinsics(dict(lowres_intrinsics))
    receipt = _seal(
        {
            "schema": CANDIDATE_INPUT_RECEIPT_SCHEMA,
            "source_id": adapter.SOURCE_ID,
            "source_role": "O0R_EVAL_CANDIDATE",
            "parent_id": visit_id,
            "video_id": video_id,
            "physical_frame_id": f"{video_id}:{timestamp_token}",
            "timestamp_token": timestamp_token,
            "color_member_binding": color_binding,
            "intrinsics_member_binding": intrinsics_binding,
            "color_decoded_content_sha256": adapter.canonical_sha256(color),
            "lowres_intrinsics": dict(lowres_intrinsics),
            "intrinsics_highres": highres,
            "effective_intrinsics_sha256": adapter.canonical_sha256(highres),
            "allowed_model_inputs": ["REGISTERED_RGB", "BOUND_EFFECTIVE_K"],
            "truth_payloads_opened": False,
        }
    )
    return validate_candidate_input_receipt(receipt)


def candidate_input_from_bound_source(
    source_frame_receipt: dict[str, Any],
    color_rgb_u8: np.ndarray,
) -> dict[str, Any]:
    """Project a full bound receipt to the RGB/K-only candidate interface."""

    source = adapter._validate_base_receipt(source_frame_receipt)
    require(source["source_role"] == "O0R_EVAL_CANDIDATE", "CANDIDATE_SOURCE_ROLE_INVALID", "candidate input projection requires a frozen eval source")
    adapter._validate_bound_decoded_payload(source, "color", np.asarray(color_rgb_u8))

    def project(role: str) -> dict[str, Any]:
        binding = source["asset_bindings"][role]
        return {key: binding[key] for key in ("container_id", "member_path", "bytes", "sha256", "crc32")}

    return build_candidate_input_receipt(
        visit_id=source["parent_id"],
        video_id=source["session_id"],
        timestamp_token=source["sensor_timestamp"]["decimal_token"],
        color_member_binding=project("color"),
        intrinsics_member_binding=project("intrinsics"),
        color_rgb_u8=color_rgb_u8,
        lowres_intrinsics=source["lowres_intrinsics_source"],
    )


def preprocess_depthart_input(
    color_rgb_u8: np.ndarray,
    intrinsics_highres_3x3: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the frozen official 448 lower-bound preprocessing to RGB/K."""

    color = np.asarray(color_rgb_u8)
    require(color.shape == (*HIGHRES_SHAPE_HW, 3) and color.dtype == np.uint8, "COLOR_INPUT_INVALID", "DepthART input must be RGB uint8 1440x1920x3")
    matrix = np.asarray(intrinsics_highres_3x3, dtype=np.float32)
    require(matrix.shape == (3, 3) and bool(np.all(np.isfinite(matrix))) and matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0, "INTRINSICS_INPUT_INVALID", "DepthART input intrinsics must be finite 3x3 pinhole K")
    height, width = color.shape[:2]
    new_width, new_height = lower_bound_size(width, height)
    require((new_height, new_width) == NATIVE_SHAPE_HW, "PREPROCESS_SHAPE_DRIFT", "frozen landscape input must resize to 448x608", actual=[new_height, new_width])
    image = color.astype(np.float32) / np.float32(255.0)
    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
    std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
    tensor = ((image - mean) / std).transpose(2, 0, 1).copy()[None]
    resized_k = matrix.copy()
    resized_k[0, :] *= np.float32(new_width / width)
    resized_k[1, :] *= np.float32(new_height / height)
    resized_k = resized_k[None]
    require(tensor.shape == (1, 3, *NATIVE_SHAPE_HW) and tensor.dtype == np.float32 and bool(np.all(np.isfinite(tensor))), "PREPROCESS_TENSOR_INVALID", "DepthART input tensor is invalid")
    require(resized_k.shape == (1, 3, 3) and resized_k.dtype == np.float32 and bool(np.all(np.isfinite(resized_k))), "PREPROCESS_INTRINSICS_INVALID", "DepthART resized intrinsics are invalid")
    return tensor, resized_k


def upsample_native_depth(native_depth_m: np.ndarray) -> np.ndarray:
    """Apply the frozen CPU form of DepthART's bilinear registration operator."""

    native = np.asarray(native_depth_m)
    require(native.shape == NATIVE_SHAPE_HW and native.dtype.kind == "f" and bool(np.all(np.isfinite(native))), "NATIVE_DEPTH_INVALID", "native DepthART output must be a finite 448x608 float raster")
    import torch
    import torch.nn.functional as torch_functional

    tensor = torch.from_numpy(np.ascontiguousarray(native, dtype=np.float32))[None, None]
    highres = torch_functional.interpolate(
        tensor,
        HIGHRES_SHAPE_HW,
        mode="bilinear",
        align_corners=True,
    )[0, 0].numpy()
    highres = np.ascontiguousarray(highres, dtype=np.float32)
    require(highres.shape == HIGHRES_SHAPE_HW and bool(np.all(np.isfinite(highres))), "HIGHRES_DEPTH_INVALID", "upsampled DepthART output is invalid")
    return highres


def deterministic_npy_gzip_bytes(array: np.ndarray) -> bytes:
    """Encode one ndarray as deterministic gzip(npy), with mtime fixed to zero."""

    value = np.asarray(array)
    require(value.dtype.kind in "biuf" and bool(np.all(np.isfinite(value))) if value.dtype.kind == "f" else value.dtype.kind in "biu", "NATIVE_BLOB_ARRAY_INVALID", "native blob array must be finite numeric")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=6, mtime=0) as stream:
        np.save(stream, value, allow_pickle=False)
    return buffer.getvalue()


def decode_npy_gzip_bytes(payload: bytes) -> np.ndarray:
    require(isinstance(payload, bytes) and bool(payload), "NATIVE_BLOB_INVALID", "native blob payload is required")
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as stream:
            value = np.load(stream, allow_pickle=False)
    except Exception as error:
        raise DepthARTRuntimeError("NATIVE_BLOB_INVALID", "native blob cannot be decoded") from error
    require(isinstance(value, np.ndarray), "NATIVE_BLOB_INVALID", "native blob must contain one ndarray")
    return value


def write_native_depth_blob(path: Path, native_depth_m: np.ndarray) -> dict[str, Any]:
    native = np.asarray(native_depth_m, dtype=np.float32)
    require(native.shape == NATIVE_SHAPE_HW and bool(np.all(np.isfinite(native))), "NATIVE_DEPTH_INVALID", "native DepthART output must be finite float32 448x608")
    payload = deterministic_npy_gzip_bytes(native)
    require(not path.exists() and not path.with_suffix(path.suffix + ".partial").exists(), "NATIVE_BLOB_COLLISION", "native output blob already exists", path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "path": path.as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest().upper(),
        "array_sha256": adapter.canonical_sha256(native),
        "shape_hw": list(NATIVE_SHAPE_HW),
        "dtype": "float32",
        "encoding": "DETERMINISTIC_GZIP_NPY_MTIME_0",
    }


def validate_depthart_inference_receipt(value: Any) -> dict[str, Any]:
    receipt = _validate_seal(value, "INFERENCE_RECEIPT_HASH_MISMATCH")
    keys = {
        "schema",
        "model_id",
        "checkpoint_sha256",
        "preprocess_id",
        "postprocess_id",
        "candidate_input_receipt_sha256",
        "parent_id",
        "video_id",
        "physical_frame_id",
        "timestamp_token",
        "input_color_decoded_content_sha256",
        "effective_intrinsics_sha256",
        "input_tensor_shape_nchw",
        "input_tensor_sha256",
        "resized_intrinsics_sha256",
        "native_output_shape_hw",
        "native_output_dtype",
        "native_output_array_sha256",
        "highres_output_shape_hw",
        "highres_output_dtype",
        "highres_output_array_sha256",
        "runtime_identity",
        "truth_alignment_used",
        "truth_payload_read",
        "content_sha256",
    }
    require(set(receipt) == keys and receipt["schema"] == INFERENCE_RECEIPT_SCHEMA, "INFERENCE_RECEIPT_KEY_SET", "DepthART inference receipt key/schema drift")
    require(receipt["model_id"] == adapter.BASELINE_MODEL_ID and receipt["checkpoint_sha256"] == adapter.BASELINE_CHECKPOINT_SHA256, "INFERENCE_MODEL_BINDING_INVALID", "DepthART baseline identity drift")
    require(receipt["preprocess_id"] == PREPROCESS_ID and receipt["postprocess_id"] == POSTPROCESS_ID, "INFERENCE_TRANSFORM_DRIFT", "DepthART transform identity drift")
    for key in (
        "checkpoint_sha256",
        "candidate_input_receipt_sha256",
        "input_color_decoded_content_sha256",
        "effective_intrinsics_sha256",
        "input_tensor_sha256",
        "resized_intrinsics_sha256",
        "native_output_array_sha256",
        "highres_output_array_sha256",
    ):
        require(isinstance(receipt[key], str) and bool(_SHA256.fullmatch(receipt[key])), "INFERENCE_RECEIPT_HASH_INVALID", "inference receipt contains a malformed hash", field=key)
    require(receipt["physical_frame_id"] == f"{receipt['video_id']}:{receipt['timestamp_token']}" and isinstance(receipt["parent_id"], str) and bool(receipt["parent_id"]), "INFERENCE_RECEIPT_IDENTITY_INVALID", "physical frame identity is invalid")
    adapter._validate_roster_identity("O0R_EVAL_CANDIDATE", receipt["parent_id"], receipt["video_id"])
    adapter.decimal_timestamp_ns(receipt["timestamp_token"])
    require(receipt["input_tensor_shape_nchw"] == [1, 3, *NATIVE_SHAPE_HW], "INFERENCE_SHAPE_INVALID", "DepthART tensor shape drift")
    require(receipt["native_output_shape_hw"] == list(NATIVE_SHAPE_HW) and receipt["native_output_dtype"] == "float32", "INFERENCE_SHAPE_INVALID", "native DepthART output shape/dtype drift")
    require(receipt["highres_output_shape_hw"] == list(HIGHRES_SHAPE_HW) and receipt["highres_output_dtype"] == "float32", "INFERENCE_SHAPE_INVALID", "registered DepthART output shape/dtype drift")
    require(isinstance(receipt["runtime_identity"], dict) and bool(receipt["runtime_identity"]), "INFERENCE_RUNTIME_IDENTITY_INVALID", "runtime identity is required")
    require(receipt["truth_alignment_used"] is False and receipt["truth_payload_read"] is False, "INFERENCE_TRUTH_LEAK", "candidate inference must remain truth-blind")
    return receipt


def build_depthart_inference_receipt(
    *,
    candidate_input_receipt: dict[str, Any],
    color_rgb_u8: np.ndarray,
    input_tensor_nchw: np.ndarray,
    resized_intrinsics_n33: np.ndarray,
    native_depth_m: np.ndarray,
    highres_depth_m: np.ndarray,
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_input = validate_candidate_input_receipt(candidate_input_receipt)
    color = np.asarray(color_rgb_u8)
    require(adapter.canonical_sha256(color) == candidate_input["color_decoded_content_sha256"], "CANDIDATE_INPUT_COLOR_MISMATCH", "candidate RGB differs from its sealed input receipt")
    tensor = np.asarray(input_tensor_nchw)
    resized_k = np.asarray(resized_intrinsics_n33)
    native = np.asarray(native_depth_m)
    highres = np.asarray(highres_depth_m)
    require(tensor.shape == (1, 3, *NATIVE_SHAPE_HW) and tensor.dtype == np.float32 and bool(np.all(np.isfinite(tensor))), "INFERENCE_INPUT_INVALID", "preprocessed DepthART tensor is invalid")
    require(resized_k.shape == (1, 3, 3) and resized_k.dtype == np.float32 and bool(np.all(np.isfinite(resized_k))), "INFERENCE_INPUT_INVALID", "resized DepthART intrinsics are invalid")
    require(native.shape == NATIVE_SHAPE_HW and native.dtype == np.float32 and bool(np.all(np.isfinite(native))), "NATIVE_DEPTH_INVALID", "native DepthART output is invalid")
    require(highres.shape == HIGHRES_SHAPE_HW and highres.dtype == np.float32 and bool(np.all(np.isfinite(highres))), "HIGHRES_DEPTH_INVALID", "registered DepthART output is invalid")
    require(adapter.canonical_sha256(upsample_native_depth(native)) == adapter.canonical_sha256(highres), "POSTPROCESS_RECOMPUTE_MISMATCH", "registered output is not the frozen upsample of native output")
    receipt = _seal(
        {
            "schema": INFERENCE_RECEIPT_SCHEMA,
            "model_id": adapter.BASELINE_MODEL_ID,
            "checkpoint_sha256": adapter.BASELINE_CHECKPOINT_SHA256,
            "preprocess_id": PREPROCESS_ID,
            "postprocess_id": POSTPROCESS_ID,
            "candidate_input_receipt_sha256": candidate_input["content_sha256"],
            "parent_id": candidate_input["parent_id"],
            "video_id": candidate_input["video_id"],
            "physical_frame_id": candidate_input["physical_frame_id"],
            "timestamp_token": candidate_input["timestamp_token"],
            "input_color_decoded_content_sha256": candidate_input["color_decoded_content_sha256"],
            "effective_intrinsics_sha256": candidate_input["effective_intrinsics_sha256"],
            "input_tensor_shape_nchw": list(tensor.shape),
            "input_tensor_sha256": adapter.canonical_sha256(tensor),
            "resized_intrinsics_sha256": adapter.canonical_sha256(resized_k),
            "native_output_shape_hw": list(native.shape),
            "native_output_dtype": "float32",
            "native_output_array_sha256": adapter.canonical_sha256(native),
            "highres_output_shape_hw": list(highres.shape),
            "highres_output_dtype": "float32",
            "highres_output_array_sha256": adapter.canonical_sha256(highres),
            "runtime_identity": dict(runtime_identity),
            "truth_alignment_used": False,
            "truth_payload_read": False,
        }
    )
    return validate_depthart_inference_receipt(receipt)


def install_timm_compat_shim() -> bool:
    try:
        from timm.models.layers.helpers import to_2tuple  # type: ignore # noqa: F401

        return False
    except (ImportError, ValueError):
        from timm.layers.helpers import to_2tuple  # type: ignore

        module = types.ModuleType("timm.models.layers.helpers")
        module.to_2tuple = to_2tuple
        sys.modules["timm.models.layers.helpers"] = module
        return True


def load_official_depthart(
    source_root: Path,
    checkpoint: Path,
    *,
    device: str = "cuda",
    seed: int = 0,
) -> tuple[Any, dict[str, Any]]:
    """Load the exact official fp32 model under frozen deterministic settings."""

    import torch

    source = source_root.resolve()
    weights = checkpoint.resolve()
    require(device == "cuda" and torch.cuda.is_available(), "CUDA_REQUIRED", "frozen TARO DepthART inference requires CUDA")
    require(source.is_dir() and weights.is_file(), "DEPTHART_ASSET_MISSING", "DepthART source/checkpoint is missing")
    commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(source), "status", "--short"], capture_output=True, text=True, check=True).stdout.strip()
    require(commit == EXPECTED_SOURCE_GIT_COMMIT and not dirty, "DEPTHART_SOURCE_DRIFT", "DepthART source identity or cleanliness drift", commit=commit, dirty=dirty)
    require(sha256_file(weights) == adapter.BASELINE_CHECKPOINT_SHA256, "DEPTHART_CHECKPOINT_DRIFT", "DepthART checkpoint SHA-256 drift")
    compat_shim = install_timm_compat_shim()
    deployment = Path(__file__).resolve().parents[1] / "hftf" / "deployment" / "depthart"
    paths = [source / "metric", source / "deploy" / "shared", source / "deploy" / "shared" / "selective_scan", deployment]
    for path in reversed(paths):
        sys.path.insert(0, str(path))
    from depthart_selective_scan import install_depthart  # type: ignore
    from model import load_model  # type: ignore
    from network import tvimblock  # type: ignore

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = load_model(str(weights), "S", "indoor", device).eval()
    model.requires_grad_(False)
    previous_scan = install_depthart(tvimblock)
    identity = {
        "source_git_commit": commit,
        "source_tree_clean": True,
        "checkpoint_bytes": weights.stat().st_size,
        "torch_version": str(torch.__version__),
        "cuda_version": str(torch.version.cuda),
        "opencv_version": str(cv2.__version__),
        "numpy_version": str(np.__version__),
        "device": device,
        "cuda_device_name": str(torch.cuda.get_device_name(torch.cuda.current_device())),
        "tf32_matmul": False,
        "tf32_cudnn": False,
        "cudnn_benchmark": False,
        "autocast": False,
        "inference_dtype": "float32",
        "seed": seed,
        "timm_compat_shim": compat_shim,
        "selective_scan_backend": "depthart_selective_scan.cross_selective_scan",
        "selective_scan_replaced": f"{previous_scan.__module__}.{previous_scan.__name__}",
    }
    return model, _canonical_copy(identity)


def infer_depthart_candidate(
    model: Any,
    *,
    candidate_input_receipt: dict[str, Any],
    color_rgb_u8: np.ndarray,
    runtime_identity: Mapping[str, Any],
    device: str = "cuda",
) -> dict[str, Any]:
    """Run and seal one truth-blind candidate; returns arrays only in memory."""

    import torch

    candidate_input = validate_candidate_input_receipt(candidate_input_receipt)
    matrix = np.asarray(candidate_input["intrinsics_highres"]["matrix_3x3"], dtype=np.float32)
    tensor, resized_k = preprocess_depthart_input(color_rgb_u8, matrix)
    with torch.inference_mode():
        prediction = model(torch.from_numpy(tensor).to(device), torch.from_numpy(resized_k).to(device))
    native = prediction.detach().float().cpu().numpy()
    require(native.shape == (1, *NATIVE_SHAPE_HW), "NATIVE_DEPTH_INVALID", "official DepthART output must be 1x448x608", actual=list(native.shape))
    native = np.ascontiguousarray(native[0], dtype=np.float32)
    require(bool(np.all(np.isfinite(native))), "NATIVE_DEPTH_INVALID", "official DepthART output contains non-finite values")
    highres = upsample_native_depth(native)
    inference_receipt = build_depthart_inference_receipt(
        candidate_input_receipt=candidate_input,
        color_rgb_u8=color_rgb_u8,
        input_tensor_nchw=tensor,
        resized_intrinsics_n33=resized_k,
        native_depth_m=native,
        highres_depth_m=highres,
        runtime_identity=runtime_identity,
    )
    return {
        "native_depth_m": native,
        "highres_depth_m": highres,
        "inference_receipt": inference_receipt,
    }


def bind_sealed_candidate_to_source(
    *,
    inference_receipt: dict[str, Any],
    native_depth_m: np.ndarray,
    source_frame_receipt: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Join a sealed candidate to full source identity only after candidate phase."""

    inference = validate_depthart_inference_receipt(inference_receipt)
    source = adapter._validate_base_receipt(source_frame_receipt)
    require(source["source_role"] == "O0R_EVAL_CANDIDATE", "CANDIDATE_SOURCE_ROLE_INVALID", "candidate/source join requires a frozen eval receipt")
    require(
        inference["parent_id"] == source["parent_id"]
        and inference["video_id"] == source["session_id"]
        and inference["physical_frame_id"] == source["physical_frame_id"]
        and inference["timestamp_token"] == source["sensor_timestamp"]["decimal_token"],
        "CANDIDATE_SOURCE_JOIN_MISMATCH",
        "sealed candidate and source identities differ",
    )
    require(inference["input_color_decoded_content_sha256"] == source["decoded_payload_bindings"]["color"]["decoded_content_sha256"], "CANDIDATE_SOURCE_JOIN_MISMATCH", "sealed candidate RGB binding differs from full source receipt")
    require(inference["effective_intrinsics_sha256"] == adapter.canonical_sha256(source["intrinsics_highres"]), "CANDIDATE_SOURCE_JOIN_MISMATCH", "sealed candidate K binding differs from full source receipt")
    native = np.asarray(native_depth_m, dtype=np.float32)
    require(native.shape == NATIVE_SHAPE_HW and adapter.canonical_sha256(native) == inference["native_output_array_sha256"], "CANDIDATE_NATIVE_JOIN_MISMATCH", "sealed native output differs from inference receipt")
    highres = upsample_native_depth(native)
    require(adapter.canonical_sha256(highres) == inference["highres_output_array_sha256"], "CANDIDATE_HIGHRES_JOIN_MISMATCH", "reconstructed candidate output differs from inference receipt")
    candidate_output_receipt = adapter.build_candidate_depth_output_receipt(highres, source, inference_receipt_sha256=inference["content_sha256"])
    require(candidate_output_receipt["output_array_sha256"] == inference["highres_output_array_sha256"], "CANDIDATE_RECEIPT_BINDING_MISMATCH", "candidate and inference output hashes differ")
    return highres, candidate_output_receipt


__all__ = [
    "DepthARTRuntimeError",
    "CANDIDATE_INPUT_RECEIPT_SCHEMA",
    "EXPECTED_SOURCE_GIT_COMMIT",
    "HIGHRES_SHAPE_HW",
    "INFERENCE_RECEIPT_SCHEMA",
    "NATIVE_SHAPE_HW",
    "POSTPROCESS_ID",
    "PREPROCESS_ID",
    "bind_sealed_candidate_to_source",
    "build_candidate_input_receipt",
    "build_depthart_inference_receipt",
    "candidate_input_from_bound_source",
    "decode_npy_gzip_bytes",
    "deterministic_npy_gzip_bytes",
    "infer_depthart_candidate",
    "load_official_depthart",
    "lower_bound_size",
    "preprocess_depthart_input",
    "sha256_file",
    "upsample_native_depth",
    "validate_depthart_inference_receipt",
    "validate_candidate_input_receipt",
    "write_native_depth_blob",
]
