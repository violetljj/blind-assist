#!/usr/bin/env python3
"""Fit a label-free, fold-local normalization artifact for the U0 dense teacher.

Only training-fold videos and frame ledgers are accepted.  Human event labels,
review/adjudication fields, held-out inputs, blind inputs and future frames are
structurally forbidden.  The resulting artifact calibrates relative-depth and
boundary ranges; it does not fit an event classifier or authorize production.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

MODULE = Path(__file__).resolve().parent / "research" / "ustrf_sc"
sys.path.insert(0, str(MODULE))

from dense_teacher_field import (  # noqa: E402
    ARTIFACT_SCHEMA,
    MODEL_LICENSE,
    MODEL_NAME,
    MODEL_VERSION,
    DepthAnythingOnnxTeacher,
    DenseTeacherError,
    canonical_json_bytes,
    decode_video_frame_rgb,
    normalized_text_sha256,
    sha256_bytes,
    sha256_file,
)


CALIBRATION_SCHEMA = "blindassist_ustrf_sc_u0_dense_teacher_calibration_inputs_v1"
TRAINING_MANIFEST_SCHEMA = "blindassist_ustrf_sc_u0_fold_training_input_manifest_v1"
TRAINING_RECEIPT_SCHEMA = "blindassist_ustrf_sc_u0_fold_training_receipt_v1"
ARM_ID = "teacher_dense_explicit_route"
ADAPTER_ID = "teacher_dense_explicit_route_adapter_v1"
FIT_POLICY = "leave_one_session_out_fit_v1"
FORBIDDEN_KEYS = {
    "should_alert", "event_label", "event_labels", "review", "reviews", "adjudication",
    "actionability", "critical_event", "blind", "blind_label", "risk_truth",
}


class FitError(ValueError):
    pass


def load_json(path: Path, *, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FitError(f"cannot load {where}: {path}") from error
    if not isinstance(value, dict):
        raise FitError(f"{where} must be an object")
    return value


def reject_forbidden_keys(value: Any, *, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise FitError(f"forbidden label/review field at {path}.{key}")
            reject_forbidden_keys(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_keys(child, path=f"{path}[{index}]")


def resolve_bound(root: Path, relative: Any, expected_sha: Any, *, where: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise FitError(f"{where} path is missing")
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FitError(f"{where} escapes inference root") from error
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise FitError(f"{where} file/hash mismatch")
    return path


def validate_inputs(
    training_manifest: Mapping[str, Any],
    calibration_inputs: Mapping[str, Any],
    *,
    training_manifest_sha256: str,
) -> None:
    reject_forbidden_keys(calibration_inputs)
    expected = {
        "schema": TRAINING_MANIFEST_SCHEMA,
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "fit_policy": FIT_POLICY,
        "held_out_inputs_used": False,
        "blind_accessed": False,
        "future_inputs_used": False,
    }
    for key, value in expected.items():
        if training_manifest.get(key) != value:
            raise FitError(f"training manifest {key} mismatch")
    calibration_expected = {
        "schema": CALIBRATION_SCHEMA,
        "contract_id": training_manifest.get("contract_id"),
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "fit_policy": FIT_POLICY,
        "held_out_session_id": training_manifest.get("held_out_session_id"),
        "training_input_manifest_sha256": training_manifest_sha256,
        "training_session_ids": training_manifest.get("training_session_ids"),
        "held_out_inputs_used": False,
        "blind_accessed": False,
        "future_inputs_used": False,
        "human_event_truth_used": False,
        "production_authorized": False,
    }
    for key, value in calibration_expected.items():
        if calibration_inputs.get(key) != value:
            raise FitError(f"calibration inputs {key} mismatch")
    held_out = training_manifest.get("held_out_session_id")
    sessions = training_manifest.get("training_session_ids")
    episodes = training_manifest.get("training_episode_ids")
    rows = calibration_inputs.get("episodes")
    if (
        not isinstance(sessions, list) or not sessions or sessions != sorted(set(sessions)) or held_out in sessions
        or not isinstance(episodes, list) or not episodes or episodes != sorted(set(episodes))
        or not isinstance(rows, list) or not rows
    ):
        raise FitError("training fold inventory is invalid")
    row_ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or row.get("session_id") not in sessions:
            raise FitError("calibration episode session is outside the training fold")
        episode_id = row.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise FitError("calibration episode ID is invalid")
        row_ids.append(episode_id)
        frames = row.get("frames")
        if not isinstance(frames, list) or not frames:
            raise FitError("calibration episode frame inventory is empty")
        previous = -1
        for frame in frames:
            if not isinstance(frame, dict):
                raise FitError("calibration frame must be an object")
            pts = frame.get("video_pts_ms")
            if not isinstance(pts, int) or isinstance(pts, bool) or pts <= previous:
                raise FitError("calibration frame timestamps must increase")
            previous = pts
    if sorted(row_ids) != episodes or len(row_ids) != len(set(row_ids)):
        raise FitError("calibration episode inventory differs from the exact training fold")


def fit_calibration(raw_depths: list[Any]) -> dict[str, float | int]:
    if not raw_depths:
        raise FitError("no training depth outputs were produced")
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as error:
        raise FitError("numpy/opencv are required for dense teacher fitting") from error
    flattened = np.concatenate([np.asarray(depth, dtype=np.float32).reshape(-1) for depth in raw_depths])
    if not np.isfinite(flattened).all():
        raise FitError("training teacher outputs contain non-finite values")
    low, high = [float(value) for value in np.quantile(flattened, [0.02, 0.98])]
    if not low < high:
        raise FitError("training depth distribution is degenerate")
    gradients: list[Any] = []
    for depth in raw_depths:
        normalized = np.clip((depth - low) / (high - low), 0.0, 1.0).astype(np.float32)
        gx = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(np.sqrt(gx * gx + gy * gy).reshape(-1))
    gradient = float(np.quantile(np.concatenate(gradients), 0.98))
    if not gradient > 0:
        raise FitError("training depth-gradient distribution is degenerate")
    return {
        "raw_depth_lower_quantile": low,
        "raw_depth_upper_quantile": high,
        "gradient_upper_quantile": gradient,
        "lower_quantile": 0.02,
        "upper_quantile": 0.98,
        "gradient_quantile": 0.98,
        "training_frame_count": len(raw_depths),
        "training_value_count": int(flattened.size),
    }


def run(args: argparse.Namespace) -> None:
    for output in (args.artifact_output, args.training_receipt_output):
        if output.exists():
            raise FitError(f"refusing to overwrite output: {output}")
    training_manifest = load_json(args.training_input_manifest, where="training input manifest")
    calibration_inputs = load_json(args.calibration_inputs, where="calibration inputs")
    training_manifest_sha = sha256_file(args.training_input_manifest)
    validate_inputs(training_manifest, calibration_inputs, training_manifest_sha256=training_manifest_sha)
    teacher = DepthAnythingOnnxTeacher(args.teacher_model)
    if teacher.model_sha256 != args.expected_teacher_model_sha256:
        raise FitError("teacher model SHA differs from the frozen command")
    raw_depths: list[Any] = []
    sample_rows: list[dict[str, Any]] = []
    inference_root = args.inference_root.resolve()
    for episode in calibration_inputs["episodes"]:
        video = resolve_bound(
            inference_root, episode.get("video_path"), episode.get("video_sha256"),
            where=f"{episode['episode_id']} video",
        )
        for frame in episode["frames"]:
            rgb, decode_ms = decode_video_frame_rgb(video, frame["video_pts_ms"])
            depth, inference_ms = teacher.infer_rgb(rgb)
            raw_depths.append(depth)
            sample_rows.append({
                "session_id": episode["session_id"],
                "episode_id": episode["episode_id"],
                "frame_id": frame["frame_id"],
                "video_pts_ms": frame["video_pts_ms"],
                "teacher_decoded_rgb_sha256": sha256_bytes(rgb.tobytes(order="C")),
                "raw_depth_sha256": sha256_bytes(depth.astype("float32").tobytes(order="C")),
                "decode_duration_ms": decode_ms,
                "inference_duration_ms": inference_ms,
            })
    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "contract_id": training_manifest["contract_id"],
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "fit_policy": FIT_POLICY,
        "held_out_session_id": training_manifest["held_out_session_id"],
        "training_session_ids": training_manifest["training_session_ids"],
        "training_episode_ids": training_manifest["training_episode_ids"],
        "training_input_manifest_sha256": training_manifest_sha,
        "training_sample_inventory_sha256": sha256_file(args.calibration_inputs),
        "teacher_model_name": MODEL_NAME,
        "teacher_model_version": MODEL_VERSION,
        "teacher_model_license": MODEL_LICENSE,
        "teacher_model_sha256": teacher.model_sha256,
        "fit_implementation_sha256": normalized_text_sha256(Path(__file__)),
        "calibration": fit_calibration(raw_depths),
        "training_samples": sample_rows,
        "held_out_inputs_used": False,
        "blind_accessed": False,
        "future_inputs_used": False,
        "human_event_truth_used": False,
        "production_authorized": False,
        "authority": "teacher_auxiliary_normalization_only",
    }
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_output.write_bytes(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")
    receipt = {
        "schema": TRAINING_RECEIPT_SCHEMA,
        "contract_id": training_manifest["contract_id"],
        "arm_id": ARM_ID,
        "candidate_adapter_id": ADAPTER_ID,
        "fit_policy": FIT_POLICY,
        "held_out_session_id": training_manifest["held_out_session_id"],
        "training_input_manifest_sha256": training_manifest_sha,
        "artifact_sha256": sha256_file(args.artifact_output),
        "fit_executed": True,
        "held_out_inputs_used": False,
        "blind_accessed": False,
        "future_inputs_used": False,
        "provenance_completed": True,
        "failure_count": 0,
    }
    args.training_receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.training_receipt_output.write_bytes(json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8") + b"\n")
    print(json.dumps({
        "status": "complete",
        "artifact": str(args.artifact_output.resolve()),
        "artifact_sha256": sha256_file(args.artifact_output),
        "training_receipt": str(args.training_receipt_output.resolve()),
        "training_frame_count": len(sample_rows),
        "authority": "teacher_auxiliary_normalization_only",
    }, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-input-manifest", type=Path, required=True)
    parser.add_argument("--calibration-inputs", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--expected-teacher-model-sha256", required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--training-receipt-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except (FitError, DenseTeacherError) as error:
        print(f"USTRF dense teacher LOSO fit failed: {error}", file=sys.stderr)
        raise SystemExit(2)
