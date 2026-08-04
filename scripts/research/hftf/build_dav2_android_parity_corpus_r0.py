#!/usr/bin/env python3
"""Build the fixed DA-V2 Android parity corpus and host references.

The corpus deliberately keeps neural-network parity (fixed normalized tensor to
raw metric depth) separate from the downstream geometry canaries, which are
derived from the clean PyTorch depth after resizing to camera coordinates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SCALE_DIR = REPO_ROOT / "scripts" / "research" / "known_camera_height_ground_scale_r0"
HFTF_DIR = REPO_ROOT / "scripts" / "research" / "hftf"
sys.path[:0] = [str(SCALE_DIR), str(HFTF_DIR)]

from evaluate_camera_conditioned_student_r0 import runtime_features
from evaluate_consumed_tartanground import strict_band_values
from evaluate_metric3d_clearance_field_a0 import clearance_field
from sealed_student import SealedScaleStudent

import core as scale_core

SOURCE_RELATIVE = Path(
    "artifacts.local/evidence/hftf/"
    "stage-c-d5-tartanground-outcome-unseen-transfer-v0/media/transfer/"
    "GothicIsland/Data_diff/P1000/image/000486.png"
)
SOURCE_SHA256 = "BC4C44551CB77A759E554516467108C270A99E0ADB0AF3AAF32AA65D117D2EAA"
CHECKPOINT_SHA256 = "B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545"
PARENT_ID = "GothicIsland/Data_diff/P1000"
ANCHOR_FRAME_ID = 486
CAMERA_HEIGHT_M = 1.0341161949454936
CAMERA_INTRINSICS = np.asarray(
    [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
INPUT_HEIGHT = 518
INPUT_WIDTH = 686
SCENARIOS = ("clean", "gaussian_sigma3", "motion_horizontal_length17")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def center_crop_camera_frame(bgr: np.ndarray) -> np.ndarray:
    if bgr.shape != (640, 640, 3):
        raise ValueError(f"expected frozen 640x640 source, got {bgr.shape}")
    return np.ascontiguousarray(bgr[80:560, :, :])


def perturb_bgr(clean: np.ndarray, scenario: str) -> np.ndarray:
    if scenario == "clean":
        return clean.copy()
    if scenario == "gaussian_sigma3":
        sigma = 3.0
        kernel = 2 * int(np.ceil(3.0 * sigma)) + 1
        return cv2.GaussianBlur(clean, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    if scenario == "motion_horizontal_length17":
        length = 17
        kernel = np.zeros((length, length), dtype=np.float32)
        kernel[length // 2, :] = 1.0 / length
        return cv2.filter2D(clean, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
    raise ValueError(f"unsupported scenario: {scenario}")


def official_preprocess(model: Any, bgr: np.ndarray) -> np.ndarray:
    tensor, original_shape = model.image2tensor(bgr, input_size=INPUT_HEIGHT)
    if original_shape != (480, 640):
        raise ValueError(f"unexpected original shape receipt: {original_shape}")
    array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    if array.shape != (1, 3, INPUT_HEIGHT, INPUT_WIDTH):
        raise ValueError(f"unexpected official input shape: {array.shape}")
    return array


def depth_to_camera_coordinates(torch: Any, raw_depth: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(raw_depth, dtype=np.float32))[None, None]
    resized = torch.nn.functional.interpolate(
        tensor, (480, 640), mode="bilinear", align_corners=True
    )[0, 0]
    return np.ascontiguousarray(resized.numpy(), dtype=np.float32)


def local_horizontal_linear(depth: np.ndarray) -> np.ndarray:
    base = np.linspace(-1.0, 1.0, depth.shape[1], dtype=np.float64)[None, :]
    multiplier = 1.0 + 0.20 * np.broadcast_to(base, depth.shape)
    multiplier /= float(np.median(multiplier))
    return np.asarray(depth, dtype=np.float64) * multiplier


def mask_lower_roi_half(depth: np.ndarray) -> np.ndarray:
    output = np.asarray(depth, dtype=np.float64).copy()
    y0 = int(np.ceil(scale_core.LOWER_ROI_START_FRACTION * output.shape[0]))
    roi_height = output.shape[0] - y0
    rows = max(1, round(roi_height * 0.50))
    output[output.shape[0] - rows :, :] = np.nan
    return output


def parity_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    expected = np.asarray(reference, dtype=np.float64)
    actual = np.asarray(candidate, dtype=np.float64)
    if actual.shape != expected.shape:
        raise ValueError(
            f"ORT/PyTorch shape mismatch: {actual.shape} != {expected.shape}"
        )
    difference = np.abs(actual - expected)
    denominator = np.maximum(np.abs(expected), 1e-6)
    return {
        "element_count": int(difference.size),
        "maximum_absolute_error_m": float(np.max(difference)),
        "mean_absolute_error_m": float(np.mean(difference)),
        "root_mean_squared_error_m": float(np.sqrt(np.mean(np.square(difference)))),
        "p95_absolute_error_m": float(np.quantile(difference, 0.95)),
        "maximum_relative_error": float(np.max(difference / denominator)),
        "allclose_rtol_1e_4_atol_1e_4": bool(
            np.allclose(actual, expected, rtol=1e-4, atol=1e-4)
        ),
    }


def downstream_reference(
    depth: np.ndarray, student: SealedScaleStudent
) -> dict[str, Any]:
    receipt = scale_core.CameraHeightReceipt(PARENT_ID, PARENT_ID, CAMERA_HEIGHT_M, 0.0)
    recovery = scale_core.recover_metric_scale(
        depth, CAMERA_INTRINSICS, receipt, PARENT_ID, PARENT_ID
    )
    result: dict[str, Any] = {
        "recovery_status": recovery.get("status"),
        "recovery_reason": recovery.get("reason"),
        "finite_depth_fraction": float(np.mean(np.isfinite(depth) & (depth > 0.0))),
    }
    if recovery.get("status") != "VALID":
        result.update(student_status="UNKNOWN", student_reason=recovery.get("reason"))
        return result
    plane = recovery["ground"]
    features = runtime_features(depth, CAMERA_HEIGHT_M, recovery)
    if features is None:
        result.update(
            student_status="UNKNOWN", student_reason="INVALID_RUNTIME_FEATURES"
        )
        return result
    prediction = student.predict(features)
    result.update(
        r0_known_height_scale=float(recovery["scale"]),
        plane={
            "normal": np.asarray(plane.normal).tolist(),
            "relative_height": float(plane.relative_height),
            "normalized_median_residual": float(plane.normalized_median_residual),
            "candidate_count": int(plane.candidate_count),
            "inlier_count": int(plane.inlier_count),
            "inlier_fraction": float(plane.inlier_fraction),
        },
        runtime_features=features.tolist(),
        student_status=prediction["status"],
        student_reason=prediction.get("reason"),
        student_log_scale=prediction.get("log_scale"),
        student_scale=prediction.get("scale"),
    )
    if prediction["status"] != "VALID":
        return result
    metric_depth = np.asarray(depth, dtype=np.float64) * float(prediction["scale"])
    field = clearance_field(
        metric_depth,
        CAMERA_INTRINSICS,
        plane_override=(
            np.asarray(plane.normal, dtype=np.float64),
            CAMERA_HEIGHT_M,
            float(plane.normalized_median_residual) * CAMERA_HEIGHT_M,
        ),
    )
    result["clearance_field_status"] = field.get("status")
    result["strict_band_clearance_m"] = strict_band_values(field)
    return result


def load_model(repo: Path, checkpoint: Path, device: str) -> tuple[Any, Any]:
    sys.path.insert(0, str(repo / "metric_depth"))
    import torch
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(torch.device(device)).eval()
    return torch, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=REPO_ROOT / SOURCE_RELATIVE)
    parser.add_argument(
        "--depth-anything-repo",
        type=Path,
        default=REPO_ROOT
        / "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPO_ROOT
        / "artifacts.local/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth",
    )
    parser.add_argument(
        "--onnx",
        type=Path,
        default=REPO_ROOT
        / "artifacts.local/models/dav2-metric-hypersim-vits-android-r0/model_518x686.onnx",
    )
    parser.add_argument(
        "--student",
        type=Path,
        default=REPO_ROOT
        / "configs/hftf/camera_conditioned_scale_student_r0_model.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts.local/evidence/hftf/dav2-android-parity-r0",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if sha256(args.source) != SOURCE_SHA256:
        raise ValueError("frozen RGB source hash mismatch")
    if sha256(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("exact DA-V2 checkpoint hash mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    torch, model = load_model(args.depth_anything_repo, args.checkpoint, args.device)

    import onnxruntime as ort

    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    input_metadata = session.get_inputs()[0]
    output_metadata = session.get_outputs()[0]
    if input_metadata.shape != [1, 3, INPUT_HEIGHT, INPUT_WIDTH]:
        raise ValueError(f"unexpected ONNX input: {input_metadata.shape}")
    if output_metadata.shape != [1, INPUT_HEIGHT, INPUT_WIDTH]:
        raise ValueError(f"unexpected ONNX output: {output_metadata.shape}")

    source = cv2.imread(str(args.source), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"unable to decode {args.source}")
    clean = center_crop_camera_frame(source)
    records = []
    clean_camera_depth = None
    with torch.inference_mode():
        for scenario in SCENARIOS:
            directory = args.output / scenario
            directory.mkdir(parents=True, exist_ok=True)
            bgr = perturb_bgr(clean, scenario)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_path = directory / "rgb_640x480.png"
            if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise OSError(f"failed to write {rgb_path}")
            raw_rgb_path = directory / "rgb_640x480_uint8.npy"
            np.save(raw_rgb_path, rgb)
            normalized = official_preprocess(model, bgr)
            input_path = directory / "normalized_nchw_fp32_1x3x518x686.npy"
            np.save(input_path, normalized)
            raw_depth = (
                model(torch.from_numpy(normalized).to(args.device))
                .detach()
                .cpu()
                .numpy()
            )
            raw_depth = np.ascontiguousarray(raw_depth, dtype=np.float32)
            pytorch_path = directory / "pytorch_fp32_raw_depth_1x518x686.npy"
            np.save(pytorch_path, raw_depth)
            ort_depth = np.asarray(
                session.run([output_metadata.name], {input_metadata.name: normalized})[
                    0
                ],
                dtype=np.float32,
            )
            ort_path = directory / "ort_cpu_raw_depth_1x518x686.npy"
            np.save(ort_path, ort_depth)
            metrics = parity_metrics(raw_depth, ort_depth)
            metrics_path = directory / "host_ort_parity.json"
            write_json(metrics_path, metrics)
            records.append(
                {
                    "scenario": scenario,
                    "files": {
                        path.name: sha256(path)
                        for path in (
                            rgb_path,
                            raw_rgb_path,
                            input_path,
                            pytorch_path,
                            ort_path,
                            metrics_path,
                        )
                    },
                    "host_ort_parity": metrics,
                }
            )
            if scenario == "clean":
                clean_camera_depth = depth_to_camera_coordinates(torch, raw_depth[0])

    if clean_camera_depth is None:
        raise AssertionError("clean scenario missing")
    downstream_dir = args.output / "downstream"
    downstream_dir.mkdir(parents=True, exist_ok=True)
    downstream_depths = {
        "clean": clean_camera_depth,
        "lower_roi_full_width_bottom_50pct_nan": mask_lower_roi_half(
            clean_camera_depth
        ),
        "local_horizontal_linear_amplitude20pct_polarity_p1": local_horizontal_linear(
            clean_camera_depth
        ),
    }
    student = SealedScaleStudent.load(args.student)
    downstream = []
    for scenario, depth in downstream_depths.items():
        path = downstream_dir / f"{scenario}_depth_640x480_fp32.npy"
        np.save(path, np.asarray(depth, dtype=np.float32))
        downstream.append(
            {
                "scenario": scenario,
                "depth_sha256": sha256(path),
                "reference": downstream_reference(depth, student),
            }
        )

    manifest = {
        "schema": "blindassist_dav2_android_parity_corpus_r0_v1",
        "status": "HOST_REFERENCE_GENERATED",
        "model": {
            "identity": "Depth Anything V2 Metric Hypersim ViT-S",
            "checkpoint_sha256": sha256(args.checkpoint),
            "onnx_sha256": sha256(args.onnx),
            "onnx_input": {"name": input_metadata.name, "shape": input_metadata.shape},
            "onnx_output": {
                "name": output_metadata.name,
                "shape": output_metadata.shape,
            },
        },
        "source": {
            "parent_id": PARENT_ID,
            "anchor_frame_id": ANCHOR_FRAME_ID,
            "source_sha256": sha256(args.source),
            "crop_xyxy": [0, 80, 640, 560],
            "camera_shape_hw": [480, 640],
            "camera_intrinsics": CAMERA_INTRINSICS.tolist(),
            "camera_height_m": CAMERA_HEIGHT_M,
        },
        "preprocess": {
            "implementation": "official DepthAnythingV2.image2tensor",
            "input_size": 518,
            "normalized_nchw_shape": [1, 3, 518, 686],
            "rgb_mean": [0.485, 0.456, 0.406],
            "rgb_std": [0.229, 0.224, 0.225],
        },
        "neural_parity_records": records,
        "downstream_reference_records": downstream,
    }
    manifest_path = args.output / "manifest.json"
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "records": records,
                "downstream": downstream,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
