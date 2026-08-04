#!/usr/bin/env python3
"""Materialize and compare every numeric layer of one real CameraX YUV frame."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HEIGHT = 480
WIDTH = 640
OUTPUT_HEIGHT = 518
OUTPUT_WIDTH = 686
ELEMENTS = 3 * OUTPUT_HEIGHT * OUTPUT_WIDTH


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    mismatch = np.flatnonzero(actual != expected)
    return {
        "elements": int(actual.size),
        "mismatch_elements": int(mismatch.size),
        "first_mismatch_index": int(mismatch[0]) if mismatch.size else None,
        "mean_abs": float(np.mean(difference)),
        "p95_abs": float(np.quantile(difference, 0.95)),
        "max_abs": float(np.max(difference)),
    }


def half_comparison(actual: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    actual_bits = np.asarray(actual, dtype=np.float16).view(np.uint16).reshape(-1)
    expected_bits = np.asarray(expected, dtype=np.float16).view(np.uint16).reshape(-1)
    mismatch = np.flatnonzero(actual_bits != expected_bits)
    result = comparison(actual.astype(np.float32), expected.astype(np.float32))
    result.update(
        bit_mismatch_elements=int(mismatch.size),
        first_bit_mismatch_index=int(mismatch[0]) if mismatch.size else None,
        first_actual_bits=(f"0x{int(actual_bits[mismatch[0]]):04x}" if mismatch.size else None),
        first_expected_bits=(f"0x{int(expected_bits[mismatch[0]]):04x}" if mismatch.size else None),
    )
    return result


def reproduce_rgb(root: Path, rotation: int) -> np.ndarray:
    y = np.fromfile(root / "camerax_y_640x480_u8.raw", dtype=np.uint8).reshape(HEIGHT, WIDTH)
    u = np.fromfile(root / "camerax_u_320x240_u8.raw", dtype=np.uint8).reshape(HEIGHT // 2, WIDTH // 2)
    v = np.fromfile(root / "camerax_v_320x240_u8.raw", dtype=np.uint8).reshape(HEIGHT // 2, WIDTH // 2)
    i420 = np.concatenate((y.reshape(-1), u.reshape(-1), v.reshape(-1))).reshape(HEIGHT * 3 // 2, WIDTH)
    rgb = cv2.cvtColor(i420, cv2.COLOR_YUV2RGB_I420)
    if rotation == 90:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        rgb = cv2.rotate(rgb, cv2.ROTATE_180)
    elif rotation == 270:
        rgb = cv2.rotate(rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation != 0:
        raise ValueError(f"unsupported rotation: {rotation}")
    crop_width = min(rgb.shape[1], rgb.shape[0] * 4 // 3)
    crop_height = min(rgb.shape[0], rgb.shape[1] * 3 // 4)
    x0 = (rgb.shape[1] - crop_width) // 2
    y0 = (rgb.shape[0] - crop_height) // 2
    crop = rgb[y0 : y0 + crop_height, x0 : x0 + crop_width]
    return np.ascontiguousarray(cv2.resize(crop, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR))


def official_preprocess(depth_anything_repo: Path, rgb: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(depth_anything_repo / "metric_depth"))
    from depth_anything_v2.dpt import DepthAnythingV2

    model = DepthAnythingV2(
        encoder="vits", features=64, out_channels=[48, 96, 192, 384], max_depth=20.0
    )
    tensor, original_shape = model.image2tensor(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), input_size=OUTPUT_HEIGHT)
    if original_shape != (HEIGHT, WIDTH):
        raise ValueError(f"official shape receipt mismatch: {original_shape}")
    value = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype=np.float32)
    if value.shape != (1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH):
        raise ValueError(f"official tensor shape mismatch: {value.shape}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument(
        "--depth-anything-repo",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main",
    )
    args = parser.parse_args()
    root = args.capture_root.resolve()
    capture = json.loads((root / "capture.json").read_text(encoding="utf-8"))
    saved_rgb = np.fromfile(root / "rgb_crop_640x480_uint8.raw", dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
    reproduced_rgb = reproduce_rgb(root, int(capture["rotation_degrees"]))
    cv2.imwrite(str(root / "rgb_crop_640x480.png"), cv2.cvtColor(saved_rgb, cv2.COLOR_RGB2BGR))

    native_resized = np.fromfile(root / "native_fast_resized_hwc_fp32.raw", dtype="<f4").reshape(
        OUTPUT_HEIGHT, OUTPUT_WIDTH, 3
    )
    host_resized = cv2.resize(
        saved_rgb.astype(np.float32) / np.float32(255.0),
        (OUTPUT_WIDTH, OUTPUT_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    means = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    normalized_from_native_resize = np.ascontiguousarray(
        ((native_resized - means) / std).transpose(2, 0, 1)[None], dtype=np.float32
    )
    fast_native_fp32 = np.fromfile(root / "native_fast_normalized_nchw_fp32.raw", dtype="<f4").reshape(
        1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH
    )
    fast_native_fp16 = np.fromfile(root / "native_fast_normalized_nchw_fp16.raw", dtype="<f2").reshape(
        1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH
    )
    native_fp32 = np.fromfile(root / "native_normalized_nchw_fp32.raw", dtype="<f4").reshape(
        1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH
    )
    native_fp16 = np.fromfile(root / "native_normalized_nchw_fp16.raw", dtype="<f2").reshape(
        1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH
    )
    official_fp32 = official_preprocess(args.depth_anything_repo, saved_rgb)
    official_fp16 = np.ascontiguousarray(official_fp32.astype(np.float16))
    np.save(root / "official_normalized_nchw_fp32.npy", official_fp32)
    np.save(root / "official_normalized_nchw_fp16.npy", official_fp16)
    official_fp32.astype("<f4").tofile(root / "official_normalized_nchw_fp32.raw")
    official_fp16.astype("<f2").tofile(root / "official_normalized_nchw_fp16.raw")

    stages: dict[str, Any] = {
        "pixel_conversion_host_opencv_vs_app_rgb": comparison(reproduced_rgb, saved_rgb),
        "fast_cubic_resize_host_opencv_vs_fast_native_resized": comparison(host_resized, native_resized),
        "fast_normalization_pack_from_native_resize_vs_fast_native_fp32": comparison(
            normalized_from_native_resize, fast_native_fp32
        ),
        "fast_native_fp32_vs_official_fp32": comparison(fast_native_fp32, official_fp32),
        "fast_native_strict_fp16_vs_official_fp16": half_comparison(fast_native_fp16, official_fp16),
        "native_fp32_vs_official_fp32": comparison(native_fp32, official_fp32),
        "native_strict_fp16_vs_official_fp16": half_comparison(native_fp16, official_fp16),
    }
    app_depth = np.fromfile(root / "app_qnn_depth_fp16.raw", dtype="<f2").astype(np.float32)
    cli_path = root / "cli_qnn_depth_fp16.raw"
    if cli_path.is_file():
        cli_depth = np.fromfile(cli_path, dtype="<f2").astype(np.float32)
        stages["app_qnn_depth_vs_cli_qnn_depth"] = comparison(app_depth, cli_depth)

    files = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name not in {"parity.json"}
    }
    gate_failures = []
    if stages["pixel_conversion_host_opencv_vs_app_rgb"]["mismatch_elements"] != 0:
        gate_failures.append("pixel_conversion")
    if stages["native_fp32_vs_official_fp32"]["max_abs"] != 0:
        gate_failures.append("native_fp32")
    if stages["native_strict_fp16_vs_official_fp16"]["bit_mismatch_elements"] != 0:
        gate_failures.append("half_rounding_input_identity")
    if "app_qnn_depth_vs_cli_qnn_depth" in stages and stages["app_qnn_depth_vs_cli_qnn_depth"]["max_abs"] != 0:
        gate_failures.append("qnn_app_cli")
    result = {
        "schema": "blindassist_camerax_depth_parity_r0",
        "capture": capture,
        "stages": stages,
        "files": files,
        "gate_pass": not gate_failures and "app_qnn_depth_vs_cli_qnn_depth" in stages,
        "gate_failures": gate_failures,
        "cli_evaluated": "app_qnn_depth_vs_cli_qnn_depth" in stages,
    }
    (root / "parity.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
