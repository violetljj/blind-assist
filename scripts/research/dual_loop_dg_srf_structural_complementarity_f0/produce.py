"""Produce truth-blind Depth Anything V2 Small maps for DG-SRF F0."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import torch

from .common import (
    PROTOCOL_ID,
    SHAPE,
    ensure_artifact_output,
    read_json,
    read_jsonl,
    resolve_repo_path,
    sha256_array,
    sha256_file,
    validate_config,
    verify_file,
    write_json,
    write_jsonl,
)
from .operators import (
    depth_health_and_proximity,
    validate_depth_direction_canary,
)


def _git_head(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _draw_canary(scene_index: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(9100 + scene_index)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * (0.31 + 0.025 * scene_index))
    sky_top = np.array([185, 145, 95], dtype=np.float64)
    sky_bottom = np.array([230, 210, 180], dtype=np.float64)
    for y in range(horizon):
        alpha = y / max(horizon - 1, 1)
        image[y, :, :] = (
            (1.0 - alpha) * sky_top + alpha * sky_bottom
        ).astype(np.uint8)
    ground_top = np.array([105, 110, 115], dtype=np.float64)
    ground_bottom = np.array([45, 55, 65], dtype=np.float64)
    for y in range(horizon, height):
        alpha = (y - horizon) / max(height - 1 - horizon, 1)
        row = (1.0 - alpha) * ground_top + alpha * ground_bottom
        texture = rng.normal(0.0, 4.0 + 5.0 * alpha, size=(width, 1))
        image[y, :, :] = np.clip(row + texture, 0, 255).astype(np.uint8)

    vanishing_x = int(width * (0.45 + 0.035 * scene_index))
    for x_bottom in range(-width // 2, width * 3 // 2, width // 8):
        cv2.line(
            image,
            (vanishing_x, horizon),
            (x_bottom, height - 1),
            (150, 150, 150),
            1,
            cv2.LINE_AA,
        )
    for fraction in (0.12, 0.22, 0.36, 0.55, 0.78):
        y = int(horizon + (height - horizon) * fraction)
        cv2.line(image, (0, y), (width - 1, y), (125, 125, 125), 1)

    near_w = int(width * (0.20 + 0.015 * (scene_index % 2)))
    near_h = int(height * (0.45 + 0.02 * (scene_index % 3)))
    near_x = int(width * (0.18 + 0.08 * (scene_index % 3)))
    near_y = height - near_h - int(height * 0.03)
    far_w = int(width * 0.065)
    far_h = int(height * 0.14)
    far_x = int(width * (0.67 - 0.04 * (scene_index % 2)))
    far_y = horizon + int(height * 0.035)
    colors = (
        (35, 65, 210),
        (190, 80, 30),
        (55, 175, 75),
        (155, 50, 160),
    )
    color = colors[scene_index % len(colors)]
    cv2.rectangle(
        image,
        (near_x, near_y),
        (near_x + near_w, near_y + near_h),
        color,
        -1,
    )
    cv2.rectangle(
        image,
        (far_x, far_y),
        (far_x + far_w, far_y + far_h),
        color,
        -1,
    )
    for x, y, w, h in (
        (near_x, near_y, near_w, near_h),
        (far_x, far_y, far_w, far_h),
    ):
        cv2.rectangle(image, (x, y), (x + w, y + h), (245, 245, 245), 2)
        cv2.line(image, (x, y), (x + w, y + h), (20, 20, 20), 2)
        cv2.line(image, (x + w, y), (x, y + h), (20, 20, 20), 2)
    cv2.ellipse(
        image,
        (near_x + near_w // 2, near_y + near_h),
        (int(near_w * 0.65), int(height * 0.035)),
        0,
        0,
        360,
        (30, 35, 40),
        -1,
    )

    near_mask = np.zeros((height, width), dtype=np.uint8)
    far_mask = np.zeros((height, width), dtype=np.uint8)
    inset_near = max(4, near_w // 8)
    inset_far = max(2, far_w // 8)
    near_mask[
        near_y + inset_near : near_y + near_h - inset_near,
        near_x + inset_near : near_x + near_w - inset_near,
    ] = 1
    far_mask[
        far_y + inset_far : far_y + far_h - inset_far,
        far_x + inset_far : far_x + far_w - inset_far,
    ] = 1
    return image, near_mask.astype(bool), far_mask.astype(bool)


def _load_model(
    repo_root: Path,
    config: Mapping[str, Any],
    device: str,
) -> tuple[Any, dict[str, Any]]:
    contract = config["model_contract"]
    archive_path = resolve_repo_path(repo_root, contract["source_archive_path"])
    verify_file(archive_path, contract["source_archive_sha256"])
    checkpoint_path = resolve_repo_path(repo_root, contract["checkpoint_path"])
    verify_file(
        checkpoint_path,
        contract["checkpoint_sha256"],
        int(contract["checkpoint_bytes"]),
    )
    source_root = resolve_repo_path(repo_root, contract["source_root"])
    for relative, expected in contract["source_file_sha256"].items():
        verify_file(source_root / relative, expected)
    sys.path.insert(0, str(source_root))
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    finally:
        sys.path.pop(0)
    model = DepthAnythingV2(
        encoder="vits",
        features=64,
        out_channels=[48, 96, 192, 384],
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    exact_parameter_count = int(
        sum(parameter.numel() for parameter in model.parameters())
    )
    if exact_parameter_count != int(contract["exact_parameter_count"]):
        raise ValueError(
            "strict-loaded model parameter count does not match frozen contract"
        )
    return model, {
        "source_archive_sha256": sha256_file(archive_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_commit": contract["source_commit"],
        "state_tensor_count": len(state),
        "exact_parameter_count": exact_parameter_count,
        "official_preprocess": {
            "input_size": int(contract["input_size"]),
            "keep_aspect_lower_bound_multiple_of_14": True,
            "input_interpolation": "INTER_CUBIC",
            "bgr_to_rgb_divide_255": True,
            "imagenet_normalization": True,
            "original_size_restore": "bilinear_align_corners_true",
            "analysis_resize": "opencv_inter_linear_to_256x256",
        },
    }


def _infer(model: Any, image: np.ndarray, input_size: int) -> np.ndarray:
    raw = model.infer_image(image, input_size)
    if raw.ndim != 2 or raw.shape != image.shape[:2]:
        raise ValueError(
            f"restored model output shape {raw.shape} does not match {image.shape[:2]}"
        )
    return np.asarray(raw, dtype=np.float32)


def _run_canary(
    *,
    model: Any,
    input_size: int,
    config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    contract = config["direction_canary"]
    width = int(contract["width"])
    height = int(contract["height"])
    scene_rows: list[dict[str, Any]] = []
    signed_margins: list[float] = []
    canary_root = output_root / "direction_canary"
    canary_root.mkdir(parents=True)
    transform_canary = _run_transform_only_canary(config=config)
    for scene_index in range(int(contract["scene_count"])):
        image, near_mask, far_mask = _draw_canary(scene_index, width, height)
        raw = _infer(model, image, input_size)
        finite = raw[np.isfinite(raw)]
        if finite.size != raw.size:
            raise ValueError("direction canary produced non-finite output")
        q05, q95 = np.quantile(finite, [0.05, 0.95])
        span = float(q95 - q05)
        if span <= 1e-6:
            normalized_margin = 0.0
        else:
            normalized_margin = float(
                (np.median(raw[near_mask]) - np.median(raw[far_mask])) / span
            )
        signed_margins.append(normalized_margin)
        cv2.imwrite(str(canary_root / f"scene_{scene_index}.png"), image)
        np.save(canary_root / f"raw_depth_{scene_index}.npy", raw)
        scene_rows.append(
            {
                "scene_index": scene_index,
                "image_sha256": sha256_file(canary_root / f"scene_{scene_index}.png"),
                "raw_depth_array_sha256": sha256_array(raw),
                "near_median": float(np.median(raw[near_mask])),
                "far_median": float(np.median(raw[far_mask])),
                "normalized_near_minus_far_margin": normalized_margin,
            }
        )
    decision = validate_depth_direction_canary(
        signed_margins,
        frozen_direction=contract["frozen_direction"],
        minimum_consistent=int(contract["minimum_consistent_scene_count"]),
        minimum_median_margin=float(
            contract["minimum_median_normalized_near_far_margin"]
        ),
    )
    overall_passed = (
        decision["passed"] and transform_canary["status"] == "PASS"
    )
    result = {
        "schema_version": (
            "blindassist.dg_srf_image_space_structural_"
            "complementarity_f0.direction_canary.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "status": "PASS" if overall_passed else "FAIL",
        "canonical_truth_accessed": False,
        "direction_selected_from_canary": False,
        "transform_only_resize_and_normalization_canary": transform_canary,
        "scene_rows": scene_rows,
        "decision": decision,
    }
    write_json(output_root / "direction_canary.json", result)
    return result


def _run_transform_only_canary(*, config: Mapping[str, Any]) -> dict[str, Any]:
    height = int(config["direction_canary"]["height"])
    width = int(config["direction_canary"]["width"])
    y = np.linspace(0.15, 0.65, height, dtype=np.float32)[:, None]
    x = np.linspace(-0.05, 0.05, width, dtype=np.float32)[None, :]
    raw = np.broadcast_to(y + x, (height, width)).copy()
    near_mask = np.zeros((height, width), dtype=np.uint8)
    far_mask = np.zeros((height, width), dtype=np.uint8)
    near_mask[int(height * 0.55) :, int(width * 0.25) : int(width * 0.55)] = 1
    far_mask[
        int(height * 0.18) : int(height * 0.34),
        int(width * 0.62) : int(width * 0.78),
    ] = 1
    raw[near_mask.astype(bool)] += 0.55
    raw[far_mask.astype(bool)] -= 0.10
    resized = cv2.resize(
        raw,
        (SHAPE[1], SHAPE[0]),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    resized_near = cv2.resize(
        near_mask,
        (SHAPE[1], SHAPE[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    resized_far = cv2.resize(
        far_mask,
        (SHAPE[1], SHAPE[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    health_a, normalized_a = depth_health_and_proximity(
        resized,
        direction=config["direction_canary"]["frozen_direction"],
        config=config,
    )
    affine = resized * np.float32(3.25) + np.float32(7.0)
    health_b, normalized_b = depth_health_and_proximity(
        affine,
        direction=config["direction_canary"]["frozen_direction"],
        config=config,
    )
    max_abs_difference = float(np.max(np.abs(normalized_a - normalized_b)))
    near_median = float(np.median(normalized_a[resized_near]))
    far_median = float(np.median(normalized_a[resized_far]))
    passed = (
        health_a["q"] == 1
        and health_b["q"] == 1
        and near_median > far_median
        and max_abs_difference <= 2e-6
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "known_inverse_depth_near_median": near_median,
        "known_inverse_depth_far_median": far_median,
        "affine_normalization_max_abs_difference": max_abs_difference,
        "analysis_shape": list(SHAPE),
    }


def run_produce(
    *,
    repo_root: Path,
    config_path: Path,
    prepared_root: Path,
    output_root: Path,
    mode: str,
    pilot_count: int,
    device: str,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    prepared_root = prepared_root.resolve()
    output_root = ensure_artifact_output(repo_root, output_root)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.mkdir(parents=True)

    config = read_json(config_path)
    validate_config(config)
    if mode not in {"pilot", "full"}:
        raise ValueError("mode must be pilot or full")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    torch.manual_seed(20260801)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260801)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    started = time.perf_counter()
    model, model_identity = _load_model(repo_root, config, device)
    canary = _run_canary(
        model=model,
        input_size=int(config["model_contract"]["input_size"]),
        config=config,
        output_root=output_root,
    )
    if (
        canary["status"] != "PASS"
        or canary["transform_only_resize_and_normalization_canary"]["status"]
        != "PASS"
    ):
        receipt = {
            "schema_version": (
                "blindassist.dg_srf_image_space_structural_"
                "complementarity_f0.producer_receipt.v1"
            ),
            "protocol_id": PROTOCOL_ID,
            "status": "NOT_EVALUABLE_DIRECTION_CANARY_FAILED",
            "mode": mode,
            "model_identity": model_identity,
            "direction_canary": canary["decision"],
            "scientific_truth_accessed": False,
        }
        write_json(output_root / "producer_receipt.json", receipt)
        return receipt

    prepare_receipt_path = prepared_root / "prepare_receipt.json"
    inference_manifest_path = prepared_root / "inference_manifest.jsonl"
    prepare_receipt = read_json(prepare_receipt_path)
    if prepare_receipt["status"] != "COMPLETE":
        raise ValueError("prepare receipt is not complete")
    if prepare_receipt["config_sha256"] != sha256_file(config_path):
        raise ValueError("prepare/config identity drift")
    if sha256_file(inference_manifest_path) != prepare_receipt["inference_manifest"]["sha256"]:
        raise ValueError("prepared inference manifest SHA mismatch")
    all_rows = read_jsonl(inference_manifest_path)
    rows = all_rows if mode == "full" else all_rows[: int(pilot_count)]
    if mode == "full" and len(rows) != int(config["input_contract"]["expected_frame_count"]):
        raise ValueError("full mode frame count mismatch")
    if not rows:
        raise ValueError("no frames selected")

    depth_path = output_root / "depth_maps.npy"
    depth_maps = np.lib.format.open_memmap(
        depth_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(rows), *SHAPE),
    )
    depth_index_rows: list[dict[str, Any]] = []
    direction = config["direction_canary"]["frozen_direction"]
    inference_started = time.perf_counter()
    for index, row in enumerate(rows):
        image_path = resolve_repo_path(repo_root, row["image_repo_relative_path"])
        verify_file(image_path, row["image_sha256"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to decode {image_path}")
        frame_started = time.perf_counter()
        raw_original = _infer(
            model,
            image,
            int(config["model_contract"]["input_size"]),
        )
        raw = cv2.resize(
            raw_original,
            (SHAPE[1], SHAPE[0]),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        depth_maps[index] = raw
        health, _ = depth_health_and_proximity(
            raw,
            direction=direction,
            config=config,
        )
        depth_index_rows.append(
            {
                "schema_version": (
                    "blindassist.dg_srf_image_space_structural_"
                    "complementarity_f0.depth_index.v1"
                ),
                "protocol_id": PROTOCOL_ID,
                "index": index,
                "view_row_id": row["view_row_id"],
                "session_id": row["session_id"],
                "frame_id": int(row["frame_id"]),
                "image_sha256": row["image_sha256"],
                "raw_depth_array_sha256": sha256_array(raw),
                "health": health,
                "inference_and_restore_ms": (
                    time.perf_counter() - frame_started
                )
                * 1000.0,
            }
        )
        if (index + 1) % 25 == 0 or index + 1 == len(rows):
            elapsed = time.perf_counter() - inference_started
            write_json(
                output_root / "progress.json",
                {
                    "phase": "DEPTH_INFERENCE",
                    "completed": index + 1,
                    "total": len(rows),
                    "elapsed_seconds": elapsed,
                    "frames_per_second": (index + 1) / max(elapsed, 1e-9),
                    "last_progress_index": index,
                },
            )
    depth_maps.flush()
    del depth_maps
    write_jsonl(output_root / "depth_index.jsonl", depth_index_rows)
    inference_elapsed = time.perf_counter() - inference_started
    total_elapsed = time.perf_counter() - started
    q_count = sum(int(row["health"]["q"]) for row in depth_index_rows)
    driver = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    device_properties = (
        torch.cuda.get_device_properties(torch.device(device))
        if device.startswith("cuda")
        else None
    )
    receipt = {
        "schema_version": (
            "blindassist.dg_srf_image_space_structural_"
            "complementarity_f0.producer_receipt.v1"
        ),
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "mode": mode,
        "git_head": _git_head(repo_root),
        "config_sha256": sha256_file(config_path),
        "prepare_receipt_sha256": sha256_file(prepare_receipt_path),
        "scientific_truth_accessed_by_producer": False,
        "prepared_manifest_role": "TRUTH_MINIMIZED_IDENTITY_AND_RGB_ONLY",
        "canonical_mask_or_packed_mask_loaded": False,
        "model_identity": model_identity,
        "direction_canary": canary["decision"],
        "depth_map": {
            "path": str(depth_path.relative_to(repo_root)).replace("\\", "/"),
            "sha256": sha256_file(depth_path),
            "shape": [len(rows), *SHAPE],
            "dtype": "float32",
        },
        "depth_index": {
            "path": str(
                (output_root / "depth_index.jsonl").relative_to(repo_root)
            ).replace("\\", "/"),
            "sha256": sha256_file(output_root / "depth_index.jsonl"),
            "row_count": len(depth_index_rows),
        },
        "health_summary": {
            "evaluable_frame_count": q_count,
            "frame_count": len(rows),
            "coverage": q_count / len(rows),
        },
        "runtime": {
            "inference_seconds": inference_elapsed,
            "total_seconds": total_elapsed,
            "frames_per_second": len(rows) / max(inference_elapsed, 1e-9),
            "device": device,
            "device_name": (
                torch.cuda.get_device_name(torch.device(device))
                if device.startswith("cuda")
                else platform.processor()
            ),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "nvidia_driver": driver,
            "python": platform.python_version(),
            "compute_capability": (
                [
                    int(device_properties.major),
                    int(device_properties.minor),
                ]
                if device_properties is not None
                else None
            ),
            "total_vram_bytes": (
                int(device_properties.total_memory)
                if device_properties is not None
                else None
            ),
            "deterministic_algorithms": True,
            "tf32_enabled": False,
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "claim_ceiling": config["claim_ceiling"],
    }
    write_json(output_root / "producer_receipt.json", receipt)
    write_json(
        output_root / "progress.json",
        {
            "phase": "COMPLETE",
            "completed": len(rows),
            "total": len(rows),
            "elapsed_seconds": total_elapsed,
            "terminal": "PRODUCER_COMPLETE",
        },
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--pilot-count", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    receipt = run_produce(
        repo_root=args.repo_root,
        config_path=args.config,
        prepared_root=args.prepared_root,
        output_root=args.output_root,
        mode=args.mode,
        pilot_count=args.pilot_count,
        device=args.device,
    )
    print(
        f"{receipt['status']} mode={receipt['mode']} "
        f"frames={receipt.get('depth_index', {}).get('row_count', 0)}"
    )


if __name__ == "__main__":
    main()
