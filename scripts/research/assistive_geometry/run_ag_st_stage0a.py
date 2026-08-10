#!/usr/bin/env python3
"""Run a small MapAnything depth-labelability pilot on frozen TRAIN sources.

This is a reversible WILD_LAB diagnostic.  Source depth is deterministically
split into an observed metric anchor and a hidden sensor-reference region.  The
hidden values never enter the Teacher input and are joined only after inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MODULE_DIR))

from arkitscenes_truth_reader import (  # noqa: E402
    TruthReaderPolicy,
    canonicalize_frame,
    depth_mm_to_metres,
    interpolate_camera_to_world,
    parse_pincam,
    parse_trajectory,
)


DEFAULT_SOURCE_MANIFEST = (
    REPO_ROOT
    / "artifacts.local"
    / "datasets"
    / "assistive-geometry-b0-arkitscenes-20260809-r2"
    / "manifest.json"
)
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts.local" / "models" / "map-anything-apache"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local" / "experiments" / "ag-st-stage0a-mapanything-apache-r0"
)
DEFAULT_PARENTS = ("42445086", "42898024", "47204445", "47334948")
DEFAULT_FRAME_INDICES_BY_PARENT = {
    "42445086": (137, 166, 196),
    "42898024": (214, 233, 253),
    "47204445": (205, 232, 260),
    "47334948": (55, 73, 92),
}
RISK_QUANTILES = (0.0, 0.25, 0.50, 0.75, 0.90)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def make_withheld_pattern(
    shape: tuple[int, int],
    *,
    block_size: int,
    modulus: int,
    residue: int,
) -> np.ndarray:
    """Return a value-independent grid of contiguous withheld blocks."""
    require(len(shape) == 2 and min(shape) > 0, "invalid mask shape")
    require(block_size > 0, "block_size must be positive")
    require(modulus >= 2, "modulus must be at least two")
    require(0 <= residue < modulus, "residue outside modulus")
    rows, columns = np.indices(shape)
    block_ids = (rows // block_size) * math.ceil(shape[1] / block_size) + columns // block_size
    return (block_ids % modulus) == residue


def split_observed_and_hidden_depth(
    depth_m: np.ndarray,
    source_valid: np.ndarray,
    withheld_pattern: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(source_valid, dtype=np.bool_)
    pattern = np.asarray(withheld_pattern, dtype=np.bool_)
    require(depth.ndim == 2 and valid.shape == depth.shape and pattern.shape == depth.shape, "depth/mask shape mismatch")
    valid = valid & np.isfinite(depth) & (depth > 0)
    hidden = valid & pattern
    observed_valid = valid & ~pattern
    observed = np.where(observed_valid, depth, 0.0).astype(np.float32)
    require(not np.any(observed[hidden] > 0), "hidden source depth leaked into Teacher input")
    return observed, observed_valid, hidden


def estimate_observed_anchor_scale(
    observed_depth_m: np.ndarray,
    raw_prediction_depth_m: np.ndarray,
    *,
    minimum_support: int,
) -> tuple[float, int]:
    observed = np.asarray(observed_depth_m, dtype=np.float32)
    prediction = np.asarray(raw_prediction_depth_m, dtype=np.float32)
    require(observed.shape == prediction.shape, "anchor/prediction shape mismatch")
    support_mask = (
        np.isfinite(observed)
        & (observed > 0)
        & np.isfinite(prediction)
        & (prediction > 0)
    )
    support = int(support_mask.sum())
    require(support >= minimum_support, "observed metric-anchor scale denominator too small")
    scale = float(np.median(observed[support_mask] / prediction[support_mask]))
    require(math.isfinite(scale) and 0.5 <= scale <= 2.0, "observed metric-anchor scale factor invalid")
    return scale, support


def select_train_videos(
    manifest: dict[str, Any], parent_ids: Iterable[str]
) -> list[dict[str, Any]]:
    requested = [str(value) for value in parent_ids]
    require(len(requested) == len(set(requested)), "duplicate parent id")
    train = {
        str(video["video_id"]): video
        for video in manifest.get("videos", [])
        if video.get("role") == "TRAIN"
    }
    missing = sorted(set(requested) - set(train))
    require(not missing, f"requested parents are not frozen TRAIN parents: {missing}")
    return [train[parent_id] for parent_id in requested]


def load_factor_source_frame(
    video: dict[str, Any],
    frame_index: int,
    trajectory: np.ndarray,
    policy: TruthReaderPolicy = TruthReaderPolicy(),
) -> dict[str, Any]:
    """Load only RGB/K/pose/depth/confidence; never derive task outcomes."""
    selected = [str(value) for value in video["selected_frame_stems"]]
    require(0 <= frame_index < len(selected), "frame index outside selected window")
    stem = selected[frame_index]
    extracted = video["extracted"]
    modalities = ("lowres_wide", "lowres_depth", "confidence", "lowres_wide_intrinsics")
    for modality in modalities:
        require(len(extracted[modality]) == len(selected), f"{modality} mapping count drift")
        require(Path(extracted[modality][frame_index]["path"]).stem == stem, f"{modality} stem drift")
    with Image.open(extracted["lowres_wide"][frame_index]["path"]) as image:
        rgb = np.asarray(image.convert("RGB"))
    with Image.open(extracted["lowres_depth"][frame_index]["path"]) as image:
        depth_raw = np.asarray(image).copy()
    with Image.open(extracted["confidence"][frame_index]["path"]) as image:
        confidence = np.asarray(image).copy()
    intrinsics, source_size = parse_pincam(
        Path(extracted["lowres_wide_intrinsics"][frame_index]["path"])
    )
    require((rgb.shape[1], rgb.shape[0]) == source_size, "RGB/pincam size drift")
    timestamp = float(stem.rsplit("_", 1)[1])
    pose, interpolation = interpolate_camera_to_world(
        trajectory, timestamp, policy.maximum_pose_bracketing_gap_seconds
    )
    canonical = canonicalize_frame(rgb, depth_raw, confidence, intrinsics, pose)
    depth_m = depth_mm_to_metres(canonical["depth_raw_mm"])
    source_valid = (
        np.isfinite(depth_m)
        & (depth_m >= policy.depth_min_m)
        & (depth_m <= policy.depth_max_m)
        & (canonical["confidence"] >= policy.minimum_sensor_confidence)
    )
    return {
        "identity": {
            "visit_id": str(video["visit_id"]),
            "video_id": str(video["video_id"]),
            "frame_index": frame_index,
            "frame_stem": stem,
            "timestamp_seconds": timestamp,
        },
        "orientation": {
            "rotation_label": canonical["rotation_label"],
            "rotation_index": canonical["rotation_index"],
        },
        "pose_interpolation": interpolation,
        "rgb_upright": canonical["rgb"],
        "intrinsics_upright": canonical["intrinsics"],
        "camera_to_world_upright": canonical["camera_to_world"],
        "depth_m_upright": depth_m,
        "depth_valid_upright": source_valid,
        "confidence_upright": canonical["confidence"],
    }


def nearest_observed_completion(depth: np.ndarray, observed_valid: np.ndarray) -> np.ndarray:
    """Deterministic source-only baseline using nearest observed depth."""
    from scipy.ndimage import distance_transform_edt

    values = np.asarray(depth, dtype=np.float32)
    observed = np.asarray(observed_valid, dtype=np.bool_)
    require(values.ndim == 2 and observed.shape == values.shape, "baseline shape mismatch")
    if not np.any(observed):
        return np.full_like(values, np.nan)
    nearest = distance_transform_edt(~observed, return_distances=False, return_indices=True)
    return values[tuple(nearest)].astype(np.float32)


def _error_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    truth_values = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    valid = (
        np.isfinite(truth_values)
        & (truth_values > 0)
        & np.isfinite(prediction_values)
        & (prediction_values > 0)
    )
    if not np.any(valid):
        return {
            "count": 0,
            "mae_m": None,
            "median_ae_m": None,
            "rmse_m": None,
            "abs_rel": None,
            "bad_0_10m_rate": None,
        }
    absolute = np.abs(prediction_values[valid] - truth_values[valid])
    return {
        "count": int(absolute.size),
        "mae_m": float(np.mean(absolute)),
        "median_ae_m": float(np.median(absolute)),
        "rmse_m": float(np.sqrt(np.mean(np.square(absolute)))),
        "abs_rel": float(np.mean(absolute / truth_values[valid])),
        "bad_0_10m_rate": float(np.mean(absolute > 0.10)),
    }


def compute_selective_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute diagnostic parent-macro risk/coverage from Teacher confidence."""
    require(bool(records), "no evaluation records")
    parents = sorted({str(record["parent_id"]) for record in records})
    hidden_total = sum(int(np.asarray(record["hidden_mask"], dtype=np.bool_).sum()) for record in records)
    require(hidden_total > 0, "hidden sensor-reference denominator is zero")

    usable_confidences: list[np.ndarray] = []
    for record in records:
        truth = np.asarray(record["truth_depth_m"])
        prediction = np.asarray(record["prediction_depth_m"])
        confidence = np.asarray(record["confidence"])
        hidden = np.asarray(record["hidden_mask"], dtype=np.bool_)
        model_mask = np.asarray(record["model_mask"], dtype=np.bool_)
        require(truth.shape == prediction.shape == confidence.shape == hidden.shape == model_mask.shape, "evaluation record shape mismatch")
        usable = hidden & model_mask & np.isfinite(prediction) & (prediction > 0) & np.isfinite(confidence)
        if np.any(usable):
            usable_confidences.append(confidence[usable].astype(np.float64))
    require(bool(usable_confidences), "Teacher has zero usable hidden predictions")
    all_confidence = np.concatenate(usable_confidences)

    curve: list[dict[str, Any]] = []
    for quantile in RISK_QUANTILES:
        threshold = float(np.quantile(all_confidence, quantile))
        accepted_truth: list[np.ndarray] = []
        accepted_prediction: list[np.ndarray] = []
        parent_rows: list[dict[str, Any]] = []
        for parent in parents:
            parent_hidden = 0
            parent_truth: list[np.ndarray] = []
            parent_prediction: list[np.ndarray] = []
            for record in records:
                if str(record["parent_id"]) != parent:
                    continue
                truth = np.asarray(record["truth_depth_m"])
                prediction = np.asarray(record["prediction_depth_m"])
                confidence = np.asarray(record["confidence"])
                hidden = np.asarray(record["hidden_mask"], dtype=np.bool_)
                model_mask = np.asarray(record["model_mask"], dtype=np.bool_)
                parent_hidden += int(hidden.sum())
                accepted = (
                    hidden
                    & model_mask
                    & np.isfinite(prediction)
                    & (prediction > 0)
                    & np.isfinite(confidence)
                    & (confidence >= threshold)
                )
                if np.any(accepted):
                    parent_truth.append(truth[accepted])
                    parent_prediction.append(prediction[accepted])
                    accepted_truth.append(truth[accepted])
                    accepted_prediction.append(prediction[accepted])
            accepted_count = sum(values.size for values in parent_truth)
            metrics = (
                _error_metrics(np.concatenate(parent_truth), np.concatenate(parent_prediction))
                if accepted_count
                else _error_metrics(np.asarray([]), np.asarray([]))
            )
            parent_rows.append(
                {
                    "parent_id": parent,
                    "hidden_count": parent_hidden,
                    "accepted_count": accepted_count,
                    "coverage": accepted_count / parent_hidden if parent_hidden else 0.0,
                    **metrics,
                }
            )
        overall = (
            _error_metrics(np.concatenate(accepted_truth), np.concatenate(accepted_prediction))
            if accepted_truth
            else _error_metrics(np.asarray([]), np.asarray([]))
        )
        macro_evaluable = all(row["count"] > 0 for row in parent_rows)
        curve.append(
            {
                "confidence_quantile": quantile,
                "confidence_threshold": threshold,
                "accepted_count": overall["count"],
                "coverage_of_hidden": overall["count"] / hidden_total,
                "parent_macro_coverage": float(np.mean([row["coverage"] for row in parent_rows])),
                "parent_macro_evaluable": macro_evaluable,
                "parent_macro_mae_m": (
                    float(np.mean([row["mae_m"] for row in parent_rows]))
                    if macro_evaluable
                    else None
                ),
                "parent_macro_bad_0_10m_rate": (
                    float(np.mean([row["bad_0_10m_rate"] for row in parent_rows]))
                    if macro_evaluable
                    else None
                ),
                "overall": overall,
                "parents": parent_rows,
            }
        )

    baseline_truth: list[np.ndarray] = []
    baseline_prediction: list[np.ndarray] = []
    for record in records:
        hidden = np.asarray(record["hidden_mask"], dtype=np.bool_)
        truth = np.asarray(record["truth_depth_m"])
        baseline = np.asarray(record["baseline_depth_m"])
        valid = hidden & np.isfinite(baseline) & (baseline > 0)
        if np.any(valid):
            baseline_truth.append(truth[valid])
            baseline_prediction.append(baseline[valid])
    baseline = _error_metrics(np.concatenate(baseline_truth), np.concatenate(baseline_prediction))
    baseline["coverage_of_hidden"] = baseline["count"] / hidden_total
    return {
        "parent_count": len(parents),
        "hidden_pixel_count": hidden_total,
        "source_only_nearest_baseline": baseline,
        "teacher_confidence_risk_coverage": curve,
    }


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _load_checkpoint_identity(model_dir: Path) -> dict[str, Any]:
    checkpoint = model_dir / "model.safetensors"
    require(checkpoint.is_file(), f"missing MapAnything checkpoint: {checkpoint}")
    return {
        "model_id": "facebook/map-anything-apache",
        "model_dir": str(model_dir.resolve()),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "weight_license": "Apache-2.0",
    }


def _preprocess_reference(
    *,
    rgb: np.ndarray,
    intrinsics: np.ndarray,
    depth_m: np.ndarray,
    source_valid: np.ndarray,
    withheld_pattern: np.ndarray,
    output_wh: tuple[int, int],
) -> dict[str, np.ndarray]:
    from mapanything.utils.cropping import crop_resize_if_necessary

    processed = crop_resize_if_necessary(
        image=rgb,
        resolution=output_wh,
        depthmap=np.asarray(depth_m, dtype=np.float32),
        intrinsics=np.asarray(intrinsics, dtype=np.float64),
        additional_quantities=[
            np.asarray(source_valid, dtype=np.uint8),
            np.asarray(withheld_pattern, dtype=np.uint8),
        ],
    )
    _, truth_depth, _, quantities = processed
    valid = quantities[0] > 0
    pattern = quantities[1] > 0
    hidden = valid & pattern
    return {
        "truth_depth_m": np.asarray(truth_depth, dtype=np.float32),
        "source_valid": valid,
        "hidden_mask": hidden,
    }


def _frame_summary(record: dict[str, Any]) -> dict[str, Any]:
    hidden = np.asarray(record["hidden_mask"], dtype=np.bool_)
    model_mask = np.asarray(record["model_mask"], dtype=np.bool_)
    prediction = np.asarray(record["prediction_depth_m"])
    usable = hidden & model_mask & np.isfinite(prediction) & (prediction > 0)
    teacher = _error_metrics(record["truth_depth_m"][usable], prediction[usable])
    raw_prediction = np.asarray(record["raw_prediction_depth_m"])
    raw_usable = hidden & model_mask & np.isfinite(raw_prediction) & (raw_prediction > 0)
    raw_teacher = _error_metrics(
        record["truth_depth_m"][raw_usable], raw_prediction[raw_usable]
    )
    baseline = _error_metrics(
        record["truth_depth_m"][hidden], record["baseline_depth_m"][hidden]
    )
    return {
        "parent_id": record["parent_id"],
        "frame_index": record["frame_index"],
        "frame_stem": record["frame_stem"],
        "orientation": record["orientation"],
        "output_hw": list(record["truth_depth_m"].shape),
        "source_valid_count": int(np.asarray(record["source_valid"], dtype=np.bool_).sum()),
        "hidden_count": int(hidden.sum()),
        "teacher_usable_count": int(usable.sum()),
        "teacher_coverage_of_hidden": float(usable.sum() / hidden.sum()),
        "anchor_scale_factor": record["anchor_scale_factor"],
        "anchor_scale_support": record["anchor_scale_support"],
        "teacher": teacher,
        "uncalibrated_teacher": raw_teacher,
        "source_only_nearest_baseline": baseline,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.geometry import (
        get_rays_in_camera_frame,
        rotation_matrix_to_quaternion,
    )
    from mapanything.utils.image import preprocess_inputs

    require(torch.cuda.is_available(), "MapAnything Stage 0A requires CUDA on this host")
    require(args.source_manifest.is_file(), f"missing source manifest: {args.source_manifest}")
    require(args.model_dir.is_dir(), f"missing model directory: {args.model_dir}")
    require(not args.output_dir.exists(), f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    require(manifest.get("schema") == "blindassist_assistive_geometry_b0_arkitscenes_pose_covered_media_manifest_v1", "source manifest schema drift")
    videos = select_train_videos(manifest, args.parents)
    frame_indices_by_parent: dict[str, list[int]] = {}
    for video in videos:
        parent_id = str(video["video_id"])
        indices = (
            list(args.frame_indices)
            if args.frame_indices is not None
            else list(DEFAULT_FRAME_INDICES_BY_PARENT[parent_id])
        )
        require(len(indices) >= 2, "each parent needs at least two views")
        for frame_index in indices:
            require(0 <= frame_index < 300, f"frame index outside frozen window: {frame_index}")
        frame_indices_by_parent[parent_id] = indices

    source_identity = {
        "manifest_path": str(args.source_manifest.resolve()),
        "manifest_sha256": sha256_file(args.source_manifest),
        "role": "TRAIN_ONLY_CONSUMED_WILD_LAB",
        "parents": list(args.parents),
        "frame_indices_by_parent": frame_indices_by_parent,
        "withheld_mask": {
            "kind": "VALUE_INDEPENDENT_CONTIGUOUS_BLOCK_GRID",
            "block_size_source_px": args.block_size,
            "modulus": args.mask_modulus,
        },
    }
    teacher_identity = _load_checkpoint_identity(args.model_dir)

    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_load_started = time.monotonic()
    model_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    model = (
        MapAnything.from_pretrained(str(args.model_dir))
        .eval()
        .to(device="cuda", dtype=model_dtype)
    )
    model_load_seconds = time.monotonic() - model_load_started

    records: list[dict[str, Any]] = []
    parent_runs: list[dict[str, Any]] = []
    inference_started = time.monotonic()
    for parent_index, video in enumerate(videos):
        parent_id = str(video["video_id"])
        parent_frame_indices = frame_indices_by_parent[parent_id]
        trajectory = parse_trajectory(Path(video["trajectory"]["path"]))
        raw_views: list[dict[str, Any]] = []
        frame_input_receipts: list[dict[str, Any]] = []
        for view_index, frame_index in enumerate(parent_frame_indices):
            frame = load_factor_source_frame(video, frame_index, trajectory)
            depth = np.asarray(frame["depth_m_upright"], dtype=np.float32)
            source_valid = np.asarray(frame["depth_valid_upright"], dtype=np.bool_)
            residue = (parent_index + view_index) % args.mask_modulus
            pattern = make_withheld_pattern(
                depth.shape,
                block_size=args.block_size,
                modulus=args.mask_modulus,
                residue=residue,
            )
            observed_depth, observed_valid, hidden = split_observed_and_hidden_depth(
                depth, source_valid, pattern
            )
            require(int(hidden.sum()) >= args.minimum_hidden_pixels, "hidden source-reference denominator too small")
            raw_views.append(
                {
                    "img": np.asarray(frame["rgb_upright"], dtype=np.uint8),
                    "intrinsics": np.asarray(frame["intrinsics_upright"], dtype=np.float32),
                    "depth_z": observed_depth,
                    "camera_poses": np.asarray(frame["camera_to_world_upright"], dtype=np.float32),
                    "is_metric_scale": torch.tensor([True], dtype=torch.bool),
                }
            )
            frame_input_receipts.append(
                {
                    "frame_index": frame_index,
                    "frame_stem": frame["identity"]["frame_stem"],
                    "timestamp_seconds": frame["identity"]["timestamp_seconds"],
                    "orientation": frame["orientation"]["rotation_label"],
                    "pose_interpolation": frame["pose_interpolation"],
                    "source_valid_count": int(source_valid.sum()),
                    "observed_input_count": int(observed_valid.sum()),
                    "hidden_count": int(hidden.sum()),
                    "mask_residue": residue,
                }
            )

        processed_views = preprocess_inputs(
            raw_views,
            resize_mode="longest_side",
            size=args.longest_side,
            patch_size=14,
            norm_type="dinov2",
        )
        for processed_view in processed_views:
            intrinsics = processed_view.pop("intrinsics")
            height, width = processed_view["img"].shape[-2:]
            _, ray_directions = get_rays_in_camera_frame(
                intrinsics, height, width, normalize_to_unit_sphere=True
            )
            processed_view["ray_directions"] = ray_directions.to(model_dtype)
            processed_view["depth_z"] = processed_view["depth_z"].to(model_dtype)
            pose = processed_view["camera_poses"]
            quaternions = rotation_matrix_to_quaternion(pose[:, :3, :3])
            processed_view["camera_poses"] = (
                quaternions.to(model_dtype),
                pose[:, :3, 3].to(model_dtype),
            )
        torch.cuda.synchronize()
        parent_started = time.monotonic()
        upstream_autocast = torch.autocast

        def low_vram_autocast(
            device_type: str,
            *autocast_args: Any,
            enabled: bool = True,
            dtype: Any = None,
            **autocast_kwargs: Any,
        ) -> Any:
            if device_type == "cuda" and not enabled:
                enabled = True
                dtype = model_dtype
            return upstream_autocast(
                device_type,
                *autocast_args,
                enabled=enabled,
                dtype=dtype,
                **autocast_kwargs,
            )

        torch.autocast = low_vram_autocast
        try:
            predictions = model.infer(
                processed_views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype=args.amp_dtype,
                apply_mask=False,
                mask_edges=False,
                apply_confidence_mask=False,
                use_multiview_confidence=False,
            )
        finally:
            torch.autocast = upstream_autocast
        torch.cuda.synchronize()
        parent_seconds = time.monotonic() - parent_started

        parent_record_indices: list[int] = []
        for view_index, (frame_index, prediction) in enumerate(zip(parent_frame_indices, predictions)):
            observed_depth = (
                processed_views[view_index]["depth_z"][0]
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            observed_valid = observed_depth > 0
            raw_predicted_depth = (
                prediction["depth_z"][0, ..., 0].detach().float().cpu().numpy()
            )
            anchor_scale_factor, anchor_scale_support = estimate_observed_anchor_scale(
                observed_depth,
                raw_predicted_depth,
                minimum_support=args.minimum_hidden_pixels,
            )
            predicted_depth = raw_predicted_depth * anchor_scale_factor
            baseline = nearest_observed_completion(observed_depth, observed_valid)
            confidence = prediction["conf"][0].detach().float().cpu().numpy()
            non_ambiguous = (
                prediction["non_ambiguous_mask"][0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.bool_)
            )

            # Hidden sensor-reference values are reloaded and joined only after
            # Teacher output and observed-anchor calibration have been fixed.
            frame = load_factor_source_frame(video, frame_index, trajectory)
            depth = np.asarray(frame["depth_m_upright"], dtype=np.float32)
            source_valid = np.asarray(frame["depth_valid_upright"], dtype=np.bool_)
            residue = (parent_index + view_index) % args.mask_modulus
            pattern = make_withheld_pattern(
                depth.shape,
                block_size=args.block_size,
                modulus=args.mask_modulus,
                residue=residue,
            )
            output_hw = tuple(int(value) for value in prediction["depth_z"].shape[1:3])
            output_wh = (output_hw[1], output_hw[0])
            reference = _preprocess_reference(
                rgb=np.asarray(frame["rgb_upright"], dtype=np.uint8),
                intrinsics=np.asarray(frame["intrinsics_upright"], dtype=np.float64),
                depth_m=depth,
                source_valid=source_valid,
                withheld_pattern=pattern,
                output_wh=output_wh,
            )
            require(not np.any(observed_valid & reference["hidden_mask"]), "processed hidden depth leaked into Teacher input")
            record = {
                "parent_id": str(video["video_id"]),
                "frame_index": frame_index,
                "frame_stem": frame["identity"]["frame_stem"],
                "orientation": frame["orientation"]["rotation_label"],
                "truth_depth_m": reference["truth_depth_m"],
                "source_valid": reference["source_valid"],
                "hidden_mask": reference["hidden_mask"],
                "observed_depth_m": observed_depth,
                "prediction_depth_m": predicted_depth,
                "raw_prediction_depth_m": raw_predicted_depth,
                "anchor_scale_factor": anchor_scale_factor,
                "anchor_scale_support": anchor_scale_support,
                "confidence": confidence,
                "model_mask": non_ambiguous,
                "baseline_depth_m": baseline,
            }
            parent_record_indices.append(len(records))
            records.append(record)
            np.savez_compressed(
                args.output_dir / f"{record['parent_id']}_{record['frame_stem']}.npz",
                truth_depth_m=record["truth_depth_m"],
                source_valid=record["source_valid"],
                hidden_mask=record["hidden_mask"],
                observed_depth_m=record["observed_depth_m"],
                prediction_depth_m=record["prediction_depth_m"],
                raw_prediction_depth_m=record["raw_prediction_depth_m"],
                anchor_scale_factor=np.asarray(record["anchor_scale_factor"], dtype=np.float32),
                teacher_confidence=record["confidence"],
                teacher_non_ambiguous_mask=record["model_mask"],
                source_only_nearest_depth_m=record["baseline_depth_m"],
            )
        parent_runs.append(
            {
                "parent_id": str(video["video_id"]),
                "view_count": len(parent_frame_indices),
                "inference_seconds": parent_seconds,
                "input_receipts": frame_input_receipts,
                "frame_summaries": [_frame_summary(records[index]) for index in parent_record_indices],
            }
        )
        print(
            json.dumps(
                {
                    "phase": "mapanything_inference",
                    "completed_parents": parent_index + 1,
                    "total_parents": len(videos),
                    "parent_id": str(video["video_id"]),
                    "seconds": parent_seconds,
                }
            ),
            flush=True,
        )
        del predictions, processed_views, raw_views
        torch.cuda.empty_cache()

    selective = compute_selective_metrics(records)
    raw_records = [
        {**record, "prediction_depth_m": record["raw_prediction_depth_m"]}
        for record in records
    ]
    raw_selective = compute_selective_metrics(raw_records)
    selective["uncalibrated_teacher_diagnostic"] = {
        "teacher_confidence_risk_coverage": raw_selective[
            "teacher_confidence_risk_coverage"
        ]
    }
    result = {
        "schema": "blindassist_ag_st_stage0a_wild_lab_result_v1",
        "status": "COMPLETED",
        "mode": "WILD_LAB_REVERSIBLE_EXPLORATION",
        "question": "Can source-anchored MapAnything produce confidence-ranked metric depth on deterministically hidden ARKitScenes TRAIN regions?",
        "source": source_identity,
        "teacher": {
            **teacher_identity,
            "repository_url": "https://github.com/facebookresearch/map-anything",
            "repository_revision": args.repository_revision,
            "role": "SOURCE_ANCHORED_PSEUDO_LABEL_TEACHER_NOT_TRUTH",
            "observed_anchor_calibration": "PER_VIEW_MEDIAN_OBSERVED_DEPTH_DIVIDED_BY_RAW_TEACHER_DEPTH_BEFORE_HIDDEN_REFERENCE_JOIN",
        },
        "execution": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "amp_dtype": args.amp_dtype,
            "low_vram_compatibility": "BF16_MODEL_PLUS_FORCED_CUDA_AUTOCAST_IN_UPSTREAM_FP32_BLOCKS",
            "longest_side_px": args.longest_side,
            "model_load_seconds": model_load_seconds,
            "inference_and_evaluation_seconds": time.monotonic() - inference_started,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        },
        "parent_runs": parent_runs,
        "metrics": selective,
        "claim_boundary": "Consumed TRAIN-only sensor-reference diagnostic. No ground-truth, factor-learnability, F1 authorization, generalization, deployment, product, or safety claim.",
        "next_decision": "Use the observed risk-coverage shape to decide whether to expand depth labelability and derive support/boundary diagnostics; do not materialize canonical pseudo-labels from this pilot.",
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--parents", nargs="+", default=list(DEFAULT_PARENTS))
    parser.add_argument(
        "--frame-indices",
        nargs="+",
        type=int,
        default=None,
        help="Optional uniform frame indices for every parent; defaults to vetted per-parent windows.",
    )
    parser.add_argument("--longest-side", type=int, default=336)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--mask-modulus", type=int, default=4)
    parser.add_argument("--minimum-hidden-pixels", type=int, default=512)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--repository-revision", default="3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    try:
        result = run(args)
        result["execution"]["total_seconds"] = time.monotonic() - started
        _write_json_exclusive(args.output_dir / "result.json", result)
        print(json.dumps({"status": result["status"], "result": str(args.output_dir / "result.json")}, indent=2))
        return 0
    except Exception as error:
        failure = {
            "schema": "blindassist_ag_st_stage0a_wild_lab_failure_v1",
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.monotonic() - started,
            "claim_boundary": "Operational failure only; no scientific conclusion.",
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "failure.json"
        if not failure_path.exists():
            _write_json_exclusive(failure_path, failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
