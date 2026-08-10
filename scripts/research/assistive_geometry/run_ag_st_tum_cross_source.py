#!/usr/bin/env python3
"""Run the AG-ST multi-Teacher confidence gate on a disjoint TUM RGB-D domain.

Four TUM sequences are diagnostics and three are held-out evaluation.  Both
Teachers receive RGB plus source-side camera evidence; deterministically hidden
source depth is joined only after inference.  The frozen R0 C-tier threshold is
applied unchanged.  If accepted pixels remain lower-risk than rejected pixels,
the runner materializes source-first depth/uncertainty/UNKNOWN labels.  Support
and boundary stay UNKNOWN because TUM gravity has not been verified.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
HFTF_DIR = MODULE_DIR.parent / "hftf"
sys.path[:0] = [str(MODULE_DIR), str(HFTF_DIR)]

from ag_st_tum_rgbd import (  # noqa: E402
    DEFAULT_TUM_COHORT_MANIFEST,
    TumSelectedPayload,
    load_tum_role_payloads,
)
from build_ag_st_factor_labels import (  # noqa: E402
    FrameBundle,
    PROVENANCE_SOURCE_NATIVE,
    PROVENANCE_TEACHER,
    TEACHER_B_QUALITY,
    TEACHER_C_QUALITY,
    TIER_A_SOURCE,
    TIER_B_ANCHORED,
    TIER_C_TEACHER,
    aggregate_multiview_residual,
    teacher_quality_signals,
)
from build_ag_st_multiteacher_factor_labels import (  # noqa: E402
    PAIR_DISAGREEMENT_SCALE,
    _compact_curve,
    _split_error,
    robust_observed_scale,
    teacher_pair_quality,
)
from produce_external_rgb_metric_depth_observations import (  # noqa: E402
    DepthAnythingV2MetricSource,
)
from run_ag_st_stage0a import (  # noqa: E402
    _error_metrics,
    _preprocess_reference,
    compute_selective_metrics,
    make_withheld_pattern,
    nearest_observed_completion,
    sha256_file,
    split_observed_and_hidden_depth,
)


DEFAULT_MAPANYTHING_MODEL = REPO_ROOT / "artifacts.local/models/map-anything-apache"
DEFAULT_DAV2_REPO = (
    REPO_ROOT / "artifacts.local/downloads/depth-lab/src/Depth-Anything-V2-main"
)
DEFAULT_DAV2_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/models/depth-anything-v2-metric-hypersim-small/depth_anything_v2_metric_hypersim_vits.pth"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "artifacts.local/experiments/ag-st-tum-cross-source-multiteacher-r1"
)

SCHEMA = "blindassist_ag_st_tum_cross_source_multiteacher_result_v1"
FROZEN_ACCEPT_THRESHOLD = TEACHER_C_QUALITY
MINIMUM_EVALUATION_COVERAGE = 0.25


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _frame_id(payload: TumSelectedPayload) -> str:
    return f"{payload.parent_id}__rgb{payload.rgb.row_index:06d}"


def _preprocess_secondary_depth(
    payload: TumSelectedPayload,
    raw_depth_m: np.ndarray,
    output_wh: tuple[int, int],
) -> np.ndarray:
    from mapanything.utils.cropping import crop_resize_if_necessary

    raw = np.asarray(raw_depth_m, dtype=np.float32)
    processed = crop_resize_if_necessary(
        image=payload.load_rgb(),
        resolution=output_wh,
        depthmap=raw,
        intrinsics=np.asarray(payload.intrinsics, dtype=np.float64),
        additional_quantities=[
            (np.isfinite(raw) & (raw > 0)).astype(np.uint8),
        ],
    )
    return np.asarray(processed[1], dtype=np.float32)


def _preprocess_reference_with_intrinsics(
    payload: TumSelectedPayload,
    depth_m: np.ndarray,
    source_valid: np.ndarray,
    withheld_pattern: np.ndarray,
    output_wh: tuple[int, int],
) -> dict[str, np.ndarray]:
    from mapanything.utils.cropping import crop_resize_if_necessary

    processed = crop_resize_if_necessary(
        image=payload.load_rgb(),
        resolution=output_wh,
        depthmap=np.asarray(depth_m, dtype=np.float32),
        intrinsics=np.asarray(payload.intrinsics, dtype=np.float64),
        additional_quantities=[
            np.asarray(source_valid, dtype=np.uint8),
            np.asarray(withheld_pattern, dtype=np.uint8),
        ],
    )
    _, truth_depth, intrinsics, quantities = processed
    valid = np.asarray(quantities[0] > 0, dtype=np.bool_)
    hidden = valid & np.asarray(quantities[1] > 0, dtype=np.bool_)
    return {
        "truth_depth_m": np.asarray(truth_depth, dtype=np.float32),
        "source_valid": valid,
        "hidden_mask": hidden,
        "intrinsics": np.asarray(intrinsics, dtype=np.float64),
    }


def _group_payloads(
    payloads: list[TumSelectedPayload],
) -> dict[str, list[TumSelectedPayload]]:
    grouped: dict[str, list[TumSelectedPayload]] = defaultdict(list)
    for payload in payloads:
        grouped[payload.parent_id].append(payload)
    for parent_id, values in grouped.items():
        values.sort(key=lambda value: value.rgb.timestamp_seconds)
        require(len(values) == 3, f"TUM parent does not have three views: {parent_id}")
    return dict(grouped)


def _load_mapanything(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.geometry import (
        get_rays_in_camera_frame,
        rotation_matrix_to_quaternion,
    )
    from mapanything.utils.image import preprocess_inputs

    require(torch.cuda.is_available(), "TUM MapAnything canary requires CUDA")
    model_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    model = (
        MapAnything.from_pretrained(str(args.mapanything_model))
        .eval()
        .to(device="cuda", dtype=model_dtype)
    )
    return (
        torch,
        model,
        model_dtype,
        preprocess_inputs,
        (get_rays_in_camera_frame, rotation_matrix_to_quaternion),
    )


def _mapanything_infer_parent(
    args: argparse.Namespace,
    parent_index: int,
    payloads: list[TumSelectedPayload],
    torch: Any,
    model: Any,
    model_dtype: Any,
    preprocess_inputs: Any,
    geometry_functions: tuple[Any, Any],
) -> list[dict[str, Any]]:
    get_rays_in_camera_frame, rotation_matrix_to_quaternion = geometry_functions
    raw_views: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    for view_index, payload in enumerate(payloads):
        rgb = payload.load_rgb()
        depth, source_valid = payload.load_depth()
        residue = (parent_index + view_index) % args.mask_modulus
        pattern = make_withheld_pattern(
            depth.shape,
            block_size=args.block_size,
            modulus=args.mask_modulus,
            residue=residue,
        )
        observed, observed_valid, hidden = split_observed_and_hidden_depth(
            depth, source_valid, pattern
        )
        require(int(hidden.sum()) >= args.minimum_hidden_pixels, "TUM hidden denominator too small")
        raw_views.append(
            {
                "img": rgb,
                "intrinsics": np.asarray(payload.intrinsics, dtype=np.float32),
                "depth_z": observed,
                "camera_poses": np.asarray(payload.camera_to_world, dtype=np.float32),
                "is_metric_scale": torch.tensor([True], dtype=torch.bool),
            }
        )
        native.append(
            {
                "payload": payload,
                "rgb": rgb,
                "depth": depth,
                "source_valid": source_valid,
                "pattern": pattern,
                "native_hidden_count": int(hidden.sum()),
                "native_observed_count": int(observed_valid.sum()),
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

    records: list[dict[str, Any]] = []
    for item, processed_view, prediction in zip(
        native, processed_views, predictions, strict=True
    ):
        payload = item["payload"]
        observed = (
            processed_view["depth_z"][0].float().cpu().numpy().astype(np.float32)
        )
        observed_valid = observed > 0
        raw_primary = prediction["depth_z"][0, ..., 0].detach().float().cpu().numpy()
        anchor_support = int(np.sum(observed_valid & np.isfinite(raw_primary) & (raw_primary > 0)))
        require(anchor_support >= args.minimum_hidden_pixels, "MapAnything anchor support too small")
        anchor_scale = float(
            np.median(observed[observed_valid] / raw_primary[observed_valid])
        )
        require(math.isfinite(anchor_scale) and 0.05 <= anchor_scale <= 20.0, "MapAnything anchor scale invalid")
        primary = (raw_primary * anchor_scale).astype(np.float32)
        confidence = prediction["conf"][0].detach().float().cpu().numpy()
        primary_valid = (
            prediction["non_ambiguous_mask"][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.bool_)
            & np.isfinite(primary)
            & (primary > 0)
        )
        output_wh = (primary.shape[1], primary.shape[0])
        reference = _preprocess_reference_with_intrinsics(
            payload,
            item["depth"],
            item["source_valid"],
            item["pattern"],
            output_wh,
        )
        require(
            not np.any(observed_valid & reference["hidden_mask"]),
            "TUM hidden source depth leaked into Teacher input",
        )
        records.append(
            {
                "payload": payload,
                "parent_id": payload.parent_id,
                "frame_id": _frame_id(payload),
                "frame_index": payload.rgb.row_index,
                "truth_depth_m": reference["truth_depth_m"],
                "source_valid": reference["source_valid"],
                "hidden_mask": reference["hidden_mask"],
                "intrinsics": reference["intrinsics"],
                "camera_to_world": payload.camera_to_world,
                "observed_depth_m": observed,
                "primary_depth_m": primary,
                "raw_primary_depth_m": raw_primary,
                "primary_confidence": confidence,
                "primary_valid": primary_valid,
                "anchor_scale": anchor_scale,
                "anchor_support": anchor_support,
                "baseline_depth_m": nearest_observed_completion(
                    observed, observed_valid
                ),
                "pose_bracketing_gap_seconds": payload.pose_bracketing_gap_seconds,
                "native_hidden_count": item["native_hidden_count"],
                "native_observed_count": item["native_observed_count"],
            }
        )
    return records


def _attach_multiview_quality(records: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["parent_id"]].append(record)
    for parent_records in grouped.values():
        bundles = [
            FrameBundle(
                parent_id=record["parent_id"],
                frame_index=record["frame_index"],
                frame_stem=record["frame_id"],
                intrinsics=record["intrinsics"],
                camera_to_world=record["camera_to_world"],
                source_depth_m=record["truth_depth_m"],
                source_valid=record["source_valid"],
                sensor_confidence=np.where(record["source_valid"], 2, 0).astype(
                    np.uint8
                ),
                observed_depth_m=record["observed_depth_m"],
                teacher_depth_m=record["primary_depth_m"],
                teacher_confidence=record["primary_confidence"],
                teacher_valid=record["primary_valid"],
                hidden_mask=record["hidden_mask"],
                baseline_depth_m=record["baseline_depth_m"],
            )
            for record in parent_records
        ]
        for index, record in enumerate(parent_records):
            multiview_residual, multiview_valid = aggregate_multiview_residual(
                index, bundles
            )
            signals = teacher_quality_signals(
                record["primary_depth_m"],
                record["primary_confidence"],
                record["primary_valid"],
                record["observed_depth_m"],
                multiview_residual,
                multiview_valid,
            )
            record["geometry_quality"] = signals["combined_quality"]
            record["anchor_quality"] = signals["anchor_quality"]
            record["anchor_residual_m"] = signals["anchor_residual_m"]
            record["multiview_quality"] = signals["multiview_quality"]
            record["multiview_residual_m"] = signals["multiview_residual_m"]
            record["multiview_valid"] = signals["multiview_valid"]


def _attach_secondary_teacher(
    args: argparse.Namespace,
    records: list[dict[str, Any]],
) -> None:
    teacher = DepthAnythingV2MetricSource(
        args.dav2_repo,
        args.dav2_checkpoint,
        args.device,
        input_size=args.dav2_input_size,
        precision=args.dav2_precision,
    )
    for record in records:
        payload = record["payload"]
        raw_secondary, _ = teacher.infer(payload.load_rgb(), {})
        primary = record["primary_depth_m"]
        secondary = _preprocess_secondary_depth(
            payload,
            raw_secondary,
            (primary.shape[1], primary.shape[0]),
        )
        secondary_scale, secondary_support = robust_observed_scale(
            record["observed_depth_m"],
            secondary,
            minimum_support=512,
        )
        secondary = (secondary * secondary_scale).astype(np.float32)
        secondary_valid = np.isfinite(secondary) & (secondary > 0)
        disagreement, pair_quality, pair_valid = teacher_pair_quality(
            primary,
            record["primary_valid"],
            secondary,
            secondary_valid,
            disagreement_scale=PAIR_DISAGREEMENT_SCALE,
        )
        combined = np.sqrt(
            np.clip(record["geometry_quality"], 0.0, 1.0)
            * np.clip(pair_quality, 0.0, 1.0)
        ).astype(np.float32)
        combined[~pair_valid] = 0.0
        record["secondary_depth_m"] = secondary
        record["secondary_valid"] = secondary_valid
        record["secondary_anchor_scale"] = secondary_scale
        record["secondary_anchor_support"] = secondary_support
        record["pair_relative_disagreement"] = disagreement
        record["pair_quality"] = pair_quality
        record["pair_valid"] = pair_valid
        record["combined_quality"] = combined


def _metric_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "parent_id": record["parent_id"],
            "truth_depth_m": record["truth_depth_m"],
            "prediction_depth_m": record["primary_depth_m"],
            "confidence": record["combined_quality"],
            "hidden_mask": record["hidden_mask"],
            "model_mask": record["pair_valid"],
            "baseline_depth_m": record["baseline_depth_m"],
        }
        for record in records
    ]


def evaluate_frozen_threshold(
    records: list[dict[str, Any]],
    threshold: float = FROZEN_ACCEPT_THRESHOLD,
) -> dict[str, Any]:
    metric_records = _metric_records(records)
    overall = _split_error(metric_records, threshold=threshold)
    parents: list[dict[str, Any]] = []
    for parent_id in sorted({record["parent_id"] for record in records}):
        parent = _split_error(
            [row for row in metric_records if row["parent_id"] == parent_id],
            threshold=threshold,
        )
        parents.append({"parent_id": parent_id, **parent})
    evaluable = [
        row
        for row in parents
        if row["accepted"]["count"] > 0 and row["rejected"]["count"] > 0
    ]
    improved = sum(
        row["accepted"]["mae_m"] < row["rejected"]["mae_m"]
        for row in evaluable
    )
    return {
        "overall": overall,
        "parents": parents,
        "evaluable_parent_count": len(evaluable),
        "accepted_lower_risk_parent_count": int(improved),
    }


def cross_source_passes(evaluation: dict[str, Any]) -> bool:
    overall = evaluation["overall"]
    accepted = overall["accepted"]
    rejected = overall["rejected"]
    return bool(
        overall["coverage"] >= MINIMUM_EVALUATION_COVERAGE
        and accepted["count"] > 0
        and rejected["count"] > 0
        and accepted["mae_m"] < rejected["mae_m"]
        and evaluation["evaluable_parent_count"] == len(evaluation["parents"])
        and evaluation["accepted_lower_risk_parent_count"]
        >= math.ceil(len(evaluation["parents"]) * 2 / 3)
    )


def build_depth_label_payload(
    record: dict[str, Any],
    threshold: float = FROZEN_ACCEPT_THRESHOLD,
) -> dict[str, np.ndarray]:
    source_valid = np.asarray(record["source_valid"], dtype=np.bool_)
    primary_valid = np.asarray(record["primary_valid"], dtype=np.bool_)
    pair_valid = np.asarray(record["pair_valid"], dtype=np.bool_)
    quality = np.asarray(record["combined_quality"], dtype=np.float32)
    teacher_candidate = ~source_valid & primary_valid & pair_valid
    teacher_accept = teacher_candidate & (quality >= threshold)
    teacher_b = (
        teacher_accept
        & (quality >= TEACHER_B_QUALITY)
        & (
            np.asarray(record["multiview_valid"], dtype=np.bool_)
            | (np.asarray(record["anchor_quality"], dtype=np.float32) >= 0.75)
        )
    )
    teacher_c = teacher_accept & ~teacher_b

    tiers = np.zeros(source_valid.shape, dtype=np.uint8)
    provenance = np.zeros(source_valid.shape, dtype=np.uint8)
    score = np.zeros(source_valid.shape, dtype=np.float32)
    tiers[source_valid] = TIER_A_SOURCE
    provenance[source_valid] = PROVENANCE_SOURCE_NATIVE
    score[source_valid] = 0.98
    tiers[teacher_b] = TIER_B_ANCHORED
    tiers[teacher_c] = TIER_C_TEACHER
    provenance[teacher_accept] = PROVENANCE_TEACHER
    score[teacher_accept] = quality[teacher_accept]

    metric_valid = source_valid | teacher_accept
    metric_depth = np.where(
        source_valid,
        record["truth_depth_m"],
        record["primary_depth_m"],
    ).astype(np.float32)
    metric_depth[~metric_valid] = np.nan
    uncertainty = np.full(metric_depth.shape, np.nan, dtype=np.float32)
    uncertainty[source_valid] = 0.015 + 0.01 * metric_depth[source_valid]
    teacher_uncertainty = (
        (0.015 + 0.02 * np.maximum(metric_depth, 0.0))
        * (1.0 + 2.5 * (1.0 - score))
    )
    pair_difference = np.abs(
        record["primary_depth_m"] - record["secondary_depth_m"]
    )
    teacher_uncertainty = np.sqrt(
        np.square(teacher_uncertainty) + np.square(0.5 * pair_difference)
    )
    uncertainty[teacher_accept] = teacher_uncertainty[teacher_accept]
    return {
        "metric_depth_m_hw": metric_depth,
        "metric_depth_valid_hw": metric_valid,
        "depth_uncertainty_proxy_m_hw": uncertainty,
        "quality_score_hw": score,
        "quality_tier_hw": tiers,
        "provenance_code_hw": provenance,
        "source_native_valid_hw": source_valid,
        "teacher_candidate_valid_hw": teacher_candidate,
        "primary_teacher_depth_m_hw": record["primary_depth_m"],
        "secondary_teacher_depth_m_hw": record["secondary_depth_m"],
        "teacher_pair_valid_hw": pair_valid,
        "teacher_pair_relative_disagreement_hw": record[
            "pair_relative_disagreement"
        ],
        "teacher_pair_quality_hw": record["pair_quality"],
        "combined_quality_hw": quality,
        "intrinsics_output": record["intrinsics"],
        "camera_to_world_output": record["camera_to_world"],
        "support_valid_hw": np.zeros(source_valid.shape, dtype=np.bool_),
        "boundary_evidence_valid_hw": np.zeros(source_valid.shape, dtype=np.bool_),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    require(args.cohort_manifest.is_file(), "TUM cohort manifest missing")
    require(args.mapanything_model.is_dir(), "MapAnything model missing")
    require(args.dav2_repo.is_dir(), "Depth Anything V2 repository missing")
    require(args.dav2_checkpoint.is_file(), "Depth Anything V2 checkpoint missing")
    require(not args.output_dir.exists(), f"output directory already exists: {args.output_dir}")
    fit_payloads, fit_receipt = load_tum_role_payloads(args.cohort_manifest, "fit")
    evaluation_payloads, evaluation_receipt = load_tum_role_payloads(
        args.cohort_manifest, "evaluation"
    )
    require(
        not (set(fit_receipt["parent_ids"]) & set(evaluation_receipt["parent_ids"])),
        "TUM FIT/evaluation parent overlap",
    )

    import torch

    torch.set_float32_matmul_precision("high")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model_load_started = time.monotonic()
    (
        torch,
        map_model,
        model_dtype,
        preprocess_inputs,
        geometry_functions,
    ) = _load_mapanything(args)
    map_model_load_seconds = time.monotonic() - model_load_started

    role_records: dict[str, list[dict[str, Any]]] = {"fit": [], "evaluation": []}
    for role_index, (role, payloads) in enumerate(
        (("fit", fit_payloads), ("evaluation", evaluation_payloads))
    ):
        for parent_index, parent_payloads in enumerate(
            _group_payloads(payloads).values()
        ):
            role_records[role].extend(
                _mapanything_infer_parent(
                    args,
                    role_index * 17 + parent_index,
                    parent_payloads,
                    torch,
                    map_model,
                    model_dtype,
                    preprocess_inputs,
                    geometry_functions,
                )
            )
        _attach_multiview_quality(role_records[role])

    del map_model
    gc.collect()
    torch.cuda.empty_cache()
    _attach_secondary_teacher(args, role_records["fit"] + role_records["evaluation"])

    fit_threshold = evaluate_frozen_threshold(role_records["fit"])
    evaluation_threshold = evaluate_frozen_threshold(role_records["evaluation"])
    passed = cross_source_passes(evaluation_threshold)
    args.output_dir.mkdir(parents=True)

    frame_receipts: list[dict[str, Any]] = []
    total_pixels = 0
    source_pixels = 0
    teacher_added_pixels = 0
    metric_pixels = 0
    if passed:
        for role, records in role_records.items():
            for record in records:
                label = build_depth_label_payload(record)
                output_path = args.output_dir / f"{role}__{record['frame_id']}.npz"
                np.savez_compressed(output_path, **label)
                source_valid = label["source_native_valid_hw"]
                metric_valid = label["metric_depth_valid_hw"]
                teacher_added = ~source_valid & metric_valid
                total_pixels += int(metric_valid.size)
                source_pixels += int(source_valid.sum())
                teacher_added_pixels += int(teacher_added.sum())
                metric_pixels += int(metric_valid.sum())
                frame_receipts.append(
                    {
                        "role": role,
                        "parent_id": record["parent_id"],
                        "frame_id": record["frame_id"],
                        "output_path": str(output_path.resolve()),
                        "output_bytes": output_path.stat().st_size,
                        "source_native_coverage": float(np.mean(source_valid)),
                        "teacher_added_coverage": float(np.mean(teacher_added)),
                        "metric_depth_coverage": float(np.mean(metric_valid)),
                        "pose_bracketing_gap_seconds": record[
                            "pose_bracketing_gap_seconds"
                        ],
                    }
                )

    risk_coverage = {
        role: compute_selective_metrics(_metric_records(records))
        for role, records in role_records.items()
    }
    primary_metrics: dict[str, Any] = {}
    secondary_metrics: dict[str, Any] = {}
    for role, records in role_records.items():
        truth_primary: list[np.ndarray] = []
        prediction_primary: list[np.ndarray] = []
        truth_secondary: list[np.ndarray] = []
        prediction_secondary: list[np.ndarray] = []
        for record in records:
            hidden = record["hidden_mask"] & record["pair_valid"]
            truth_primary.append(record["truth_depth_m"][hidden])
            prediction_primary.append(record["primary_depth_m"][hidden])
            truth_secondary.append(record["truth_depth_m"][hidden])
            prediction_secondary.append(record["secondary_depth_m"][hidden])
        primary_metrics[role] = _error_metrics(
            np.concatenate(truth_primary), np.concatenate(prediction_primary)
        )
        secondary_metrics[role] = _error_metrics(
            np.concatenate(truth_secondary), np.concatenate(prediction_secondary)
        )

    result = {
        "schema": SCHEMA,
        "status": (
            "TUM_CROSS_SOURCE_SELECTIVE_RISK_SUPPORTED_DEPTH_LABELS_MATERIALIZED"
            if passed
            else "TUM_CROSS_SOURCE_SELECTIVE_RISK_NOT_SUPPORTED"
        ),
        "mode": "WILD_LAB_CROSS_SOURCE_HELDOUT_EVALUATION",
        "question": "Does the frozen ARKit-derived multi-Teacher quality signal still separate reliable from unreliable depth on disjoint TUM RGB-D sequences?",
        "cohort": {
            "manifest_path": str(args.cohort_manifest.resolve()),
            "manifest_sha256": sha256_file(args.cohort_manifest),
            "fit": fit_receipt,
            "evaluation": evaluation_receipt,
            "evaluation_used_for_threshold_selection": False,
        },
        "teachers": {
            "primary": {
                "model_id": "facebook/map-anything-apache",
                "checkpoint_sha256": sha256_file(
                    args.mapanything_model / "model.safetensors"
                ),
                "role": "PRIMARY_SOURCE_ANCHORED_MULTIVIEW_GEOMETRY",
            },
            "secondary": {
                "model_id": "depth-anything-v2-metric-hypersim-vits",
                "checkpoint_sha256": sha256_file(args.dav2_checkpoint),
                "role": "INDEPENDENT_DISAGREEMENT_EVIDENCE_NOT_TRUTH",
            },
        },
        "frozen_gate": {
            "threshold": FROZEN_ACCEPT_THRESHOLD,
            "origin": "AG_ST_R0_C_TEACHER_THRESHOLD_UNCHANGED",
            "minimum_evaluation_coverage": MINIMUM_EVALUATION_COVERAGE,
            "fit": fit_threshold,
            "evaluation": evaluation_threshold,
        },
        "diagnostic_teacher_error": {
            "primary": primary_metrics,
            "secondary": secondary_metrics,
            "no_requirement_to_beat_either_teacher": True,
        },
        "risk_coverage": {
            role: {
                "compact_curve": _compact_curve(metrics),
                "full": metrics,
            }
            for role, metrics in risk_coverage.items()
        },
        "decision": {
            "cross_source_selective_risk_supported": passed,
            "depth_labels_materialized": passed,
            "support_materialized": False,
            "boundary_materialized": False,
            "reason_support_boundary_unknown": "TUM gravity basis has not been verified",
            "student_training_authorized": False,
            "next_execution": (
                "Add another independent geometry Teacher and expand source coverage before any student training."
                if passed
                else "Do not tune on held-out TUM evaluation; inspect FIT-only signal design or add an independent Teacher, then use a new held-out source."
            ),
        },
        "materialization": {
            "frame_count": len(frame_receipts),
            "source_native_coverage": (
                source_pixels / total_pixels if total_pixels else None
            ),
            "teacher_added_coverage": (
                teacher_added_pixels / total_pixels if total_pixels else None
            ),
            "metric_depth_coverage": (
                metric_pixels / total_pixels if total_pixels else None
            ),
            "support_and_boundary_are_all_unknown": True,
        },
        "execution": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "mapanything_model_load_seconds": map_model_load_seconds,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "elapsed_seconds": time.monotonic() - started,
        },
        "frame_receipts": frame_receipts,
        "claim_boundary": "Cross-source TUM depth pseudo-label and selective-risk evidence only. It is not support/boundary truth, complete geometry truth, formal F1 authorization, task utility, deployment, product, or safety evidence.",
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_TUM_COHORT_MANIFEST)
    parser.add_argument("--mapanything-model", type=Path, default=DEFAULT_MAPANYTHING_MODEL)
    parser.add_argument("--dav2-repo", type=Path, default=DEFAULT_DAV2_REPO)
    parser.add_argument("--dav2-checkpoint", type=Path, default=DEFAULT_DAV2_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dav2-precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--dav2-input-size", type=int, default=518)
    parser.add_argument("--longest-side", type=int, default=336)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--mask-modulus", type=int, default=4)
    parser.add_argument("--minimum-hidden-pixels", type=int, default=512)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run(args)
        result_path = args.output_dir / "result.json"
        _write_json_exclusive(result_path, result)
        print(
            json.dumps(
                {
                    "status": result["status"],
                    "result": str(result_path),
                    "evaluation": result["frozen_gate"]["evaluation"]["overall"],
                    "materialization": result["materialization"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except Exception as error:
        failure = {
            "schema": "blindassist_ag_st_tum_cross_source_failure_v1",
            "status": "FAILED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "failure.json"
        if not failure_path.exists():
            _write_json_exclusive(failure_path, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
