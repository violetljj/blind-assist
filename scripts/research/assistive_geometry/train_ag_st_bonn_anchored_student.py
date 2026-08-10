#!/usr/bin/env python3
"""Train an identity-gated AG-ST student with Bonn source-depth anchors."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build_ag_st_factor_labels import PROVENANCE_UNKNOWN
from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_student_bonn_depth import (
    BONN_HEIGHT,
    BONN_WIDTH,
    DEFAULT_BONN_ARCHIVE,
    DEFAULT_BONN_CATALOG,
    DEFAULT_BONN_RECEIPT,
    DEFAULT_BONN_ROOT,
    extract_rgb_only_feature,
    fixed_frame_pairs,
    load_cohort_indices,
    load_depth_native,
    load_rgb_native,
    validate_source_receipts,
)
from train_ag_st_masked_student import (
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    DEPTHART_SHARED_CHANNELS,
    MAX_BOUNDARY_DISTANCE_PX,
    PROVENANCE_SOURCE_NATIVE,
    TIER_A_SOURCE,
    TIER_UNKNOWN,
    CachedFrame,
    FrameDescriptor,
    MaskedFactorStudent,
    aggregate_label_digest,
    build_frame_descriptor_batches,
    calibrate_support_head,
    compute_training_priors,
    evaluate_frames,
    extract_depthart_features,
    load_depthart_backbone,
    save_checkpoint_exclusive,
    train_student,
    write_json_exclusive,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE0A_RESULTS = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-stage0a-mapanything-apache-train16-block64-r1/result.json",
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-stage0a-mapanything-spatial-train16-block64-r1/result.json",
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-stage0a-mapanything-b0-development8-block64-r1/result.json",
)
DEFAULT_LABEL_DIRS = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-superteacher-factor-labels-train16-r5",
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-superteacher-factor-labels-spatial-train16-r0",
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-superteacher-factor-labels-b0-development8-r0",
)
DEFAULT_COHORT_MANIFEST = (
    REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_ST_BONN_MIXED_DOMAIN_COHORT_R0_2026-08-10.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-bonn-anchored-identity-gated-student-r0"
)


def _unknown_factor_targets(
    depth_m: np.ndarray,
    depth_valid: np.ndarray,
) -> dict[str, torch.Tensor]:
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.asarray(depth_valid, dtype=np.bool_)
    require(depth.shape == (BONN_HEIGHT, BONN_WIDTH), "Bonn depth target shape drift")
    require(valid.shape == depth.shape, "Bonn depth validity shape drift")
    metric_tier = np.full(depth.shape, TIER_UNKNOWN, dtype=np.uint8)
    metric_tier[valid] = TIER_A_SOURCE
    zeros = np.zeros(depth.shape, dtype=np.float32)
    false = np.zeros(depth.shape, dtype=np.bool_)
    unknown = np.full(depth.shape, TIER_UNKNOWN, dtype=np.uint8)
    return {
        "metric_depth_m": torch.from_numpy(depth)[None, None],
        "metric_valid": torch.from_numpy(valid)[None, None],
        "metric_tier": torch.from_numpy(metric_tier)[None, None],
        "metric_provenance": torch.from_numpy(
            np.where(valid, PROVENANCE_SOURCE_NATIVE, PROVENANCE_UNKNOWN).astype(
                np.uint8
            )
        )[None, None],
        "support": torch.from_numpy(zeros.copy())[None, None],
        "support_valid": torch.from_numpy(false.copy())[None, None],
        "support_tier": torch.from_numpy(unknown.copy())[None, None],
        "boundary": torch.from_numpy(zeros.copy())[None, None],
        "boundary_distance_px": torch.full(
            (1, 1, BONN_HEIGHT, BONN_WIDTH),
            MAX_BOUNDARY_DISTANCE_PX,
            dtype=torch.float32,
        ),
        "obstacle": torch.from_numpy(zeros.copy())[None, None],
        "evidence_valid": torch.from_numpy(false.copy())[None, None],
        "evidence_tier": torch.from_numpy(unknown.copy())[None, None],
    }


def extract_bonn_anchor_frames(
    cohort_manifest: Path,
    dataset_root: Path,
    archive: Path,
    catalog: Path,
    receipt: Path,
    depthart_source: Path,
    depthart_checkpoint: Path,
    device: torch.device,
    seed: int,
) -> tuple[list[CachedFrame], dict[str, Any]]:
    frame_indices = load_cohort_indices(cohort_manifest, "fit")
    _, _, provenance = validate_source_receipts(
        dataset_root,
        archive,
        catalog,
        receipt,
        set(frame_indices),
    )
    pairs_by_parent = fixed_frame_pairs(dataset_root, frame_indices)
    extractor, scan = load_depthart_backbone(
        depthart_source,
        depthart_checkpoint,
        device,
        seed,
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    started = time.perf_counter()
    frames: list[CachedFrame] = []
    frame_receipts: list[dict[str, Any]] = []
    for parent_id, pairs in pairs_by_parent.items():
        for pair in pairs:
            rgb = load_rgb_native(pair.rgb.absolute_path)
            feature, base_depth = extract_rgb_only_feature(
                extractor,
                rgb,
                "shared",
                device,
                amp_dtype,
            )
            depth_m, depth_valid = load_depth_native(pair.depth.absolute_path)
            descriptor = FrameDescriptor(
                parent_id=parent_id,
                frame_index=pair.rgb.row_index,
                frame_stem=f"{parent_id}:{pair.rgb.row_index}",
                output_hw=(BONN_HEIGHT, BONN_WIDTH),
                label_path=pair.depth.absolute_path,
                video={"source": "BONN_RGBD_DYNAMIC"},
            )
            frames.append(
                CachedFrame(
                    descriptor=descriptor,
                    feature=feature[0].to(dtype=torch.float16, device="cpu"),
                    base_depth_m=base_depth[0].float().cpu(),
                    targets=_unknown_factor_targets(depth_m, depth_valid),
                )
            )
            frame_receipts.append(
                {
                    "parent_id": parent_id,
                    "rgb_row_index_zero_based": pair.rgb.row_index,
                    "rgb_relative_path": pair.rgb.relative_path.as_posix(),
                    "depth_relative_path": pair.depth.relative_path.as_posix(),
                    "rgb_depth_delta_seconds": pair.association_delta_seconds,
                    "depth_valid_fraction": float(depth_valid.mean()),
                }
            )
    del extractor
    torch.cuda.empty_cache()
    require(len(frames) == 3 * len(frame_indices), "Bonn anchor frame count drift")
    return frames, {
        "cohort_manifest_path": str(cohort_manifest),
        "cohort_manifest_sha256": sha256_file(cohort_manifest),
        "parent_ids": list(frame_indices),
        "parent_count": len(frame_indices),
        "frame_count": len(frames),
        "feature_profile": "shared",
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "elapsed_seconds": time.perf_counter() - started,
        "scan_backend": scan,
        "source_provenance": provenance,
        "frame_receipts": frame_receipts,
        "factor_validity": {
            "metric_depth": "A_SOURCE_WHERE_UINT16_GT_ZERO",
            "support": "UNKNOWN",
            "boundary": "UNKNOWN",
            "obstacle": "UNKNOWN",
        },
    }


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), "mixed-domain output collision")
    output_dir.mkdir(parents=True, exist_ok=False)
    stage0a_results = [value.resolve() for value in args.stage0a_result]
    label_dirs = [value.resolve() for value in args.label_dir]
    require(len(stage0a_results) == len(label_dirs) == 3, "expected three ARKit batches")
    cohort_manifest = args.cohort_manifest.resolve()
    depthart_source = args.depthart_source.resolve()
    depthart_checkpoint = args.depthart_checkpoint.resolve()
    require(cohort_manifest.is_file(), "Bonn cohort manifest missing")
    require(depthart_source.is_dir(), "DepthART source missing")
    require(depthart_checkpoint.is_file(), "DepthART checkpoint missing")

    descriptors, source_batches = build_frame_descriptor_batches(
        stage0a_results,
        label_dirs,
    )
    arkit_parents = {row.parent_id for row in descriptors}
    require(len(arkit_parents) == 40 and len(descriptors) == 120, "ARKit 40/120 drift")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "mixed-domain training requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    arkit_frames, arkit_extraction = extract_depthart_features(
        descriptors,
        depthart_source,
        depthart_checkpoint,
        device,
        args.seed,
        feature_profile="shared",
    )
    bonn_frames, bonn_extraction = extract_bonn_anchor_frames(
        cohort_manifest,
        args.dataset_root.resolve(),
        args.archive.resolve(),
        args.catalog.resolve(),
        args.receipt.resolve(),
        depthart_source,
        depthart_checkpoint,
        device,
        args.seed,
    )
    bonn_parents = {row.descriptor.parent_id for row in bonn_frames}
    require(not (arkit_parents & bonn_parents), "mixed-domain parent collision")
    bonn_repeat_factor = int(math.ceil(len(arkit_frames) / len(bonn_frames)))
    optimization_frames = [*arkit_frames, *(bonn_frames * bonn_repeat_factor)]
    priors, class_weights = compute_training_priors(arkit_frames)

    torch.manual_seed(args.seed)
    model = MaskedFactorStudent(
        channels=DEPTHART_SHARED_CHANNELS,
        hidden=32,
        depth_mode="residual",
        head_profile="basic",
        use_base_depth_feature=True,
        depth_gate_profile="identity_sigmoid",
    ).to(device)
    model.initialize_priors(priors)
    for unused_head in (
        model.boundary_logits,
        model.boundary_distance_logits,
        model.obstacle_logits,
    ):
        unused_head.requires_grad_(False)
    before_arkit = evaluate_frames(model, arkit_frames, device)
    before_bonn_fit = evaluate_frames(model, bonn_frames, device)
    history, training = train_student(
        model,
        optimization_frames,
        class_weights,
        device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        objective_profile="depth_support_precision",
    )
    support_calibration = calibrate_support_head(
        model,
        arkit_frames,
        device,
        steps=args.support_calibration_steps,
    )
    after_arkit = evaluate_frames(model, arkit_frames, device)
    after_bonn_fit = evaluate_frames(model, bonn_frames, device)

    fit_parents = sorted(arkit_parents | bonn_parents)
    split = {
        "method": "ALL_40_CONSUMED_ARKIT_PLUS_HASH_FROZEN_BONN_DEPTH_ANCHORS",
        "fit_parents": fit_parents,
        "train_parents": fit_parents,
        "selection_parents": [],
        "canary_parents": [],
        "arkit_fit_parents": sorted(arkit_parents),
        "bonn_depth_anchor_parents": sorted(bonn_parents),
        "external_evaluation_parents_read": False,
    }
    architecture = {
        "frozen_encoder": "FROZEN_DEPTHART_S_METRIC_INDOOR",
        "input_feature_channels": DEPTHART_SHARED_CHANNELS,
        "feature_profile": "shared",
        "head_hidden_channels": 32,
        "head_profile": "basic",
        "use_base_depth_feature": True,
        "depth_gate_profile": "identity_sigmoid",
        "depth_mode": "residual",
        "objective_profile": "depth_support_precision",
        "outputs": [
            "metric_depth_residual",
            "depth_identity_gate",
            "support_logit",
            "boundary_logit",
            "boundary_distance_px",
            "obstacle_logit",
        ],
    }
    checkpoint = save_checkpoint_exclusive(
        output_dir / "masked-factor-head.pt",
        {
            "schema": "blindassist_ag_st_masked_factor_student_checkpoint_v1",
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "architecture": architecture,
            "split": split,
            "priors": priors,
            "class_weights": class_weights,
            "seed": args.seed,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "objective_profile": "depth_support_precision",
            "support_calibration": support_calibration,
        },
    )
    result = {
        "schema": "blindassist_ag_st_bonn_anchored_identity_gated_student_result_v1",
        "status": "MIXED_DOMAIN_FIT_COMPLETED_AWAITING_DISJOINT_BONN_EVALUATION",
        "mode": "WILD_LAB_DEVELOPMENT",
        "question": "Can source-native Bonn A-tier depth anchors plus an identity-initialized correction gate prevent ARKit-specific residual collapse?",
        "inputs": {
            "arkit_source_batches": source_batches,
            "arkit_factor_label_payloads": aggregate_label_digest(
                row.label_path for row in descriptors
            ),
            "bonn_depth_anchors": bonn_extraction,
            "depthart_checkpoint_path": str(depthart_checkpoint),
            "depthart_checkpoint_sha256": sha256_file(depthart_checkpoint),
            "trainer_path": str(Path(__file__).resolve()),
            "trainer_sha256": sha256_file(Path(__file__).resolve()),
        },
        "split": split,
        "architecture": architecture,
        "checkpoint": checkpoint,
        "training": {
            **training,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "unique_frame_count": len(arkit_frames) + len(bonn_frames),
            "optimizer_frame_visits_per_epoch": len(optimization_frames),
            "arkit_frame_count": len(arkit_frames),
            "bonn_anchor_frame_count": len(bonn_frames),
            "bonn_anchor_repeat_factor": bonn_repeat_factor,
            "history": history,
            "support_calibration": support_calibration,
        },
        "metrics": {
            "before_arkit_fit": before_arkit,
            "after_arkit_fit": after_arkit,
            "before_bonn_depth_anchor_fit": before_bonn_fit,
            "after_bonn_depth_anchor_fit": after_bonn_fit,
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "arkit_feature_extraction": arkit_extraction,
            "total_seconds": time.perf_counter() - started,
        },
        "claim_boundary": {
            "bonn_fit_depth_is_A_tier_source_native": True,
            "bonn_support_boundary_obstacle_are_UNKNOWN": True,
            "disjoint_bonn_evaluation_read": False,
            "cross_dataset_transfer_claim_authorized": False,
            "task_deployment_product_safety_claim_authorized": False,
        },
    }
    write_json_exclusive(output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "result": str(output_dir / "result.json"),
                "checkpoint": checkpoint,
                "after_arkit": after_arkit["parent_macro"],
                "after_bonn_fit": after_bonn_fit["parent_macro"],
                "total_seconds": result["execution"]["total_seconds"],
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage0a-result",
        type=Path,
        action="append",
        default=list(DEFAULT_STAGE0A_RESULTS),
    )
    parser.add_argument(
        "--label-dir",
        type=Path,
        action="append",
        default=list(DEFAULT_LABEL_DIRS),
    )
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_BONN_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_BONN_CATALOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_BONN_RECEIPT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--support-calibration-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1729)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
