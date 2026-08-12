#!/usr/bin/env python3
"""Evaluate a frozen AG-ST no-regret selector on parent-disjoint Bonn depth."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_student_bonn_depth import (
    DEFAULT_BONN_ARCHIVE,
    DEFAULT_BONN_CATALOG,
    DEFAULT_BONN_RECEIPT,
    DEFAULT_BONN_ROOT,
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    build_students,
    checkpoint_architecture,
    checkpoint_parent_ids,
    extract_rgb_only_feature,
    fixed_frame_pairs,
    load_cohort_indices,
    load_depth_native,
    load_rgb_native,
    load_depthart_backbone,
    validate_source_receipts,
)
from train_ag_st_bonn_anchored_student import DEFAULT_COHORT_MANIFEST
from train_ag_st_masked_student import write_json_exclusive
from train_ag_st_no_regret_selector import (
    SELECTOR_SCHEMA,
    NoRegretDepthSelector,
    SelectorObservation,
    summarize_selector_observations,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTOR_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-no-regret-selector-two-domain-r0/no-regret-selector.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-no-regret-selector-bonn-evaluation-r0.json"
)


def load_selector(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[dict[str, Any], NoRegretDepthSelector, float, Path, dict[str, Any], torch.nn.Module]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(isinstance(payload, dict), "selector checkpoint root invalid")
    require(payload.get("schema") == SELECTOR_SCHEMA, "selector checkpoint schema drift")
    architecture = payload.get("architecture")
    require(isinstance(architecture, dict), "selector architecture missing")
    expert_receipt = payload.get("expert")
    require(isinstance(expert_receipt, dict), "selector expert receipt missing")
    expert_path = Path(str(expert_receipt["checkpoint_path"])).resolve()
    require(expert_path.is_file(), "selector expert checkpoint missing")
    require(
        sha256_file(expert_path) == str(expert_receipt["checkpoint_sha256"]),
        "selector expert checkpoint hash drift",
    )
    expert_payload = torch.load(expert_path, map_location="cpu", weights_only=False)
    require(isinstance(expert_payload, dict), "selector expert payload invalid")
    expert_architecture = checkpoint_architecture(expert_payload)
    _, expert = build_students(expert_payload, expert_architecture, device)
    expert.eval().requires_grad_(False)
    selector = NoRegretDepthSelector(
        feature_channels=int(architecture["feature_channels"]),
        hidden=int(architecture["hidden_channels"]),
        global_context_profile=str(architecture.get("global_context_profile", "none")),
    ).to(device)
    incompatible = selector.load_state_dict(payload["state_dict"], strict=True)
    require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "selector state-dict drift",
    )
    selector.eval().requires_grad_(False)
    threshold = float(payload["threshold"])
    require(0.0 < threshold <= 1.001, "selector frozen threshold invalid")
    return payload, selector, threshold, expert_path, expert_payload, expert


def selector_consumed_parents(payload: dict[str, Any]) -> set[str]:
    split = payload.get("split", {})
    require(isinstance(split, dict), "selector split missing")
    return {
        str(parent)
        for role in ("selector_fit_parents", "selector_calibration_parents")
        for parent in split.get(role, [])
    }


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    dataset_root = args.dataset_root.resolve()
    archive = args.archive.resolve()
    catalog = args.catalog.resolve()
    receipt = args.receipt.resolve()
    cohort_manifest = args.cohort_manifest.resolve()
    selector_checkpoint = args.selector_checkpoint.resolve()
    depthart_source = args.depthart_source.resolve()
    depthart_checkpoint = args.depthart_checkpoint.resolve()
    output = args.output.resolve()
    require(selector_checkpoint.is_file(), "selector checkpoint missing")
    require(depthart_source.is_dir(), "DepthART source missing")
    require(depthart_checkpoint.is_file(), "DepthART checkpoint missing")
    require(not output.exists(), "selector evaluation output collision")
    require(torch.cuda.is_available(), "selector Bonn evaluation requires CUDA")

    frame_indices = load_cohort_indices(cohort_manifest, args.cohort_role)
    _, source_receipt, source_provenance = validate_source_receipts(
        dataset_root,
        archive,
        catalog,
        receipt,
        set(frame_indices),
    )
    pairs_by_parent = fixed_frame_pairs(dataset_root, frame_indices)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    (
        selector_payload,
        selector,
        threshold,
        expert_path,
        expert_payload,
        expert,
    ) = load_selector(selector_checkpoint, device)
    evaluation_parents = set(frame_indices)
    selector_overlap = sorted(selector_consumed_parents(selector_payload) & evaluation_parents)
    expert_overlap = sorted(checkpoint_parent_ids(expert_payload) & evaluation_parents)
    require(not selector_overlap, f"selector/evaluation parent overlap: {selector_overlap}")
    require(not expert_overlap, f"expert/evaluation parent overlap: {expert_overlap}")

    extractor, scan = load_depthart_backbone(
        depthart_source,
        depthart_checkpoint,
        device,
        int(selector_payload["seed"]),
    )
    expert_architecture = checkpoint_architecture(expert_payload)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    observations: list[SelectorObservation] = []
    frame_receipts: list[dict[str, Any]] = []
    for parent_id, pairs in pairs_by_parent.items():
        for pair in pairs:
            rgb = load_rgb_native(pair.rgb.absolute_path)
            feature, base_depth = extract_rgb_only_feature(
                extractor,
                rgb,
                expert_architecture["feature_profile"],
                device,
                amp_dtype,
            )
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
            ):
                expert_outputs = expert(
                    feature,
                    base_depth,
                    (base_depth.shape[-2], base_depth.shape[-1]),
                )
                selector_outputs = selector(
                    feature,
                    base_depth,
                    expert_outputs["depth_m"],
                    expert_outputs["depth_identity_gate"],
                    (base_depth.shape[-2], base_depth.shape[-1]),
                )
            # Source truth is opened only after all RGB/K-only predictions exist.
            truth_depth_m, valid = load_depth_native(pair.depth.absolute_path)
            base_np = base_depth[0, 0].float().cpu().numpy()
            expert_np = expert_outputs["depth_m"][0, 0].float().cpu().numpy()
            probability_np = (
                selector_outputs["selector_probability"][0, 0].float().cpu().numpy()
            )
            observations.append(
                SelectorObservation(
                    parent_id=parent_id,
                    domain="BONN_RGBD_DYNAMIC",
                    truth_depth_m=truth_depth_m,
                    valid=valid,
                    base_depth_m=base_np,
                    expert_depth_m=expert_np,
                    selector_probability=probability_np,
                )
            )
            frame_receipts.append(
                {
                    "parent_id": parent_id,
                    "rgb_row_index_zero_based": pair.rgb.row_index,
                    "rgb_relative_path": pair.rgb.relative_path.as_posix(),
                    "rgb_sha256": sha256_file(pair.rgb.absolute_path),
                    "depth_row_index_zero_based": pair.depth.row_index,
                    "depth_relative_path": pair.depth.relative_path.as_posix(),
                    "depth_sha256": sha256_file(pair.depth.absolute_path),
                    "rgb_depth_delta_seconds": pair.association_delta_seconds,
                    "source_valid_pixel_count": int(valid.sum()),
                    "selector_selected_fraction_on_valid": float(
                        ((probability_np >= threshold) & valid).sum() / valid.sum()
                    ),
                }
            )
    metrics = summarize_selector_observations(observations, threshold)
    parent_macro = metrics["parent_macro"]
    non_regret = (
        parent_macro["selected_mae_delta_vs_base_m"] <= 0.0
        and parent_macro["selected_bad_delta_vs_base"] <= 0.0
    )
    result = {
        "schema": "blindassist_ag_st_no_regret_selector_bonn_evaluation_result_v1",
        "status": (
            "NO_REGRET_SELECTOR_CROSS_DATASET_PASS"
            if non_regret and parent_macro["selected_coverage_fraction"] > 0.0
            else "NO_REGRET_SELECTOR_CROSS_DATASET_NOT_SUPPORTED"
        ),
        "mode": "WILD_LAB_DEVELOPMENT",
        "question": (
            "Does the frozen selector preserve or improve frozen metric DepthART on "
            "parent-disjoint Bonn depth while retaining nonzero correction coverage?"
        ),
        "provenance": {
            **source_provenance,
            "dataset_root": str(dataset_root),
            "source_receipt_schema": source_receipt.get("schema"),
            "cohort_manifest_path": str(cohort_manifest),
            "cohort_manifest_sha256": sha256_file(cohort_manifest),
            "cohort_role": args.cohort_role,
            "selector_checkpoint_path": str(selector_checkpoint),
            "selector_checkpoint_sha256": sha256_file(selector_checkpoint),
            "expert_checkpoint_path": str(expert_path),
            "expert_checkpoint_sha256": sha256_file(expert_path),
            "depthart_checkpoint_path": str(depthart_checkpoint),
            "depthart_checkpoint_sha256": sha256_file(depthart_checkpoint),
            "evaluator_path": str(Path(__file__).resolve()),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "parent_firewall": {
            "selector_consumed_parents": sorted(selector_consumed_parents(selector_payload)),
            "expert_consumed_parents": sorted(checkpoint_parent_ids(expert_payload)),
            "evaluation_parents": sorted(evaluation_parents),
            "selector_overlap": selector_overlap,
            "expert_overlap": expert_overlap,
            "evaluation_used_for_selector_fit_or_threshold": False,
        },
        "frozen_decision": {
            "threshold": threshold,
            "threshold_decision": selector_payload["threshold_calibration_decision"],
            "base_fallback_below_threshold": True,
        },
        "cohort": {
            "parent_count": len(frame_indices),
            "frame_count": len(observations),
            "frame_indices_zero_based": {
                parent: list(indices) for parent, indices in frame_indices.items()
            },
            "frame_receipts": frame_receipts,
        },
        "metrics": metrics,
        "decision": {
            "nonzero_correction_coverage": parent_macro[
                "selected_coverage_fraction"
            ]
            > 0.0,
            "mae_no_regret": parent_macro["selected_mae_delta_vs_base_m"] <= 0.0,
            "bad_rate_no_regret": parent_macro["selected_bad_delta_vs_base"] <= 0.0,
            "cross_dataset_no_regret_supported": non_regret
            and parent_macro["selected_coverage_fraction"] > 0.0,
        },
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "scan_backend": scan,
            "total_seconds": time.perf_counter() - started,
            "training_performed": False,
            "threshold_selected": False,
        },
        "claim_boundary": {
            "depth_only": True,
            "complete_truth_required": False,
            "source_depth_is_model_input": False,
            "support_boundary_obstacle_evaluated": False,
            "freshness_scope": "PARENT_DISJOINT_FROM_SELECTOR_AND_EXPERT_BUT_PREVIOUSLY_CONSUMED_FOR_AG_ST_DEVELOPMENT",
            "confirmation_claim_authorized": False,
            "task_deployment_product_safety_claim_authorized": False,
        },
    }
    write_json_exclusive(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "threshold": threshold,
                "parent_macro": parent_macro,
                "decision": result["decision"],
                "total_seconds": result["execution"]["total_seconds"],
            },
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selector-checkpoint", type=Path, default=DEFAULT_SELECTOR_CHECKPOINT)
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_COHORT_MANIFEST)
    parser.add_argument("--cohort-role", default="evaluation")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_BONN_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_BONN_ARCHIVE)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_BONN_CATALOG)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_BONN_RECEIPT)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
