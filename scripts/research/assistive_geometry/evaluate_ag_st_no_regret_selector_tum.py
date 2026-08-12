#!/usr/bin/env python3
"""Evaluate a frozen AG-ST no-regret selector on parent-disjoint TUM RGB-D."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from ag_st_tum_rgbd import DEFAULT_TUM_COHORT_MANIFEST, load_tum_role_payloads
from download_b0_arkitscenes_assets import require, sha256_file
from evaluate_ag_st_no_regret_selector_bonn import (
    load_selector,
    selector_consumed_parents,
)
from evaluate_ag_st_student_bonn_depth import (
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    checkpoint_architecture,
    checkpoint_parent_ids,
    extract_rgb_only_feature_with_intrinsics,
    load_depthart_backbone,
)
from train_ag_st_masked_student import write_json_exclusive
from train_ag_st_no_regret_selector import (
    SelectorObservation,
    summarize_selector_observations,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SELECTOR_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-no-regret-selector-three-domain-r0/no-regret-selector.pt"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts.local/experiments/ag-st-no-regret-selector-three-domain-tum-evaluation-r0.json"
)


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    selector_checkpoint = args.selector_checkpoint.resolve()
    cohort_manifest = args.cohort_manifest.resolve()
    depthart_source = args.depthart_source.resolve()
    depthart_checkpoint = args.depthart_checkpoint.resolve()
    output = args.output.resolve()
    require(selector_checkpoint.is_file(), "TUM selector checkpoint missing")
    require(not output.exists(), "TUM selector evaluation output collision")
    require(torch.cuda.is_available(), "TUM selector evaluation requires CUDA")

    payloads, source_provenance = load_tum_role_payloads(
        cohort_manifest,
        "evaluation",
    )
    evaluation_parents = {payload.parent_id for payload in payloads}
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
    selector_overlap = sorted(selector_consumed_parents(selector_payload) & evaluation_parents)
    expert_overlap = sorted(checkpoint_parent_ids(expert_payload) & evaluation_parents)
    require(not selector_overlap, f"selector/TUM evaluation overlap: {selector_overlap}")
    require(not expert_overlap, f"expert/TUM evaluation overlap: {expert_overlap}")

    extractor, scan = load_depthart_backbone(
        depthart_source,
        depthart_checkpoint,
        device,
        int(selector_payload["seed"]),
    )
    expert_architecture = checkpoint_architecture(expert_payload)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    observations: list[SelectorObservation] = []
    frame_receipts = []
    for payload in payloads:
        rgb = payload.load_rgb()
        feature, base_depth = extract_rgb_only_feature_with_intrinsics(
            extractor,
            rgb,
            payload.intrinsics,
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
        # The registered depth is decoded only after RGB/K-only inference.
        truth_depth_m, valid = payload.load_depth()
        base_np = base_depth[0, 0].float().cpu().numpy()
        expert_np = expert_outputs["depth_m"][0, 0].float().cpu().numpy()
        probability_np = (
            selector_outputs["selector_probability"][0, 0].float().cpu().numpy()
        )
        observations.append(
            SelectorObservation(
                parent_id=payload.parent_id,
                domain="TUM_RGBD",
                truth_depth_m=truth_depth_m,
                valid=valid,
                base_depth_m=base_np,
                expert_depth_m=expert_np,
                selector_probability=probability_np,
            )
        )
        frame_receipts.append(
            {
                "parent_id": payload.parent_id,
                "rgb_row_index_zero_based": payload.rgb.row_index,
                "depth_row_index_zero_based": payload.depth.row_index,
                "rgb_relative_path": payload.rgb.relative_path,
                "depth_relative_path": payload.depth.relative_path,
                "rgb_depth_delta_seconds": payload.association_delta_seconds,
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
        "schema": "blindassist_ag_st_no_regret_selector_tum_evaluation_result_v1",
        "status": (
            "NO_REGRET_SELECTOR_TUM_CROSS_DATASET_PASS"
            if non_regret and parent_macro["selected_coverage_fraction"] > 0.0
            else "NO_REGRET_SELECTOR_TUM_CROSS_DATASET_NOT_SUPPORTED"
        ),
        "mode": "WILD_LAB_DEVELOPMENT",
        "question": (
            "Does the frozen three-domain selector preserve or improve metric "
            "DepthART on parent-disjoint TUM RGB-D with nonzero correction coverage?"
        ),
        "provenance": {
            **source_provenance,
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
            "parent_count": len(evaluation_parents),
            "frame_count": len(observations),
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
            "project_global_fresh_claim": False,
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
    parser.add_argument("--cohort-manifest", type=Path, default=DEFAULT_TUM_COHORT_MANIFEST)
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint", type=Path, default=DEFAULT_DEPTHART_CHECKPOINT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
