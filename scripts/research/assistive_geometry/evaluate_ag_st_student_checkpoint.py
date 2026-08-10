#!/usr/bin/env python3
"""Evaluate an AG-ST factor head on a parent-disjoint label batch."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from download_b0_arkitscenes_assets import require, sha256_file
from train_ag_st_masked_student import (
    DEFAULT_DEPTHART_CHECKPOINT,
    DEFAULT_DEPTHART_SOURCE,
    DEPTHART_PYRAMID_CHANNELS,
    DEPTHART_SHARED_CHANNELS,
    MaskedFactorStudent,
    aggregate_label_digest,
    build_frame_descriptors,
    evaluate_frames,
    extract_depthart_features,
    relative_improvement,
    stable_hash,
    write_json_exclusive,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STAGE0A_RESULT = (
    REPO_ROOT
    / "artifacts.local"
    / "experiments"
    / "ag-st-stage0a-mapanything-spatial-train16-block64-r1"
    / "result.json"
)
DEFAULT_LABEL_DIR = (
    REPO_ROOT
    / "artifacts.local"
    / "experiments"
    / "ag-st-superteacher-factor-labels-spatial-train16-r0"
)
DEFAULT_STUDENT_CHECKPOINT = (
    REPO_ROOT
    / "artifacts.local"
    / "experiments"
    / "ag-st-masked-student-depthart-s-r1"
    / "masked-factor-head.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "artifacts.local"
    / "experiments"
    / "ag-st-masked-student-depthart-s-fresh-zero-shot-r0"
)

EVALUATION_MODES = (
    "fresh_zero_shot",
    "consumed_development_comparison",
)


def _checkpoint_parent_ids(payload: dict[str, Any]) -> set[str]:
    split = payload["split"]
    return {
        str(parent)
        for role in (
            "train_parents",
            "selection_parents",
            "canary_parents",
            "fit_parents",
        )
        for parent in split.get(role, [])
    }


def _diagnostic_parent_split(
    parents: set[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, Any]]:
    require(len(parents) == 16, "diagnostic split requires 16 parents")
    token = "AG_ST_FRESH_ZERO_SHOT_DIAGNOSTIC_R0"
    ranked = tuple(
        sorted(parents, key=lambda parent: stable_hash(f"{token}:{parent}"))
    )
    fit = tuple(sorted(ranked[:12]))
    selection = tuple(sorted(ranked[12:14]))
    canary = tuple(sorted(ranked[14:]))
    return fit, selection, canary, {
        "method": "PARENT_ID_SHA256_12_2_2_DIAGNOSTIC_ONLY",
        "split_token": token,
        "fit_parents": list(fit),
        "selection_parents": list(selection),
        "canary_parents": list(canary),
        "ranked_parents": list(ranked),
    }


def _improvements(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float | None]:
    before_macro = before["parent_macro"]
    after_macro = after["parent_macro"]
    pairs = {
        "depth_mae": "depth_mae_m",
        "support_bce": "support_bce",
        "boundary_bce": "boundary_bce",
        "boundary_soft_bce": "boundary_soft_bce",
        "boundary_distance_mae": "boundary_distance_mae_px",
        "obstacle_bce": "obstacle_bce",
    }
    return {
        output: relative_improvement(before_macro[metric], after_macro[metric])
        for output, metric in pairs.items()
    }


def _core_factor_names(objective_profile: str) -> tuple[str, ...]:
    if objective_profile in {"depth_support", "depth_support_precision"}:
        return ("depth_mae", "support_bce")
    if objective_profile == "boundary_only":
        return ("boundary_soft_bce", "boundary_distance_mae")
    return ("depth_mae", "support_bce", "obstacle_bce")


def _evaluation_context(mode: str, parent_count: int) -> dict[str, Any]:
    require(mode in EVALUATION_MODES, f"unsupported evaluation mode: {mode}")
    if mode == "fresh_zero_shot":
        return {
            "schema": "blindassist_ag_st_student_fresh_zero_shot_wild_lab_result_v1",
            "status_prefix": "FRESH_PARENT_ZERO_SHOT",
            "question": (
                "Does the frozen AG-ST factor head transfer without fitting to "
                f"{parent_count} completely disjoint source parents?"
            ),
            "labels_previously_consumed": False,
            "fresh_claim_authorized": True,
            "claim_boundary": (
                f"Zero-shot pseudo-label transfer across {parent_count} "
                "parent-disjoint ARKitScenes source videos. This is not objective "
                "truth, cross-dataset generalization, deterministic-reducer task "
                "utility, deployment, product, or safety evidence."
            ),
        }
    return {
        "schema": (
            "blindassist_ag_st_student_consumed_development_comparison_"
            "wild_lab_result_v1"
        ),
        "status_prefix": "CONSUMED_PARENT_ZERO_SHOT_DEVELOPMENT",
        "question": (
            "Does the new frozen AG-ST head improve over its initialized baseline "
            f"on {parent_count} parent-disjoint but previously consumed evaluation "
            "parents?"
        ),
        "labels_previously_consumed": True,
        "fresh_claim_authorized": False,
        "claim_boundary": (
            f"Architecture-development comparison on {parent_count} parent-disjoint "
            "ARKitScenes videos whose outcomes were consumed by an earlier experiment. "
            "This is not fresh evidence, a new generalization result, objective truth, "
            "cross-dataset evidence, task utility, deployment, product, or safety evidence."
        ),
    }


def execute(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    stage0a_result = args.stage0a_result.resolve()
    label_dir = args.label_dir.resolve()
    label_result = label_dir / "result.json"
    student_checkpoint = args.student_checkpoint.resolve()
    depthart_source = args.depthart_source.resolve()
    depthart_checkpoint = args.depthart_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    for path, description in (
        (stage0a_result, "Stage 0A result"),
        (label_result, "factor-label result"),
        (student_checkpoint, "student checkpoint"),
        (depthart_checkpoint, "DepthART checkpoint"),
    ):
        require(path.is_file(), f"{description} missing: {path}")
    require(depthart_source.is_dir(), "DepthART source missing")
    require(not output_dir.exists(), "evaluation output collision")

    descriptors, stage0a, _ = build_frame_descriptors(stage0a_result, label_dir)
    fresh_parents = {row.parent_id for row in descriptors}
    require(len(fresh_parents) >= 4, "evaluation requires at least four parents")
    evaluation = _evaluation_context(args.evaluation_mode, len(fresh_parents))
    checkpoint = torch.load(student_checkpoint, map_location="cpu", weights_only=False)
    require(
        checkpoint.get("schema")
        == "blindassist_ag_st_masked_factor_student_checkpoint_v1",
        "student checkpoint schema drift",
    )
    trained_parents = _checkpoint_parent_ids(checkpoint)
    overlap = sorted(fresh_parents & trained_parents)
    require(not overlap, f"fresh evaluation parent overlap: {overlap}")
    architecture = checkpoint["architecture"]
    require(
        architecture["frozen_encoder"] == "FROZEN_DEPTHART_S_METRIC_INDOOR",
        "checkpoint encoder drift",
    )
    feature_channels = int(architecture["input_feature_channels"])
    feature_profile = str(architecture.get("feature_profile", "shared"))
    expected_feature_channels = (
        DEPTHART_PYRAMID_CHANNELS
        if feature_profile == "decoder_pyramid"
        else DEPTHART_SHARED_CHANNELS
    )
    require(
        feature_channels == expected_feature_channels,
        "feature channel/profile drift",
    )
    head_hidden_channels = int(architecture.get("head_hidden_channels", 32))
    head_profile = str(architecture.get("head_profile", "basic"))
    use_base_depth_feature = bool(
        architecture.get("use_base_depth_feature", False)
    )
    depth_gate_profile = str(architecture.get("depth_gate_profile", "none"))
    objective_profile = str(
        architecture.get(
            "objective_profile",
            checkpoint.get("objective_profile", "multifactor"),
        )
    )

    parent_shapes: dict[str, tuple[int, int]] = {}
    for descriptor in descriptors:
        previous = parent_shapes.setdefault(descriptor.parent_id, descriptor.output_hw)
        require(previous == descriptor.output_hw, "within-parent output shape drift")
    diagnostic_roles: dict[str, tuple[str, ...]] = {}
    diagnostic_split: dict[str, Any] | None = None
    if len(parent_shapes) == 16:
        fit_parents, selection_parents, canary_parents, diagnostic_split = (
            _diagnostic_parent_split(set(parent_shapes))
        )
        diagnostic_roles = {
            "diagnostic_fit": fit_parents,
            "diagnostic_selection": selection_parents,
            "diagnostic_canary": canary_parents,
        }

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "AG-ST checkpoint evaluation requires CUDA")
    cached, extraction = extract_depthart_features(
        descriptors,
        depthart_source,
        depthart_checkpoint,
        device,
        int(checkpoint["seed"]),
        feature_profile=feature_profile,
    )

    depth_mode = str(architecture["depth_mode"])
    baseline = MaskedFactorStudent(
        channels=feature_channels,
        hidden=head_hidden_channels,
        depth_mode=depth_mode,
        head_profile=head_profile,
        use_base_depth_feature=use_base_depth_feature,
        depth_gate_profile=depth_gate_profile,
    ).to(device)
    baseline.initialize_priors(checkpoint["priors"])
    before_all = evaluate_frames(baseline, cached, device)
    before_diagnostics = {
        role: evaluate_frames(
            baseline,
            [row for row in cached if row.descriptor.parent_id in parents],
            device,
        )
        for role, parents in diagnostic_roles.items()
    }

    model = MaskedFactorStudent(
        channels=feature_channels,
        hidden=head_hidden_channels,
        depth_mode=depth_mode,
        head_profile=head_profile,
        use_base_depth_feature=use_base_depth_feature,
        depth_gate_profile=depth_gate_profile,
    ).to(device)
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    require(
        not incompatible.missing_keys and not incompatible.unexpected_keys,
        "student state-dict drift",
    )
    after_all = evaluate_frames(model, cached, device)
    after_diagnostics = {
        role: evaluate_frames(
            model,
            [row for row in cached if row.descriptor.parent_id in parents],
            device,
        )
        for role, parents in diagnostic_roles.items()
    }

    all_improvements = _improvements(before_all, after_all)
    core_factors = _core_factor_names(objective_profile)
    core_signals = {
        name: all_improvements[name] is not None and all_improvements[name] > 0.0
        for name in core_factors
    }
    supported = sum(core_signals.values())
    status_prefix = evaluation["status_prefix"]
    if objective_profile in {"depth_support", "depth_support_precision"}:
        status = (
            f"{status_prefix}_DEPTH_SUPPORT_SIGNAL_SUPPORTED"
            if supported == len(core_factors)
            else f"{status_prefix}_PARTIAL_DEPTH_SUPPORT_SIGNAL_SUPPORTED"
            if supported > 0
            else f"{status_prefix}_DEPTH_SUPPORT_SIGNAL_NOT_SUPPORTED"
        )
    elif objective_profile == "boundary_only":
        status = (
            f"{status_prefix}_BOUNDARY_SIGNAL_SUPPORTED"
            if supported == len(core_factors)
            else f"{status_prefix}_PARTIAL_BOUNDARY_SIGNAL_SUPPORTED"
            if supported > 0
            else f"{status_prefix}_BOUNDARY_SIGNAL_NOT_SUPPORTED"
        )
    else:
        status = (
            f"{status_prefix}_DEPTH_SUPPORT_OBSTACLE_SIGNAL_SUPPORTED"
            if supported == len(core_factors)
            else f"{status_prefix}_PARTIAL_FACTOR_SIGNAL_SUPPORTED"
            if supported > 0
            else f"{status_prefix}_FACTOR_SIGNAL_NOT_SUPPORTED"
        )
    label_digest = aggregate_label_digest(row.label_path for row in descriptors)
    metric_receipt = {
        "before_all_evaluation": before_all,
        "after_all_evaluation": after_all,
        "all_evaluation_parent_macro_relative_improvement": all_improvements,
        "before_diagnostics": before_diagnostics,
        "after_diagnostics": after_diagnostics,
        "core_signals": core_signals,
    }
    if args.evaluation_mode == "fresh_zero_shot":
        metric_receipt.update(
            {
                "before_all_fresh": before_all,
                "after_all_fresh": after_all,
                "all_fresh_parent_macro_relative_improvement": all_improvements,
            }
        )
    result = {
        "schema": evaluation["schema"],
        "status": status,
        "mode": "WILD_LAB_REVERSIBLE_EXPLORATION",
        "evaluation_mode": args.evaluation_mode,
        "question": evaluation["question"],
        "inputs": {
            "stage0a_result_path": str(stage0a_result),
            "stage0a_result_sha256": sha256_file(stage0a_result),
            "factor_label_result_path": str(label_result),
            "factor_label_result_sha256": sha256_file(label_result),
            "factor_label_payloads": label_digest,
            "source_manifest_sha256": stage0a["source"]["manifest_sha256"],
            "student_checkpoint_path": str(student_checkpoint),
            "student_checkpoint_sha256": sha256_file(student_checkpoint),
            "depthart_checkpoint_sha256": sha256_file(depthart_checkpoint),
            "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "parent_firewall": {
            "checkpoint_consumed_parents": sorted(trained_parents),
            "evaluation_parents": sorted(fresh_parents),
            "overlap": overlap,
            "evaluation_labels_previously_consumed": evaluation[
                "labels_previously_consumed"
            ],
            "labels_used_for_current_checkpoint_fitting_or_threshold_selection": False,
            "fresh_claim_authorized": evaluation["fresh_claim_authorized"],
        },
        "diagnostic_split": diagnostic_split,
        "objective_profile": objective_profile,
        "execution": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "feature_extraction": extraction,
            "total_seconds": time.perf_counter() - started,
        },
        "metrics": metric_receipt,
        "decision": {
            "complete_truth_required": False,
            "fresh_student_training_performed": False,
            "optimizer_constructed": False,
            "threshold_selected": False,
            "supported_core_factor_count": supported,
            "total_core_factor_count": len(core_factors),
            "boundary_is_diagnostic_not_a_rescue_factor": True,
            "obstacle_is_diagnostic_not_a_rescue_factor": (
                objective_profile
                in {"depth_support", "depth_support_precision"}
            ),
            "next_step": (
                "Compare against the frozen prior architecture only; do not promote "
                "this consumed cohort as fresh generalization evidence."
                if args.evaluation_mode == "consumed_development_comparison"
                else
                "Preserve this fresh cohort as consumed and scale only after the depth/support result is summarized."
                if objective_profile
                in {"depth_support", "depth_support_precision"}
                and supported == len(core_factors)
                else "Combine the two disjoint TRAIN-source batches for a larger student fit while reserving a new source for evaluation."
                if supported == len(core_factors)
                else "Do not scale this student architecture until the failed fresh factors are localized."
            ),
        },
        "claim_boundary": evaluation["claim_boundary"],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_exclusive(output_dir / "result.json", result)
    print(
        json.dumps(
            {
                "status": status,
                "result": str((output_dir / "result.json").resolve()),
                "all_evaluation_parent_macro_relative_improvement": all_improvements,
                "before": before_all["parent_macro"],
                "after": after_all["parent_macro"],
                "total_seconds": result["execution"]["total_seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage0a-result", type=Path, default=DEFAULT_STAGE0A_RESULT)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument(
        "--student-checkpoint",
        type=Path,
        default=DEFAULT_STUDENT_CHECKPOINT,
    )
    parser.add_argument("--depthart-source", type=Path, default=DEFAULT_DEPTHART_SOURCE)
    parser.add_argument(
        "--depthart-checkpoint",
        type=Path,
        default=DEFAULT_DEPTHART_CHECKPOINT,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--evaluation-mode",
        choices=EVALUATION_MODES,
        default="fresh_zero_shot",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(execute(parse_args()))
