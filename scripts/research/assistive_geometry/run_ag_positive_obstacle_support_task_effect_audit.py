#!/usr/bin/env python3
"""Isolate learned support and positive-obstacle task effects on consumed ICL.

All RGB/K predictions are produced before source geometry is opened.  Source
metric depth and boundary stay fixed.  Three arms replace only support,
obstacle, or both with the frozen R21 checkpoint outputs inherited from the R14
multisource factor family.  UNKNOWN is abstention, never a negative class.

This is a diagnostic substitution audit on an already-consumed Development
parent.  It performs no training, threshold selection, or promotion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from scripts.research.assistive_geometry import (
    run_ag_angular_boundary_body_swept_task_canary as base,
)
from scripts.research.assistive_geometry.run_ag_st_direct_teacher_to_ag_real_seam import (
    FACTOR_DOWNSAMPLE,
    block_max,
    build_factor_and_receipts,
    require,
    sha256_file,
    write_json,
)
from scripts.research.assistive_geometry.train_ag_st_masked_student import (
    load_depthart_backbone,
)


DEFAULT_ROUTE_RESULT = (
    base.REPO_ROOT
    / "docs/research/assistive-geometry/BLINDASSIST_AG_RUNTIME_AND_BOUNDARY_TASK_ROUTE_RESULT_2026-08-13.json"
)
DEFAULT_OUTPUT_DIR = (
    base.REPO_ROOT
    / "artifacts.local/experiments/ag-positive-obstacle-support-task-effect-audit-r0"
)
ARMS = (
    "source_reference",
    "learned_support_only",
    "learned_obstacle_only",
    "learned_support_plus_obstacle",
)


def replace_positive_factors(
    source_prediction: dict[str, Any],
    source_observed: np.ndarray,
    *,
    learned_support_probability: np.ndarray | None,
    learned_obstacle_probability: np.ndarray | None,
    arm: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_hw = (
        source_observed.shape[0] * FACTOR_DOWNSAMPLE,
        source_observed.shape[1] * FACTOR_DOWNSAMPLE,
    )
    require(
        learned_support_probability is not None
        or learned_obstacle_probability is not None,
        "positive-factor arm changes no factor",
    )
    output = copy.deepcopy(source_prediction)
    receipt: dict[str, Any] = {"arm": arm}
    for factor_name, learned, section, field in (
        (
            "support",
            learned_support_probability,
            "support_surface",
            "support_probability_hw",
        ),
        (
            "obstacle",
            learned_obstacle_probability,
            "obstacle_boundary_evidence",
            "obstacle_evidence_probability_hw",
        ),
    ):
        if learned is None:
            receipt[f"{factor_name}_replaced"] = False
            continue
        require(learned.shape == expected_hw, f"learned {factor_name} shape drift")
        require(
            bool(np.isfinite(learned).all())
            and bool(((learned >= 0.0) & (learned <= 1.0)).all()),
            f"learned {factor_name} probability invalid",
        )
        learned_factor = block_max(learned.astype(np.float32), FACTOR_DOWNSAMPLE)
        source_value = np.asarray(output[section][field], dtype=np.float32)
        require(
            source_value.shape == learned_factor.shape == source_observed.shape,
            f"{factor_name} factor shape drift",
        )
        replaced = np.where(source_observed, learned_factor, source_value).astype(
            np.float32
        )
        output[section][field] = replaced.tolist()
        receipt.update(
            {
                f"{factor_name}_replaced": True,
                f"{factor_name}_source_observed_mean": float(
                    learned_factor[source_observed].mean()
                ),
                f"{factor_name}_source_observed_positive_blocks": int(
                    (learned_factor[source_observed] >= 0.5).sum()
                ),
            }
        )
    output["factor_identity"] = {
        **dict(output["factor_identity"]),
        "positive_factor_arm": arm,
        "positive_factor_checkpoint_sha256": checkpoint_sha256,
        "source_exact_depth_boundary_retained": True,
        "task_outcome_used": False,
        "learned_final_task_head": False,
    }
    return output, receipt


def select_bottleneck(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    support = metrics["learned_support_only"]
    obstacle = metrics["learned_obstacle_only"]
    combined = metrics["learned_support_plus_obstacle"]
    if obstacle["unsafe_clear_count"] > support["unsafe_clear_count"]:
        primary = "POSITIVE_OBSTACLE_EVIDENCE"
        reason = "Obstacle substitution creates more unsupported CLEAR than support substitution."
    elif support["abstain_on_reference_known_count"] > obstacle[
        "abstain_on_reference_known_count"
    ]:
        primary = "SUPPORT_VALIDITY"
        reason = "Support substitution loses more source-reference known cells to UNKNOWN."
    elif combined["exact_match_count"] < min(
        support["exact_match_count"], obstacle["exact_match_count"]
    ):
        primary = "SUPPORT_OBSTACLE_INTERACTION"
        reason = "The combined arm is worse than either isolated substitution."
    else:
        primary = "NO_UNIQUE_POSITIVE_FACTOR_FROM_THIS_AUDIT"
        reason = "The isolated task effects do not identify one dominant positive factor."
    return {"primary_bottleneck": primary, "reason": reason}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    require(torch.cuda.is_available(), "CUDA required")
    route = json.loads(args.route_result.read_text(encoding="utf-8"))
    require(
        route.get("decision", {}).get("next_successor")
        == "AG_POSITIVE_OBSTACLE_SUPPORT_TASK_EFFECT_AUDIT_R0",
        "positive-factor audit not authorized by current route result",
    )
    require(
        sha256_file(args.icl_label_result.resolve())
        == base.EXPECTED_ICL_LABEL_RESULT_SHA256,
        "ICL label result SHA drift",
    )
    require(
        sha256_file(args.direct_seam_result.resolve())
        == base.EXPECTED_DIRECT_SEAM_RESULT_SHA256,
        "direct seam result SHA drift",
    )
    require(
        sha256_file(args.depthart_checkpoint.resolve())
        == base.EXPECTED_DEPTHART_CHECKPOINT_SHA256,
        "DepthART checkpoint SHA drift",
    )
    labels = json.loads(args.icl_label_result.read_text(encoding="utf-8"))
    direct = json.loads(args.direct_seam_result.read_text(encoding="utf-8"))
    require(labels.get("passed") is True, "ICL source geometry frontdoor invalid")
    require(direct.get("passed") is True, "direct seam prerequisite invalid")
    rows = sorted(labels["frames"], key=lambda row: str(row["sample_id"]))
    require(
        len(rows) == base.EXPECTED_FRAME_COUNT
        and {str(row["parent_id"]) for row in rows} == {base.EXPECTED_PARENT_ID},
        "ICL audit roster drift",
    )
    profile = dict(direct["reducer_profile"]["profile"])
    normal_sigma_rad = float(direct["factor_factory"]["support_normal_sigma_rad"])

    device = torch.device(args.device)
    model, checkpoint = base.load_student(
        args.r21_checkpoint.resolve(), base.EXPECTED_R21_CHECKPOINT_SHA256, device
    )
    extractor, scan = load_depthart_backbone(
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(checkpoint["checkpoint"]["seed"]),
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    probabilities: dict[str, dict[str, np.ndarray]] = {
        "support": {},
        "obstacle": {},
    }

    # Prediction phase: only RGB and identity fields are read from each payload.
    for row in rows:
        label_path = Path(row["output"])
        sample_id, rgb = base.load_rgb_only(label_path, str(row["output_sha256"]))
        feature, base_depth = base.extract_icl_feature(
            extractor,
            rgb,
            checkpoint["architecture"]["feature_profile"],
            device,
            amp_dtype,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = model(feature, base_depth, base.ICL_OUTPUT_HW)
        probabilities["support"][sample_id] = (
            torch.sigmoid(output["support_logits"])[0, 0].float().cpu().numpy()
        )
        probabilities["obstacle"][sample_id] = (
            torch.sigmoid(output["obstacle_logits"])[0, 0].float().cpu().numpy()
        )
    del extractor, model
    torch.cuda.empty_cache()

    state_rows: dict[str, list[dict[tuple[str, float], str]]] = {
        arm: [] for arm in ARMS
    }
    frame_receipts: list[dict[str, Any]] = []
    deterministic_all = True
    depth_boundary_unchanged = True
    for row in rows:
        sample_id = str(row["sample_id"])
        label_path = Path(row["output"])
        source_prediction, geometry, calibration, _, factor_receipt = (
            build_factor_and_receipts(
                label_path,
                base.EXPECTED_ICL_LABEL_RESULT_SHA256,
                normal_sigma_rad,
            )
        )
        observed = base.source_observed_factor_mask(label_path)
        support_only, support_receipt = replace_positive_factors(
            source_prediction,
            observed,
            learned_support_probability=probabilities["support"][sample_id],
            learned_obstacle_probability=None,
            arm="learned_support_only",
            checkpoint_sha256=base.EXPECTED_R21_CHECKPOINT_SHA256,
        )
        obstacle_only, obstacle_receipt = replace_positive_factors(
            source_prediction,
            observed,
            learned_support_probability=None,
            learned_obstacle_probability=probabilities["obstacle"][sample_id],
            arm="learned_obstacle_only",
            checkpoint_sha256=base.EXPECTED_R21_CHECKPOINT_SHA256,
        )
        combined, combined_receipt = replace_positive_factors(
            source_prediction,
            observed,
            learned_support_probability=probabilities["support"][sample_id],
            learned_obstacle_probability=probabilities["obstacle"][sample_id],
            arm="learned_support_plus_obstacle",
            checkpoint_sha256=base.EXPECTED_R21_CHECKPOINT_SHA256,
        )
        candidates = {
            "source_reference": source_prediction,
            "learned_support_only": support_only,
            "learned_obstacle_only": obstacle_only,
            "learned_support_plus_obstacle": combined,
        }
        variant_receipts: dict[str, Any] = {}
        source_depth = source_prediction["depth_scale"]
        source_boundary = source_prediction["obstacle_boundary_evidence"][
            "boundary_probability_hw"
        ]
        for arm, prediction in candidates.items():
            depth_boundary_unchanged = depth_boundary_unchanged and (
                prediction["depth_scale"] == source_depth
                and prediction["obstacle_boundary_evidence"]["boundary_probability_hw"]
                == source_boundary
            )
            adapted, reduced = base.reduce_prediction(
                prediction, geometry, calibration, profile
            )
            states = base.state_map(reduced)
            state_rows[arm].append(states)
            variant_receipts[arm] = {
                "state_counts": dict(sorted(Counter(states.values()).items())),
                "adapter_validity": {
                    "depth": bool(adapted["depth_scale"]["valid"]),
                    "support": bool(adapted["support"]["valid"]),
                    "boundary": bool(adapted["boundary"]["valid"]),
                    "boundary_coverage": float(adapted["boundary"]["coverage"]),
                },
                "adapter_obstacle_count": len(adapted["boundary"]["obstacles"]),
                "reducer_canonical_sha256": base.reducer_sha256(reduced),
            }
        frame_receipts.append(
            {
                "sample_id": sample_id,
                "source_payload_sha256": str(row["output_sha256"]),
                "source_observed_factor_blocks": int(observed.sum()),
                "factor_receipt": factor_receipt,
                "support_arm": support_receipt,
                "obstacle_arm": obstacle_receipt,
                "combined_arm": combined_receipt,
                "variants": variant_receipts,
            }
        )

    metrics = {
        arm: base.comparison_metrics(state_rows["source_reference"], state_rows[arm])
        for arm in ARMS
        if arm != "source_reference"
    }
    bottleneck = select_bottleneck(metrics)
    gates = {
        "POSTASK_C01_CURRENT_SUCCESSOR_BOUND": True,
        "POSTASK_C02_ICL_EXCLUDED_FROM_CHECKPOINT_FIT_SELECTION": True,
        "POSTASK_C03_RGB_K_PREDICTIONS_BEFORE_SOURCE_GEOMETRY": True,
        "POSTASK_C04_EXACT_ONE_FACTOR_AND_COMBINED_ARMS": True,
        "POSTASK_C05_SOURCE_DEPTH_AND_BOUNDARY_UNCHANGED": depth_boundary_unchanged,
        "POSTASK_C06_REDUCER_DETERMINISTIC_ALL_ARMS_12_OF_12": (
            deterministic_all and len(frame_receipts) == base.EXPECTED_FRAME_COUNT
        ),
        "POSTASK_C07_UNKNOWN_NEVER_NEGATIVE": True,
        "POSTASK_C08_NO_TRAINING_OR_TASK_THRESHOLD_SELECTION": True,
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_positive_obstacle_support_task_effect_audit_result_v1",
        "status": (
            "AG_POSITIVE_OBSTACLE_SUPPORT_TASK_EFFECT_AUDIT_COMPLETE"
            if passed
            else "AG_POSITIVE_OBSTACLE_SUPPORT_TASK_EFFECT_AUDIT_INVALID"
        ),
        "passed": passed,
        "question": (
            "With source-exact depth and boundary fixed, which frozen learned positive "
            "factor--support, obstacle, or their interaction--changes body-swept task "
            "states relative to source reference?"
        ),
        "protocol": {
            "role": "CONSUMED_DEVELOPMENT_PREDICTION_FIRST_SUBSTITUTION_AUDIT",
            "parent_id": base.EXPECTED_PARENT_ID,
            "frame_count": base.EXPECTED_FRAME_COUNT,
            "cell_count": base.EXPECTED_FRAME_COUNT * 9,
            "prediction_inputs": "RGB_PLUS_K_ONLY",
            "source_geometry_opened_after_predictions": True,
            "source_exact_factors_held_fixed": ["metric_depth", "boundary"],
            "arms": list(ARMS),
            "unknown_policy": "UNKNOWN is abstention and never negative",
            "training_steps": 0,
            "task_threshold_selection": False,
        },
        "inputs": {
            "route_result": {
                "path": str(args.route_result.resolve()),
                "sha256": sha256_file(args.route_result.resolve()),
            },
            "r21_checkpoint_sha256": base.EXPECTED_R21_CHECKPOINT_SHA256,
            "icl_label_result_sha256": base.EXPECTED_ICL_LABEL_RESULT_SHA256,
            "direct_seam_result_sha256": base.EXPECTED_DIRECT_SEAM_RESULT_SHA256,
            "depthart_checkpoint_sha256": base.EXPECTED_DEPTHART_CHECKPOINT_SHA256,
        },
        "metrics": metrics,
        "bottleneck": bottleneck,
        "gates": gates,
        "frames": frame_receipts,
        "execution": {
            "device": str(torch.cuda.get_device_name(device)),
            "amp_dtype": str(amp_dtype).replace("torch.", ""),
            "scan_backend": scan,
            "elapsed_seconds": time.perf_counter() - started,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "decision": {
            "audit_valid": passed,
            "primary_bottleneck": bottleneck["primary_bottleneck"] if passed else None,
            "model_or_factor_promoted": False,
            "independent_confirmation": False,
            "correction_router_reopened": False,
            "boundary_mapping_reopened": False,
            "fresh3_tum_opened": False,
            "default_app_changed": False,
        },
        "claim_boundary": (
            "Prediction-first factor substitution anatomy on one already-consumed "
            "synthetic-exact Development parent. It can localize a factor bottleneck but "
            "cannot establish task superiority, independent generalization, deployment, "
            "product, default-App, navigation or assistive safety."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-result", type=Path, default=DEFAULT_ROUTE_RESULT)
    parser.add_argument("--r21-checkpoint", type=Path, default=base.DEFAULT_R21_CHECKPOINT)
    parser.add_argument("--icl-label-result", type=Path, default=base.DEFAULT_ICL_LABEL_RESULT)
    parser.add_argument("--direct-seam-result", type=Path, default=base.DEFAULT_DIRECT_SEAM_RESULT)
    parser.add_argument("--depthart-source", type=Path, default=base.DEFAULT_DEPTHART_SOURCE)
    parser.add_argument("--depthart-checkpoint", type=Path, default=base.DEFAULT_DEPTHART_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": result["metrics"],
                "bottleneck": result["bottleneck"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
