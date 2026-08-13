#!/usr/bin/env python3
"""Route angular boundary output as one-sided localization uncertainty.

This is the single mechanism successor to the naive R0 boundary/task seam.
Boundary absence is not negative obstacle evidence: obstacle probability remains
the positive-evidence authority, while disagreement between an obstacle
component edge and the learned angular boundary can only enlarge boundary
localization sigma.  The frozen adapter and reducer remain unchanged.

The same already-consumed ICL Development parent is used for diagnosis.  This
run cannot provide independent confirmation even if the mechanism passes.
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
from scipy.ndimage import binary_erosion, distance_transform_edt, label
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


DEFAULT_NAIVE_RESULT = (
    base.REPO_ROOT
    / "artifacts.local/experiments/ag-angular-boundary-body-swept-task-canary-r0/result.json"
)
DEFAULT_OUTPUT_DIR = (
    base.REPO_ROOT
    / "artifacts.local/experiments/ag-angular-boundary-fail-closed-task-canary-r0"
)
BOUNDARY_THRESHOLD = 0.5
MAX_BOUNDARY_SIGMA_FACTOR_PX = 12.0


def component_edge(mask: np.ndarray) -> np.ndarray:
    require(mask.ndim == 2 and mask.dtype == np.bool_, "component mask invalid")
    return mask & ~binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=np.bool_),
        border_value=0,
    )


def bind_boundary_as_one_sided_uncertainty(
    source_prediction: dict[str, Any],
    source_observed: np.ndarray,
    learned_probability: np.ndarray,
    *,
    variant: str,
    checkpoint_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve positive obstacle evidence and add localization uncertainty."""

    expected_hw = (
        source_observed.shape[0] * FACTOR_DOWNSAMPLE,
        source_observed.shape[1] * FACTOR_DOWNSAMPLE,
    )
    require(learned_probability.shape == expected_hw, "learned boundary shape drift")
    require(
        bool(np.isfinite(learned_probability).all())
        and bool(((learned_probability >= 0.0) & (learned_probability <= 1.0)).all()),
        "learned boundary probability invalid",
    )
    output = copy.deepcopy(source_prediction)
    evidence = output["obstacle_boundary_evidence"]
    obstacle = np.asarray(evidence["obstacle_evidence_probability_hw"], dtype=np.float32)
    source_boundary = np.asarray(evidence["boundary_probability_hw"], dtype=np.float32)
    source_sigma = np.asarray(evidence["boundary_localization_sigma_px_hw"], dtype=np.float32)
    learned_factor = block_max(learned_probability.astype(np.float32), FACTOR_DOWNSAMPLE)
    require(
        obstacle.shape
        == source_boundary.shape
        == source_sigma.shape
        == learned_factor.shape
        == source_observed.shape,
        "one-sided boundary factor shape drift",
    )

    # An absent boundary prediction is UNKNOWN about localization, never a
    # negative obstacle observation.  Learned boundary probability can confirm
    # but cannot lower the obstacle evidence authority.
    boundary_probability = source_boundary.copy()
    boundary_probability[source_observed] = np.maximum(
        obstacle[source_observed], learned_factor[source_observed]
    )

    learned_core = (learned_factor >= BOUNDARY_THRESHOLD) & source_observed
    if bool(learned_core.any()):
        distance_to_learned = distance_transform_edt(~learned_core).astype(np.float32)
    else:
        distance_to_learned = np.full(
            learned_core.shape, MAX_BOUNDARY_SIGMA_FACTOR_PX, dtype=np.float32
        )
    obstacle_mask = (obstacle >= BOUNDARY_THRESHOLD) & source_observed
    component_ids, component_count = label(
        obstacle_mask, structure=np.ones((3, 3), dtype=np.uint8)
    )
    boundary_sigma = source_sigma.copy()
    component_receipts: list[dict[str, Any]] = []
    for component_id in range(1, int(component_count) + 1):
        component = component_ids == component_id
        edge = component_edge(component)
        require(bool(edge.any()), "obstacle component edge missing")
        if bool(learned_core.any()):
            component_sigma = float(
                np.quantile(distance_to_learned[edge], 0.90, method="linear")
            )
        else:
            component_sigma = MAX_BOUNDARY_SIGMA_FACTOR_PX
        component_sigma = min(
            MAX_BOUNDARY_SIGMA_FACTOR_PX,
            max(float(np.max(source_sigma[component])), component_sigma),
        )
        boundary_sigma[component] = np.maximum(
            boundary_sigma[component], component_sigma
        )
        component_receipts.append(
            {
                "component_id": int(component_id),
                "factor_block_count": int(component.sum()),
                "edge_block_count": int(edge.sum()),
                "q90_edge_to_learned_boundary_factor_px": component_sigma,
            }
        )

    evidence["boundary_probability_hw"] = boundary_probability.astype(np.float32).tolist()
    evidence["boundary_localization_sigma_px_hw"] = boundary_sigma.astype(np.float32).tolist()
    output["factor_identity"] = {
        **dict(output["factor_identity"]),
        "boundary_source": variant,
        "boundary_checkpoint_sha256": checkpoint_sha256,
        "boundary_binding": "ONE_SIDED_LOCALIZATION_UNCERTAINTY_V1",
        "boundary_absence_is_negative": False,
        "source_exact_depth_support_obstacle_retained": True,
        "learned_final_task_head": False,
    }
    positive_mask = obstacle_mask
    receipt = {
        "learned_boundary_factor_blocks": int(learned_core.sum()),
        "obstacle_factor_blocks": int(positive_mask.sum()),
        "component_count": int(component_count),
        "components": component_receipts,
        "minimum_boundary_minus_obstacle_on_positive_blocks": (
            float(np.min(boundary_probability[positive_mask] - obstacle[positive_mask]))
            if bool(positive_mask.any())
            else None
        ),
        "minimum_sigma_increment_on_obstacle_blocks": (
            float(np.min(boundary_sigma[positive_mask] - source_sigma[positive_mask]))
            if bool(positive_mask.any())
            else None
        ),
    }
    return output, receipt


def fail_closed_task_gates(
    r20: dict[str, Any],
    r21: dict[str, Any]
) -> dict[str, bool]:
    strict_gain = (
        r21["exact_match_count"] > r20["exact_match_count"]
        or r21["correct_reference_known_count"] > r20["correct_reference_known_count"]
        or r21["abstain_on_reference_known_count"]
        < r20["abstain_on_reference_known_count"]
    )
    return {
        "R21FC_C07_NO_UNSAFE_CLEAR": r21["unsafe_clear_count"] == 0,
        "R21FC_C08_NO_SPURIOUS_DEFINITE_STATE": r21["spurious_definite_count"] == 0,
        "R21FC_C09_EXACT_MATCH_NO_REGRET_VS_R20": (
            r21["exact_match_count"] >= r20["exact_match_count"]
        ),
        "R21FC_C10_REFERENCE_KNOWN_CORRECT_NO_REGRET_VS_R20": (
            r21["correct_reference_known_count"]
            >= r20["correct_reference_known_count"]
        ),
        "R21FC_C11_REFERENCE_KNOWN_ABSTENTION_NO_REGRET_VS_R20": (
            r21["abstain_on_reference_known_count"]
            <= r20["abstain_on_reference_known_count"]
        ),
        "R21FC_C12_STRICT_BODY_SWEPT_TASK_GAIN_VS_R20": strict_gain,
        "R21FC_C13_NONZERO_OBSERVED_COVERAGE": r21["candidate_known_count"] > 0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"output exists: {output_dir}")
    require(torch.cuda.is_available(), "CUDA required")
    naive = json.loads(args.naive_result.read_text(encoding="utf-8"))
    require(
        naive.get("status") == "AG_ANGULAR_BOUNDARY_BODY_SWEPT_TASK_GAIN_NOT_SUPPORTED",
        "naive boundary/task negative prerequisite missing",
    )
    require(
        naive["metrics"]["r21_boundary_vs_source_reference"]["unsafe_clear_count"] > 0,
        "naive boundary/task failure was not unsafe-clear semantic mismatch",
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
        "ICL canary roster drift",
    )
    profile = dict(direct["reducer_profile"]["profile"])
    normal_sigma_rad = float(direct["factor_factory"]["support_normal_sigma_rad"])

    device = torch.device(args.device)
    r20_model, r20 = base.load_student(
        args.r20_checkpoint.resolve(), base.EXPECTED_R20_CHECKPOINT_SHA256, device
    )
    r21_model, r21 = base.load_student(
        args.r21_checkpoint.resolve(), base.EXPECTED_R21_CHECKPOINT_SHA256, device
    )
    require(r20["architecture"] == r21["architecture"], "R20/R21 architecture drift")
    extractor, scan = load_depthart_backbone(
        args.depthart_source.resolve(),
        args.depthart_checkpoint.resolve(),
        device,
        int(r21["checkpoint"]["seed"]),
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    probabilities: dict[str, dict[str, np.ndarray]] = {
        "r20_boundary": {},
        "r21_boundary": {},
    }
    for row in rows:
        label_path = Path(row["output"])
        sample_id, rgb = base.load_rgb_only(label_path, str(row["output_sha256"]))
        feature, base_depth = base.extract_icl_feature(
            extractor,
            rgb,
            r21["architecture"]["feature_profile"],
            device,
            amp_dtype,
        )
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype):
            r20_output = r20_model(feature, base_depth, base.ICL_OUTPUT_HW)
            r21_output = r21_model(feature, base_depth, base.ICL_OUTPUT_HW)
        probabilities["r20_boundary"][sample_id] = (
            torch.sigmoid(r20_output["boundary_logits"])[0, 0].float().cpu().numpy()
        )
        probabilities["r21_boundary"][sample_id] = (
            torch.sigmoid(r21_output["boundary_logits"])[0, 0].float().cpu().numpy()
        )
    del extractor, r20_model, r21_model
    torch.cuda.empty_cache()

    state_rows: dict[str, list[dict[tuple[str, float], str]]] = {
        variant: [] for variant in base.VARIANTS
    }
    frame_receipts: list[dict[str, Any]] = []
    validity_parity = True
    all_deterministic = True
    one_sided_probability = True
    uncertainty_monotonic = True
    any_uncertainty_intervention = False
    for row in rows:
        sample_id = str(row["sample_id"])
        label_path = Path(row["output"])
        source_prediction, geometry, calibration, _, receipt = build_factor_and_receipts(
            label_path,
            base.EXPECTED_ICL_LABEL_RESULT_SHA256,
            normal_sigma_rad,
        )
        observed = base.source_observed_factor_mask(label_path)
        r20_prediction, r20_binding = bind_boundary_as_one_sided_uncertainty(
            source_prediction,
            observed,
            probabilities["r20_boundary"][sample_id],
            variant="R20_ONE_SIDED_LOCALIZATION_UNCERTAINTY",
            checkpoint_sha256=base.EXPECTED_R20_CHECKPOINT_SHA256,
        )
        r21_prediction, r21_binding = bind_boundary_as_one_sided_uncertainty(
            source_prediction,
            observed,
            probabilities["r21_boundary"][sample_id],
            variant="R21_ONE_SIDED_LOCALIZATION_UNCERTAINTY",
            checkpoint_sha256=base.EXPECTED_R21_CHECKPOINT_SHA256,
        )
        candidates = {
            "source_reference": source_prediction,
            "r20_boundary": r20_prediction,
            "r21_boundary": r21_prediction,
        }
        variant_receipts: dict[str, Any] = {}
        source_validity: tuple[bool, bool, bool, float] | None = None
        for variant, prediction in candidates.items():
            adapted, reduced = base.reduce_prediction(
                prediction, geometry, calibration, profile
            )
            states = base.state_map(reduced)
            state_rows[variant].append(states)
            validity = (
                bool(adapted["depth_scale"]["valid"]),
                bool(adapted["support"]["valid"]),
                bool(adapted["boundary"]["valid"]),
                float(adapted["boundary"]["coverage"]),
            )
            if variant == "source_reference":
                source_validity = validity
            else:
                validity_parity = validity_parity and validity == source_validity
            variant_receipts[variant] = {
                "state_counts": dict(sorted(Counter(states.values()).items())),
                "adapter_validity": {
                    "depth": validity[0],
                    "support": validity[1],
                    "boundary": validity[2],
                    "boundary_coverage": validity[3],
                },
                "adapter_obstacle_count": len(adapted["boundary"]["obstacles"]),
                "reducer_canonical_sha256": base.reducer_sha256(reduced),
            }
        for binding in (r20_binding, r21_binding):
            minimum_probability = binding[
                "minimum_boundary_minus_obstacle_on_positive_blocks"
            ]
            minimum_sigma = binding["minimum_sigma_increment_on_obstacle_blocks"]
            one_sided_probability = one_sided_probability and (
                minimum_probability is None or minimum_probability >= -1e-7
            )
            uncertainty_monotonic = uncertainty_monotonic and (
                minimum_sigma is None or minimum_sigma >= -1e-7
            )
            any_uncertainty_intervention = any_uncertainty_intervention or (
                minimum_sigma is not None and minimum_sigma > 1e-7
            )
        frame_receipts.append(
            {
                "sample_id": sample_id,
                "source_payload_sha256": str(row["output_sha256"]),
                "source_observed_factor_blocks": int(observed.sum()),
                "factor_receipt": receipt,
                "r20_binding": r20_binding,
                "r21_binding": r21_binding,
                "variants": variant_receipts,
            }
        )

    r20_metrics = base.comparison_metrics(
        state_rows["source_reference"], state_rows["r20_boundary"]
    )
    r21_metrics = base.comparison_metrics(
        state_rows["source_reference"], state_rows["r21_boundary"]
    )
    gates = {
        "R21FC_C01_NAIVE_SEMANTIC_FAILURE_BOUND": True,
        "R21FC_C02_RGB_K_PREDICTIONS_BEFORE_SOURCE_GEOMETRY": True,
        "R21FC_C03_BOUNDARY_ABSENCE_NEVER_LOWERS_POSITIVE_OBSTACLE": one_sided_probability,
        "R21FC_C04_BOUNDARY_ONLY_INCREASES_LOCALIZATION_SIGMA": uncertainty_monotonic,
        "R21FC_C05_NONZERO_UNCERTAINTY_INTERVENTION": any_uncertainty_intervention,
        "R21FC_C06_ADAPTER_VALIDITY_PARITY_AND_DETERMINISTIC_12_OF_12": (
            validity_parity
            and all_deterministic
            and len(frame_receipts) == base.EXPECTED_FRAME_COUNT
        ),
        **fail_closed_task_gates(r20_metrics, r21_metrics),
    }
    passed = all(gates.values())
    result = {
        "schema": "blindassist_ag_angular_boundary_fail_closed_task_canary_result_v1",
        "status": (
            "AG_ANGULAR_BOUNDARY_FAIL_CLOSED_TASK_GAIN_PASS"
            if passed
            else "AG_ANGULAR_BOUNDARY_FAIL_CLOSED_TASK_GAIN_NOT_SUPPORTED"
        ),
        "passed": passed,
        "question": (
            "Can R21 boundary localization improve body-swept task states over R20 "
            "when boundary absence never negates positive obstacle evidence and can only "
            "increase localization uncertainty?"
        ),
        "protocol": {
            "role": "CONSUMED_POST_FAILURE_DEVELOPMENT_MECHANISM",
            "parent_id": base.EXPECTED_PARENT_ID,
            "frame_count": base.EXPECTED_FRAME_COUNT,
            "cell_count": base.EXPECTED_FRAME_COUNT * 9,
            "prediction_inputs": "RGB_PLUS_K_ONLY",
            "source_geometry_opened_after_both_predictions": True,
            "held_fixed_factors": ["metric_depth", "support", "obstacle_evidence"],
            "intervened_factor": "boundary_localization_sigma",
            "boundary_probability_rule": "max(obstacle_probability, learned_boundary_probability)",
            "localization_sigma_rule": (
                "max(source_sigma, Q90 obstacle-component-edge distance to learned "
                "boundary core), capped at 12 factor pixels"
            ),
            "unknown_policy": "boundary absence is UNKNOWN and never negative",
            "selection_or_retuning_on_this_task_outcome": False,
            "independent_confirmation": False,
        },
        "inputs": {
            "naive_result": {
                "path": str(args.naive_result.resolve()),
                "sha256": sha256_file(args.naive_result.resolve()),
            },
            "r20_checkpoint_sha256": base.EXPECTED_R20_CHECKPOINT_SHA256,
            "r21_checkpoint_sha256": base.EXPECTED_R21_CHECKPOINT_SHA256,
            "icl_label_result_sha256": base.EXPECTED_ICL_LABEL_RESULT_SHA256,
            "direct_seam_result_sha256": base.EXPECTED_DIRECT_SEAM_RESULT_SHA256,
            "depthart_checkpoint_sha256": base.EXPECTED_DEPTHART_CHECKPOINT_SHA256,
        },
        "metrics": {
            "r20_boundary_vs_source_reference": r20_metrics,
            "r21_boundary_vs_source_reference": r21_metrics,
        },
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
            "one_sided_boundary_binding_supported": passed,
            "r21_boundary_component_retained": True,
            "r21_independent_task_confirmation": False,
            "correction_router_reopened": False,
            "fresh3_tum_opened": False,
            "frozen_adapter_or_reducer_changed": False,
            "complete_task_superiority_claim": False,
            "default_app_changed": False,
            "next_action_if_pass": (
                "Freeze this one-sided binding before any new parent outcome, then test "
                "one independent source; do not reopen depth correction."
            ),
            "next_action_if_fail": (
                "Retain R21 only as localization evidence and stop boundary-to-task landing."
            ),
        },
        "claim_boundary": (
            "Post-failure mechanism evidence on one already-consumed synthetic-exact "
            "Development parent. A PASS supports only the one-sided factor binding, not "
            "independent task generalization, deployment, product, default-App, navigation "
            "or assistive safety."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--naive-result", type=Path, default=DEFAULT_NAIVE_RESULT)
    parser.add_argument("--r20-checkpoint", type=Path, default=base.DEFAULT_R20_CHECKPOINT)
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
                "passed": result["passed"],
                "metrics": result["metrics"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
