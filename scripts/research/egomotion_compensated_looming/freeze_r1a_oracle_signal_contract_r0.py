#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build() -> dict[str, Any]:
    return {
        "schema_version": "egomotion_compensated_looming_r1a_oracle_signal_contract_r0",
        "goal_id": "EGOMOTION_COMPENSATED_LOOMING_SIGNAL_R1",
        "stage": "R1A_ORACLE_PHYSICAL_SIGNAL_ONLY",
        "input_pair_contract": {
            "pairing": "CONSECUTIVE_RGB_WITHIN_FROZEN_UNIT",
            "minimum_delta_seconds": 0.020,
            "maximum_delta_seconds": 0.050,
            "rgb_resolution": [640, 480],
            "grayscale_conversion": "OPENCV_BGR2GRAY",
            "lens_distortion_policy": (
                "SOURCE_NATIVE_PIXELS_NO_POST_OUTCOME_UNDISTORTION"
            ),
            "validation_or_holdout_open_allowed": False,
            "old_15_pair_window_reads_allowed": False,
        },
        "flow_producer": {
            "library": "opencv-python-headless",
            "version": "4.13.0.92",
            "algorithm": "calcOpticalFlowFarneback",
            "parameters": {
                "pyr_scale": 0.5,
                "levels": 3,
                "winsize": 15,
                "iterations": 3,
                "poly_n": 5,
                "poly_sigma": 1.2,
                "flags": 0,
            },
            "forward_backward_quality": {
                "backward_flow_same_parameters": True,
                "maximum_roundtrip_error_pixels": 1.5,
                "minimum_common_support_fraction": 0.50,
            },
        },
        "spatial_contract": {
            "border_exclusion_pixels": 16,
            "principal_point_exclusion_radius_pixels": 32,
            "radial_origin": "SOURCE_INTRINSICS_PRINCIPAL_POINT",
            "radial_rate_formula": (
                "dot(flow_pixels, radial_unit) / "
                "(delta_seconds * radius_pixels)"
            ),
            "continuous_summary_quantile": 0.90,
            "positive_clipping_for_primary_looming_summary": True,
            "signed_summary_also_required": True,
            "roi_grid": [8, 6],
            "grid_cells_are_reporting_units_not_selection_units": True,
        },
        "arms": [
            {
                "arm_id": "RAW_FLOW_ENERGY",
                "truth_namespace_allowed": False,
                "output": "Q90_FLOW_MAGNITUDE_PIXELS_PER_SECOND",
            },
            {
                "arm_id": "BBOX_LOG_AREA_GROWTH",
                "truth_namespace_allowed": False,
                "output": "DELTA_LOG_BBOX_AREA_PER_SECOND",
                "missing_detection_policy": "UNIT_ABSTAIN_NOT_ZERO",
                "bonn_static_surface_policy": (
                    "SOURCE_CLAIM_ABSTAIN_NO_FROZEN_TARGET_BBOX"
                ),
            },
            {
                "arm_id": "UNCOMPENSATED_LOCAL_RADIAL_EXPANSION",
                "truth_namespace_allowed": False,
                "output": "Q90_LOCAL_RADIAL_RATE_PER_SECOND",
            },
            {
                "arm_id": "ORACLE_ROTATION_COMPENSATION",
                "truth_namespace_allowed": "ORIENTATION_ONLY",
                "rotation_flow": (
                    "K * R_CURRENT_FROM_PREVIOUS * inverse(K)"
                ),
                "output": (
                    "Q90_LOCAL_RADIAL_RATE_AFTER_ROTATION_FLOW_SUBTRACTION"
                ),
            },
            {
                "arm_id": "FULL_6DOF_RESIDUAL_DIAGNOSTIC",
                "truth_namespace_allowed": (
                    "FULL_POSE_AND_SOURCE_DEPTH_DIAGNOSTIC_ONLY"
                ),
                "output": (
                    "Q90_RESIDUAL_FLOW_AND_RADIAL_RATE_AFTER_RIGID_FLOW_"
                    "SUBTRACTION"
                ),
                "acceptance_authority": False,
            },
        ],
        "namespace_firewall": {
            "base_arms_may_read": ["RGB", "INTRINSICS"],
            "base_arms_may_not_read": [
                "POSE",
                "DEPTH",
                "CELL_LABEL",
                "CLOSING_TRUTH",
                "OUTCOME",
            ],
            "oracle_rotation_may_read": ["ORIENTATION_TRUTH"],
            "full_6dof_diagnostic_may_read": [
                "FULL_POSE_TRUTH",
                "SOURCE_DEPTH",
            ],
            "truth_join_occurs_after_arm_trace_hash_freeze": True,
            "deployable_rotation_estimator_allowed": False,
        },
        "common_support": {
            "compare_only_pairs_evaluated_by_all_claim_applicable_arms": True,
            "source_inapplicable_arm_does_not_zero_other_arms": True,
            "missing_or_low_quality_units_are_abstained": True,
            "source_and_session_weighting": "EQUAL_WEIGHT",
            "no_missing_value_imputation": True,
        },
        "continuous_analysis": {
            "alarm_threshold_selection": False,
            "required_reports": [
                "SIGNED_SPEARMAN_WITH_INDEPENDENT_CONTINUOUS_TRUTH",
                "THEIL_SEN_SLOPE",
                "SESSION_BLOCK_BOOTSTRAP_95_PERCENT_INTERVAL",
                "PURE_ROTATION_LEAKAGE",
                "TRUE_CLOSING_RETENTION",
                "SOURCE_SESSION_CONCORDANCE",
                "WORST_SOURCE",
                "SUPPORT_AND_ABSTENTION",
                "LEAVE_ONE_SOURCE_OR_SESSION_OUT",
            ],
        },
        "claim_gates": {
            "C1_ROTATION_LEAKAGE_SUPPRESSION": {
                "required_families": [
                    "CONTROLLED_RIGID_TARGET_CAPTURE",
                    "BONN_RGBD_DYNAMIC",
                ],
                "clear_gain": (
                    "PER_FAMILY_MEDIAN_ORACLE_LEAKAGE_AT_LEAST_20_PERCENT_"
                    "LOWER_THAN_UNCOMPENSATED"
                ),
                "uncertainty": (
                    "SESSION_BLOCK_BOOTSTRAP_95_PERCENT_LOWER_BOUND_ABOVE_ZERO"
                ),
            },
            "C2_STATIC_SURFACE_CLOSING_RETENTION": {
                "required_families": [
                    "CONTROLLED_RIGID_TARGET_CAPTURE",
                    "BONN_RGBD_DYNAMIC",
                ],
                "oracle_truth_association": (
                    "PER_FAMILY_SIGNED_SPEARMAN_AT_LEAST_0_30_AND_95_PERCENT_"
                    "LOWER_BOUND_ABOVE_ZERO"
                ),
                "retention": (
                    "ORACLE_STANDARDIZED_POSITIVE_SLOPE_AND_NOT_MORE_THAN_"
                    "0_05_SPEARMAN_WORSE_THAN_UNCOMPENSATED"
                ),
            },
            "C2_ACTIVE_TARGET_AND_LATERAL_PASS": {
                "required_families": [
                    "CONTROLLED_RIGID_TARGET_CAPTURE",
                    "REVEL",
                ],
                "status": "INPUT_AUTHORITY_PENDING",
            },
        },
        "stop_rules": [
            "ORACLE_FAILS_ANY_REQUIRED_FAMILY_FOR_A_CORE_CLAIM",
            "WORST_SOURCE_BOOTSTRAP_INTERVAL_CROSSES_NO_EFFECT",
            "ONLY_ONE_FAMILY_SHOWS_THE_REQUIRED_DIRECTION",
            "FLOW_OR_COMPENSATION_QUALITY_SENSITIVITY_REVERSES_THE_SIGN",
            "COMMON_SUPPORT_FRACTION_BELOW_0_50_IN_ANY_REQUIRED_FAMILY",
            "TRUTH_SIGNAL_DERIVATION_CHAIN_OVERLAP_IS_DISCOVERED",
        ],
        "current_source_effect": {
            "Bonn_C1": "ABSTAIN_NO_PURE_ROTATION_DISCOVERY_WINDOW",
            "Bonn_C2": (
                "INPUT_AUTHORITY_AVAILABLE_SIGNAL_NOT_COMPUTED"
            ),
            "controlled": "HOLD_CONTROLLED_CAPTURE_HARDWARE_RECEIPT",
            "REveL_dynamic": (
                "PRIOR_INSPECTED_DISCOVERY_OR_MIGRATION_ONLY"
            ),
        },
        "hard_boundaries": {
            "candidate_signal_computed_at_freeze": False,
            "old_window_selection_tuning_acceptance_reads": 0,
            "alarm_threshold_selected": False,
            "route_or_event_truth_used": False,
            "app_or_lifecycle_connected": False,
        },
        "terminal": "R1A_ORACLE_SIGNAL_CONTRACT_FROZEN_EXECUTION_PENDING",
        "status": "VALID",
    }


def validate(receipt: dict[str, Any]) -> None:
    if receipt["stage"] != "R1A_ORACLE_PHYSICAL_SIGNAL_ONLY":
        raise ValueError("wrong R1-A stage")
    arms = [item["arm_id"] for item in receipt["arms"]]
    expected = [
        "RAW_FLOW_ENERGY",
        "BBOX_LOG_AREA_GROWTH",
        "UNCOMPENSATED_LOCAL_RADIAL_EXPANSION",
        "ORACLE_ROTATION_COMPENSATION",
        "FULL_6DOF_RESIDUAL_DIAGNOSTIC",
    ]
    if arms != expected:
        raise ValueError("R1-A arm contract changed")
    firewall = receipt["hard_boundaries"]
    if firewall["candidate_signal_computed_at_freeze"]:
        raise ValueError("signal ran before contract freeze")
    if firewall["old_window_selection_tuning_acceptance_reads"] != 0:
        raise ValueError("old-window firewall violated")
    if (
        firewall["alarm_threshold_selected"]
        or firewall["route_or_event_truth_used"]
        or firewall["app_or_lifecycle_connected"]
    ):
        raise ValueError("downstream authority opened")
    if receipt["arms"][-1]["acceptance_authority"]:
        raise ValueError("6DoF diagnostic gained acceptance authority")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    receipt = build()
    validate(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "terminal": receipt["terminal"],
                "arm_count": len(receipt["arms"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
