#!/usr/bin/env python3
"""Validate the design-only RCLE R2 counterfactual freeze.

This module deliberately contains no generator or RCLE execution path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
DOC_ROOT = REPO_ROOT / "docs" / "research" / "rcle"
DEFAULT_CONTRACT = (
    DOC_ROOT
    / "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_CONTRACT_2026-07-28.json"
)
DEFAULT_GEOMETRY = (
    DOC_ROOT
    / (
        "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_"
        "GEOMETRY_VALIDATION_R0_2026-07-28.json"
    )
)
DEFAULT_BUDGET = (
    DOC_ROOT
    / "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_RUN_BUDGET_R0_2026-07-28.json"
)
PROTOCOL_ID = "RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2"
ARMS = [
    "STATIC_CAMERA__CLEAN",
    "STATIC_CAMERA__BLUR",
    "STATIC_CAMERA__LOW_TEXTURE",
    "PERIODIC_6DOF_SELF_MOTION__CLEAN",
    "PERIODIC_6DOF_SELF_MOTION__BLUR",
    "PERIODIC_6DOF_SELF_MOTION__LOW_TEXTURE",
]
BLOCKS = ["ADVIO_13", "ADVIO_14", "ADVIO_15", "ADVIO_17"]
TERMINALS = [
    "INTERVENTION_NOT_EVALUABLE",
    "MIXED",
    "MOTION_SUPPORTED",
    "QUALITY_SUPPORTED",
    "NO_SEPARATION_HOLD",
]
TERMINAL_DRIVING_FAMILY = [
    "MOTION_CLEAN",
    "BLUR_STATIC",
    "LOW_TEXTURE_STATIC",
    "MOTION_X_BLUR",
    "MOTION_X_LOW_TEXTURE",
    "MOTION_BLUR_VS_STATIC_CLEAN",
    "MOTION_LOW_TEXTURE_VS_STATIC_CLEAN",
    "BLUR_FAILURE_UNION_STATIC",
    "LOW_TEXTURE_FAILURE_UNION_STATIC",
]
TERMINAL_RULES = {
    "INTERVENTION_NOT_EVALUABLE": (
        "Any required 3D geometry gate fails; no global response-blind blur or "
        "low-texture strength satisfies its frozen calibration grid; any "
        "formal quality manipulation check fails; either clean arm fails the "
        "clean-arm block gate; or the source-known positive guardrail fails."
    ),
    "MIXED": (
        "MOTION_CLEAN and at least one of BLUR_STATIC or LOW_TEXTURE_STATIC are "
        "SUPPORTED, or a positive interaction is SUPPORTED and its "
        "corresponding motion-plus-degradation versus static-clean combination "
        "check is also SUPPORTED."
    ),
    "MOTION_SUPPORTED": (
        "MOTION_CLEAN is SUPPORTED; both quality simple effects and both "
        "positive interactions are RULED_OUT_AS_MATERIAL; the clean-motion "
        "tracking gate passes. At least 3 of 4 motion-block point effects must "
        "meet the 0.10 threshold by the common SUPPORTED rule."
    ),
    "QUALITY_SUPPORTED": (
        "At least one quality simple effect is SUPPORTED with its corresponding "
        "quality-accompaniment failure metric supported; MOTION_CLEAN and both "
        "positive interactions are RULED_OUT_AS_MATERIAL. Report subtype BLUR, "
        "LOW_TEXTURE or BOTH."
    ),
    "NO_SEPARATION_HOLD": (
        "Every other complete and valid pattern, including an effect that is "
        "supported while a competing mechanism remains INCONCLUSIVE, or no "
        "effect that reaches the frozen support rule."
    ),
}
SIMULTANEOUS_INTERVAL = (
    "For each of nine terminal-driving contrasts j, let theta_j be the "
    "observed equal-block estimate and s_j the sample SD of its 20000 "
    "bootstrap estimates. For replicate r compute "
    "z_rj=(theta_star_rj-theta_j)/s_j for every s_j>0 and "
    "T_r=max_j(abs(z_rj)). Let c be the type-7 linearly interpolated 95th "
    "percentile of the 20000 T_r values. The familywise interval is "
    "[theta_j-c*s_j, theta_j+c*s_j]."
)
GEOMETRY_GATES = {f"G{index:02d}_" for index in range(1, 15)}
TRAJECTORY_HASHES = {
    "ADVIO_13": (
        "566836258c6411f024f25b663b5713f57eca0c98f6828bce03699e6ba4d9cd77",
        "b632a2648d963e27dc2e4ab142ba451a2f8b8fa08b095722b931d182f3d6a1f8",
    ),
    "ADVIO_14": (
        "be90300cc697470a61d9bdba41be20ade48205c51726fcf7ae99d6352ec80f0e",
        "3c3087ee3905a212dce4c71e67e2bb355115ac67648956cd8b9b5fba31b21e8d",
    ),
    "ADVIO_15": (
        "cffa6c6d9e453b0bfd4ed7bca33bf54952840edc1c1039eec3b7bb2911a59ce9",
        "c410785639e813c35d2c14ad62a7864db1eb19a39ceb7da513fa2a5e852ad643",
    ),
    "ADVIO_17": (
        "225cc502937c121ac95d56b5671a33e5250f2a764e6b61bc9008cf0a6a1d8f3d",
        "0a8a2d1dabf1100070146c946bfca2f099974b4decace5de1dce1e2e23e66b16",
    ),
}


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON_OBJECT_REQUIRED:{path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _error(errors: list[str], code: str) -> None:
    if code not in errors:
        errors.append(code)


def _check_authorization_false(
    value: Any, errors: list[str], path: str = "root"
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key == "formal_execution_authorized" and item is not False:
                _error(errors, f"FORMAL_AUTHORIZATION_NOT_FALSE:{child}")
            _check_authorization_false(item, errors, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _check_authorization_false(item, errors, f"{path}[{index}]")


def validate_bundle(
    contract: dict[str, Any],
    geometry: dict[str, Any],
    budget: dict[str, Any],
    *,
    verify_dependencies: bool = True,
) -> list[str]:
    errors: list[str] = []
    for name, document in (
        ("contract", contract),
        ("geometry", geometry),
        ("budget", budget),
    ):
        if document.get("protocol_id") != PROTOCOL_ID:
            _error(errors, f"PROTOCOL_ID:{name}")
        _check_authorization_false(document, errors, name)

    if contract.get("result_model", {}).get("execution_validity") != "NOT_RUN":
        _error(errors, "CONTRACT_EXECUTION_NOT_NOT_RUN")
    if contract.get("result_model", {}).get("scientific_outcome") != "NOT_RUN":
        _error(errors, "CONTRACT_OUTCOME_NOT_NOT_RUN")
    if contract.get("result_model", {}).get("mechanism_terminal") != "NOT_RUN":
        _error(errors, "CONTRACT_MECHANISM_TERMINAL_NOT_NOT_RUN")

    design = contract.get("factorial_design", {})
    if design.get("arms_in_fixed_order") != ARMS:
        _error(errors, "ARMS")
    if design.get("motion_blocks") != BLOCKS:
        _error(errors, "BLOCKS")
    ordinals = design.get("scene_seed_ordinals_per_block")
    if ordinals != list(range(20)):
        _error(errors, "MAIN_SEED_ORDINALS")
    if design.get("analysis_cluster_count") != 80:
        _error(errors, "ANALYSIS_CLUSTER_COUNT")
    if design.get("main_sequence_count") != 480:
        _error(errors, "MAIN_SEQUENCE_COUNT")

    trajectory = contract.get("trajectory_blocks", {})
    if sorted(key for key in trajectory if key != "common") != sorted(BLOCKS):
        _error(errors, "TRAJECTORY_BLOCKS")
    common = trajectory.get("common", {})
    if common.get("frame_count") != 602 or common.get("pair_count") != 601:
        _error(errors, "TRAJECTORY_COUNTS")
    if common.get("time_scale") != 1.0:
        _error(errors, "TIME_SCALE")
    if common.get("translation_scale") != 1.0:
        _error(errors, "TRANSLATION_SCALE")
    if common.get("rotation_scale") != 1.0:
        _error(errors, "ROTATION_SCALE")
    if common.get("clipping") != "FORBIDDEN":
        _error(errors, "TRAJECTORY_CLIPPING")
    for block, (frames_sha, pose_sha) in TRAJECTORY_HASHES.items():
        fields = trajectory.get(block, {})
        if fields.get("frames_csv_sha256") != frames_sha:
            _error(errors, f"TRAJECTORY_FRAMES_HASH:{block}")
        if fields.get("pose_csv_sha256") != pose_sha:
            _error(errors, f"TRAJECTORY_POSE_HASH:{block}")

    algorithm = contract.get("unchanged_algorithm_lock", {})
    if algorithm.get("implementation") != (
        "ADVIO_WXYZ_TCAMIMU_VALIDMASK_CONTINUOUS_R3"
    ):
        _error(errors, "R3_IMPLEMENTATION")
    if algorithm.get("response_field") != (
        "compensated_expansion_median_per_s"
    ):
        _error(errors, "R3_RESPONSE_FIELD")
    if algorithm.get("threshold_per_s") != 0.01:
        _error(errors, "R3_THRESHOLD")
    if algorithm.get("threshold_operator") != "strict_greater_than":
        _error(errors, "R3_THRESHOLD_OPERATOR")
    if algorithm.get("required_consecutive_evaluable_pairs") != 3:
        _error(errors, "R3_CONSECUTIVE_PAIRS")
    if algorithm.get("lookahead_pairs") != 0:
        _error(errors, "R3_LOOKAHEAD")
    if algorithm.get("reset_rule") != (
        "Any abstention, arm or sequence boundary, or pair at or below "
        "0.01/s resets the streak to zero."
    ):
        _error(errors, "R3_RESET_RULE")
    if algorithm.get("pair_state") != (
        "one fresh continuous PairState per sequence arm; no reset inside an arm"
    ):
        _error(errors, "R3_PAIR_STATE")
    if algorithm.get("implementation_lock_status") != "NOT_CREATED":
        _error(errors, "IMPLEMENTATION_LOCK_PREMATURE")

    calibration = contract.get("quality_interventions", {})
    if calibration.get("calibration_lock_status") != "NOT_CREATED":
        _error(errors, "CALIBRATION_LOCK_PREMATURE")
    if calibration.get("calibration_seed_ordinals_per_block") != list(range(4)):
        _error(errors, "CALIBRATION_SEEDS")
    metric_definitions = calibration.get(
        "response_blind_metric_definitions", {}
    )
    edge_spread = metric_definitions.get("source_known_edge_spread", "")
    if not all(
        token in edge_spread
        for token in (
            "CAL scenes and the analytic calibration fixture",
            "CAL/fixture-only",
            "not required in main scenes",
        )
    ):
        _error(errors, "EDGE_SPREAD_SCOPE")
    invalid_rule = metric_definitions.get("invalid_metric_rule", "")
    if not all(
        token in invalid_rule
        for token in (
            "zero clean denominator",
            "no valid tile or edge",
            "non-finite",
            "never imputed, clipped or omitted",
        )
    ):
        _error(errors, "INVALID_METRIC_FAIL_CLOSED")
    main_check = calibration.get("formal_main_manipulation_check", {})
    if not isinstance(main_check, dict):
        _error(errors, "FORMAL_MANIPULATION_CHECK")
        main_check = {}
    low_texture_pass = main_check.get("low_texture_sequence_pass", "")
    if not all(
        token in low_texture_pass
        for token in (
            "multiscale-gradient-density target ratio",
            "main scenes do not contain CAL plates",
            "edge-spread is not recomputed",
        )
    ):
        _error(errors, "MAIN_LOW_TEXTURE_METRIC")
    low_texture_identity = main_check.get(
        "low_texture_no_blur_identity", ""
    )
    if not all(
        token in low_texture_identity
        for token in (
            "implementation hash",
            "selected alpha",
            "no PSF operator",
            "quality-geometry identity gate",
            "INVALID",
        )
    ):
        _error(errors, "MAIN_LOW_TEXTURE_IDENTITY")

    non_goals = " ".join(contract.get("non_goals", [])).lower()
    for token, code in (
        ("sequence16", "SEQUENCE16_NOT_CLOSED"),
        ("cotracker", "COTRACKER_NOT_CLOSED"),
        ("android", "ANDROID_NOT_CLOSED"),
    ):
        if token not in non_goals:
            _error(errors, code)

    statistics = contract.get("statistical_plan", {})
    bootstrap = statistics.get("bootstrap", {})
    if bootstrap.get("replicates") != 20000:
        _error(errors, "BOOTSTRAP_REPLICATES")
    if bootstrap.get("seed") != 20260728:
        _error(errors, "BOOTSTRAP_SEED")
    if bootstrap.get("simultaneous_interval") != SIMULTANEOUS_INTERVAL:
        _error(errors, "SIMULTANEOUS_INTERVAL_FORMULA")
    if statistics.get("independent_analysis_unit") != (
        "SCENE_SEED_X_MOTION_BLOCK"
    ):
        _error(errors, "ANALYSIS_UNIT")
    classes = statistics.get("effect_classification", {})
    if set(classes) != {"SUPPORTED", "RULED_OUT_AS_MATERIAL", "INCONCLUSIVE"}:
        _error(errors, "EFFECT_CLASSIFICATION")

    estimands = contract.get("estimands", {}).get("unit_level_contrasts", {})
    if set(estimands) != {
        "MOTION_CLEAN",
        "BLUR_STATIC",
        "LOW_TEXTURE_STATIC",
        "MOTION_X_BLUR",
        "MOTION_X_LOW_TEXTURE",
    }:
        _error(errors, "ESTIMANDS")
    family = contract.get("estimands", {}).get("terminal_driving_family", [])
    if family != TERMINAL_DRIVING_FAMILY:
        _error(errors, "TERMINAL_DRIVING_FAMILY")

    decision = contract.get("decision_rules", {})
    if decision.get("scientific_terminal_precedence") != TERMINALS:
        _error(errors, "TERMINAL_PRECEDENCE")
    for terminal, expected in TERMINAL_RULES.items():
        if decision.get(terminal) != expected:
            _error(errors, f"TERMINAL_RULE:{terminal}")
    if contract.get("claims_allowed") != [
        "IMPLEMENTATION_READY_FOR_CONFIRMATION",
        "IMPLEMENTATION_NOT_READY",
        "NOT_EVALUABLE",
    ]:
        _error(errors, "CLAIMS_ALLOWED")
    if contract.get("claim_ceiling") != (
        "CONTROLLED_GENERATOR_INTERNAL_MECHANISM_DEVELOPMENT_ONLY"
    ):
        _error(errors, "CLAIM_CEILING")
    successor = contract.get("successor_policy", {})
    for field in (
        "automatic_confirmation_authority",
        "automatic_algorithm_change_authority",
        "automatic_android_authority",
    ):
        if successor.get(field) is not False:
            _error(errors, f"SUCCESSOR_AUTHORITY:{field}")

    gate_ids = {
        gate.get("id")
        for gate in geometry.get("required_gates", [])
        if isinstance(gate, dict)
    }
    if len(gate_ids) != 14:
        _error(errors, "GEOMETRY_GATE_COUNT")
    for prefix in GEOMETRY_GATES:
        if not any(
            isinstance(gate_id, str) and gate_id.startswith(prefix)
            for gate_id in gate_ids
        ):
            _error(errors, f"GEOMETRY_GATE_MISSING:{prefix}")
    forbidden = " ".join(geometry.get("forbidden_shortcuts", [])).lower()
    if "homography" not in forbidden or "warpperspective" not in forbidden:
        _error(errors, "PLANAR_SHORTCUT_NOT_FORBIDDEN")

    counts = budget.get("count_budget", {})
    expected_counts = {
        "main_analysis_clusters": 80,
        "main_sequences": 480,
        "main_rendered_frames": 480 * 602,
        "main_scheduled_pairs": 480 * 601,
        "guardrail_analysis_clusters": 8,
        "guardrail_sequences": 16,
        "formal_total_sequences": 496,
        "formal_total_rendered_frames": 496 * 602,
        "formal_total_scheduled_pairs": 496 * 601,
        "quality_calibration_candidate_image_evaluations": (
            4 * 4 * 2 * 12 * 16
        ),
        "quality_calibration_final_six_arm_panel_images": (
            4 * 4 * 2 * 3 * 16
        ),
        "runtime_preflight_total_sequences": 8,
        "runtime_preflight_frames": 8 * 602,
        "runtime_preflight_pairs": 8 * 601,
        "all_planned_sequences_including_preflight": 504,
        "all_planned_frames_including_preflight": 504 * 602,
        "all_planned_pairs_including_preflight": 504 * 601,
    }
    for field, expected in expected_counts.items():
        if counts.get(field) != expected:
            _error(errors, f"BUDGET_COUNT:{field}")
    if budget.get("current_terminal") != "EXECUTION_NOT_AUTHORIZED":
        _error(errors, "BUDGET_CURRENT_TERMINAL")
    phases = budget.get("phase_budget", [])
    if not phases or phases[0].get("allowed_now") is not True:
        _error(errors, "P0_NOT_ALLOWED")
    if any(phase.get("allowed_now") is not False for phase in phases[1:]):
        _error(errors, "FUTURE_PHASE_ALLOWED")
    if budget.get("host_policy", {}).get("launcher_required") != (
        "scripts/run_guarded_host_research.ps1"
    ):
        _error(errors, "GUARDED_LAUNCHER")
    host_policy = budget.get("host_policy", {})
    if host_policy.get("candidate_profiles") != {
        "interactive": 4,
        "balanced": 8,
    }:
        _error(errors, "PREFLIGHT_WORKER_PROFILES")
    scheduling = budget.get("required_preflight", {}).get(
        "scheduling_comparison", ""
    )
    if not all(
        token in scheduling
        for token in ("4 and 8 workers", "12 and 16 workers are prohibited")
    ):
        _error(errors, "PREFLIGHT_SCHEDULING_COMPARISON")
    if budget.get("storage_budget", {}).get(
        "maximum_concurrent_staged_sequences"
    ) != 8:
        _error(errors, "PREFLIGHT_MAXIMUM_CONCURRENCY")
    canonical_progress = {
        "completed_units",
        "total_units",
        "throughput",
        "eta_seconds",
        "last_progress_at",
        "status",
    }
    progress_fields = set(
        budget.get("progress_contract", {}).get(
            "machine_readable_fields", []
        )
    )
    if not canonical_progress.issubset(progress_fields):
        _error(errors, "CANONICAL_PROGRESS_FIELDS")

    linked = contract.get("linked_specs", {})
    if linked.get("geometry_validation") != str(
        DEFAULT_GEOMETRY.relative_to(REPO_ROOT)
    ).replace("\\", "/"):
        _error(errors, "GEOMETRY_LINK")
    if linked.get("run_budget") != str(
        DEFAULT_BUDGET.relative_to(REPO_ROOT)
    ).replace("\\", "/"):
        _error(errors, "BUDGET_LINK")
    for name, path in (
        ("geometry_validation_sha256", DEFAULT_GEOMETRY),
        ("run_budget_sha256", DEFAULT_BUDGET),
        (
            "preregistration_sha256",
            REPO_ROOT / linked.get("preregistration", ""),
        ),
    ):
        expected = linked.get(name)
        if not isinstance(expected, str) or not path.is_file():
            _error(errors, f"LINKED_SPEC_HASH_FIELD:{name}")
        elif _sha256(path) != expected:
            _error(errors, f"LINKED_SPEC_HASH:{name}")

    if verify_dependencies:
        for name, dependency in contract.get("frozen_dependencies", {}).items():
            if name == "repository_head_at_design_freeze":
                continue
            if not isinstance(dependency, dict):
                _error(errors, f"DEPENDENCY_OBJECT:{name}")
                continue
            relative = dependency.get("path")
            expected = dependency.get("sha256")
            if not isinstance(relative, str) or not isinstance(expected, str):
                _error(errors, f"DEPENDENCY_FIELDS:{name}")
                continue
            path = (REPO_ROOT / relative).resolve()
            try:
                path.relative_to(REPO_ROOT)
            except ValueError:
                _error(errors, f"DEPENDENCY_SCOPE:{name}")
                continue
            if not path.is_file():
                _error(errors, f"DEPENDENCY_MISSING:{name}")
            elif _sha256(path) != expected:
                _error(errors, f"DEPENDENCY_HASH:{name}")

    return sorted(errors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        errors = validate_bundle(
            _load(args.contract.resolve()),
            _load(args.geometry.resolve()),
            _load(args.budget.resolve()),
        )
        payload = {
            "status": "VALID" if not errors else "INVALID",
            "protocol_id": PROTOCOL_ID,
            "formal_execution_authorized": False,
            "errors": errors,
        }
    except (OSError, ValueError, json.JSONDecodeError) as error:
        payload = {
            "status": "INVALID",
            "protocol_id": PROTOCOL_ID,
            "formal_execution_authorized": False,
            "errors": [f"LOAD:{type(error).__name__}:{error}"],
        }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["status"] == "VALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
