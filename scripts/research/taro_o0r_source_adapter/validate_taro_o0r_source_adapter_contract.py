#!/usr/bin/env python3
"""Validate the non-execution TARO O0R ARKitScenes source-adapter lock."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


CONTRACT_PATH = Path(
    "docs/research/taro/"
    "TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK_2026-08-10.json"
)
SCHEMA = "blindassist.taro.o0r.source_adapter_contract.v1"
PROTOCOL_ID = "TARO_O0R_ARKITSCENES_SOURCE_AND_ADAPTER_CONTRACT_LOCK"
SUCCESSOR_ID = "TARO_O0R_ARKITSCENES_SOURCE_ADAPTER_IMPLEMENTATION_LOCK"
ROLE_COUNTS = {"ADAPTER_FIT": 8, "O0R_EVAL_CANDIDATE": 16}
ARMS = [
    "NONE",
    "SCALE",
    "SUPPORT",
    "BOUNDARY",
    "SCALE_SUPPORT",
    "SCALE_BOUNDARY",
    "SUPPORT_BOUNDARY",
    "SCALE_SUPPORT_BOUNDARY",
]
ORACLE_MODES = [
    "VALUE_ONLY_COMMON_SUPPORT",
    "FULL_BLOCK_VALUE_VALIDITY_UNCERTAINTY",
]
REQUIRED_ASSETS = [
    "upsampling/Training/{video_id}.zip",
    "raw/Training/{video_id}/lowres_wide_intrinsics.zip",
    "raw/Training/{video_id}/lowres_wide.traj",
]
PATHSPECS = [
    ":(glob)docs/**/*.md",
    ":(glob)docs/**/*.json",
    ":(glob)scripts/**/*.md",
    ":(glob)scripts/**/*.json",
    "DATASET_MASTER_LEDGER.csv",
]
ID_PATTERN = re.compile(r"(?<!\d)\d{6,8}(?!\d)")


def require(condition: bool, code: str) -> None:
    if not condition:
        raise ValueError(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _rank(salt: str, role: str, visit_id: str, video_id: str) -> str:
    value = f"{salt}:{role}:{visit_id}:{video_id}"
    return hashlib.sha256(value.encode("ascii")).hexdigest().upper()


def validate_contract(payload: dict[str, Any]) -> None:
    require(payload.get("schema") == SCHEMA, "CONTRACT_SCHEMA")
    require(payload.get("protocol_id") == PROTOCOL_ID, "PROTOCOL_ID")
    require(payload.get("version") == "R0", "VERSION")
    require(payload.get("stage") == "O0R_SOURCE_AND_ADAPTER_CONTRACT_LOCK", "STAGE")
    require(payload.get("status") == "PROTOCOL_FROZEN_SCIENTIFIC_NOT_RUN", "STATUS")
    require(payload.get("scientific_status") == "NOT_RUN", "SCIENTIFIC_STATUS")
    require(payload.get("outcome_access_started") is False, "OUTCOME_ACCESS")

    source = payload.get("source_contract", {})
    require(source.get("source_id") == "ARKITSCENES_UPSAMPLING_V1_TRAIN", "SOURCE_ID")
    require(source.get("official_fold") == "Training", "SOURCE_FOLD")
    require(source.get("metadata_rows") == 2257, "SOURCE_METADATA_ROWS")
    require(source.get("required_assets_per_video") == REQUIRED_ASSETS, "SOURCE_ASSETS")
    require(source.get("teacher_or_model_generated_truth_allowed") is False, "SOURCE_TRUTH_SHORTCUT")

    selection = payload.get("selection_contract", {})
    require(selection.get("exclusion_snapshot_commit") == "1cc126e768c0e89af934835d96763b0ec2fcdd38", "EXCLUSION_COMMIT")
    require(selection.get("matched_official_identity_count") == 100, "EXCLUSION_COUNT")
    require(selection.get("matched_official_identities_sha256") == "E9D948F2F6286DE3D307C6FD379DE0EAE34322693030310EF3DC8770DC3C7A7D", "EXCLUSION_SHA")
    require(selection.get("selection_salt") == "TARO_O0R_ARKITSCENES_R0", "SELECTION_SALT")
    require(selection.get("role_order") == list(ROLE_COUNTS), "ROLE_ORDER")
    roles = selection.get("roles", {})
    require(set(roles) == set(ROLE_COUNTS), "ROLE_KEYS")
    all_visits: list[str] = []
    all_videos: list[str] = []
    for role, expected_count in ROLE_COUNTS.items():
        rows = roles.get(role)
        require(isinstance(rows, list) and len(rows) == expected_count, f"ROLE_COUNT:{role}")
        ranks: list[str] = []
        for row in rows:
            require(set(row) == {"visit_id", "video_id", "official_fold", "selection_rank_sha256"}, f"ROLE_FIELDS:{role}")
            visit = str(row["visit_id"])
            video = str(row["video_id"])
            require(row["official_fold"] == "Training", f"ROLE_FOLD:{role}")
            expected_rank = _rank(selection["selection_salt"], role, visit, video)
            require(row["selection_rank_sha256"] == expected_rank, f"ROLE_RANK:{role}:{video}")
            all_visits.append(visit)
            all_videos.append(video)
            ranks.append(expected_rank)
        require(ranks == sorted(ranks), f"ROLE_RANK_ORDER:{role}")
    require(len(all_visits) == len(set(all_visits)) == 24, "VISIT_OVERLAP")
    require(len(all_videos) == len(set(all_videos)) == 24, "VIDEO_OVERLAP")
    invariants = selection.get("invariants", {})
    require(invariants.get("adapter_fit_parent_count") == 8, "FIT_PARENT_COUNT")
    require(invariants.get("o0r_eval_candidate_parent_count") == 16, "EVAL_PARENT_COUNT")
    require(invariants.get("replacement_allowed") is False, "REPLACEMENT_AUTHORITY")
    require(invariants.get("role_reassignment_allowed") is False, "REASSIGNMENT_AUTHORITY")
    require(invariants.get("model_output_influence") is False, "MODEL_SELECTION_INFLUENCE")
    require(invariants.get("confirmation_role_allocated") is False, "CONFIRMATION_ALLOCATION")

    role_contract = payload.get("role_contract", {})
    require(role_contract.get("ADAPTER_FIT", {}).get("model_outputs_forbidden") is True, "FIT_MODEL_FIREWALL")
    require(role_contract.get("ADAPTER_FIT", {}).get("task_metric_forbidden") is True, "FIT_TASK_FIREWALL")
    require(role_contract.get("O0R_EVAL_CANDIDATE", {}).get("drop_after_model_output_forbidden") is True, "EVAL_DROP_FIREWALL")
    require(role_contract.get("O0R_EVAL_CANDIDATE", {}).get("replacement_after_truth_access_forbidden") is True, "EVAL_REPLACEMENT_FIREWALL")

    receipt = payload.get("frame_receipt_adapter", {})
    require(receipt.get("site_id") == "visit_id" and receipt.get("parent_id") == "visit_id", "PARENT_SITE_IDENTITY")
    require(receipt.get("session_id") == "video_id" and receipt.get("capture_id") == "video_id", "SESSION_CAPTURE_IDENTITY")
    require("Decimal" in str(receipt.get("sensor_timestamp_ns")), "TIMESTAMP_DECIMAL")
    require("Decimal" in str(receipt.get("pose_bracket_timestamp_ns")), "POSE_TIMESTAMP_DECIMAL")
    require(receipt.get("max_source_timestamp_rule") == "MAX_FRAME_AND_RIGHT_POSE_BRACKET", "POSE_CAUSAL_WATERMARK")
    require(receipt.get("p0_frame_receipt_projection") == "NOT_AVAILABLE_IN_O0R_SOURCE_CHARACTERIZATION", "P0_RECEIPT_OVERCLAIM")
    require(receipt.get("base_receipt_schema") == "blindassist.taro.o0r.source_frame_receipt.v1", "BASE_RECEIPT_SCHEMA")
    require(receipt.get("query_receipt_schema") == "blindassist.taro.o0r.query_receipt.v1", "QUERY_RECEIPT_SCHEMA")
    require(receipt.get("query_receipts_per_physical_frame") == 9, "QUERY_RECEIPT_CARDINALITY")
    require(receipt.get("receipt_failure") == "UNKNOWN_NOT_ADMITTED", "RECEIPT_FAIL_CLOSED")

    query = payload.get("query_contract", {})
    swept = query.get("swept_volume", {})
    require(swept == {"kind": "CAPSULE_CHAIN", "half_width_m": 0.25, "height_m": 1.8, "inflation_m": 0.05}, "QUERY_SWEPT_VOLUME")
    require(query.get("horizon_m") == 2.0, "QUERY_HORIZON")
    require(query.get("clear_margin_m") == 0.05 and query.get("occupied_margin_m") == 0.0, "QUERY_MARGINS")
    require(query.get("path_lateral_offsets_m") == [-0.35, 0.0, 0.35], "QUERY_LATERAL_GRID")
    require(query.get("path_yaw_degrees") == [-10.0, 0.0, 10.0], "QUERY_YAW_GRID")
    require(query.get("queries_per_frame") == 9, "QUERY_COUNT")
    require(query.get("query_grid_order") == "LATERAL_MAJOR_THEN_YAW_ASCENDING", "QUERY_ORDER")
    require(query.get("minimum_forward_m") == 0.2 and query.get("clearance_cap_m") == 2.0, "QUERY_METRIC_DOMAIN")
    require("0.30 m" in str(query.get("signed_clearance_rule")), "QUERY_SIGNED_CLEARANCE")
    require("all nine" in str(query.get("complete_frame_rule")), "QUERY_COMPLETE_FRAME")
    require(query.get("truth_source") == "registered FARO highres_depth plus bound K/pose only", "QUERY_TRUTH_SOURCE")
    require(query.get("unknown_is_negative") is False, "QUERY_UNKNOWN_NEGATIVE")

    truth = payload.get("factor_truth_contract", {})
    require(truth.get("state_blocks") == ["SCALE", "SUPPORT", "BOUNDARY"], "FACTOR_BLOCKS")
    for block in ("SCALE", "SUPPORT", "BOUNDARY"):
        contract = truth.get(block, {})
        require("FARO" in str(contract.get("value_truth")), f"FACTOR_VALUE_TRUTH:{block}")
        require(bool(contract.get("validity_truth")), f"FACTOR_VALIDITY_TRUTH:{block}")
        require("ADAPTER_FIT" in str(contract.get("uncertainty_truth")) or block == "SUPPORT", f"FACTOR_UNCERTAINTY_TRUTH:{block}")
    scale = truth.get("SCALE", {})
    require(scale.get("truth_only_value_kind") == "ABSOLUTE_FARO_METRIC_REFERENCE", "SCALE_TRUTH_ONLY_MODEL_FIREWALL")
    require(scale.get("truth_only_log_metric_scale") == 0.0, "SCALE_ABSOLUTE_METRIC_VALUE")
    require(scale.get("candidate_relative_correction_after_truth_only_result") is True, "SCALE_RELATIVE_ORDER")
    require(scale.get("candidate_relative_correction_may_change_selection") is False, "SCALE_RELATIVE_SELECTION")
    cells = truth.get("uncertainty_fit_cells", {})
    require(cells.get("confidence_values") == [0, 1, 2], "UNCERTAINTY_CONFIDENCE_CELLS")
    require(cells.get("range_edges_m") == [0.0, 1.0, 2.0, 3.0, 6.0], "UNCERTAINTY_RANGE_CELLS")
    require(cells.get("minimum_independent_parents_per_cell") == 4, "UNCERTAINTY_PARENT_GATE")
    require(cells.get("minimum_samples_per_cell") == 128, "UNCERTAINTY_SAMPLE_GATE")
    require(cells.get("quantile") == 0.95, "UNCERTAINTY_QUANTILE")
    require(cells.get("range_edge_ownership") == "LEFT_CLOSED_RIGHT_OPEN_LAST_BIN_RIGHT_CLOSED", "UNCERTAINTY_EDGE_OWNERSHIP")
    require(cells.get("parent_aggregation") == "PARENT_MACRO_Q95_OF_WITHIN_PARENT_Q95", "UNCERTAINTY_PARENT_MACRO")
    require(cells.get("fallback_order") == [
        "EXACT_CONFIDENCE_EXACT_RANGE",
        "SAME_CONFIDENCE_SYMMETRIC_CONTIGUOUS_RANGE_EXPANSION",
        "ALL_CONFIDENCE_EXACT_RANGE",
        "ALL_CONFIDENCE_SYMMETRIC_CONTIGUOUS_RANGE_EXPANSION",
        "GLOBAL",
    ], "UNCERTAINTY_FALLBACK_ORDER")
    canonical = truth.get("canonicalization", {})
    require(canonical == {
        "json_sort_keys": True,
        "float_decimal_places": 12,
        "negative_zero_to_zero": True,
        "nan_or_infinity": "REJECT",
    }, "CANONICALIZATION")
    require(truth.get("truth_materialization_before_model_output") is True, "TRUTH_BEFORE_MODEL")
    require(truth.get("teacher_or_model_output_as_truth") is False, "MODEL_AS_TRUTH")
    require(truth.get("invented_constant_uncertainty") is False, "CONSTANT_UNCERTAINTY")

    baseline = payload.get("baseline_contract", {})
    require(baseline.get("model_id") == "depthart-s-metric-indoor-448-official-fp32", "BASELINE_ID")
    require(baseline.get("checkpoint_sha256") == "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65", "BASELINE_SHA")
    require(baseline.get("training_steps") == 0, "BASELINE_TRAINING")
    require(baseline.get("checkpoint_selection_allowed") is False, "BASELINE_SELECTION")
    require(baseline.get("truth_alignment_or_per_frame_fitting_allowed") is False, "BASELINE_TRUTH_FIT")
    require(baseline.get("generation_order") == "only after the truth-only admission result is signed", "BASELINE_GENERATION_ORDER")
    require(baseline.get("truth_only_bundle_immutable_before_baseline") is True, "BASELINE_TRUTH_IMMUTABLE")
    require(baseline.get("relative_scale_correction_uses_preexisting_truth_only_common_support") is True, "BASELINE_SCALE_COMMON_SUPPORT")

    factorial = payload.get("factorial_contract", {})
    require(factorial.get("arms") == ARMS, "FACTORIAL_ARMS")
    require(factorial.get("oracle_modes") == ORACLE_MODES, "ORACLE_MODES")
    require(factorial.get("primary_mode") == ORACLE_MODES[0], "PRIMARY_MODE")
    require(factorial.get("primary_comparison") == "SCALE_SUPPORT_BOUNDARY_VERSUS_NONE", "PRIMARY_COMPARISON")
    require(factorial.get("single_block_diff_required") is True, "SINGLE_BLOCK_DIFF")
    require(factorial.get("common_support_identity_required") is True, "COMMON_SUPPORT")
    require("separate negative control" in str(factorial.get("receipt_k_corruption_control")), "K_FACTORIAL_CONTAMINATION")
    require(factorial.get("post_outcome_arm_selection_forbidden") is True, "POST_OUTCOME_ARM_SELECTION")

    registration = payload.get("registration_and_geometry_contract", {})
    require(registration.get("highres_color_and_faro_shape_hw") == [1440, 1920], "FARO_SHAPE")
    require(registration.get("apple_depth_and_confidence_shape_hw") == [192, 256], "APPLE_SHAPE")
    require(registration.get("lowres_to_highres_scale_xy") == [7.5, 7.5], "REGISTRATION_SCALE")
    require(registration.get("depth_valid_range_m") == [0.25, 6.0], "DEPTH_VALID_RANGE")
    require(registration.get("boundary_sign") == "POSITIVE_OUTSIDE_OBSTACLE_NEGATIVE_INSIDE_OBSTACLE_ZERO_AT_BOUNDARY", "BOUNDARY_SIGN")
    require(registration.get("minimum_boundary_local_valid_fraction") == 0.8, "BOUNDARY_COMPLETENESS")
    require(registration.get("model_teacher_or_task_fields_in_truth_input") == "REJECT_RECURSIVELY", "TRUTH_INPUT_FIREWALL")

    interface = payload.get("implementation_interface_contract", {})
    require(interface.get("new_runtime_module") == "scripts/research/taro_o0r_source_adapter_runtime", "RUNTIME_MODULE")
    require(interface.get("source_io_in_implementation_lock") is False, "IMPLEMENTATION_SOURCE_IO")
    require(interface.get("static_tests_use_synthetic_arrays_only") is True, "IMPLEMENTATION_TEST_DATA")
    require(interface.get("legacy_geometry_r2_reducer_runtime_role") == "REFERENCE_ONLY_NOT_TARO_QUERY_REDUCER", "LEGACY_REDUCER_ROLE")
    require(interface.get("taro_specific_query_reducer_required") is True, "TARO_QUERY_REDUCER")
    require(interface.get("p0_full_frame_receipt_claim") is False, "P0_FULL_RECEIPT_CLAIM")
    require(set(interface.get("legacy_truth_reader_forbidden_interfaces", [])) == {
        "load_manifest_frame",
        "derive_assistive_truth three-band output",
        "binary-float filename timestamp parsing",
    }, "LEGACY_READER_FIREWALL")

    gates = payload.get("truth_only_admission_gates", {})
    require(gates.get("all_adapter_fit_parents_present") == 8, "GATE_FIT_PARENTS")
    require(gates.get("minimum_evaluable_o0r_parents") == 12, "GATE_EVAL_PARENTS")
    require(gates.get("minimum_truth_clear_parents") == 6, "GATE_CLEAR_PARENTS")
    require(gates.get("minimum_truth_occupied_parents") == 6, "GATE_OCCUPIED_PARENTS")
    require(gates.get("minimum_exact_timestamp_frames_per_evaluable_parent") == 12, "GATE_FRAME_COUNT")
    require(gates.get("minimum_complete_factor_query_fraction_within_source_eligible_frames") == 1.0, "GATE_COMPLETE_TRUTH")
    require(gates.get("depthart_or_other_model_outputs_absent") is True, "GATE_MODEL_ABSENCE")
    require(gates.get("query_receipts_required_per_source_eligible_frame") == 9, "GATE_QUERY_RECEIPTS")
    require(gates.get("max_source_timestamp_includes_right_pose_bracket") is True, "GATE_POSE_WATERMARK")
    require(gates.get("model_independent_scale_reference_complete") is True, "GATE_SCALE_REFERENCE")
    require(gates.get("parent_state_roles_may_overlap") is True, "GATE_PARENT_STATE_ROLES")
    require(bool(gates.get("exact_timestamp_frame_denominator")), "GATE_EXACT_DENOMINATOR")
    require(bool(gates.get("source_eligible_frame_denominator")), "GATE_SOURCE_DENOMINATOR")
    require(bool(gates.get("admitted_frame_denominator")), "GATE_ADMITTED_DENOMINATOR")
    require(gates.get("undefined_denominator") == "FAIL_NOT_DROP", "GATE_UNDEFINED")

    metrics = payload.get("o0r_metrics_and_gates", {})
    require(metrics.get("minimum_meaningful_effect_m") == 0.02, "MINIMUM_EFFECT")
    bootstrap = metrics.get("bootstrap", {})
    require(bootstrap == {"replicates": 20000, "seed": 271828, "unit": "parent", "two_sided_alpha": 0.05}, "BOOTSTRAP")
    guardrails = metrics.get("guardrails", {})
    require(guardrails.get("false_clear_difference_upper_confidence_bound_max") == 0.01, "FALSE_CLEAR_GUARDRAIL")
    require(guardrails.get("known_coverage_difference_lower_confidence_bound_min") == -0.02, "COVERAGE_GUARDRAIL")
    require(guardrails.get("minimum_favorable_parent_fraction") == 0.75, "PARENT_SIGN_GATE")
    require(guardrails.get("all_unknown_forbidden") is True, "ALL_UNKNOWN_GATE")

    budget = payload.get("resource_budget", {})
    require(budget.get("training_steps") == 0, "BUDGET_TRAINING")
    require(budget.get("device_or_android") is False, "BUDGET_DEVICE")
    require(int(budget.get("maximum_compressed_source_bytes", 0)) <= 21474836480, "SOURCE_BYTE_BUDGET")

    isolation = payload.get("artifact_isolation", {})
    roots = [value for key, value in isolation.items() if key.endswith("_root")]
    require(len(roots) == len(set(roots)) == 5, "ARTIFACT_ROOT_COLLISION")
    require(isolation.get("historical_o0m_read_only") is True, "O0M_READ_ONLY")
    require(isolation.get("overwrite_or_rerun_historical_o0m") is False, "O0M_RERUN")

    authority = payload.get("execution_authority", {})
    require(authority.get("contract_design") is True, "CONTRACT_DESIGN_AUTHORITY")
    for key, value in authority.items():
        if key not in {"contract_design", "metadata_and_signed_receipt_read"}:
            require(value is False, f"UNAUTHORIZED_EXECUTION:{key}")
    require(payload.get("unique_successor") == SUCCESSOR_ID, "SUCCESSOR")


def validate_bindings(repo_root: Path, payload: dict[str, Any]) -> None:
    for name, binding in payload.get("bindings", {}).items():
        path = repo_root / str(binding.get("path"))
        require(path.is_file(), f"BINDING_MISSING:{name}")
        require(path.stat().st_size == int(binding.get("bytes", -1)), f"BINDING_BYTES:{name}")
        require(sha256_file(path) == binding.get("sha256"), f"BINDING_SHA:{name}")


def _git_numeric_tokens(repo_root: Path, commit: str) -> list[str]:
    command = [
        "git",
        "-C",
        str(repo_root),
        "grep",
        "-h",
        "-o",
        "-P",
        r"(?<![0-9])[0-9]{6,8}(?![0-9])",
        commit,
        "--",
        *PATHSPECS,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, encoding="utf-8")
    return result.stdout.splitlines()


def validate_selection(repo_root: Path, payload: dict[str, Any]) -> None:
    selection = payload["selection_contract"]
    metadata_binding = payload["bindings"]["arkitscenes_upsampling_split"]
    metadata_path = repo_root / metadata_binding["path"]
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 2257, "METADATA_ROW_COUNT")
    require(set(rows[0]) == {"video_id", "visit_id", "fold"}, "METADATA_SCHEMA")
    known_ids = {
        value
        for row in rows
        for value in (row["visit_id"], row["video_id"])
        if value != "NA"
    }
    tokens = _git_numeric_tokens(repo_root, selection["exclusion_snapshot_commit"])
    excluded = sorted({token for token in tokens if token in known_ids})
    digest = hashlib.sha256(("\n".join(excluded) + "\n").encode("utf-8")).hexdigest().upper()
    require(len(excluded) == selection["matched_official_identity_count"], "RECOMPUTED_EXCLUSION_COUNT")
    require(digest == selection["matched_official_identities_sha256"], "RECOMPUTED_EXCLUSION_SHA")

    excluded_set = set(excluded)
    used_visits: set[str] = set()
    used_videos: set[str] = set()
    recomputed: dict[str, list[dict[str, str]]] = {}
    for role, count in ROLE_COUNTS.items():
        eligible: list[dict[str, str]] = []
        for row in rows:
            visit = row["visit_id"]
            video = row["video_id"]
            if (
                row["fold"] != "Training"
                or visit == "NA"
                or visit in excluded_set
                or video in excluded_set
                or visit in used_visits
                or video in used_videos
            ):
                continue
            eligible.append(
                {
                    "visit_id": visit,
                    "video_id": video,
                    "official_fold": row["fold"],
                    "selection_rank_sha256": _rank(selection["selection_salt"], role, visit, video),
                }
            )
        eligible.sort(key=lambda row: row["selection_rank_sha256"])
        picked: list[dict[str, str]] = []
        role_visits: set[str] = set()
        for row in eligible:
            if row["visit_id"] in role_visits:
                continue
            picked.append(row)
            role_visits.add(row["visit_id"])
            if len(picked) == count:
                break
        require(len(picked) == count, f"RECOMPUTED_ROLE_INSUFFICIENT:{role}")
        recomputed[role] = picked
        used_visits.update(row["visit_id"] for row in picked)
        used_videos.update(row["video_id"] for row in picked)
    require(recomputed == selection["roles"], "RECOMPUTED_ROSTER_DRIFT")


def validate_artifact_roots(repo_root: Path, payload: dict[str, Any]) -> None:
    isolation = payload["artifact_isolation"]
    historical = repo_root / isolation["historical_o0m_root"]
    require(historical.is_dir(), "HISTORICAL_O0M_ROOT_MISSING")
    for key in (
        "future_source_root",
        "future_work_root",
        "future_truth_evidence_root",
        "future_o0r_evidence_root",
    ):
        require(not (repo_root / isolation[key]).exists(), f"FUTURE_ROOT_PRESENT:{key}")


def validate_repository(repo_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    validate_contract(payload)
    validate_bindings(repo_root, payload)
    validate_selection(repo_root, payload)
    validate_artifact_roots(repo_root, payload)
    return {
        "schema": "blindassist.taro.o0r.source_adapter_static_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID",
        "scientific_status": "NOT_RUN",
        "adapter_fit_parents": 8,
        "o0r_eval_candidate_parents": 16,
        "future_roots_absent": True,
        "execution_authority": False,
        "semantic_seams_frozen": [
            "MODEL_FREE_SCALE_TRUTH_ONLY",
            "RIGHT_POSE_BRACKET_WATERMARK",
            "NINE_QUERY_BOUND_RECEIPTS",
            "SOURCE_SPECIFIC_RECEIPT_NO_P0_OVERCLAIM",
            "NEW_TARO_QUERY_REDUCER",
        ],
        "unique_successor": SUCCESSOR_ID,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    payload = json.loads((repo_root / CONTRACT_PATH).read_text(encoding="utf-8"))
    print(json.dumps(validate_repository(repo_root, payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
