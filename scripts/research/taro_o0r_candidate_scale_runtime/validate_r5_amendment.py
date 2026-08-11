"""Validate the pre-outcome TARO R5 role-amendment lock."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOCK_PATH = REPO_ROOT / (
    "docs/research/taro/"
    "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_"
    "AMENDMENT_2026-08-11.json"
)
SCHEMA = "blindassist.taro.o0r.direct_apple_hybrid_adapter_fit_confirmation_r5_amendment.v1"
LOCK_ID = "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_AMENDMENT"
SUCCESSOR = "TARO_O0R_ARKITSCENES_DIRECT_APPLE_HYBRID_ADAPTER_FIT_CONFIRMATION_R5_IMPLEMENTATION_LOCK"
POLICY_ID = "DIRECT_WHEN_SOURCE_SUPPORT_AVAILABLE_ELSE_R1_BASELINE_V1"
R3_ROOT = REPO_ROOT / "artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3"

EXPECTED_TOP_LEVEL_KEYS = {
    "schema", "lock_id", "date", "research_mode", "status", "scientific_objective",
    "authority_interpretation", "role_amendment", "exact_cohort", "frozen_algorithm",
    "phase_contract", "confirmation_gates", "predecessor_bindings",
    "execution_contract_skeleton", "execution_authority", "claim_ceiling", "unique_successor",
}
EXPECTED_PARENT_ORDER = [
    {"visit_id": "470974", "video_id": "47332075", "physical_frame_count": 25},
    {"visit_id": "469216", "video_id": "47332946", "physical_frame_count": 16},
    {"visit_id": "423614", "video_id": "42898071", "physical_frame_count": 11},
    {"visit_id": "467370", "video_id": "47333776", "physical_frame_count": 24},
    {"visit_id": "469460", "video_id": "47333043", "physical_frame_count": 23},
    {"visit_id": "438794", "video_id": "44358241", "physical_frame_count": 26},
    {"visit_id": "467346", "video_id": "47333876", "physical_frame_count": 43},
    {"visit_id": "472473", "video_id": "47204786", "physical_frame_count": 43},
]
EXPECTED_SEQUENCE_HASHES = {
    "canonical_frame_identity_sequence_sha256": "52CFCC0CC37ED9DF2B7B3A5C99A617661062E600EB75B5790FC96225D7765B6F",
    "canonical_bound_source_sequence_sha256": "5C2E33FEB2DEF1FA48AF0F683AA342E236AC1B192874390830A32AF92513196A",
    "source_frame_receipt_sequence_sha256": "910F132065D8F17C8B5AA041E37520BC12D9DE998CF43EC5635B46F5C19194DD",
    "color_binding_sequence_sha256": "7D77CA90B23396C3B31D802E466A612F715E42B6DEA83DCA08327BFC70134ADF",
    "prior_eval_parent_denylist_sha256": "03B652A234DE72AA3638897443A2B4ECDBB69C97A516CC84CA900674C336B589",
}
EXPECTED_BINDINGS = {
    "DATA_USE_AUTHORIZATION_R1": ("docs/research/taro/TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_R1_RECEIPT_2026-08-10.json", 2713, "502EA4D66C74FB9D63FABFC7B75297B7EB370FE05DE2D4C01182764FA033E66D"),
    "AVAILABILITY_SUCCESSOR_R1": ("docs/research/taro/TARO_O0R_ARKITSCENES_AVAILABILITY_SUCCESSOR_R1_LOCK_2026-08-10.json", 2330, "5DF6DE1695DEEAEAD36F1EA2CBAD69C7481FDA70DFA5D4F783AE683FBBD2BB30"),
    "TRUTH_PREFLIGHT_R1": ("docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK_2026-08-10.json", 7282, "D8F63088B5F766A098DE06665DFAB06BF47E5F43ABB736C0DE65954432A77F65"),
    "R3_FACTOR_IMPLEMENTATION_LOCK": ("docs/research/taro/TARO_O0R_ARKITSCENES_FACTOR_HEADROOM_R3_IMPLEMENTATION_LOCK_2026-08-10.json", 14262, "23F35A86B570FE1E16782EF6CF68A02D26AFFDED4F6D24120879994295F7AC42"),
    "R3_FACTOR_EXECUTION_LOCK": ("docs/research/taro/TARO_O0R_ARKITSCENES_FACTOR_HEADROOM_R3_ONE_SHOT_EXECUTION_LOCK_2026-08-10.json", 8813, "5598DE46A9C4E6EF16763B8E62B02FDFC7F922D43BD8C633FE5124D528EB8DD6"),
    "R3_EXACT_FRAME_PLAN": ("artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/exact-frame-plan.json.gz", 11228, "BBAC08A9832A6CE465E8A8C6532FFF857AF878E6E821E683B24A83981C3C946C"),
    "R3_MANIFEST": ("artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/manifest.json", 139459, "429DE2ECBA3EA10E15ECC36A2625B3E022D4CF20D3B10557FD727E7E8F507298"),
    "R3_COMPLETION": ("artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/completion-receipt.json", 310, "0204FD7AC4198FC0EEE580E791A62BC73847458E6D3D518B3BE84457872DAF21"),
    "R3_UNCERTAINTY_RECEIPT": ("artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-receipt.json", 14820, "8E1C97C0961DEF6C4E7B3FCF2EADA6ECA022F0DFF2140221D221FB4C8B6A8CAE"),
    "R3_UNCERTAINTY_ARTIFACT": ("artifacts.local/evidence/taro/o0r-arkitscenes-source-adapter-r3/uncertainty-model-artifact.json.gz", 58084, "833CA7074E178D3D2FE6FEB66A386985C0CAEB8AA6878089E7C1B08984FD5E59"),
    "R4_MANIFEST": ("artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r4-full-cohort/manifest.json", 99718, "EB87FC6141723D2B44DCB384DF594FE3BFD65436262B07BCF2C0E3809709F760"),
    "R4_RESULT": ("artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r4-full-cohort/result.json", 1020, "F8BDDCB58534D5C6436A40C4509F72E21B5CB7BBFEFC29522DBAFEBE12483C3C"),
    "R4_SUMMARY": ("artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r4-full-cohort/summary.json", 6970, "6C586B35F6622F785BA6C463BFF901BD2CFED0C2AFBEBB5A9AFEFADB1A9A0CC8"),
    "R4_QUERY_RECORDS": ("artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-support-r4-full-cohort/full-cohort-query-records.json.gz", 588166, "323FFB7F456517C4EAEAED301A18FC96A1B2813526D963230B3EB43B8D58F2A7"),
    "R4A_MANIFEST": ("artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-r4a/manifest.json", 735, "F9260F32BEEA9B5AB749F3D8F67DEC83C1623F159D3B9BFA7A2BEDD52BEDF309"),
    "R4A_RESULT": ("artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-r4a/result.json", 872, "F2030EE00B5C0F63B5807D5DF6E361B1828C45462D079F59A482A4C62CB71684"),
    "R4A_SUMMARY": ("artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-r4a/summary.json", 8471, "BB0B7BE96F32290CC2C1CAA8119090AF6D0220AABCE4189C94AFC7D8FBAA1A2E"),
    "R4A_QUERY_RECORDS": ("artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-r4a/hybrid-query-records.json.gz", 568348, "F1D4095584D32805F3EDF266C313872210D764BDBEA9C8DFE8200B73E3AF651F"),
    "R4A_POLICY_CODE": ("scripts/research/taro_o0r_candidate_scale_runtime/direct_apple_hybrid.py", 17710, "891746B2DFD776931C13AA751CB018F027AFDC223B79241A4ECD780B82E2E6FB"),
    "R3_DIRECT_SUPPORT_CODE": ("scripts/research/taro_o0r_candidate_scale_runtime/direct_apple_support.py", 38656, "0495064DB81970FBEDCAFC3EB760AFC1AE8689C1D23D5FDFA4BA31843E7FAE93"),
    "R4_FULL_COHORT_CODE": ("scripts/research/taro_o0r_candidate_scale_runtime/direct_apple_full_cohort.py", 23669, "92D815E76FA5678F1162B65655D4E50DF3B6178A9BF8B5E3C6506E5E74AA7A69"),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest().upper()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _require(errors: list[str], condition: bool, code: str) -> None:
    if not condition:
        errors.append(code)


def _recompute_cohort(errors: list[str]) -> dict[str, str]:
    plan_path = R3_ROOT / "exact-frame-plan.json.gz"
    if not plan_path.is_file():
        errors.append("R3_EXACT_FRAME_PLAN_MISSING")
        return {}
    plan = _load_gzip_json(plan_path)
    fit = [entry for entry in plan if entry.get("parent", {}).get("role") == "ADAPTER_FIT"]
    prior_eval = [entry["parent"] for entry in plan if entry.get("parent", {}).get("role") == "O0R_EVAL_CANDIDATE"]
    observed_order = [
        {
            "visit_id": entry["parent"]["visit_id"],
            "video_id": entry["parent"]["video_id"],
            "physical_frame_count": len(entry["frame_plan"]["exact_timestamp_tokens"]),
        }
        for entry in fit
    ]
    _require(errors, observed_order == EXPECTED_PARENT_ORDER, "R5_SOURCE_PLAN_PARENT_ORDER_DRIFT")
    _require(errors, len(prior_eval) == 16, "R5_PRIOR_EVAL_DENYLIST_COUNT_DRIFT")
    entries: list[dict[str, str]] = []
    for parent_entry in fit:
        parent = parent_entry["parent"]
        for token in parent_entry["frame_plan"]["exact_timestamp_tokens"]:
            relative = (
                f"source-frames/adapter-fit/{parent['visit_id']}/{parent['video_id']}/{token}.json.gz"
            )
            path = R3_ROOT / relative
            if not path.is_file():
                errors.append(f"R5_SOURCE_RECORD_MISSING:{relative}")
                continue
            record = _load_gzip_json(path)
            source = record.get("source_frame_receipt", {})
            envelope = record.get("bound_source_frame_envelope", {})
            _require(errors, record.get("model_outputs_absent") is True, f"R5_PREDECESSOR_MODEL_OUTPUT_PRESENT:{relative}")
            _require(errors, source.get("source_role") == "ADAPTER_FIT", f"R5_SOURCE_ROLE_DRIFT:{relative}")
            _require(errors, source.get("physical_frame_id") == f"{parent['video_id']}:{token}", f"R5_PHYSICAL_FRAME_ID_DRIFT:{relative}")
            try:
                entries.append(
                    {
                        "visit_id": parent["visit_id"],
                        "video_id": parent["video_id"],
                        "timestamp_token": token,
                        "physical_frame_id": source["physical_frame_id"],
                        "source_record_relative_path": relative,
                        "source_record_sha256": _sha256_bytes(path.read_bytes()),
                        "source_frame_receipt_sha256": source["content_sha256"],
                        "bound_source_frame_envelope_sha256": envelope["content_sha256"],
                        "color_member_sha256": source["asset_bindings"]["color"]["sha256"],
                        "color_decoded_content_sha256": source["decoded_payload_bindings"]["color"]["decoded_content_sha256"],
                    }
                )
            except (KeyError, TypeError):
                errors.append(f"R5_SOURCE_RECORD_SCHEMA_DRIFT:{relative}")
    identities = [
        {key: entry[key] for key in ("visit_id", "video_id", "timestamp_token", "physical_frame_id")}
        for entry in entries
    ]
    colors = [
        {key: entry[key] for key in ("physical_frame_id", "color_member_sha256", "color_decoded_content_sha256")}
        for entry in entries
    ]
    return {
        "physical_frame_count": str(len(entries)),
        "canonical_frame_identity_sequence_sha256": _sha256_bytes(_canonical(identities)),
        "canonical_bound_source_sequence_sha256": _sha256_bytes(_canonical(entries)),
        "source_frame_receipt_sequence_sha256": _sha256_bytes(_canonical([entry["source_frame_receipt_sha256"] for entry in entries])),
        "color_binding_sequence_sha256": _sha256_bytes(_canonical(colors)),
        "prior_eval_parent_denylist_sha256": _sha256_bytes(_canonical(prior_eval)),
    }


def validate_payload(payload: Mapping[str, Any], *, verify_files: bool = True) -> list[str]:
    errors: list[str] = []
    _require(errors, set(payload) == EXPECTED_TOP_LEVEL_KEYS, "R5_TOP_LEVEL_KEY_SET_DRIFT")
    _require(errors, payload.get("schema") == SCHEMA, "R5_SCHEMA_DRIFT")
    _require(errors, payload.get("lock_id") == LOCK_ID, "R5_LOCK_ID_DRIFT")
    _require(errors, payload.get("date") == "2026-08-11", "R5_DATE_DRIFT")
    _require(errors, payload.get("research_mode") == "WILD_LAB", "R5_RESEARCH_MODE_DRIFT")
    _require(errors, payload.get("status") == "PROTOCOL_AMENDMENT_FROZEN_EXECUTION_FALSE", "R5_STATUS_DRIFT")
    _require(errors, payload.get("unique_successor") == SUCCESSOR, "R5_SUCCESSOR_DRIFT")

    authority = payload.get("authority_interpretation", {})
    _require(errors, authority.get("previous_model_or_task_authority") is False, "R5_PREVIOUS_AUTHORITY_EXPANDED")
    _require(errors, authority.get("amendment_does_not_activate_execution") is True, "R5_AMENDMENT_ACTIVATED_EXECUTION")
    _require(errors, authority.get("separate_hash_bound_implementation_lock_required") is True, "R5_IMPLEMENTATION_LOCK_NOT_REQUIRED")
    _require(errors, authority.get("separate_one_shot_execution_lock_and_user_model_execution_authority_required") is True, "R5_EXECUTION_LOCK_NOT_REQUIRED")
    _require(errors, authority.get("training_authorized") is False, "R5_TRAINING_AUTHORITY_EXPANDED")

    role = payload.get("role_amendment", {})
    _require(errors, role.get("predecessor_role") == "ADAPTER_FIT", "R5_PREDECESSOR_ROLE_DRIFT")
    _require(errors, role.get("successor_role") == "R5_TASK_METRIC_CONFIRMATION", "R5_SUCCESSOR_ROLE_DRIFT")
    _require(errors, role.get("predecessor_uncertainty_role_unchanged") is True, "R5_PREDECESSOR_ROLE_MUTATED")
    _require(errors, role.get("predecessor_contract_mutated") is False, "R5_PREDECESSOR_CONTRACT_MUTATED")
    forbidden = set(role.get("forbidden", []))
    for required in ("uncertainty refit", "selector fitting", "training", "policy modification", "reading or enumerating the sixteen O0R_EVAL_CANDIDATE truth or outcome roots"):
        _require(errors, required in forbidden, f"R5_ROLE_FORBIDDEN_RULE_MISSING:{required}")

    cohort = payload.get("exact_cohort", {})
    _require(errors, cohort.get("official_fold") == "Training", "R5_FOLD_DRIFT")
    _require(errors, cohort.get("parent_count") == 8, "R5_PARENT_COUNT_DRIFT")
    _require(errors, cohort.get("physical_frame_count") == 211, "R5_FRAME_COUNT_DRIFT")
    _require(errors, cohort.get("queries_per_frame") == 9, "R5_QUERY_SLOT_MULTIPLICITY_DRIFT")
    _require(errors, cohort.get("query_slot_count") == 1899, "R5_QUERY_SLOT_COUNT_DRIFT")
    _require(errors, cohort.get("parent_order") == EXPECTED_PARENT_ORDER, "R5_PARENT_ORDER_DRIFT")
    _require(errors, cohort.get("unknown_records_retained") is True, "R5_UNKNOWN_DROP_ALLOWED")
    _require(errors, cohort.get("drop_allowed") is False, "R5_DROP_ALLOWED")
    for key, expected in EXPECTED_SEQUENCE_HASHES.items():
        _require(errors, cohort.get(key) == expected, f"R5_LOCKED_SEQUENCE_HASH_DRIFT:{key}")

    algorithm = payload.get("frozen_algorithm", {})
    _require(errors, algorithm.get("policy_id") == POLICY_ID, "R5_POLICY_DRIFT")
    _require(errors, algorithm.get("free_parameter_count") == 0, "R5_FREE_PARAMETER_DRIFT")
    _require(errors, algorithm.get("threshold_count") == 0, "R5_THRESHOLD_DRIFT")
    _require(errors, algorithm.get("training_steps") == 0, "R5_TRAINING_STEP_DRIFT")
    _require(errors, algorithm.get("selection_field_allowlist") == ["source_support_available"], "R5_SELECTION_ALLOWLIST_DRIFT")
    _require(errors, algorithm.get("outcome_dependent_reselection_forbidden") is True, "R5_OUTCOME_RESELECTION_ALLOWED")
    _require(errors, algorithm.get("direct_extraction_failure_after_selection_action") == "RETAIN_DIRECT_UNKNOWN_NEVER_FALL_BACK", "R5_POST_OUTCOME_FALLBACK_ALLOWED")
    candidate = algorithm.get("candidate_identity", {})
    _require(errors, candidate.get("model_id") == "depthart-s-metric-indoor-448-official-fp32", "R5_MODEL_ID_DRIFT")
    _require(errors, candidate.get("source_commit") == "0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c", "R5_MODEL_SOURCE_DRIFT")
    _require(errors, candidate.get("checkpoint_sha256") == "597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65", "R5_CHECKPOINT_DRIFT")

    phases = payload.get("phase_contract", {})
    phase_a = phases.get("phase_a", {})
    phase_b = phases.get("phase_b", {})
    _require(errors, phase_a.get("all_candidates_before_decisions") is True, "R5_CANDIDATE_COMPLETION_ORDER_DRIFT")
    _require(errors, phase_a.get("all_records_before_phase_b") is True, "R5_PHASE_A_COMPLETION_NOT_REQUIRED")
    _require(errors, set(phase_a.get("required_zero_read_counts", [])) == {"FARO", "QUERY_TRUTH", "COMPACT_TRUTH", "TASK_METRIC", "PRIOR_EVAL_OUTCOME"}, "R5_PHASE_A_ZERO_READ_SET_DRIFT")
    _require(errors, "all prior eval truth roots" in phase_a.get("forbidden_payload_reads", []), "R5_PRIOR_EVAL_READ_NOT_FORBIDDEN")
    _require(errors, phase_b.get("phase_a_completion_required") is True, "R5_PHASE_B_WITHOUT_COMPLETION")
    _require(errors, phase_b.get("phase_a_hash_revalidation_required") is True, "R5_PHASE_A_HASH_REVALIDATION_MISSING")
    _require(errors, phase_b.get("prior_eval_truth_access_forbidden") is True, "R5_PHASE_B_PRIOR_EVAL_ACCESS_ALLOWED")
    _require(errors, phase_b.get("branch_reselection_forbidden") is True, "R5_PHASE_B_RESELECTION_ALLOWED")
    _require(errors, phase_b.get("aggregation_order") == "QUERY_TO_PHYSICAL_FRAME_THEN_PHYSICAL_FRAME_TO_PARENT_THEN_MEDIAN_ACROSS_PARENTS", "R5_AGGREGATION_ORDER_DRIFT")

    gates = payload.get("confirmation_gates", {})
    _require(errors, gates.get("exact_parent_denominator") == 8, "R5_GATE_PARENT_DENOMINATOR_DRIFT")
    _require(errors, gates.get("height_parents_with_paired_metric_required") == 8, "R5_HEIGHT_DENOMINATOR_DRIFT")
    _require(errors, gates.get("normal_parents_with_paired_metric_required") == 8, "R5_NORMAL_DENOMINATOR_DRIFT")
    _require(errors, gates.get("height_error_reduction_median_of_parent_medians_rule") == "> 0", "R5_HEIGHT_GATE_DRIFT")
    _require(errors, gates.get("normal_error_reduction_median_of_parent_medians_rule") == "> 0", "R5_NORMAL_GATE_DRIFT")
    _require(errors, gates.get("parents_jointly_positive_height_and_normal_required") == 8, "R5_JOINT_PARENT_GATE_DRIFT")
    _require(errors, gates.get("hybrid_extraction_evaluable_count_rule") == ">= baseline_extraction_evaluable_count", "R5_EXTRACTION_GATE_DRIFT")
    _require(errors, gates.get("hybrid_query_known_count_rule") == ">= baseline_query_known_count", "R5_KNOWNNESS_GATE_DRIFT")
    _require(errors, gates.get("unknown_is_negative") is False, "R5_UNKNOWN_BECAME_NEGATIVE")
    _require(errors, gates.get("undefined_required_denominator_action") == "NOT_EVALUABLE", "R5_UNDEFINED_DENOMINATOR_DRIFT")
    _require(errors, gates.get("gate_change_after_phase_a_forbidden") is True, "R5_POST_PHASE_A_GATE_CHANGE_ALLOWED")

    execution = payload.get("execution_contract_skeleton", {})
    argv = execution.get("unique_argv", [])
    _require(errors, len(argv) == 5 and argv[1:4] == ["-m", "scripts.research.taro_o0r_candidate_scale_runtime.run_direct_apple_hybrid_adapter_fit_confirmation", "--execution-lock"], "R5_UNIQUE_ARGV_DRIFT")
    _require(errors, execution.get("argv_alternatives") == [], "R5_ARGV_ALTERNATIVE_ALLOWED")
    _require(errors, execution.get("exclusive_evidence_root") == "artifacts.local/evidence/taro/o0r-arkitscenes-direct-apple-hybrid-adapter-fit-r5", "R5_EVIDENCE_ROOT_DRIFT")
    for key in ("root_must_be_absent_at_activation", "one_shot_consumed_on_root_creation"):
        _require(errors, execution.get(key) is True, f"R5_ONE_SHOT_RULE_DRIFT:{key}")
    for key in ("overwrite", "rerun"):
        _require(errors, execution.get(key) is False, f"R5_DESTRUCTIVE_RULE_DRIFT:{key}")
    _require(errors, execution.get("network_requests") == 0 and execution.get("training_steps") == 0, "R5_RESOURCE_SIDE_EFFECT_DRIFT")

    execution_authority = payload.get("execution_authority", {})
    expected_authority = {
        "implementation": True, "implementation_lock": False, "depthart_inference": False,
        "phase_a_source_decisions": False, "phase_b_truth_scoring": False, "training": False,
        "network": False, "device": False, "product": False, "safety": False,
    }
    _require(errors, execution_authority == expected_authority, "R5_EXECUTION_AUTHORITY_DRIFT")

    bindings = payload.get("predecessor_bindings", [])
    observed_bindings: dict[str, tuple[str, int, str]] = {}
    for binding in bindings if isinstance(bindings, list) else []:
        if not isinstance(binding, Mapping) or not isinstance(binding.get("role"), str):
            errors.append("R5_BINDING_SCHEMA_DRIFT")
            continue
        role_name = binding["role"]
        if role_name in observed_bindings:
            errors.append(f"R5_DUPLICATE_BINDING:{role_name}")
            continue
        observed_bindings[role_name] = (binding.get("path"), binding.get("bytes"), binding.get("sha256"))
    _require(errors, observed_bindings == EXPECTED_BINDINGS, "R5_PREDECESSOR_BINDING_SET_DRIFT")
    if verify_files:
        for role_name, (relative, expected_bytes, expected_sha) in EXPECTED_BINDINGS.items():
            path = REPO_ROOT / relative
            if not path.is_file():
                errors.append(f"R5_BOUND_FILE_MISSING:{role_name}")
                continue
            content = path.read_bytes()
            _require(errors, len(content) == expected_bytes, f"R5_BOUND_FILE_BYTES_DRIFT:{role_name}")
            _require(errors, _sha256_bytes(content) == expected_sha, f"R5_BOUND_FILE_HASH_DRIFT:{role_name}")
        recomputed = _recompute_cohort(errors)
        _require(errors, recomputed.get("physical_frame_count") == "211", "R5_RECOMPUTED_FRAME_COUNT_DRIFT")
        for key, expected in EXPECTED_SEQUENCE_HASHES.items():
            _require(errors, recomputed.get(key) == expected, f"R5_RECOMPUTED_SEQUENCE_HASH_DRIFT:{key}")
        r4_result = _load_json(REPO_ROOT / EXPECTED_BINDINGS["R4_RESULT"][0])
        r4a_result = _load_json(REPO_ROOT / EXPECTED_BINDINGS["R4A_RESULT"][0])
        r4a_summary = _load_json(REPO_ROOT / EXPECTED_BINDINGS["R4A_SUMMARY"][0])
        _require(errors, r4_result.get("terminal") == "TARO_O0R_DIRECT_APPLE_SUPPORT_R4_FULL_COHORT_COMPLETE", "R5_R4_TERMINAL_DRIFT")
        _require(errors, r4a_result.get("terminal") == "TARO_O0R_DIRECT_APPLE_HYBRID_R4A_COMPLETE", "R5_R4A_TERMINAL_DRIFT")
        _require(errors, r4a_result.get("policy_id") == POLICY_ID, "R5_R4A_RESULT_POLICY_DRIFT")
        _require(errors, r4a_result.get("training_steps") == 0 and r4a_result.get("threshold_count") == 0, "R5_R4A_WAS_FITTED")
        _require(errors, r4a_summary.get("selection_metric_fields_read") == [], "R5_R4A_SELECTION_METRIC_READ")
    return errors


def validate_file(path: Path = DEFAULT_LOCK_PATH, *, verify_files: bool = True) -> list[str]:
    try:
        payload = _load_json(path)
    except (OSError, json.JSONDecodeError) as error:
        return [f"R5_LOCK_READ_FAILED:{error}"]
    if not isinstance(payload, Mapping):
        return ["R5_LOCK_NOT_OBJECT"]
    return validate_payload(payload, verify_files=verify_files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK_PATH)
    parser.add_argument("--skip-file-verification", action="store_true")
    args = parser.parse_args(argv)
    errors = validate_file(args.lock, verify_files=not args.skip_file_verification)
    result = {
        "schema": "blindassist.taro.o0r.r5_amendment_validation_result.v1",
        "lock": str(args.lock),
        "passed": not errors,
        "error_count": len(errors),
        "errors": errors,
        "terminal": "TARO_O0R_R5_AMENDMENT_VALID" if not errors else "TARO_O0R_R5_AMENDMENT_INVALID",
    }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
