#!/usr/bin/env python3
"""Fail-closed validator for the non-executing TARO R11 protocol lock."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r9_clear_runtime import clear_enrichment_fit as r9_fit
from scripts.research.taro_o1r_r11_abstention_runtime import abstention_candidate
from scripts.research.taro_o1r_r11_abstention_runtime import development_replay
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool


PROTOCOL_RELATIVE = "docs/research/taro/TARO_O1R_R11_POSITIVE_OCCUPANCY_ABSTENTION_AND_FRESH_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK_2026-08-12.json"
SCHEMA = "blindassist.taro.o1r.r11_positive_occupancy_abstention_protocol_lock.v1"
LOCK_ID = "TARO_O1R_R11_POSITIVE_OCCUPANCY_ABSTENTION_AND_FRESH_DUAL_CLASS_CONFIRMATION_PROTOCOL_LOCK"
STATUS = "LOCKED_NON_EXECUTING_PRE_NEW_NETWORK_PRE_SOURCE_PRE_OUTCOME"
EXPECTED_BINDINGS = {
    "R7_POSITIVE_FACTOR": "scripts/research/taro_o1r_r7_canary_runtime/positive_occupancy_factor.py",
    "R9_DEVELOPMENT_RESULT": "docs/research/taro/TARO_O1R_R9_CLEAR_ENRICHMENT_DEVELOPMENT_RESULT_2026-08-12.json",
    "R10_FORMAL_RESULT": development_replay.R10_FORMAL_RESULT,
    "R11_DEVELOPMENT_RESULT": "docs/research/taro/TARO_O1R_R11_WEAK_DISTAL_ABSTENTION_DEVELOPMENT_RESULT_2026-08-12.json",
    "R11_CANDIDATE_RUNTIME": "scripts/research/taro_o1r_r11_abstention_runtime/abstention_candidate.py",
    "R11_DEVELOPMENT_REPLAY": "scripts/research/taro_o1r_r11_abstention_runtime/development_replay.py",
    "R11_FRESH_POOL": "scripts/research/taro_o1r_r11_abstention_runtime/fresh_pool.py",
}
EXPECTED_FORMATION = {
    "role": "TUNED_ON_CONSUMED_R10_DEVELOPMENT_ONLY",
    "sole_committed_false_positive_used": True,
    "full_r10_replay_used_only_for_candidate_accounting": True,
    "additional_r10_threshold_search_authorized": False,
    "identity_specific_exception_allowed": False,
    "r10_counterfactual_pass_claim_allowed": False,
}
EXPECTED_PHASE_ORDER = [
    "METADATA_ONLY_POOL_FREEZE",
    "ZERO_BODY_HEAD",
    "BOUNDED_SOURCE_DOWNLOAD",
    "READ_ONLY_CONTAINER_INVENTORY",
    "ALL_48_PARENT_SOURCE_AND_DEPTHART_PHASE_A",
    "SEAL_R7_BASE_R11_CANDIDATE_AND_R9_PARENT_SCORES",
    "SOURCE_ONLY_TOP24_SELECTION_SEAL",
    "SELECTED_TOP24_FARO_PHASE_B",
    "FRAME_PARENT_AWARE_FIXED_GATE_REDUCTION",
]
EXPECTED_FIREWALL = {
    "phase_a_faro_reads": 0,
    "seal_all_48_source_records_before_faro": True,
    "seal_all_48_parent_scores_before_faro": True,
    "seal_top24_before_faro": True,
    "read_unselected_faro": False,
    "source_reselection_after_faro": False,
    "parent_reselection_after_faro": False,
    "candidate_or_threshold_reselection_after_faro": False,
    "unknown_is_negative": False,
}
EXPECTED_EVALUABILITY = {
    "selected_parent_count": 24,
    "minimum_evaluable_parents": 16,
    "minimum_parents_with_definite_occupied": 12,
    "minimum_definite_occupied_queries": 200,
    "minimum_parents_with_definite_clear": 4,
    "minimum_physical_frames_with_definite_clear": 12,
    "minimum_definite_clear_queries": 20,
}
EXPECTED_GATES = {
    "minimum_candidate_occupied_precision": 0.9,
    "minimum_one_sided_95_wilson_candidate_occupied_precision_lower_bound": 0.8,
    "minimum_candidate_occupied_recall": 0.9,
    "minimum_parent_macro_definite_occupied_recall": 0.9,
    "maximum_micro_occupied_recall_loss_vs_r7": 0.01,
    "maximum_parent_macro_occupied_recall_loss_vs_r7": 0.01,
    "candidate_false_positives_must_not_exceed_r7": True,
    "minimum_query_clear_specificity": 0.9,
    "minimum_clear_frame_specificity": 0.9,
    "minimum_one_sided_95_wilson_clear_frame_specificity_lower_bound": 0.8,
    "minimum_parent_macro_clear_frame_specificity": 0.9,
    "maximum_clear_outputs": 0,
    "unknown_is_negative": False,
}
EXPECTED_EFFECT_REPORTING = {
    "minimum_abstained_definite_clear_frames_for_effect_claim": 2,
    "minimum_parents_with_abstained_definite_clear_frame_for_effect_claim": 2,
    "effect_claim_not_required_for_absolute_candidate_confirmation": True,
    "if_challenge_absent": "ABSTENTION_EFFECT_NOT_EVALUABLE",
    "absolute_candidate_metrics_still_reduced": True,
}
EXPECTED_TERMINALS = [
    "EXECUTION_INVALID",
    "NOT_EVALUABLE_DUAL_CLASS_COVERAGE",
    "FAIL_FIXED_CONFIRMATION_GATE",
    "WILD_LAB_RESEARCH_FACTOR_CONFIRMATION_PASS",
]
EXPECTED_AUTHORITY = {
    "protocol_validation": True,
    "metadata_only_pool_recompute": True,
    "data_use_authorized": False,
    "network": False,
    "head": False,
    "source_body": False,
    "model": False,
    "faro": False,
    "truth_label_construction": False,
    "training": False,
    "device": False,
    "deployment": False,
    "product": False,
    "safety": False,
}
EXPECTED_RESOURCE_CEILING = {
    "maximum_compressed_source_bytes": 12 * 1024**3,
    "maximum_materialized_bytes": 30 * 1024**3,
    "maximum_wall_seconds": 16 * 60 * 60,
    "maximum_peak_rss_bytes": 16 * 1024**3,
    "maximum_peak_vram_bytes": 12 * 1024**3,
    "maximum_evidence_bytes": 2 * 1024**3,
}


class ProtocolLockError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise ProtocolLockError(code, message)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_bindings(protocol: Mapping[str, Any], root: Path) -> None:
    bindings = protocol.get("bindings")
    require(isinstance(bindings, list) and len(bindings) == len(EXPECTED_BINDINGS), "R11_BINDING_COUNT", "binding count drift")
    seen: set[str] = set()
    for row in bindings:
        role = row.get("role")
        relative = row.get("path")
        require(
            set(row) == {"role", "path", "bytes", "sha256"}
            and isinstance(role, str)
            and role not in seen
            and EXPECTED_BINDINGS.get(role) == relative,
            "R11_BINDING_ROLE",
            "binding role or path drift",
        )
        seen.add(role)
        path = root / str(relative)
        require(
            path.is_file()
            and path.stat().st_size == row["bytes"]
            and development_replay.sha256_file(path) == row["sha256"],
            "R11_BINDING_HASH",
            f"binding hash drift: {relative}",
        )
    require(seen == set(EXPECTED_BINDINGS), "R11_BINDING_SET", "binding role set drift")


def validate_protocol(value: Mapping[str, Any], *, repo_root: Path, recompute_pool: bool = True) -> dict[str, Any]:
    root = repo_root.resolve()
    protocol = copy.deepcopy(dict(value))
    claimed = protocol.pop("content_sha256", None)
    require(
        protocol.get("schema") == SCHEMA
        and protocol.get("lock_id") == LOCK_ID
        and protocol.get("status") == STATUS
        and protocol.get("scientific_outcome") == "NOT_RUN"
        and claimed == adapter.canonical_sha256(protocol),
        "R11_PROTOCOL_IDENTITY",
        "protocol identity, status, outcome, or content seal drift",
    )
    protocol["content_sha256"] = claimed
    _validate_bindings(protocol, root)

    predecessors = protocol.get("predecessors", {})
    r10 = _read_json(root / EXPECTED_BINDINGS["R10_FORMAL_RESULT"])
    require(
        r10.get("terminal") == development_replay.EXPECTED_R10_TERMINAL
        and r10.get("passed") is False
        and r10.get("scientifically_evaluable") is False
        and r10.get("interpretation", {}).get("r10_consumed_and_not_retargetable") is True
        and predecessors.get("r10", {}).get("terminal") == r10["terminal"]
        and predecessors.get("r10", {}).get("consumed") is True
        and predecessors.get("r10", {}).get("terminal_immutable") is True
        and predecessors.get("r10", {}).get("all_32_source_pool_parents_excluded_from_r11") is True,
        "R11_PREDECESSOR_TERMINAL_DRIFT",
        "R10 terminal, consumption, or full-pool exclusion drift",
    )
    development = development_replay.validate_development_result(_read_json(root / EXPECTED_BINDINGS["R11_DEVELOPMENT_RESULT"]))
    require(
        predecessors.get("r11_development", {}).get("result_content_sha256") == development["content_sha256"]
        and predecessors.get("r11_development", {}).get("role") == "TUNED_ON_CONSUMED_R10_DEVELOPMENT_ONLY"
        and predecessors.get("r11_development", {}).get("fresh_confirmation_authority") is False,
        "R11_DEVELOPMENT_LINEAGE_DRIFT",
        "R11 development result lineage or authority drift",
    )
    r9 = _read_json(root / EXPECTED_BINDINGS["R9_DEVELOPMENT_RESULT"])
    r9_protocol = predecessors.get("r9_selector", {})
    require(
        r9.get("selector", {}).get("selector_id") == r9_fit.SELECTOR_ID
        and r9.get("selector", {}).get("selector_sha256") == r9_protocol.get("selector_content_sha256")
        and r9.get("selector", {}).get("rule_id") == r9_protocol.get("rule_id")
        and r9_protocol.get("use") == "PARENT_RANKING_ONLY"
        and r9_protocol.get("query_label_or_abstention_authority") is False,
        "R11_R9_SELECTOR_DRIFT",
        "R9 selector identity or parent-only authority drift",
    )
    require(protocol.get("formation") == EXPECTED_FORMATION, "R11_FORMATION_ROLE_DRIFT", "development formation role or search boundary drift")
    require(protocol.get("candidate") == abstention_candidate.FROZEN_ALGORITHM, "R11_MARGIN_RULE_DRIFT", "candidate margin rule drift")

    pool = fresh_pool.build_pool(root) if recompute_pool else None
    frontdoor = protocol.get("fresh_frontdoor", {})
    require(
        frontdoor.get("official_fold") == "Training"
        and frontdoor.get("exclusion_snapshot_commit") == fresh_pool.EXCLUSION_COMMIT
        and frontdoor.get("snapshot_excluded_identity_count") == fresh_pool.EXPECTED_SNAPSHOT_EXCLUDED_COUNT
        and frontdoor.get("r10_source_pool_parent_count_explicitly_excluded") == 32
        and frontdoor.get("union_excluded_identity_count") == fresh_pool.EXPECTED_UNION_EXCLUDED_COUNT
        and frontdoor.get("union_excluded_identities_sha256") == fresh_pool.EXPECTED_UNION_EXCLUDED_SHA256
        and frontdoor.get("eligible_metadata_row_count") == fresh_pool.EXPECTED_ELIGIBLE_COUNT
        and frontdoor.get("pool_parent_count") == fresh_pool.PARENT_COUNT
        and frontdoor.get("selected_parent_count") == fresh_pool.SELECTED_PARENT_COUNT
        and frontdoor.get("selection", {}).get("selector_id") == r9_fit.SELECTOR_ID
        and frontdoor.get("selection", {}).get("rule_id") == r9_protocol.get("rule_id")
        and frontdoor.get("selection", {}).get("use") == "PARENT_RANKING_ONLY"
        and frontdoor.get("no_replacement_after_head_or_source_or_outcome") is True
        and frontdoor.get("separate_exact_data_use_authorization_required") is True
        and frontdoor.get("authorization_implied_by_protocol") is False,
        "R11_FRESH_FRONTDOOR_DRIFT",
        "fresh roster, exclusion, ranking, or authorization boundary drift",
    )
    if pool is not None:
        require(
            frontdoor.get("pool_content_sha256") == pool["pool_content_sha256"]
            and frontdoor.get("request_count") == pool["request_plan"]["request_count"]
            and frontdoor.get("request_plan_sha256") == pool["request_plan"]["expanded_requests_sha256"],
            "R11_FRESH_POOL_RECOMPUTE_DRIFT",
            "fresh pool or request plan recomputation drift",
        )

    require(protocol.get("phase_order") == EXPECTED_PHASE_ORDER, "R11_PHASE_ORDER_DRIFT", "phase order drift")
    require(protocol.get("phase_firewall") == EXPECTED_FIREWALL, "R11_PHASE_FIREWALL_DRIFT", "source/FARO firewall drift")
    require(protocol.get("dual_class_evaluability") == EXPECTED_EVALUABILITY, "R11_DUAL_CLASS_GATE_DRIFT", "dual-class evaluability drift")
    require(protocol.get("confirmation_gates") == EXPECTED_GATES, "R11_CONFIRMATION_GATE_DRIFT", "confirmation gate drift")
    require(protocol.get("abstention_effect_reporting") == EXPECTED_EFFECT_REPORTING, "R11_EFFECT_CLAIM_DRIFT", "abstention effect reporting boundary drift")
    require(protocol.get("terminal_precedence") == EXPECTED_TERMINALS, "R11_TERMINAL_PRECEDENCE_DRIFT", "terminal precedence drift")
    require(protocol.get("resource_ceiling") == EXPECTED_RESOURCE_CEILING, "R11_RESOURCE_CEILING_DRIFT", "resource ceiling drift")
    require(protocol.get("execution_authority") == EXPECTED_AUTHORITY, "R11_EXECUTION_AUTHORITY_DRIFT", "protocol granted execution or promotion authority")
    require(
        protocol.get("clear_frame_definition")
        == "A physical frame with at least one definite CLEAR query is one clear frame; it succeeds only when no definite CLEAR query in that frame is predicted OCCUPIED."
        and isinstance(protocol.get("forbidden"), list)
        and len(protocol["forbidden"]) == 8
        and "count UNKNOWN as negative" in protocol["forbidden"]
        and protocol.get("claim_ceiling")
        == "Non-executing R11 WILD_LAB protocol and R10-tuned source-only candidate only; no fresh result, route promotion, deployment, device, product, or safety claim.",
        "R11_CLAIM_CEILING_DRIFT",
        "cluster guard, forbidden actions, or claim ceiling drift",
    )
    return protocol


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    validate_protocol(_read_json(root / PROTOCOL_RELATIVE), repo_root=root)
    print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
