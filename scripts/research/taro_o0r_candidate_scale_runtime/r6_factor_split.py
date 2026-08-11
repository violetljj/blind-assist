"""Roster-independent deterministic compositor for frozen TARO R6 factor ownership."""

from __future__ import annotations

import copy
import re
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.research.taro_o0r_candidate_scale_runtime import r5_confirmation as r5
from scripts.research.taro_o0r_candidate_scale_runtime import source_factor
from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter


POLICY_ID = "R5_SELECTED_SUPPORT_BOUNDARY_PLUS_ALWAYS_R1_QUERY_CLEARANCE_V1"
PROTOCOL_LOCK_SHA256 = "5F2802F2585861F4D2D1EB002D1AFA7050278CBD33732F665DB6AF9CA32A101C"
COMPONENT_SCHEMA = "blindassist.taro.o0r.r6_factor_components.v1"
COMPOSITE_SCHEMA = "blindassist.taro.o0r.r6_factor_split_query.v1"
SUMMARY_SCHEMA = "blindassist.taro.o0r.r6_factor_split_summary.v1"
FORMATION_REPLAY = "FORMATION_REPLAY"
UNTOUCHED_CONFIRMATION = "UNTOUCHED_CONFIRMATION"
_ROLES = {FORMATION_REPLAY, UNTOUCHED_CONFIRMATION}
_SHA256 = re.compile(r"^[0-9A-F]{64}$")

FORBIDDEN_FORMATION_PARENTS = frozenset(
    {
        "410690", "411257", "411536", "414297", "415639", "421113", "424461", "433755",
        "442420", "450420", "452612", "456632", "466965", "467246", "471342", "472345",
        "423614", "438794", "467346", "467370", "469216", "469460", "470974", "472473",
    }
)


class R6FactorSplitError(RuntimeError):
    def __init__(self, code: str, message: str, **context: Any) -> None:
        super().__init__(message)
        self.code = code
        self.context = context


def require(condition: bool, code: str, message: str, **context: Any) -> None:
    if not condition:
        raise R6FactorSplitError(code, message, **context)


def _hash(value: Any, field: str) -> str:
    require(isinstance(value, str) and bool(_SHA256.fullmatch(value)), "R6_HASH_INVALID", "R6 SHA-256 binding is malformed", field=field)
    return value


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(dict(value))
    require("content_sha256" not in record, "R6_SEAL_COLLISION", "caller supplied R6 content hash")
    record["content_sha256"] = adapter.canonical_sha256(record)
    return record


def _validate_seal(value: Any, schema: str) -> dict[str, Any]:
    require(isinstance(value, dict), "R6_RECORD_INVALID", "R6 sealed record must be an object")
    record = copy.deepcopy(value)
    observed = record.pop("content_sha256", None)
    require(isinstance(observed, str) and bool(_SHA256.fullmatch(observed)) and adapter.canonical_sha256(record) == observed, "R6_SEAL_MISMATCH", "R6 record content hash drift", schema=schema)
    record["content_sha256"] = observed
    require(record.get("schema") == schema, "R6_SCHEMA_DRIFT", "R6 record schema drift", expected=schema)
    return record


def _identity(parent_id: Any, physical_frame_id: Any, query_id: Any, grid_index: Any) -> tuple[str, str, str, int]:
    require(isinstance(parent_id, str) and bool(parent_id), "R6_IDENTITY_INVALID", "R6 parent identity is missing")
    require(isinstance(physical_frame_id, str) and ":" in physical_frame_id, "R6_IDENTITY_INVALID", "R6 physical frame identity is malformed")
    require(isinstance(query_id, str) and query_id.startswith(f"{physical_frame_id}:"), "R6_IDENTITY_INVALID", "R6 query identity is malformed")
    require(isinstance(grid_index, int) and not isinstance(grid_index, bool) and 0 <= grid_index < 9, "R6_IDENTITY_INVALID", "R6 query grid index is invalid")
    return parent_id, physical_frame_id, query_id, grid_index


def build_factor_components(
    *,
    analysis_role: str,
    parent_id: str,
    physical_frame_id: str,
    query_id: str,
    grid_index: int,
    source_frame_receipt_sha256: str,
    candidate_frame_record_sha256: str,
    r6_phase_a_policy_seal_sha256: str,
    query_receipt_sha256: str,
    truth_scoring_record_sha256: str,
    source_support_available: bool,
    phase_a_selected_branch: str,
    baseline: Mapping[str, Any],
    selected_support_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    _identity(parent_id, physical_frame_id, query_id, grid_index)
    require(analysis_role in _ROLES, "R6_ANALYSIS_ROLE_INVALID", "R6 analysis role is invalid")
    for field, value in (
        ("source_frame_receipt_sha256", source_frame_receipt_sha256),
        ("candidate_frame_record_sha256", candidate_frame_record_sha256),
        ("r6_phase_a_policy_seal_sha256", r6_phase_a_policy_seal_sha256),
        ("query_receipt_sha256", query_receipt_sha256),
        ("truth_scoring_record_sha256", truth_scoring_record_sha256),
    ):
        _hash(value, field)
    require(isinstance(source_support_available, bool), "R6_BRANCH_INVALID", "R6 source support availability is not boolean")
    expected_branch = "DIRECT_APPLE_SUPPORT" if source_support_available else "R1_BASELINE"
    require(phase_a_selected_branch == expected_branch, "R6_BRANCH_INVALID", "R6 support/boundary branch differs from source-only Phase A")
    baseline_mode = r5._validate_mode_result(dict(baseline), "r6.baseline")
    selected_mode = r5._validate_mode_result(dict(selected_support_boundary), "r6.selected_support_boundary")
    if expected_branch == "R1_BASELINE":
        require(adapter.canonical_sha256(baseline_mode) == adapter.canonical_sha256(selected_mode), "R6_BASELINE_BRANCH_DRIFT", "R6 baseline branch does not copy the baseline component")
    confirmation_eligible = analysis_role == UNTOUCHED_CONFIRMATION
    if confirmation_eligible:
        require(parent_id not in FORBIDDEN_FORMATION_PARENTS, "R6_CONFIRMATION_PARENT_OVERLAP", "R6 untouched confirmation parent overlaps formation data", parent_id=parent_id)
    record = _seal(
        {
            "schema": COMPONENT_SCHEMA,
            "policy_id": POLICY_ID,
            "protocol_lock_sha256": PROTOCOL_LOCK_SHA256,
            "analysis_role": analysis_role,
            "confirmation_eligible": confirmation_eligible,
            "parent_id": parent_id,
            "physical_frame_id": physical_frame_id,
            "query_id": query_id,
            "grid_index": grid_index,
            "source_frame_receipt_sha256": source_frame_receipt_sha256,
            "candidate_frame_record_sha256": candidate_frame_record_sha256,
            "r6_phase_a_policy_seal_sha256": r6_phase_a_policy_seal_sha256,
            "query_receipt_sha256": query_receipt_sha256,
            "truth_scoring_record_sha256": truth_scoring_record_sha256,
            "source_support_available": source_support_available,
            "phase_a_selected_branch": phase_a_selected_branch,
            "support_boundary_owner": expected_branch,
            "query_clearance_owner": "R1_BASELINE",
            "baseline": baseline_mode,
            "selected_support_boundary": selected_mode,
            "baseline_component_sha256": adapter.canonical_sha256(baseline_mode),
            "selected_support_boundary_component_sha256": adapter.canonical_sha256(selected_mode),
            "owner_selection_fields_read": {"SUPPORT_BOUNDARY": ["source_support_available"], "QUERY_CLEARANCE": []},
            "forbidden_selection_reads_confirmed_absent": True,
            "branch_reselection_after_truth": False,
        }
    )
    return validate_factor_components(record)


def factor_components_from_r5_query_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt consumed R5 evidence for formation replay only, never confirmation."""

    row = r5.validate_query_record(dict(value))
    return build_factor_components(
        analysis_role=FORMATION_REPLAY,
        parent_id=row["parent_id"],
        physical_frame_id=row["physical_frame_id"],
        query_id=row["query_id"],
        grid_index=row["grid_index"],
        source_frame_receipt_sha256=row["source_frame_receipt_sha256"],
        candidate_frame_record_sha256=row["candidate_frame_record_sha256"],
        r6_phase_a_policy_seal_sha256=row["source_decision_sha256"],
        query_receipt_sha256=row["query_receipt_sha256"],
        truth_scoring_record_sha256=row["content_sha256"],
        source_support_available=row["source_support_available"],
        phase_a_selected_branch=row["phase_a_selected_branch"],
        baseline=row["baseline"],
        selected_support_boundary=row["selected_hybrid"],
    )


def validate_factor_components(value: Any) -> dict[str, Any]:
    record = _validate_seal(value, COMPONENT_SCHEMA)
    expected = {
        "schema", "policy_id", "protocol_lock_sha256", "analysis_role", "confirmation_eligible",
        "parent_id", "physical_frame_id", "query_id", "grid_index", "source_frame_receipt_sha256",
        "candidate_frame_record_sha256", "r6_phase_a_policy_seal_sha256", "query_receipt_sha256",
        "truth_scoring_record_sha256", "source_support_available", "phase_a_selected_branch",
        "support_boundary_owner", "query_clearance_owner", "baseline", "selected_support_boundary",
        "baseline_component_sha256", "selected_support_boundary_component_sha256",
        "owner_selection_fields_read", "forbidden_selection_reads_confirmed_absent",
        "branch_reselection_after_truth", "content_sha256",
    }
    require(set(record) == expected, "R6_COMPONENT_KEY_SET", "R6 factor component fields drift")
    require(record["policy_id"] == POLICY_ID and record["protocol_lock_sha256"] == PROTOCOL_LOCK_SHA256, "R6_POLICY_DRIFT", "R6 policy/protocol identity drift")
    _identity(record["parent_id"], record["physical_frame_id"], record["query_id"], record["grid_index"])
    require(record["analysis_role"] in _ROLES and record["confirmation_eligible"] == (record["analysis_role"] == UNTOUCHED_CONFIRMATION), "R6_ANALYSIS_ROLE_INVALID", "R6 analysis role/eligibility disagree")
    if record["confirmation_eligible"]:
        require(record["parent_id"] not in FORBIDDEN_FORMATION_PARENTS, "R6_CONFIRMATION_PARENT_OVERLAP", "R6 confirmation parent overlaps formation data")
    for field in (
        "source_frame_receipt_sha256", "candidate_frame_record_sha256", "r6_phase_a_policy_seal_sha256",
        "query_receipt_sha256", "truth_scoring_record_sha256", "baseline_component_sha256",
        "selected_support_boundary_component_sha256",
    ):
        _hash(record[field], field)
    require(isinstance(record["source_support_available"], bool), "R6_BRANCH_INVALID", "R6 source support availability drift")
    expected_branch = "DIRECT_APPLE_SUPPORT" if record["source_support_available"] else "R1_BASELINE"
    require(record["phase_a_selected_branch"] == record["support_boundary_owner"] == expected_branch and record["query_clearance_owner"] == "R1_BASELINE", "R6_BRANCH_INVALID", "R6 factor owner drift")
    baseline = r5._validate_mode_result(record["baseline"], "r6.baseline")
    selected = r5._validate_mode_result(record["selected_support_boundary"], "r6.selected_support_boundary")
    require(adapter.canonical_sha256(baseline) == record["baseline_component_sha256"] and adapter.canonical_sha256(selected) == record["selected_support_boundary_component_sha256"], "R6_COMPONENT_HASH_DRIFT", "R6 component hash drift")
    if expected_branch == "R1_BASELINE":
        require(record["baseline_component_sha256"] == record["selected_support_boundary_component_sha256"], "R6_BASELINE_BRANCH_DRIFT", "R6 baseline branch component drift")
    require(record["owner_selection_fields_read"] == {"SUPPORT_BOUNDARY": ["source_support_available"], "QUERY_CLEARANCE": []}, "R6_SELECTION_FIELD_DRIFT", "R6 selection fields drift")
    require(record["forbidden_selection_reads_confirmed_absent"] is True and record["branch_reselection_after_truth"] is False, "R6_SELECTION_FIREWALL_DRIFT", "R6 outcome selection firewall drift")
    return record


def build_composite_query(value: Mapping[str, Any]) -> dict[str, Any]:
    components = validate_factor_components(dict(value))
    selected = components["selected_support_boundary"]
    baseline = components["baseline"]
    support = copy.deepcopy(selected["support"])
    boundary = copy.deepcopy(selected["boundary"])
    query = copy.deepcopy(baseline["query_point_clearance"])
    record = _seal(
        {
            "schema": COMPOSITE_SCHEMA,
            "policy_id": POLICY_ID,
            "protocol_lock_sha256": PROTOCOL_LOCK_SHA256,
            "analysis_role": components["analysis_role"],
            "confirmation_eligible": components["confirmation_eligible"],
            "parent_id": components["parent_id"],
            "physical_frame_id": components["physical_frame_id"],
            "query_id": components["query_id"],
            "grid_index": components["grid_index"],
            "factor_components_sha256": components["content_sha256"],
            "source_frame_receipt_sha256": components["source_frame_receipt_sha256"],
            "r6_phase_a_policy_seal_sha256": components["r6_phase_a_policy_seal_sha256"],
            "query_receipt_sha256": components["query_receipt_sha256"],
            "factor_owners": {"SUPPORT": components["support_boundary_owner"], "BOUNDARY": components["support_boundary_owner"], "QUERY_CLEARANCE": "R1_BASELINE"},
            "factor_depth_sha256": {"SUPPORT": selected["depth_array_sha256"], "BOUNDARY": selected["depth_array_sha256"], "QUERY_CLEARANCE": baseline["depth_array_sha256"]},
            "support_boundary_extraction": {"evaluable": selected["extraction_evaluable"], "reason_codes": copy.deepcopy(selected["reason_codes"])},
            "support": support,
            "boundary": boundary,
            "query_clearance": query,
            "support_block_sha256": adapter.canonical_sha256(support),
            "boundary_block_sha256": adapter.canonical_sha256(boundary),
            "query_clearance_block_sha256": adapter.canonical_sha256(query),
            "deterministic_exact_copy": True,
            "outcome_dependent_reselection": False,
        }
    )
    return validate_composite_query(record, factor_components=components)


def validate_composite_query(value: Any, *, factor_components: Mapping[str, Any] | None = None) -> dict[str, Any]:
    record = _validate_seal(value, COMPOSITE_SCHEMA)
    expected = {
        "schema", "policy_id", "protocol_lock_sha256", "analysis_role", "confirmation_eligible", "parent_id",
        "physical_frame_id", "query_id", "grid_index", "factor_components_sha256", "source_frame_receipt_sha256",
        "r6_phase_a_policy_seal_sha256", "query_receipt_sha256", "factor_owners", "factor_depth_sha256",
        "support_boundary_extraction", "support", "boundary", "query_clearance", "support_block_sha256",
        "boundary_block_sha256", "query_clearance_block_sha256", "deterministic_exact_copy",
        "outcome_dependent_reselection", "content_sha256",
    }
    require(set(record) == expected, "R6_COMPOSITE_KEY_SET", "R6 composite fields drift")
    require(record["policy_id"] == POLICY_ID and record["protocol_lock_sha256"] == PROTOCOL_LOCK_SHA256, "R6_POLICY_DRIFT", "R6 composite policy drift")
    _identity(record["parent_id"], record["physical_frame_id"], record["query_id"], record["grid_index"])
    require(record["analysis_role"] in _ROLES and record["confirmation_eligible"] == (record["analysis_role"] == UNTOUCHED_CONFIRMATION), "R6_ANALYSIS_ROLE_INVALID", "R6 composite role drift")
    for field in ("factor_components_sha256", "source_frame_receipt_sha256", "r6_phase_a_policy_seal_sha256", "query_receipt_sha256", "support_block_sha256", "boundary_block_sha256", "query_clearance_block_sha256"):
        _hash(record[field], field)
    require(set(record["factor_owners"]) == {"SUPPORT", "BOUNDARY", "QUERY_CLEARANCE"} and record["factor_owners"]["QUERY_CLEARANCE"] == "R1_BASELINE", "R6_FACTOR_OWNER_DRIFT", "R6 composite factor owners drift")
    require(set(record["factor_depth_sha256"]) == {"SUPPORT", "BOUNDARY", "QUERY_CLEARANCE"}, "R6_FACTOR_DEPTH_LINEAGE_DRIFT", "R6 factor depth lineage fields drift")
    for field, value in record["factor_depth_sha256"].items():
        _hash(value, f"factor_depth_sha256.{field}")
    require(record["support_block_sha256"] == adapter.canonical_sha256(record["support"]) and record["boundary_block_sha256"] == adapter.canonical_sha256(record["boundary"]) and record["query_clearance_block_sha256"] == adapter.canonical_sha256(record["query_clearance"]), "R6_FACTOR_BLOCK_HASH_DRIFT", "R6 factor block hash drift")
    require(record["deterministic_exact_copy"] is True and record["outcome_dependent_reselection"] is False, "R6_COMPOSITOR_FIREWALL_DRIFT", "R6 compositor firewall drift")
    if factor_components is not None:
        components = validate_factor_components(dict(factor_components))
        require(record["factor_components_sha256"] == components["content_sha256"], "R6_COMPONENT_BINDING_DRIFT", "R6 composite does not bind supplied components")
        selected = components["selected_support_boundary"]
        baseline = components["baseline"]
        require(
            record["support"] == selected["support"]
            and record["boundary"] == selected["boundary"]
            and record["query_clearance"] == baseline["query_point_clearance"]
            and record["factor_depth_sha256"] == {"SUPPORT": selected["depth_array_sha256"], "BOUNDARY": selected["depth_array_sha256"], "QUERY_CLEARANCE": baseline["depth_array_sha256"]},
            "R6_EXACT_COPY_DRIFT",
            "R6 composite factor blocks do not exact-copy their frozen owners",
        )
    return record


def _parent_macro(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    by_frame: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = row["effects"].get(field)
        if r5._finite(value):
            by_frame.setdefault((row["parent_id"], row["physical_frame_id"]), []).append(float(value))
    by_parent: dict[str, list[float]] = {}
    for (parent, _), values in by_frame.items():
        by_parent.setdefault(parent, []).append(float(np.median(np.asarray(values, dtype=np.float64))))
    parent_values = []
    for parent in sorted({row["parent_id"] for row in rows}):
        values = by_parent.get(parent, [])
        parent_values.append({"parent_id": parent, "paired_frame_count": len(values), "median_frame_effect": float(np.median(np.asarray(values, dtype=np.float64))) if values else None})
    usable = [row["median_frame_effect"] for row in parent_values if row["median_frame_effect"] is not None]
    return {"parents_with_metric": len(usable), "median_of_parent_medians": float(np.median(np.asarray(usable, dtype=np.float64))) if usable else None, "parent_values": parent_values}


def summarize_factor_split_pairs(
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    analysis_role: str,
    expected_parent_frame_counts: Mapping[str, int],
) -> dict[str, Any]:
    require(analysis_role in _ROLES, "R6_ANALYSIS_ROLE_INVALID", "R6 summary role is invalid")
    validated: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for components_value, composite_value in pairs:
        components = validate_factor_components(dict(components_value))
        composite = validate_composite_query(dict(composite_value), factor_components=components)
        require(components["analysis_role"] == composite["analysis_role"] == analysis_role, "R6_SUMMARY_ROLE_DRIFT", "R6 summary mixes analysis roles")
        validated.append((components, composite))
    expected_counts = dict(expected_parent_frame_counts)
    require(
        bool(expected_counts)
        and all(isinstance(parent, str) and parent and isinstance(count, int) and not isinstance(count, bool) and count > 0 for parent, count in expected_counts.items()),
        "R6_SUMMARY_EXPECTED_COHORT_INVALID",
        "R6 expected parent/frame cohort is invalid",
    )
    if analysis_role == UNTOUCHED_CONFIRMATION:
        require(len(expected_counts) >= 8, "R6_CONFIRMATION_PARENT_FLOOR", "R6 untouched confirmation requires at least eight parents")
        require(not (set(expected_counts) & FORBIDDEN_FORMATION_PARENTS), "R6_CONFIRMATION_PARENT_OVERLAP", "R6 expected confirmation cohort overlaps formation data")
    expected_frames = sum(expected_counts.values())
    expected_queries = expected_frames * 9
    keys = {(row[1]["physical_frame_id"], row[1]["query_id"]) for row in validated}
    frames = {row[1]["physical_frame_id"] for row in validated}
    parents = {row[1]["parent_id"] for row in validated}
    require(len(validated) == len(keys) == expected_queries and len(frames) == expected_frames and parents == set(expected_counts), "R6_SUMMARY_COHORT_DRIFT", "R6 summary cohort differs")
    observed = Counter(row[1]["parent_id"] for row in validated)
    require(all(observed[parent] == count * 9 for parent, count in expected_counts.items()), "R6_SUMMARY_PARENT_COUNT_DRIFT", "R6 summary parent counts differ")
    by_frame: dict[str, list[int]] = {}
    for _, composite in validated:
        by_frame.setdefault(composite["physical_frame_id"], []).append(composite["grid_index"])
    require(all(sorted(indices) == list(range(9)) for indices in by_frame.values()), "R6_SUMMARY_GRID_DRIFT", "R6 summary frame grid differs")

    effect_rows = []
    baseline_extraction = composite_extraction = baseline_known = composite_known = 0
    baseline_boundary = composite_boundary = 0
    for components, composite in validated:
        baseline = components["baseline"]
        selected = components["selected_support_boundary"]
        baseline_extraction += bool(baseline["extraction_evaluable"])
        composite_extraction += bool(composite["support_boundary_extraction"]["evaluable"])
        baseline_known += bool(baseline["query_point_clearance"]["evaluable"])
        composite_known += bool(composite["query_clearance"]["evaluable"])
        baseline_boundary += bool(baseline["boundary"]["evaluable"])
        composite_boundary += bool(composite["boundary"]["evaluable"])
        effect_rows.append(
            {
                "parent_id": composite["parent_id"],
                "physical_frame_id": composite["physical_frame_id"],
                "effects": {
                    "height_error_reduction_vs_baseline_m": source_factor._difference(baseline, selected, "support", "height_abs_error_m"),
                    "normal_error_reduction_vs_baseline_rad": source_factor._difference(baseline, selected, "support", "normal_angular_error_rad"),
                },
            }
        )
    height = _parent_macro(effect_rows, "height_error_reduction_vs_baseline_m")
    normal = _parent_macro(effect_rows, "normal_error_reduction_vs_baseline_rad")
    height_parent = {row["parent_id"]: row["median_frame_effect"] for row in height["parent_values"]}
    normal_parent = {row["parent_id"]: row["median_frame_effect"] for row in normal["parent_values"]}
    jointly_positive = sum(r5._finite(height_parent[parent]) and float(height_parent[parent]) > 0.0 and r5._finite(normal_parent[parent]) and float(normal_parent[parent]) > 0.0 for parent in parents)
    denominator_defined = height["parents_with_metric"] == normal["parents_with_metric"] == len(expected_counts)
    gate_values = [
        ("EXACT_COHORT_AND_LINEAGE", True),
        ("PHASE_FIREWALL", all(components["forbidden_selection_reads_confirmed_absent"] and not components["branch_reselection_after_truth"] for components, _ in validated)),
        ("PARENT_METRIC_DENOMINATORS", denominator_defined),
        ("HEIGHT_PARENT_MACRO_POSITIVE", denominator_defined and float(height["median_of_parent_medians"]) > 0.0),
        ("NORMAL_PARENT_MACRO_POSITIVE", denominator_defined and float(normal["median_of_parent_medians"]) > 0.0),
        ("ALL_PARENTS_JOINTLY_POSITIVE", jointly_positive == len(expected_counts)),
        ("EXTRACTION_COVERAGE_NO_REGRET", composite_extraction >= baseline_extraction),
        ("BOUNDARY_EVALUABILITY_NO_REGRET", composite_boundary >= baseline_boundary),
        ("QUERY_KNOWN_COVERAGE_NO_REGRET", composite_known >= baseline_known),
    ]
    common = {
        "schema": SUMMARY_SCHEMA,
        "policy_id": POLICY_ID,
        "protocol_lock_sha256": PROTOCOL_LOCK_SHA256,
        "analysis_role": analysis_role,
        "parent_count": len(expected_counts),
        "physical_frame_count": expected_frames,
        "query_record_count": expected_queries,
        "baseline_extraction_evaluable_query_count": baseline_extraction,
        "composite_extraction_evaluable_query_count": composite_extraction,
        "baseline_boundary_evaluable_query_count": baseline_boundary,
        "composite_boundary_evaluable_query_count": composite_boundary,
        "baseline_query_known_count": baseline_known,
        "composite_query_known_count": composite_known,
        "height_error_reduction_vs_baseline_parent_macro_m": height,
        "normal_error_reduction_vs_baseline_parent_macro_rad": normal,
        "parents_jointly_positive_height_and_normal": jointly_positive,
    }
    if analysis_role == FORMATION_REPLAY:
        common.update(
            {
                "post_hoc": True,
                "promotion_allowed": False,
                "requires_untouched_confirmation": True,
                "gate_landscape": [{"id": gate, "would_pass": passed} for gate, passed in gate_values],
                "all_gate_landscape_would_pass": all(passed for _, passed in gate_values),
                "pass_fail_terminal_absent": True,
            }
        )
    else:
        if not denominator_defined:
            terminal = "TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_CONFIRMATION_NOT_EVALUABLE"
        elif all(passed for _, passed in gate_values):
            terminal = "TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_CONFIRMATION_PASS"
        else:
            terminal = "TARO_O0R_R6_FACTOR_SPLIT_UNTOUCHED_CONFIRMATION_FAIL"
        common.update({"post_hoc": False, "promotion_allowed": terminal.endswith("_PASS"), "gates": [{"id": gate, "passed": passed} for gate, passed in gate_values], "terminal": terminal, "passed": terminal.endswith("_PASS")})
    return _seal(common)


__all__ = [
    "COMPONENT_SCHEMA", "COMPOSITE_SCHEMA", "FORMATION_REPLAY", "FORBIDDEN_FORMATION_PARENTS",
    "POLICY_ID", "PROTOCOL_LOCK_SHA256", "R6FactorSplitError", "SUMMARY_SCHEMA",
    "UNTOUCHED_CONFIRMATION", "build_composite_query", "build_factor_components",
    "factor_components_from_r5_query_record", "summarize_factor_split_pairs",
    "validate_composite_query", "validate_factor_components",
]
