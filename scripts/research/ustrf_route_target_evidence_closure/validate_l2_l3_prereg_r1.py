from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


L2_CONFIG = Path("configs/ustrf_route_target_l2_fresh_selection_prereg_r1.json")
L3_TEMPLATE = Path("configs/ustrf_route_target_l3_confirmation_lockbox_template_r1.json")
L2_SCHEMA = Path("schemas/ustrf_route_target_l2_fresh_selection_prereg_r1.schema.json")
L3_SCHEMA = Path("schemas/ustrf_route_target_l3_confirmation_lockbox_template_r1.schema.json")

REQUIRED_METRICS = [
    "event_recall",
    "critical_miss",
    "clearance",
    "repeat_within_observation",
    "event_regeneration_after_clear",
    "false_alerts_per_minute",
    "evidence_age",
    "unknown_or_stale_active_alert",
]

CANDIDATE_ORDER = [
    "C1_CAUSAL_ROUTE_RELATION_FSM",
    "C2_ROUTE_OCCUPANCY_EPISODE_FSM",
    "C3_DUAL_KEY_CLEARANCE_FSM",
]

L2_TOTAL_FLOORS = {
    "event_recall_events": 20,
    "critical_events": 5,
    "clearance_events": 15,
    "complete_repeat_events": 15,
    "complete_regeneration_intervals": 15,
    "negative_exposure_minutes": 20.0,
    "matched_pairs_for_relative_claim": 10,
}

L2_PER_FAMILY_FLOORS = {
    "event_recall_events": 5,
    "critical_events": 1,
    "clearance_events": 3,
    "complete_repeat_events": 3,
    "complete_regeneration_intervals": 3,
    "negative_exposure_minutes": 5.0,
}

L2_PERFORMANCE_GATES = {
    "event_recall": {
        "field": "event_recall",
        "operator": ">=",
        "threshold": 0.9,
    },
    "critical_miss": {
        "field": "critical_miss_count",
        "operator": "=",
        "threshold": 0,
    },
    "clearance_rate": {
        "field": "clearance_rate",
        "operator": ">=",
        "threshold": 0.9,
    },
    "clearance_p95_ms": {
        "field": "clearance_p95_ms",
        "operator": "<=",
        "threshold": 1500.0,
    },
    "repeat_within_observation": {
        "field": "repeat_alert_count",
        "operator": "=",
        "threshold": 0,
    },
    "event_regeneration_after_clear": {
        "field": "event_regeneration_count",
        "operator": "=",
        "threshold": 0,
    },
    "false_alerts_per_minute": {
        "field": "false_alerts_per_minute",
        "operator": "<=",
        "threshold": 0.5,
    },
    "evidence_age": {
        "field": "evidence_age_p95_ms",
        "operator": "<=",
        "threshold": 200.0,
    },
    "unknown_or_stale_active_alert": {
        "field": "active_alert_on_unknown_or_stale_route_frames",
        "operator": "=",
        "threshold": 0,
    },
}

L2_TIE_BREAK = [
    {
        "metric": "false_alerts_per_minute",
        "field": "false_alerts_per_minute",
        "direction": "minimize",
    },
    {
        "metric": "clearance",
        "field": "clearance_rate",
        "direction": "maximize",
    },
    {
        "metric": "clearance",
        "field": "clearance_p95_ms",
        "direction": "minimize",
    },
    {
        "metric": "evidence_age",
        "field": "evidence_age_p95_ms",
        "direction": "minimize",
    },
    {
        "metric": "candidate_order",
        "field": "candidate_order_index",
        "direction": "minimize",
    },
]

L2_HARD_VETOES = [
    "any_observed_critical_miss",
    "unknown_route_active_alert",
    "stale_route_active_alert",
    "unresolved_person_active_alert",
    "truth_identity_or_route_invariant_violation",
    "future_information_leakage",
    "canonical_input_or_hash_binding_drift",
    "candidate_implementation_hash_drift",
    "more_than_one_run_per_candidate",
    "selective_or_partial_replay",
]


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON root must be an object: {path}")
    return value


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _schema_errors(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _matches_type(value, expected_type):
        return [f"{path}: expected {expected_type}"]

    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: const mismatch")
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: enum mismatch")

    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        errors.append(f"{path}: string shorter than minLength")

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems")
        if schema.get("uniqueItems"):
            encoded = [json.dumps(item, sort_keys=True) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: items are not unique")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(_schema_errors(item, item_schema, f"{path}[{index}]"))

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required property {key}")
        if len(value) < schema.get("minProperties", 0):
            errors.append(f"{path}: fewer than minProperties")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: additional property {key}")
        for key, child_schema in properties.items():
            if key in value and isinstance(child_schema, dict):
                errors.extend(_schema_errors(value[key], child_schema, f"{path}.{key}"))
    return errors


def _validate_schema_document(schema: dict[str, Any], label: str) -> None:
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise RuntimeError(f"{label} schema draft drifted")
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise RuntimeError(f"{label} schema root is not fail-closed")
    if not isinstance(schema.get("$id"), str) or not schema["$id"].startswith(
        "blindassist://schemas/"
    ):
        raise RuntimeError(f"{label} schema id missing")


def _expect(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise RuntimeError(message)


def _assert_all_false(boundary: dict[str, Any], message: str) -> None:
    if not boundary or any(value is not False for value in boundary.values()):
        raise RuntimeError(message)


def validate_contracts(
    repo: Path,
    l2: dict[str, Any],
    l3: dict[str, Any],
    l2_schema: dict[str, Any] | None = None,
    l3_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    repo = repo.resolve()
    l2_schema = l2_schema or _load(repo / L2_SCHEMA)
    l3_schema = l3_schema or _load(repo / L3_SCHEMA)
    _validate_schema_document(l2_schema, "L2")
    _validate_schema_document(l3_schema, "L3")

    l2_errors = _schema_errors(l2, l2_schema)
    if l2_errors:
        raise RuntimeError(f"L2 schema validation failed: {l2_errors[0]}")
    l3_errors = _schema_errors(l3, l3_schema)
    if l3_errors:
        raise RuntimeError(f"L3 schema validation failed: {l3_errors[0]}")

    _expect(l2["required_metrics"], REQUIRED_METRICS, "L2 required metric roster drifted")
    _expect(
        l2["candidate_execution"]["candidate_order"],
        CANDIDATE_ORDER,
        "L2 candidate order drifted",
    )
    _expect(
        l2["candidate_execution"]["runs_per_candidate"],
        1,
        "L2 one-shot rule drifted",
    )
    _expect(
        l2["candidate_execution"]["candidate_id"],
        None,
        "L2 selected a candidate before execution",
    )
    _expect(
        l2["performance_gates"],
        L2_PERFORMANCE_GATES,
        "L2 performance gates drifted",
    )
    _expect(
        l2["support_floors"]["totals"],
        L2_TOTAL_FLOORS,
        "L2 total support floors drifted",
    )
    _expect(
        l2["support_floors"]["per_family"],
        L2_PER_FAMILY_FLOORS,
        "L2 per-family support floors drifted",
    )
    _expect(
        l2["support_floors"]["minimum_independent_session_families"],
        2,
        "L2 family floor drifted",
    )
    _expect(
        l2["support_floors"]["maximum_single_family_share"],
        0.7,
        "L2 family share cap drifted",
    )
    _expect(
        l2["source_acquisition_budget"],
        {
            "maximum_new_source_families": 2,
            "maximum_canaries_per_source": 2,
            "default_maximum_download_bytes": 2147483648,
            "stop_after_consecutive_ineligible_source_families": 2,
            "candidate_specific_source_search": False,
            "budget_override_requires_new_candidate_blind_preregistration": True,
            "download_or_materialization_performed_in_this_stage": False,
        },
        "L2 source acquisition budget drifted",
    )
    _expect(l2["hard_vetoes"], L2_HARD_VETOES, "L2 hard-veto roster drifted")
    _expect(
        l2["selection_rule"]["primary_metric"],
        {"metric": "event_recall", "field": "event_recall", "direction": "maximize"},
        "L2 primary metric drifted",
    )
    _expect(
        l2["selection_rule"]["tie_break_order"],
        L2_TIE_BREAK,
        "L2 tie-break order drifted",
    )
    _expect(
        l2["selection_rule"]["allowed_selection_decision"],
        "PROVISIONAL_SELECTION_FOR_FRESH_CONFIRMATION_ONLY",
        "L2 selection decision semantics drifted",
    )
    if (
        l2["dataset_role"] != "fresh_selection"
        or l2["role_isolation"]["allowed_roles"] != ["fresh_selection"]
        or not l2["role_isolation"]["r1_and_l1_opened_data_forbidden"]
        or not l2["role_isolation"]["selection_data_forbidden_from_future_confirmation"]
    ):
        raise RuntimeError("L2 fresh-selection role isolation drifted")
    if (
        l2["candidate_outputs_visible_at_freeze"]
        or not l2["frozen_before_any_new_c1_c3_output"]
        or l2["execution_authorized_now"]
        or l2["candidate_execution"]["candidate_execution_performed_in_this_stage"]
    ):
        raise RuntimeError("L2 pre-output freeze or current execution boundary opened")
    _assert_all_false(
        l2["current_stage_claim_boundary"],
        "L2 current-stage claim boundary opened",
    )

    _expect(l3["required_metrics"], REQUIRED_METRICS, "L3 required metric roster drifted")
    if l3["executable"] is not False:
        raise RuntimeError("L3 template became executable")
    if l3["candidate_id"] is not None:
        raise RuntimeError("L3 candidate_id was bound")
    if l3["dataset_role"] != "fresh_confirmation_lockbox":
        raise RuntimeError("L3 dataset role drifted")
    if not l3["role_isolation"]["strictly_disjoint_from_l2_selection_data"]:
        raise RuntimeError("L3/L2 data-role isolation opened")
    _expect(
        l3["lockbox_floors"],
        {
            "sessions": 6,
            "minimum_provenance_families": 2,
            "complete_positive_negative_matched_pairs": 60,
            "complete_repeat_events": 60,
            "complete_regeneration_intervals": 60,
            "minimum_scenario_strata": 5,
            "loso_folds": 6,
            "maximum_single_family_share": 0.6,
            "minimum_critical_events": 59,
            "each_required_metric_minimum_provenance_families": 2,
        },
        "L3 lockbox floors drifted",
    )
    if (
        not l3["pair_and_fold_contract"]["positive_and_matched_negative_remain_in_same_fold"]
        or not l3["pair_and_fold_contract"]["loso"]
        or not l3["pair_and_fold_contract"]["session_count_equals_fold_count"]
    ):
        raise RuntimeError("L3 matched-pair or LOSO contract drifted")
    stats = l3["statistical_contract"]
    if (
        stats["bootstrap_iterations"] != 10000
        or stats["bootstrap_seed"] != 20260723
        or stats["bootstrap_stratify_by"] != "provenance_family"
        or stats["bootstrap_resample_unit"] != "session_within_family"
        or stats["provenance_family_random_resampling_enabled"]
    ):
        raise RuntimeError("L3 bootstrap contract drifted")
    if (
        not stats["worst_family_sentinel_required"]
        or not stats["loso_worst_session_sentinel_required"]
    ):
        raise RuntimeError("L3 worst-family or LOSO sentinel disabled")
    gate = l3["generation_gate"]
    if (
        not gate["requires_independent_l2_pass"]
        or gate["required_l2_decision"]
        != l2["selection_rule"]["allowed_selection_decision"]
        or gate["current_l2_pass_available"]
        or not gate["executable_prereg_may_only_be_generated_after_future_l2_pass"]
        or not gate["template_itself_cannot_be_mutated_into_executable"]
        or not gate["candidate_id_may_only_be_bound_in_new_versioned_executable_prereg"]
    ):
        raise RuntimeError("L3 future executable-prereg generation gate drifted")
    _assert_all_false(
        l3["current_stage_claim_boundary"],
        "L3 current-stage claim boundary opened",
    )

    return {
        "decision": "VALID_L2_L3_PREREG_R1",
        "l2_required_metrics": len(REQUIRED_METRICS),
        "l2_candidate_count": len(CANDIDATE_ORDER),
        "l2_runs_per_candidate": 1,
        "l3_executable": False,
        "l3_candidate_id": None,
        "l3_sessions": l3["lockbox_floors"]["sessions"],
        "l3_loso_folds": l3["lockbox_floors"]["loso_folds"],
        "l3_bootstrap_iterations": stats["bootstrap_iterations"],
        "new_data_or_candidate_execution": False,
    }


def _resolve(repo: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo / path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate frozen L2 fresh-selection and non-executable L3 contracts."
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--l2-config", type=Path, default=L2_CONFIG)
    parser.add_argument("--l3-template", type=Path, default=L3_TEMPLATE)
    parser.add_argument("--l2-schema", type=Path, default=L2_SCHEMA)
    parser.add_argument("--l3-schema", type=Path, default=L3_SCHEMA)
    args = parser.parse_args()

    repo = args.repo.resolve()
    result = validate_contracts(
        repo,
        _load(_resolve(repo, args.l2_config)),
        _load(_resolve(repo, args.l3_template)),
        _load(_resolve(repo, args.l2_schema)),
        _load(_resolve(repo, args.l3_schema)),
    )
    print(
        "VALID_L2_L3_PREREG_R1 "
        f"l2_metrics={result['l2_required_metrics']} "
        f"candidates={result['l2_candidate_count']} "
        f"runs_per_candidate={result['l2_runs_per_candidate']} "
        f"l3_executable={str(result['l3_executable']).lower()} "
        f"l3_candidate_id=null "
        f"l3_sessions={result['l3_sessions']} "
        f"l3_loso_folds={result['l3_loso_folds']} "
        f"bootstrap={result['l3_bootstrap_iterations']} "
        "data_or_candidate_execution=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
