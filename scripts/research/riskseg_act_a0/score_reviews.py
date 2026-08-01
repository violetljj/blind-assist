from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .common import (
    CELL_IDS,
    CONDITION_PASSES,
    CONDITIONS,
    DERIVED_LABELS,
    PASS_CONDITION,
    PASS_IDS,
    PROTOCOL_ID,
    A0Error,
    assert_no_forbidden_public_fields,
    load_json,
    read_jsonl,
    sha256_file,
    unknown_to_non_actionable_violation,
    validate_review_row,
    write_json,
)


EXPECTED_DECLARATION = {
    "only_assigned_pass_media_used": True,
    "source_event_session_bucket_interval_not_seen": True,
    "truth_oracle_model_outputs_not_seen": True,
    "other_pass_or_review_not_seen": True,
}

RUNTIME_GATES = {
    "alertable_exact_agreement": 0.85,
    "passed_exact_agreement": 0.85,
    "knownness_exact_agreement": 0.85,
    "boundary_relation_exact_agreement": 0.85,
    "intrusion_spatial_f1": 0.80,
    "derived_exact_agreement": 0.80,
    "derived_macro_jaccard": 0.70,
    "actionable_union_agreement": 0.75,
    "parent_event_sequence_match": 0.80,
    "union_abstain_burden_max": 0.30,
    "each_hidden_stratum_derived_agreement": 0.60,
}
HINDSIGHT_GATES = {
    "alertable_exact_agreement": 0.90,
    "passed_exact_agreement": 0.90,
    "boundary_relation_exact_agreement": 0.90,
    "intrusion_spatial_f1": 0.85,
    "derived_exact_agreement": 0.90,
    "parent_event_sequence_match": 0.90,
}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _exact(a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]], field: str) -> dict:
    matches = sum(a[field] == b[field] for a, b in zip(a_rows, b_rows, strict=True))
    return {"matches": matches, "total": len(a_rows), "value": _ratio(matches, len(a_rows))}


def _binary_f1(
    a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    tp = fp = fn = excluded_unknown = 0
    for a, b in zip(a_rows, b_rows, strict=True):
        for cell in CELL_IDS:
            left = a["intrusion_cells"][cell]
            right = b["intrusion_cells"][cell]
            if "UNKNOWN" in {left, right}:
                excluded_unknown += 1
                continue
            tp += left == right == "INTRUDING"
            fp += left == "INTRUDING" and right == "NON_INTRUDING"
            fn += left == "NON_INTRUDING" and right == "INTRUDING"
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "evaluated_cell_pairs": tp + fp + fn
        + sum(
            a["intrusion_cells"][cell] == b["intrusion_cells"][cell] == "NON_INTRUDING"
            for a, b in zip(a_rows, b_rows, strict=True)
            for cell in CELL_IDS
        ),
        "excluded_unknown_cell_pairs": excluded_unknown,
        "value": _ratio(2 * tp, denominator),
        "zero_denominator_fail_closed": denominator == 0,
    }


def _boundary_exact(
    a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    pairs = [
        (a, b)
        for a, b in zip(a_rows, b_rows, strict=True)
        if a["boundary_relation"] != "NOT_APPLICABLE"
        or b["boundary_relation"] != "NOT_APPLICABLE"
    ]
    matches = sum(a["boundary_relation"] == b["boundary_relation"] for a, b in pairs)
    return {
        "matches": matches,
        "union_applicable_total": len(pairs),
        "value": _ratio(matches, len(pairs)),
        "zero_denominator_fail_closed": not pairs,
    }


def _derived_metrics(
    a_rows: list[dict[str, Any]], b_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    confusion = {left: {right: 0 for right in DERIVED_LABELS} for left in DERIVED_LABELS}
    for a, b in zip(a_rows, b_rows, strict=True):
        confusion[a["derived_actionability"]][b["derived_actionability"]] += 1
    exact = sum(
        a["derived_actionability"] == b["derived_actionability"]
        for a, b in zip(a_rows, b_rows, strict=True)
    )
    jaccard: dict[str, dict[str, Any]] = {}
    for label in DERIVED_LABELS:
        intersection = confusion[label][label]
        union = sum(confusion[label].values()) + sum(
            confusion[other][label] for other in DERIVED_LABELS
        ) - intersection
        jaccard[label] = {
            "intersection": intersection,
            "union": union,
            "value": _ratio(intersection, union),
            "zero_denominator_fail_closed": union == 0,
        }
    macro = sum(value["value"] for value in jaccard.values()) / len(DERIVED_LABELS)
    actionable_intersection = confusion["ACTIONABLE_NOW"]["ACTIONABLE_NOW"]
    actionable_union = sum(confusion["ACTIONABLE_NOW"].values()) + sum(
        confusion[other]["ACTIONABLE_NOW"] for other in DERIVED_LABELS
    ) - actionable_intersection
    abstain_union = sum(
        a["derived_actionability"] == "ABSTAIN_NOT_EVALUABLE"
        or b["derived_actionability"] == "ABSTAIN_NOT_EVALUABLE"
        for a, b in zip(a_rows, b_rows, strict=True)
    )
    return {
        "confusion_a_rows_b_columns": confusion,
        "exact_agreement": {
            "matches": exact,
            "total": len(a_rows),
            "value": _ratio(exact, len(a_rows)),
        },
        "classwise_jaccard": jaccard,
        "macro_jaccard": macro,
        "actionable_union_agreement": {
            "intersection": actionable_intersection,
            "union": actionable_union,
            "value": _ratio(actionable_intersection, actionable_union),
            "zero_denominator_fail_closed": actionable_union == 0,
        },
        "union_abstain_burden": {
            "union_abstain": abstain_union,
            "total": len(a_rows),
            "value": _ratio(abstain_union, len(a_rows)),
        },
    }


def _unknown_burden(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counters = {
        "intrusion_cells": (
            sum(
                row["intrusion_cells"][cell] == "UNKNOWN"
                for row in rows
                for cell in CELL_IDS
            ),
            len(rows) * len(CELL_IDS),
        ),
        "alertable": (sum(row["alertable"] == "UNKNOWN" for row in rows), len(rows)),
        "passed": (sum(row["passed"] == "UNKNOWN" for row in rows), len(rows)),
        "knownness": (sum(row["knownness"] == "UNKNOWN" for row in rows), len(rows)),
        "hazard_aux": (sum(row["hazard_aux"] == "UNRESOLVED" for row in rows), len(rows)),
        "boundary_relation": (
            sum(row["boundary_relation"] == "AMBIGUOUS" for row in rows),
            len(rows),
        ),
    }
    result = {
        field: {"unknown": n, "total": d, "value": _ratio(n, d)}
        for field, (n, d) in counters.items()
    }
    total_unknown = sum(n for n, _ in counters.values())
    total_slots = sum(d for _, d in counters.values())
    result["combined"] = {
        "unknown": total_unknown,
        "total": total_slots,
        "value": _ratio(total_unknown, total_slots),
    }
    return result


def _load_key(path: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    key = load_json(path)
    if (
        key.get("schema_version") != "blindassist.riskseg_act_a0.scoring_key.v1"
        or key.get("protocol_id") != PROTOCOL_ID
        or tuple(key.get("passes", [])) != PASS_IDS
    ):
        raise A0Error("invalid A0 scoring key")
    units = key.get("units")
    if not isinstance(units, list):
        raise A0Error("scoring key units missing")
    by_pass: dict[str, list[dict[str, Any]]] = {pass_id: [] for pass_id in PASS_IDS}
    seen_ids: set[str] = set()
    for unit in units:
        pass_id = unit.get("pass_id")
        if pass_id not in by_pass or unit.get("condition") != PASS_CONDITION[pass_id]:
            raise A0Error("scoring key pass/condition mismatch")
        item_id = unit.get("review_item_id")
        if not isinstance(item_id, str) or item_id in seen_ids:
            raise A0Error("scoring key opaque IDs must be globally unique")
        seen_ids.add(item_id)
        by_pass[pass_id].append(unit)
    expected = key.get("anchor_count")
    if any(len(rows) != expected for rows in by_pass.values()):
        raise A0Error("scoring key pass cardinality mismatch")
    for rows in by_pass.values():
        parent_slots: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            parent_slots[row["parent_id"]].add(row["anchor_slot"])
        if set(parent_slots) != {row["parent_id"] for row in rows} or any(
            slots != {0, 1, 2, 3} for slots in parent_slots.values()
        ):
            raise A0Error("each hidden parent event must have exactly four anchors")
    return key, by_pass


def _load_bundle_integrity(
    bundle_root: Path,
    scoring_key_path: Path,
) -> dict[str, Any]:
    receipt_path = bundle_root / "bundle_receipt.json"
    if not receipt_path.is_file():
        raise A0Error("bundle_receipt.json is required")
    receipt = load_json(receipt_path)
    if (
        receipt.get("schema_version")
        != "blindassist.riskseg_act_a0.review_bundle.v1"
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "READY_FOR_SIX_ISOLATED_REVIEWS"
    ):
        raise A0Error("invalid bundle receipt identity/status")
    scoring = receipt.get("scoring_key", {})
    expected_key = (bundle_root / scoring.get("path", "")).resolve()
    if (
        expected_key != scoring_key_path.resolve()
        or scoring.get("sha256") != sha256_file(scoring_key_path)
        or scoring.get("must_not_be_shared_with_reviewers") is not True
    ):
        raise A0Error("bundle scoring-key binding mismatch")
    pass_rows = receipt.get("passes")
    if not isinstance(pass_rows, list) or {
        row.get("pass_id") for row in pass_rows
    } != set(PASS_IDS):
        raise A0Error("bundle pass ledger mismatch")
    pass_manifest_hashes: dict[str, str] = {}
    for row in pass_rows:
        pass_id = row["pass_id"]
        if (
            row.get("information_condition") != PASS_CONDITION[pass_id]
            or row.get("path") != f"passes/{pass_id}"
        ):
            raise A0Error(f"{pass_id}: bundle pass identity mismatch")
        pass_root = bundle_root / row["path"]
        manifest_path = pass_root / "pass_manifest.json"
        template_path = pass_root / "review_template.jsonl"
        if (
            not manifest_path.is_file()
            or not template_path.is_file()
            or row.get("manifest_sha256") != sha256_file(manifest_path)
            or row.get("review_template_sha256") != sha256_file(template_path)
        ):
            raise A0Error(f"{pass_id}: public review packet hash mismatch")
        public_manifest = load_json(manifest_path)
        assert_no_forbidden_public_fields(public_manifest)
        if (
            public_manifest.get("pass_id") != pass_id
            or public_manifest.get("information_condition")
            != PASS_CONDITION[pass_id]
            or public_manifest.get("item_count") != receipt.get("anchor_count_per_pass")
        ):
            raise A0Error(f"{pass_id}: public review packet identity mismatch")
        pass_manifest_hashes[pass_id] = row["manifest_sha256"]
    return {
        "bundle_receipt_sha256": sha256_file(receipt_path),
        "pass_manifest_sha256": pass_manifest_hashes,
    }


def _load_submission(
    pass_id: str,
    root: Path,
    key_units: list[dict[str, Any]],
    expected_pass_manifest_sha256: str,
) -> tuple[str, list[dict[str, Any]], dict[str, str]]:
    review_path = root / "review.jsonl"
    receipt_path = root / "submission_receipt.json"
    if not review_path.is_file() or not receipt_path.is_file():
        raise A0Error(f"{pass_id}: submission directory requires review.jsonl and submission_receipt.json")
    receipt = load_json(receipt_path)
    if (
        receipt.get("schema_version")
        != "blindassist.riskseg_act_a0.submission_receipt.v1"
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("pass_id") != pass_id
        or receipt.get("assigned_pass_manifest_sha256")
        != expected_pass_manifest_sha256
        or receipt.get("fresh_agent_for_this_pass") is not True
        or receipt.get("isolation_declaration") != EXPECTED_DECLARATION
        or receipt.get("review_jsonl_path") != "review.jsonl"
        or receipt.get("review_jsonl_sha256") != sha256_file(review_path)
    ):
        raise A0Error(f"{pass_id}: isolation receipt or review binding is invalid")
    reviewer = receipt.get("reviewer_identity")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise A0Error(f"{pass_id}: reviewer_identity is missing")
    rows = [
        validate_review_row(row, where=f"{review_path}:{index + 1}")
        for index, row in enumerate(read_jsonl(review_path))
    ]
    expected_ids = {unit["review_item_id"] for unit in key_units}
    actual_ids = [row["review_item_id"] for row in rows]
    if len(actual_ids) != len(expected_ids) or set(actual_ids) != expected_ids:
        raise A0Error(f"{pass_id}: review completion/ID coverage mismatch")
    if len(set(actual_ids)) != len(actual_ids):
        raise A0Error(f"{pass_id}: duplicate review item")
    by_id = {row["review_item_id"]: row for row in rows}
    ordered = [by_id[unit["review_item_id"]] for unit in key_units]
    return reviewer, ordered, {
        "review_jsonl": sha256_file(review_path),
        "submission_receipt": sha256_file(receipt_path),
    }


def _condition_metrics(
    condition: str,
    key_by_pass: dict[str, list[dict[str, Any]]],
    review_by_pass: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    pass_a, pass_b = CONDITION_PASSES[condition]
    key_a = key_by_pass[pass_a]
    key_b = key_by_pass[pass_b]
    map_a = {
        (unit["parent_id"], unit["anchor_slot"]): review
        for unit, review in zip(key_a, review_by_pass[pass_a], strict=True)
    }
    map_b = {
        (unit["parent_id"], unit["anchor_slot"]): review
        for unit, review in zip(key_b, review_by_pass[pass_b], strict=True)
    }
    keys = sorted(map_a)
    if set(keys) != set(map_b):
        raise A0Error(f"{condition}: passes do not cover the same hidden anchors")
    a_rows = [map_a[key] for key in keys]
    b_rows = [map_b[key] for key in keys]
    meta_by_key = {
        (unit["parent_id"], unit["anchor_slot"]): unit for unit in key_a
    }
    derived = _derived_metrics(a_rows, b_rows)
    parent_matches = 0
    first_deltas: list[int] = []
    parent_ids = sorted({parent for parent, _ in keys})
    for parent_id in parent_ids:
        parent_keys = [(parent_id, slot) for slot in range(4)]
        seq_a = [map_a[key]["derived_actionability"] for key in parent_keys]
        seq_b = [map_b[key]["derived_actionability"] for key in parent_keys]
        parent_matches += seq_a == seq_b
        actionable_a = [key for key in parent_keys if map_a[key]["derived_actionability"] == "ACTIONABLE_NOW"]
        actionable_b = [key for key in parent_keys if map_b[key]["derived_actionability"] == "ACTIONABLE_NOW"]
        if actionable_a and actionable_b:
            frame_a = meta_by_key[actionable_a[0]]["anchor_frame_index"]
            frame_b = meta_by_key[actionable_b[0]]["anchor_frame_index"]
            first_deltas.append(abs(frame_a - frame_b))

    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in sorted({meta_by_key[key]["hidden_source_stratum"] for key in keys}):
        stratum_keys = [key for key in keys if meta_by_key[key]["hidden_source_stratum"] == stratum]
        matches = sum(
            map_a[key]["derived_actionability"] == map_b[key]["derived_actionability"]
            for key in stratum_keys
        )
        by_stratum[stratum] = {
            "matches": matches,
            "total": len(stratum_keys),
            "value": _ratio(matches, len(stratum_keys)),
        }
    violations = sum(
        unknown_to_non_actionable_violation(row)
        for row in a_rows + b_rows
    )
    return {
        "passes": [pass_a, pass_b],
        "completion_rate": 1.0,
        "alertable_exact_agreement": _exact(a_rows, b_rows, "alertable"),
        "passed_exact_agreement": _exact(a_rows, b_rows, "passed"),
        "knownness_exact_agreement": _exact(a_rows, b_rows, "knownness"),
        "boundary_relation_exact_agreement": _boundary_exact(a_rows, b_rows),
        "intrusion_spatial_f1": _binary_f1(a_rows, b_rows),
        "derived": derived,
        "parent_event_sequence_match": {
            "matches": parent_matches,
            "total": len(parent_ids),
            "value": _ratio(parent_matches, len(parent_ids)),
        },
        "first_actionable_anchor_delta_frames": {
            "both_actionable_parent_events": len(first_deltas),
            "absolute_deltas": first_deltas,
            "mean": _ratio(sum(first_deltas), len(first_deltas)),
            "maximum": max(first_deltas, default=None),
        },
        "unknown_burden_by_pass": {
            pass_a: _unknown_burden(a_rows),
            pass_b: _unknown_burden(b_rows),
        },
        "unknown_to_non_actionable_violation_count": violations,
        "hidden_source_stratum_derived_agreement": by_stratum,
    }


def _consensus_by_condition(
    condition: str,
    key_by_pass: dict[str, list[dict[str, Any]]],
    reviews: dict[str, list[dict[str, Any]]],
) -> dict[tuple[str, int], str]:
    pass_a, pass_b = CONDITION_PASSES[condition]
    a = {
        (unit["parent_id"], unit["anchor_slot"]): row["derived_actionability"]
        for unit, row in zip(key_by_pass[pass_a], reviews[pass_a], strict=True)
    }
    b = {
        (unit["parent_id"], unit["anchor_slot"]): row["derived_actionability"]
        for unit, row in zip(key_by_pass[pass_b], reviews[pass_b], strict=True)
    }
    return {key: a[key] if a[key] == b[key] else "REVIEWER_DISAGREEMENT" for key in a}


def _cross_condition(
    key_by_pass: dict[str, list[dict[str, Any]]],
    reviews: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    consensus = {
        condition: _consensus_by_condition(condition, key_by_pass, reviews)
        for condition in CONDITIONS
    }
    pairs = (
        ("CURRENT_ONLY", "CAUSAL_HISTORY"),
        ("CAUSAL_HISTORY", "HINDSIGHT_REFERENCE"),
    )
    result: dict[str, Any] = {}
    for left, right in pairs:
        if set(consensus[left]) != set(consensus[right]):
            raise A0Error("cross-condition hidden anchor coverage mismatch")
        matrix: dict[str, Counter[str]] = defaultdict(Counter)
        exact = 0
        for key in sorted(consensus[left]):
            before, after = consensus[left][key], consensus[right][key]
            matrix[before][after] += 1
            exact += before == after
        total = len(consensus[left])
        name = f"{left}_TO_{right}"
        result[name] = {
            "consensus_transition_matrix": {
                before: dict(sorted(after.items())) for before, after in sorted(matrix.items())
            },
            "exact_consensus_transition": {
                "matches": exact,
                "total": total,
                "value": _ratio(exact, total),
            },
            "changed_fraction": _ratio(total - exact, total),
        }
    causal_hindsight = result["CAUSAL_HISTORY_TO_HINDSIGHT_REFERENCE"]
    result["causal_versus_hindsight"] = {
        "agreement": causal_hindsight["exact_consensus_transition"],
        "future_dependent_fraction": causal_hindsight["changed_fraction"],
        "interpretation": "REFERENCE_DIAGNOSTIC_ONLY_NOT_RUNTIME_OR_TRAINING_TRUTH",
    }
    return result


def _gate_condition(condition: str, metrics: dict[str, Any]) -> dict[str, Any]:
    thresholds = HINDSIGHT_GATES if condition == "HINDSIGHT_REFERENCE" else RUNTIME_GATES
    observed = {
        "alertable_exact_agreement": metrics["alertable_exact_agreement"]["value"],
        "passed_exact_agreement": metrics["passed_exact_agreement"]["value"],
        "knownness_exact_agreement": metrics["knownness_exact_agreement"]["value"],
        "boundary_relation_exact_agreement": metrics["boundary_relation_exact_agreement"]["value"],
        "intrusion_spatial_f1": metrics["intrusion_spatial_f1"]["value"],
        "derived_exact_agreement": metrics["derived"]["exact_agreement"]["value"],
        "derived_macro_jaccard": metrics["derived"]["macro_jaccard"],
        "actionable_union_agreement": metrics["derived"]["actionable_union_agreement"]["value"],
        "parent_event_sequence_match": metrics["parent_event_sequence_match"]["value"],
        "union_abstain_burden_max": metrics["derived"]["union_abstain_burden"]["value"],
        "each_hidden_stratum_derived_agreement": min(
            (
                value["value"]
                for value in metrics["hidden_source_stratum_derived_agreement"].values()
            ),
            default=0.0,
        ),
    }
    checks: dict[str, dict[str, Any]] = {}
    for name, threshold in thresholds.items():
        value = observed[name]
        passed = value <= threshold if name.endswith("_max") else value >= threshold
        checks[name] = {"observed": value, "threshold": threshold, "pass": passed}
    checks["completion_rate"] = {
        "observed": metrics["completion_rate"],
        "threshold": 1.0,
        "pass": metrics["completion_rate"] == 1.0,
    }
    checks["unknown_to_non_actionable_violation_count"] = {
        "observed": metrics["unknown_to_non_actionable_violation_count"],
        "threshold": 0,
        "pass": metrics["unknown_to_non_actionable_violation_count"] == 0,
    }
    return {"pass": all(check["pass"] for check in checks.values()), "checks": checks}


def score(
    *,
    bundle_root: Path,
    scoring_key_path: Path,
    submissions: dict[str, Path],
) -> dict[str, Any]:
    if set(submissions) != set(PASS_IDS):
        raise A0Error(f"exactly these six submissions are required: {PASS_IDS}")
    bundle_binding = _load_bundle_integrity(bundle_root, scoring_key_path)
    key, key_by_pass = _load_key(scoring_key_path)
    reviews: dict[str, list[dict[str, Any]]] = {}
    identities: dict[str, str] = {}
    bindings: dict[str, Any] = {}
    for pass_id in PASS_IDS:
        identity, rows, hashes = _load_submission(
            pass_id,
            submissions[pass_id],
            key_by_pass[pass_id],
            bundle_binding["pass_manifest_sha256"][pass_id],
        )
        identities[pass_id] = identity
        reviews[pass_id] = rows
        bindings[pass_id] = {
            "submission_root": str(submissions[pass_id]),
            **hashes,
        }
    if len(set(identities.values())) != len(PASS_IDS):
        raise A0Error("fresh_agent_per_pass/same_agent_cross_condition invariant failed")

    condition_metrics = {
        condition: _condition_metrics(condition, key_by_pass, reviews)
        for condition in CONDITIONS
    }
    gates = {
        condition: _gate_condition(condition, condition_metrics[condition])
        for condition in CONDITIONS
    }
    current_pass = gates["CURRENT_ONLY"]["pass"]
    causal_pass = gates["CAUSAL_HISTORY"]["pass"]
    hindsight_pass = gates["HINDSIGHT_REFERENCE"]["pass"]
    current_macro = condition_metrics["CURRENT_ONLY"]["derived"]["macro_jaccard"]
    causal_macro = condition_metrics["CAUSAL_HISTORY"]["derived"]["macro_jaccard"]
    strata = condition_metrics["CURRENT_ONLY"]["hidden_source_stratum_derived_agreement"]
    causal_strata = condition_metrics["CAUSAL_HISTORY"]["hidden_source_stratum_derived_agreement"]
    max_stratum_drop = max(
        (strata[name]["value"] - causal_strata[name]["value"] for name in strata),
        default=1.0,
    )
    causal_improvement = {
        "macro_jaccard_gain": causal_macro - current_macro,
        "minimum_required_gain": 0.10,
        "maximum_hidden_stratum_agreement_drop": max_stratum_drop,
        "maximum_allowed_drop": 0.05,
    }
    causal_improvement["pass"] = (
        causal_improvement["macro_jaccard_gain"] >= 0.10
        and max_stratum_drop <= 0.05
    )
    if current_pass:
        terminal = "A0_CURRENT_OBSERVATION_LABELS_READY_BUT_ONSET_COHORT_REQUIRED"
    elif causal_pass and causal_improvement["pass"]:
        terminal = "A0_CAUSAL_OBSERVATION_LABELS_READY_BUT_ONSET_COHORT_REQUIRED"
    elif hindsight_pass:
        terminal = "STOP_RUNTIME_LABEL_NOT_RELIABLE_HINDSIGHT_ONLY"
    else:
        terminal = "STOP_ACTIONABILITY_REFERENCE_CONSTRUCT_UNSTABLE"
    return {
        "schema_version": "blindassist.riskseg_act_a0.score.v1",
        "protocol_id": PROTOCOL_ID,
        "status": terminal,
        "claim_ceiling": "CONSUMED_DEVELOPMENT_LABEL_READINESS_ONLY_NO_TRAINING_AUTHORITY",
        "training_authority": False,
        "bindings": {
            **bundle_binding,
            "scoring_key_sha256": sha256_file(scoring_key_path),
            "submissions": bindings,
        },
        "reviewer_identities": identities,
        "condition_metrics": condition_metrics,
        "cross_condition": _cross_condition(key_by_pass, reviews),
        "gates": gates,
        "causal_improvement_if_current_fails": causal_improvement,
        "onset_readiness": {
            "status": "NOT_EVALUABLE_ON_CURRENT_LEFT_TRUNCATED_COHORT",
            "training_readiness_pass": False,
        },
        "third_agent_adjudication_used": False,
    }


def _parse_assignments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        pass_id, separator, path = value.partition("=")
        if not separator or pass_id not in PASS_IDS or pass_id in result:
            raise A0Error("--submission must be unique PASS_ID=directory assignments")
        result[pass_id] = Path(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--scoring-key", type=Path, required=True)
    parser.add_argument(
        "--submission",
        action="append",
        default=[],
        help="Repeat exactly six times as PASS_ID=submission_directory",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = score(
        bundle_root=args.bundle_root,
        scoring_key_path=args.scoring_key,
        submissions=_parse_assignments(args.submission),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
