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
    load_json,
    read_jsonl,
    sha256_file,
    validate_review_row,
)


DECLARATION = {
    "only_assigned_pass_media_used": True,
    "source_event_session_bucket_interval_not_seen": True,
    "truth_oracle_model_outputs_not_seen": True,
    "other_pass_or_review_not_seen": True,
}


def _ratio(n: int, d: int) -> float:
    return n / d if d else 0.0


def _assignments(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        key, separator, raw_path = value.partition("=")
        if not separator or key not in PASS_IDS or key in result:
            raise A0Error("submission assignment must be unique PASS_ID=directory")
        result[key] = Path(raw_path)
    if set(result) != set(PASS_IDS):
        raise A0Error("all six isolated submissions are required")
    return result


def _load_inputs(
    key_path: Path,
    submissions: dict[str, Path],
    pass_manifest_hashes: dict[str, str],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, str],
]:
    key = load_json(key_path)
    if (
        key.get("schema_version") != "blindassist.riskseg_act_a0.scoring_key.v1"
        or key.get("protocol_id") != PROTOCOL_ID
    ):
        raise A0Error("invalid scoring key")
    key_by_pass: dict[str, list[dict[str, Any]]] = {name: [] for name in PASS_IDS}
    for unit in key.get("units", []):
        if unit.get("pass_id") not in key_by_pass:
            raise A0Error("unknown pass in scoring key")
        key_by_pass[unit["pass_id"]].append(unit)
    reviews: dict[str, dict[str, dict[str, Any]]] = {}
    identities: dict[str, str] = {}
    for pass_id in PASS_IDS:
        root = submissions[pass_id]
        review_path = root / "review.jsonl"
        receipt_path = root / "submission_receipt.json"
        receipt = load_json(receipt_path)
        if (
            receipt.get("pass_id") != pass_id
            or receipt.get("assigned_pass_manifest_sha256")
            != pass_manifest_hashes[pass_id]
            or receipt.get("fresh_agent_for_this_pass") is not True
            or receipt.get("isolation_declaration") != DECLARATION
            or receipt.get("review_jsonl_sha256") != sha256_file(review_path)
        ):
            raise A0Error(f"{pass_id}: independent isolation revalidation failed")
        identity = receipt.get("reviewer_identity")
        if not isinstance(identity, str) or not identity:
            raise A0Error(f"{pass_id}: missing identity")
        identities[pass_id] = identity
        rows = [
            validate_review_row(row, where=f"{pass_id}:{index + 1}")
            for index, row in enumerate(read_jsonl(review_path))
        ]
        reviews[pass_id] = {row["review_item_id"]: row for row in rows}
        expected = {unit["review_item_id"] for unit in key_by_pass[pass_id]}
        if set(reviews[pass_id]) != expected or len(rows) != len(expected):
            raise A0Error(f"{pass_id}: independent completeness revalidation failed")
    if len(set(identities.values())) != 6:
        raise A0Error("six distinct fresh reviewer identities are required")
    return key_by_pass, reviews, identities


def _verify_bundle(
    bundle_root: Path,
    scoring_key_path: Path,
) -> tuple[str, dict[str, str]]:
    receipt_path = bundle_root / "bundle_receipt.json"
    receipt = load_json(receipt_path)
    if (
        receipt.get("schema_version")
        != "blindassist.riskseg_act_a0.review_bundle.v1"
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("status") != "READY_FOR_SIX_ISOLATED_REVIEWS"
    ):
        raise A0Error("independent bundle identity/status validation failed")
    scoring = receipt.get("scoring_key", {})
    if (
        (bundle_root / scoring.get("path", "")).resolve()
        != scoring_key_path.resolve()
        or scoring.get("sha256") != sha256_file(scoring_key_path)
    ):
        raise A0Error("independent scoring-key binding validation failed")
    rows = receipt.get("passes")
    if not isinstance(rows, list) or {row.get("pass_id") for row in rows} != set(PASS_IDS):
        raise A0Error("independent pass-ledger validation failed")
    hashes: dict[str, str] = {}
    for row in rows:
        pass_id = row["pass_id"]
        manifest_path = bundle_root / "passes" / pass_id / "pass_manifest.json"
        template_path = bundle_root / "passes" / pass_id / "review_template.jsonl"
        if (
            row.get("path") != f"passes/{pass_id}"
            or row.get("information_condition") != PASS_CONDITION[pass_id]
            or row.get("manifest_sha256") != sha256_file(manifest_path)
            or row.get("review_template_sha256") != sha256_file(template_path)
        ):
            raise A0Error(f"{pass_id}: independent public packet binding failed")
        hashes[pass_id] = row["manifest_sha256"]
    return sha256_file(receipt_path), hashes


def _paired(
    condition: str,
    key_by_pass: dict[str, list[dict[str, Any]]],
    reviews: dict[str, dict[str, dict[str, Any]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[tuple[str, int]],
    dict[tuple[str, int], dict[str, Any]],
]:
    pass_a, pass_b = CONDITION_PASSES[condition]
    map_a = {
        (unit["parent_id"], unit["anchor_slot"]): reviews[pass_a][unit["review_item_id"]]
        for unit in key_by_pass[pass_a]
    }
    map_b = {
        (unit["parent_id"], unit["anchor_slot"]): reviews[pass_b][unit["review_item_id"]]
        for unit in key_by_pass[pass_b]
    }
    if set(map_a) != set(map_b):
        raise A0Error(f"{condition}: hidden anchor mismatch")
    keys = sorted(map_a)
    metadata = {
        (unit["parent_id"], unit["anchor_slot"]): unit
        for unit in key_by_pass[pass_a]
    }
    return [map_a[k] for k in keys], [map_b[k] for k in keys], keys, metadata


def _recompute_condition(
    condition: str,
    key_by_pass: dict[str, list[dict[str, Any]]],
    reviews: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    a_rows, b_rows, keys, metadata = _paired(condition, key_by_pass, reviews)

    def exact(field: str) -> dict[str, Any]:
        count = sum(a[field] == b[field] for a, b in zip(a_rows, b_rows, strict=True))
        return {"matches": count, "total": len(a_rows), "value": _ratio(count, len(a_rows))}

    boundary_pairs = [
        (a, b)
        for a, b in zip(a_rows, b_rows, strict=True)
        if a["boundary_relation"] != "NOT_APPLICABLE"
        or b["boundary_relation"] != "NOT_APPLICABLE"
    ]
    boundary_matches = sum(a["boundary_relation"] == b["boundary_relation"] for a, b in boundary_pairs)
    tp = fp = fn = excluded = evaluated = 0
    for a, b in zip(a_rows, b_rows, strict=True):
        for cell in CELL_IDS:
            left, right = a["intrusion_cells"][cell], b["intrusion_cells"][cell]
            if left == "UNKNOWN" or right == "UNKNOWN":
                excluded += 1
                continue
            evaluated += 1
            tp += left == right == "INTRUDING"
            fp += left == "INTRUDING" and right == "NON_INTRUDING"
            fn += left == "NON_INTRUDING" and right == "INTRUDING"
    f1_denominator = 2 * tp + fp + fn

    confusion = {x: {y: 0 for y in DERIVED_LABELS} for x in DERIVED_LABELS}
    for a, b in zip(a_rows, b_rows, strict=True):
        confusion[a["derived_actionability"]][b["derived_actionability"]] += 1
    derived_exact = sum(
        a["derived_actionability"] == b["derived_actionability"]
        for a, b in zip(a_rows, b_rows, strict=True)
    )
    classwise: dict[str, dict[str, Any]] = {}
    for label in DERIVED_LABELS:
        intersection = confusion[label][label]
        union = (
            sum(confusion[label].values())
            + sum(confusion[other][label] for other in DERIVED_LABELS)
            - intersection
        )
        classwise[label] = {
            "intersection": intersection,
            "union": union,
            "value": _ratio(intersection, union),
            "zero_denominator_fail_closed": union == 0,
        }
    actionable_intersection = confusion["ACTIONABLE_NOW"]["ACTIONABLE_NOW"]
    actionable_union = (
        sum(confusion["ACTIONABLE_NOW"].values())
        + sum(confusion[other]["ACTIONABLE_NOW"] for other in DERIVED_LABELS)
        - actionable_intersection
    )
    abstain_union = sum(
        a["derived_actionability"] == "ABSTAIN_NOT_EVALUABLE"
        or b["derived_actionability"] == "ABSTAIN_NOT_EVALUABLE"
        for a, b in zip(a_rows, b_rows, strict=True)
    )
    parent_ids = sorted({parent for parent, _ in keys})
    row_a = dict(zip(keys, a_rows, strict=True))
    row_b = dict(zip(keys, b_rows, strict=True))
    sequence_matches = sum(
        all(
            row_a[(parent, slot)]["derived_actionability"]
            == row_b[(parent, slot)]["derived_actionability"]
            for slot in range(4)
        )
        for parent in parent_ids
    )
    first_deltas: list[int] = []
    for parent in parent_ids:
        actionable_a = [
            (parent, slot)
            for slot in range(4)
            if row_a[(parent, slot)]["derived_actionability"] == "ACTIONABLE_NOW"
        ]
        actionable_b = [
            (parent, slot)
            for slot in range(4)
            if row_b[(parent, slot)]["derived_actionability"] == "ACTIONABLE_NOW"
        ]
        if actionable_a and actionable_b:
            first_deltas.append(
                abs(
                    metadata[actionable_a[0]]["anchor_frame_index"]
                    - metadata[actionable_b[0]]["anchor_frame_index"]
                )
            )
    strata: dict[str, dict[str, Any]] = {}
    for stratum in sorted({metadata[key]["hidden_source_stratum"] for key in keys}):
        subset = [key for key in keys if metadata[key]["hidden_source_stratum"] == stratum]
        matches = sum(
            row_a[key]["derived_actionability"] == row_b[key]["derived_actionability"]
            for key in subset
        )
        strata[stratum] = {"matches": matches, "total": len(subset), "value": _ratio(matches, len(subset))}

    def unknown_burden(rows: list[dict[str, Any]]) -> dict[str, Any]:
        counters = {
            "intrusion_cells": (
                sum(
                    row["intrusion_cells"][cell] == "UNKNOWN"
                    for row in rows
                    for cell in CELL_IDS
                ),
                len(rows) * len(CELL_IDS),
            ),
            "alertable": (
                sum(row["alertable"] == "UNKNOWN" for row in rows),
                len(rows),
            ),
            "passed": (
                sum(row["passed"] == "UNKNOWN" for row in rows),
                len(rows),
            ),
            "knownness": (
                sum(row["knownness"] == "UNKNOWN" for row in rows),
                len(rows),
            ),
            "hazard_aux": (
                sum(row["hazard_aux"] == "UNRESOLVED" for row in rows),
                len(rows),
            ),
            "boundary_relation": (
                sum(row["boundary_relation"] == "AMBIGUOUS" for row in rows),
                len(rows),
            ),
        }
        burden = {
            field: {"unknown": n, "total": d, "value": _ratio(n, d)}
            for field, (n, d) in counters.items()
        }
        total_unknown = sum(n for n, _ in counters.values())
        total_slots = sum(d for _, d in counters.values())
        burden["combined"] = {
            "unknown": total_unknown,
            "total": total_slots,
            "value": _ratio(total_unknown, total_slots),
        }
        return burden
    pass_a, pass_b = CONDITION_PASSES[condition]
    return {
        "passes": [pass_a, pass_b],
        "completion_rate": 1.0,
        "alertable_exact_agreement": exact("alertable"),
        "passed_exact_agreement": exact("passed"),
        "knownness_exact_agreement": exact("knownness"),
        "boundary_relation_exact_agreement": {
            "matches": boundary_matches,
            "union_applicable_total": len(boundary_pairs),
            "value": _ratio(boundary_matches, len(boundary_pairs)),
            "zero_denominator_fail_closed": not boundary_pairs,
        },
        "intrusion_spatial_f1": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "evaluated_cell_pairs": evaluated,
            "excluded_unknown_cell_pairs": excluded,
            "value": _ratio(2 * tp, f1_denominator),
            "zero_denominator_fail_closed": f1_denominator == 0,
        },
        "derived": {
            "confusion_a_rows_b_columns": confusion,
            "exact_agreement": {
                "matches": derived_exact,
                "total": len(a_rows),
                "value": _ratio(derived_exact, len(a_rows)),
            },
            "classwise_jaccard": classwise,
            "macro_jaccard": sum(v["value"] for v in classwise.values()) / 3,
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
        },
        "parent_event_sequence_match": {
            "matches": sequence_matches,
            "total": len(parent_ids),
            "value": _ratio(sequence_matches, len(parent_ids)),
        },
        "first_actionable_anchor_delta_frames": {
            "both_actionable_parent_events": len(first_deltas),
            "absolute_deltas": first_deltas,
            "mean": _ratio(sum(first_deltas), len(first_deltas)),
            "maximum": max(first_deltas, default=None),
        },
        "unknown_burden_by_pass": {
            pass_a: unknown_burden(a_rows),
            pass_b: unknown_burden(b_rows),
        },
        "unknown_to_non_actionable_violation_count": 0,
        "hidden_source_stratum_derived_agreement": strata,
    }


def _recompute_cross(
    key_by_pass: dict[str, list[dict[str, Any]]],
    reviews: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    consensus: dict[str, dict[tuple[str, int], str]] = {}
    for condition in CONDITIONS:
        a_rows, b_rows, keys, _ = _paired(condition, key_by_pass, reviews)
        consensus[condition] = {
            key: (
                a["derived_actionability"]
                if a["derived_actionability"] == b["derived_actionability"]
                else "REVIEWER_DISAGREEMENT"
            )
            for key, a, b in zip(keys, a_rows, b_rows, strict=True)
        }
    result: dict[str, Any] = {}
    for left, right in (
        ("CURRENT_ONLY", "CAUSAL_HISTORY"),
        ("CAUSAL_HISTORY", "HINDSIGHT_REFERENCE"),
    ):
        matrix: dict[str, Counter[str]] = defaultdict(Counter)
        matches = 0
        for key in sorted(consensus[left]):
            before, after = consensus[left][key], consensus[right][key]
            matrix[before][after] += 1
            matches += before == after
        total = len(consensus[left])
        result[f"{left}_TO_{right}"] = {
            "consensus_transition_matrix": {
                before: dict(sorted(after.items())) for before, after in sorted(matrix.items())
            },
            "exact_consensus_transition": {
                "matches": matches,
                "total": total,
                "value": _ratio(matches, total),
            },
            "changed_fraction": _ratio(total - matches, total),
        }
    causal = result["CAUSAL_HISTORY_TO_HINDSIGHT_REFERENCE"]
    result["causal_versus_hindsight"] = {
        "agreement": causal["exact_consensus_transition"],
        "future_dependent_fraction": causal["changed_fraction"],
        "interpretation": "REFERENCE_DIAGNOSTIC_ONLY_NOT_RUNTIME_OR_TRAINING_TRUTH",
    }
    return result


def validate(
    *,
    bundle_root: Path,
    result_path: Path,
    scoring_key_path: Path,
    submissions: dict[str, Path],
) -> dict[str, Any]:
    bundle_receipt_sha256, pass_manifest_hashes = _verify_bundle(
        bundle_root, scoring_key_path
    )
    result = load_json(result_path)
    if (
        result.get("schema_version") != "blindassist.riskseg_act_a0.score.v1"
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("training_authority") is not False
        or result.get("third_agent_adjudication_used") is not False
    ):
        raise A0Error("result identity/authority boundary mismatch")
    if result.get("bindings", {}).get("scoring_key_sha256") != sha256_file(scoring_key_path):
        raise A0Error("result scoring-key hash binding mismatch")
    if (
        result.get("bindings", {}).get("bundle_receipt_sha256")
        != bundle_receipt_sha256
        or result.get("bindings", {}).get("pass_manifest_sha256")
        != pass_manifest_hashes
    ):
        raise A0Error("result bundle/public-packet hash binding mismatch")
    for pass_id, root in submissions.items():
        bound = result["bindings"]["submissions"][pass_id]
        if (
            bound["review_jsonl"] != sha256_file(root / "review.jsonl")
            or bound["submission_receipt"] != sha256_file(root / "submission_receipt.json")
        ):
            raise A0Error(f"{pass_id}: result input hash binding mismatch")
    key_by_pass, reviews, identities = _load_inputs(
        scoring_key_path, submissions, pass_manifest_hashes
    )
    if result.get("reviewer_identities") != identities:
        raise A0Error("reviewer identity ledger mismatch")
    for condition in CONDITIONS:
        recomputed = _recompute_condition(condition, key_by_pass, reviews)
        recorded = result["condition_metrics"][condition]
        for field in recomputed:
            if recorded[field] != recomputed[field]:
                raise A0Error(f"{condition}: independently recomputed {field} mismatch")
    cross = _recompute_cross(key_by_pass, reviews)
    if result.get("cross_condition") != cross:
        raise A0Error("independently recomputed cross-condition metrics mismatch")
    return {
        "schema_version": "blindassist.riskseg_act_a0.independent_validation.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "VALID",
        "result_sha256": sha256_file(result_path),
        "bundle_receipt_sha256": bundle_receipt_sha256,
        "scoring_key_sha256": sha256_file(scoring_key_path),
        "six_fresh_isolated_passes": True,
        "key_metrics_independently_recomputed": True,
        "cross_condition_transitions_independently_recomputed": True,
        "training_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--scoring-key", type=Path, required=True)
    parser.add_argument("--submission", action="append", default=[])
    args = parser.parse_args()
    receipt = validate(
        bundle_root=args.bundle_root,
        result_path=args.result,
        scoring_key_path=args.scoring_key,
        submissions=_assignments(args.submission),
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
