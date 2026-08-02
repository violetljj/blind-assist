from __future__ import annotations

"""Fail-close two independent RGB-only pHash candidate reviews.

The finalizer is intentionally JSON-only.  It does not open RGB assets,
masks, action facts, model outputs, oracle outputs, or feedback traces.  A
candidate screen is resolved only when both reviewers independently mark every
fixed pair as DISTINCT_CAPTURE.  SAME_CAPTURE, UNKNOWN, disagreement, missing
coverage, or substituted material remains a HOLD.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import (
    PHASH_PRIVATE_MAP_SCHEMA,
    PHASH_RESOLUTION_SCHEMA,
    PHASH_REVIEW_PACKET_SCHEMA,
    PHASH_REVIEW_SUBMISSION_SCHEMA,
    PROTOCOL_ID,
    read_json,
    sha256_file,
    sha256_json,
)
from .prepare_phash_manual_review import ADMISSION_HOLD_STATUS, PACKET_STATUS, PRIVATE_STATUS


PASSED_STATUS = "PHASH_MANUAL_REVIEW_PASSED"
HOLD_STATUS = "HOLD_EVAL_VALIDITY_DATA"
DECISIONS = {"SAME_CAPTURE", "DISTINCT_CAPTURE", "UNKNOWN"}


class PHashReviewFinalizationError(ValueError):
    """Raised when a manual-review input is malformed or disclosure-unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PHashReviewFinalizationError(message)


def _validate_admission(admission: dict[str, Any]) -> tuple[str, Counter[str]]:
    _require(admission.get("schema_version") == "blindassist.eval_validity_r0.data_admission_receipt.v1", "admission schema mismatch")
    _require(admission.get("protocol_id") == PROTOCOL_ID and admission.get("status") == ADMISSION_HOLD_STATUS, "admission is not a frozen pHash HOLD")
    _require(admission.get("candidate_outputs_opened") is False, "admission records forbidden output access")
    checks, evidence = admission.get("checks"), admission.get("evidence")
    _require(isinstance(checks, dict) and isinstance(evidence, dict), "admission checks/evidence are missing")
    expected_true = {
        "session_disjoint", "old_truth_session_disjoint", "parent_identity_disjoint", "exact_rgb_disjoint",
        "decoded_rgb_disjoint", "exact_source_mask_disjoint", "p_hash_prior_session_coverage_complete",
        "p_hash_prior_decode_complete", "p_hash_new_decode_complete",
    }
    _require(all(checks.get(field) is True for field in expected_true), "admission has a non-pHash failure")
    _require(checks.get("p_hash_no_unresolved_new_to_excluded_candidate") is False, "admission is not a pHash candidate HOLD")
    rows = evidence.get("p_hash_candidates")
    _require(evidence.get("p_hash_candidate_enumeration_complete") is True and isinstance(rows, list) and rows, "pHash candidate evidence is incomplete")
    _require(evidence.get("p_hash_candidate_count_lower_bound") == len(rows), "pHash candidate count is not exact")
    return sha256_json(admission), Counter(sha256_json(row) for row in rows)


def _validate_packet(packet: dict[str, Any], *, role: str, expected_map: dict[str, dict[str, Any]]) -> None:
    _require(packet.get("schema_version") == PHASH_REVIEW_PACKET_SCHEMA and packet.get("protocol_id") == PROTOCOL_ID, f"{role}: packet schema/protocol mismatch")
    _require(packet.get("reviewer_role") == role and packet.get("status") == PACKET_STATUS, f"{role}: packet role/status mismatch")
    disclosures = packet.get("disclosures")
    expected_disclosures = {
        "raw_rgb_only": True, "source_mask_visible": False, "action_fact_visible": False,
        "model_or_oracle_output_visible": False, "source_session_or_event_identity_visible": False,
        "other_reviewer_visible": False, "candidate_threshold_or_hamming_visible": False,
    }
    _require(isinstance(disclosures, dict) and all(disclosures.get(field) is value for field, value in expected_disclosures.items()), f"{role}: packet disclosure mismatch")
    items = packet.get("items")
    _require(isinstance(items, list) and len(items) == len(expected_map), f"{role}: packet coverage mismatch")
    observed: set[str] = set()
    for item in items:
        _require(isinstance(item, dict) and set(item) == {"review_item_id", "rgb_pair", "response_field"}, f"{role}: malformed packet item")
        opaque_id = item.get("review_item_id")
        _require(isinstance(opaque_id, str) and opaque_id in expected_map and opaque_id not in observed, f"{role}: packet opaque ID mismatch")
        pair = item.get("rgb_pair")
        _require(isinstance(pair, list) and len(pair) == 2 and all(isinstance(path, str) and path.endswith(".png") for path in pair), f"{role}: packet pair shape mismatch")
        _require(item.get("response_field") == {"same_natural_capture": ["SAME_CAPTURE", "DISTINCT_CAPTURE", "UNKNOWN"]}, f"{role}: packet response ontology mismatch")
        observed.add(opaque_id)
    _require(observed == set(expected_map), f"{role}: packet/private map coverage mismatch")


def _validate_private(
    private: dict[str, Any], *, admission_sha: str, packet_a_sha: str, packet_b_sha: str, expected_candidate_edges: Counter[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str]]:
    _require(private.get("schema_version") == PHASH_PRIVATE_MAP_SCHEMA and private.get("protocol_id") == PROTOCOL_ID, "private map schema/protocol mismatch")
    _require(private.get("status") == PRIVATE_STATUS and private.get("admission_receipt_sha256") == admission_sha, "private map admission binding mismatch")
    _require(private.get("packet_a_sha256") == packet_a_sha and private.get("packet_b_sha256") == packet_b_sha, "private map packet binding mismatch")
    count = private.get("candidate_case_count")
    _require(isinstance(count, int) and count > 0, "private map candidate case count is invalid")
    maps: list[dict[str, dict[str, Any]]] = []
    fingerprints: list[set[str]] = []
    for field, role in (("reviewer_a_map", "PHASH_RGB_REVIEW_A"), ("reviewer_b_map", "PHASH_RGB_REVIEW_B")):
        mapping = private.get(field)
        _require(isinstance(mapping, dict) and len(mapping) == count, f"{role}: private map coverage mismatch")
        case_fingerprints: set[str] = set()
        candidate_edges: Counter[str] = Counter()
        for opaque_id, item in mapping.items():
            _require(isinstance(opaque_id, str) and opaque_id.startswith(role.lower()) and isinstance(item, dict), f"{role}: private map opaque identity mismatch")
            fingerprint = item.get("case_fingerprint")
            _require(isinstance(fingerprint, str) and len(fingerprint) == 64 and fingerprint not in case_fingerprints, f"{role}: private map case binding mismatch")
            _require(isinstance(item.get("left_is_new"), bool), f"{role}: private map left/right identity is missing")
            _require(isinstance(item.get("candidate_evidence"), list) and item["candidate_evidence"], f"{role}: private map candidate evidence is missing")
            candidate_edges.update(sha256_json(edge) for edge in item["candidate_evidence"])
            case_fingerprints.add(fingerprint)
        _require(candidate_edges == expected_candidate_edges, f"{role}: private map does not cover each frozen pHash edge exactly once")
        maps.append(mapping)
        fingerprints.append(case_fingerprints)
    _require(fingerprints[0] == fingerprints[1], "reviewer private maps do not cover the same fixed cases")
    _require(private.get("candidate_case_fingerprints_sha256") == sha256_json(sorted(fingerprints[0])), "private map case-set binding mismatch")
    return maps[0], maps[1], fingerprints[0]


def _index_submission(
    review: dict[str, Any], *, role: str, admission_sha: str, expected_map: dict[str, dict[str, Any]],
) -> dict[str, str]:
    required = {
        "schema_version", "protocol_id", "reviewer_role", "admission_receipt_sha256", "isolated_context",
        "other_review_visible_before_submission", "model_or_oracle_output_visible", "items",
    }
    _require(set(review) == required, f"{role}: submission fields mismatch")
    _require(review.get("schema_version") == PHASH_REVIEW_SUBMISSION_SCHEMA and review.get("protocol_id") == PROTOCOL_ID, f"{role}: submission schema/protocol mismatch")
    _require(review.get("reviewer_role") == role and review.get("admission_receipt_sha256") == admission_sha, f"{role}: submission role/admission mismatch")
    _require(review.get("isolated_context") is True and review.get("other_review_visible_before_submission") is False and review.get("model_or_oracle_output_visible") is False, f"{role}: reviewer isolation/output disclosure failure")
    items = review.get("items")
    _require(isinstance(items, list) and len(items) == len(expected_map), f"{role}: submission coverage mismatch")
    result: dict[str, str] = {}
    observed: set[str] = set()
    for item in items:
        _require(isinstance(item, dict) and set(item) == {"review_item_id", "same_natural_capture"}, f"{role}: malformed submission item")
        opaque_id, decision = item.get("review_item_id"), item.get("same_natural_capture")
        _require(isinstance(opaque_id, str) and opaque_id in expected_map and opaque_id not in observed, f"{role}: submission opaque identity mismatch")
        _require(decision in DECISIONS, f"{role}: invalid review decision")
        fingerprint = expected_map[opaque_id]["case_fingerprint"]
        _require(fingerprint not in result, f"{role}: duplicate private case response")
        result[fingerprint] = str(decision)
        observed.add(opaque_id)
    _require(observed == set(expected_map), f"{role}: submission does not cover every packet item")
    return result


def finalize_review(
    *, admission: dict[str, Any], private_map: dict[str, Any], packet_a: dict[str, Any], packet_b: dict[str, Any],
    review_a: dict[str, Any], review_b: dict[str, Any], packet_a_sha256: str, packet_b_sha256: str,
    review_a_sha256: str, review_b_sha256: str,
) -> dict[str, Any]:
    admission_sha, expected_candidate_edges = _validate_admission(admission)
    map_a, map_b, fingerprints = _validate_private(
        private_map, admission_sha=admission_sha, packet_a_sha=packet_a_sha256, packet_b_sha=packet_b_sha256,
        expected_candidate_edges=expected_candidate_edges,
    )
    _validate_packet(packet_a, role="PHASH_RGB_REVIEW_A", expected_map=map_a)
    _validate_packet(packet_b, role="PHASH_RGB_REVIEW_B", expected_map=map_b)
    answers_a = _index_submission(review_a, role="PHASH_RGB_REVIEW_A", admission_sha=admission_sha, expected_map=map_a)
    answers_b = _index_submission(review_b, role="PHASH_RGB_REVIEW_B", admission_sha=admission_sha, expected_map=map_b)
    _require(set(answers_a) == fingerprints == set(answers_b), "submission/private candidate coverage mismatch")
    outcomes = []
    for fingerprint in sorted(fingerprints):
        left, right = answers_a[fingerprint], answers_b[fingerprint]
        outcomes.append({"case_fingerprint": fingerprint, "review_a": left, "review_b": right, "exact_agreement": left == right, "resolved_distinct": left == right == "DISTINCT_CAPTURE"})
    passed = all(item["resolved_distinct"] for item in outcomes)
    counts = Counter(item["review_a"] for item in outcomes)
    return {
        "schema_version": PHASH_RESOLUTION_SCHEMA, "protocol_id": PROTOCOL_ID,
        "status": PASSED_STATUS if passed else HOLD_STATUS,
        "admission_receipt_sha256": admission_sha,
        "candidate_outputs_opened": False,
        "input_sha256": {
            "packet_a": packet_a_sha256, "packet_b": packet_b_sha256,
            "review_a": review_a_sha256, "review_b": review_b_sha256,
            "private_map": sha256_json(private_map),
        },
        "evidence": {
            "candidate_case_count": len(outcomes), "reviewer_a_decision_counts": dict(sorted(counts.items())),
            "exact_agreement_count": sum(item["exact_agreement"] for item in outcomes),
            "all_cases_resolved_distinct": passed, "reviewers_isolated": True,
            "model_or_oracle_output_visible": False, "source_mask_visible": False,
            "outcomes": outcomes,
        },
        "next_required_gate": (
            "Reconcile the frozen admission receipt with this passed RGB-only pHash resolution; do not render any model or oracle trace."
            if passed else "HOLD. Do not modify the candidate set, pHash threshold, source sessions, or reviewer answers; do not render model/oracle output."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission-receipt", type=Path, required=True)
    parser.add_argument("--private-map", type=Path, required=True)
    parser.add_argument("--packet-a", type=Path, required=True)
    parser.add_argument("--packet-b", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    result = finalize_review(
        admission=read_json(args.admission_receipt), private_map=read_json(args.private_map),
        packet_a=read_json(args.packet_a), packet_b=read_json(args.packet_b),
        review_a=read_json(args.review_a), review_b=read_json(args.review_b),
        packet_a_sha256=sha256_file(args.packet_a), packet_b_sha256=sha256_file(args.packet_b),
        review_a_sha256=sha256_file(args.review_a), review_b_sha256=sha256_file(args.review_b),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} candidate_cases={result['evidence']['candidate_case_count']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
