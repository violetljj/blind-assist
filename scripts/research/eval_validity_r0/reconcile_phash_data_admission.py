from __future__ import annotations

"""Create a passed admission receipt only from a passed fixed pHash review.

This is not a rerun of discovery or pHash screening.  It preserves the frozen
HOLD receipt byte-for-byte as an input hash and accepts only a two-reviewer
resolution that covers every enumerated candidate as DISTINCT_CAPTURE.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from .common import ADMISSION_RECONCILIATION_SCHEMA, PHASH_RESOLUTION_SCHEMA, PROTOCOL_ID, read_json, sha256_file, sha256_json
from .finalize_phash_manual_review import PASSED_STATUS


PASSED_STATUS_AFTER_REVIEW = "EVAL_VALIDITY_DATA_ADMISSION_PASSED_AFTER_PHASH_MANUAL_REVIEW"


class AdmissionReconciliationError(ValueError):
    """Raised when the frozen HOLD and manual resolution cannot be bound."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionReconciliationError(message)


def reconcile(*, held_admission: dict[str, Any], phash_resolution: dict[str, Any]) -> dict[str, Any]:
    _require(held_admission.get("schema_version") == "blindassist.eval_validity_r0.data_admission_receipt.v1", "held admission schema mismatch")
    _require(held_admission.get("protocol_id") == PROTOCOL_ID and held_admission.get("status") == "HOLD_EVAL_VALIDITY_DATA", "input is not a frozen data-admission HOLD")
    _require(held_admission.get("candidate_outputs_opened") is False, "held admission records forbidden output access")
    checks = held_admission.get("checks")
    evidence = held_admission.get("evidence")
    _require(isinstance(checks, dict) and isinstance(evidence, dict), "held admission checks/evidence are missing")
    expected_true = {
        "session_disjoint", "old_truth_session_disjoint", "parent_identity_disjoint", "exact_rgb_disjoint",
        "decoded_rgb_disjoint", "exact_source_mask_disjoint", "p_hash_prior_session_coverage_complete",
        "p_hash_prior_decode_complete", "p_hash_new_decode_complete",
    }
    _require(all(checks.get(field) is True for field in expected_true), "held admission has a failure other than manual pHash resolution")
    _require(checks.get("p_hash_no_unresolved_new_to_excluded_candidate") is False, "held admission did not record a pHash candidate HOLD")
    _require(evidence.get("p_hash_candidate_enumeration_complete") is True and isinstance(evidence.get("p_hash_candidates"), list) and evidence["p_hash_candidates"], "held pHash evidence is incomplete")
    held_sha = sha256_json(held_admission)

    _require(phash_resolution.get("schema_version") == PHASH_RESOLUTION_SCHEMA and phash_resolution.get("protocol_id") == PROTOCOL_ID, "pHash resolution schema/protocol mismatch")
    _require(phash_resolution.get("status") == PASSED_STATUS and phash_resolution.get("admission_receipt_sha256") == held_sha, "pHash resolution does not cleanly bind this held admission")
    _require(phash_resolution.get("candidate_outputs_opened") is False, "pHash resolution records forbidden output access")
    review_evidence = phash_resolution.get("evidence")
    _require(isinstance(review_evidence, dict) and review_evidence.get("all_cases_resolved_distinct") is True, "pHash review did not resolve every case as distinct")
    _require(review_evidence.get("reviewers_isolated") is True and review_evidence.get("model_or_oracle_output_visible") is False and review_evidence.get("source_mask_visible") is False, "pHash review disclosure/isolation failure")
    outcomes = review_evidence.get("outcomes")
    _require(isinstance(outcomes, list) and outcomes and all(isinstance(item, dict) and item.get("resolved_distinct") is True for item in outcomes), "pHash review outcomes are incomplete")

    reconciled_checks = dict(checks)
    reconciled_checks["p_hash_no_unresolved_new_to_excluded_candidate"] = True
    reconciled_checks["p_hash_manual_all_cases_distinct"] = True
    _require(all(value is True for value in reconciled_checks.values()), "reconciled checks are not all true")
    return {
        "schema_version": ADMISSION_RECONCILIATION_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": PASSED_STATUS_AFTER_REVIEW,
        "screening_cohort_sha256": held_admission.get("screening_cohort_sha256"),
        "materialized_manifest_sha256": held_admission.get("materialized_manifest_sha256"),
        "candidate_outputs_opened": False,
        "source_session_count": held_admission.get("source_session_count"),
        "frame_counts": held_admission.get("frame_counts"),
        "checks": reconciled_checks,
        "evidence": {
            "held_admission_receipt_sha256": held_sha,
            "p_hash_manual_resolution_sha256": sha256_json(phash_resolution),
            "raw_p_hash_candidate_count": len(evidence["p_hash_candidates"]),
            "manual_candidate_case_count": review_evidence.get("candidate_case_count"),
            "manual_reviewers_isolated": True,
            "manual_all_cases_resolved_distinct": True,
        },
        "next_required_gate": "Generate two separate opaque causal RGB P0 reviewer packets; no model/oracle trace may be materialized.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-admission-receipt", type=Path, required=True)
    parser.add_argument("--phash-resolution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), f"refusing to overwrite {args.output}")
    result = reconcile(held_admission=read_json(args.held_admission_receipt), phash_resolution=read_json(args.phash_resolution))
    result["input_sha256"] = {
        "held_admission_receipt": sha256_file(args.held_admission_receipt),
        "p_hash_resolution": sha256_file(args.phash_resolution),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
