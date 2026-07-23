#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import exploratory_profiles_r2_l1 as r1
from r2_l1x_l2p import (
    READY_STATE,
    build_context,
    build_terminal,
    load_json,
    output_root,
)


REQUIRED_REVIEW_CHECKS = {
    "sha_bindings",
    "authority_boundaries",
    "discontinuity_resets",
    "per_ledger_coverage",
    "replacement_attack",
    "terminal_state",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--review-receipt", required=True, type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    prereg, context = build_context(repo, args.config)
    root = output_root(repo, prereg)
    terminal_path = root / "terminal-receipt-r2-l1x-l2p.json"
    if terminal_path.exists():
        raise r1.ExecutionAborted("terminal_receipt_already_exists")
    exploratory_path = root / "exploratory-terminal-receipt-r2.json"
    audit_path = root / "mechanism-gap-audit-r1.json"
    if not exploratory_path.is_file() or not audit_path.is_file():
        raise r1.ExecutionAborted("exploratory_or_mechanism_audit_missing")
    exploratory = load_json(exploratory_path)
    if exploratory.get("terminal_state") != "EXPLORATORY_PROFILES_COMPLETE":
        raise r1.ExecutionAborted("exploratory_profiles_not_complete")
    if exploratory["verified_scope"].get("fully_input_verified_sequence_ledgers") != 41:
        raise r1.ExecutionAborted("exploratory_ledger_coverage_incomplete")
    if exploratory["verified_scope"].get("fully_input_verified_frames") != 62229:
        raise r1.ExecutionAborted("exploratory_frame_coverage_incomplete")
    if len(exploratory.get("discontinuity_resets", [])) != 15:
        raise r1.ExecutionAborted("exploratory_discontinuity_reset_coverage_incomplete")
    if exploratory["candidate_execution"].get("authoritative_trace_count") != 123:
        raise r1.ExecutionAborted("exploratory_trace_coverage_incomplete")
    if len(exploratory.get("profiles", [])) != 3:
        raise r1.ExecutionAborted("exploratory_profile_count_incomplete")
    review_path = args.review_receipt.resolve()
    review = load_json(review_path)
    if review.get("schema") != "blindassist_ustrf_route_target_r2_l1x_l2p_independent_review_r1":
        raise r1.ExecutionAborted("independent_review_schema_invalid")
    if review.get("status") != "PASS":
        raise r1.ExecutionAborted("independent_review_not_pass")
    if review.get("review_mode") != "independent_agent_read_only":
        raise r1.ExecutionAborted("independent_review_mode_invalid")
    checks = review.get("checks", {})
    if set(checks) != REQUIRED_REVIEW_CHECKS or any(
        checks[name] is not True for name in REQUIRED_REVIEW_CHECKS
    ):
        raise r1.ExecutionAborted("independent_review_checks_incomplete")
    if review.get("exploratory_terminal_receipt_sha256") != r1.sha256_file(
        exploratory_path
    ):
        raise r1.ExecutionAborted("independent_review_exploratory_binding_drift")
    if review.get("mechanism_gap_audit_sha256") != r1.sha256_file(audit_path):
        raise r1.ExecutionAborted("independent_review_audit_binding_drift")
    receipt = build_terminal(READY_STATE, prereg, context, root, None)
    receipt["exploratory_profile_closure"] = {
        "path": str(exploratory_path.relative_to(repo)).replace("\\", "/"),
        "sha256": r1.sha256_file(exploratory_path),
        "candidate_count": 3,
        "authoritative_trace_count": 123,
    }
    receipt["mechanism_gap_audit"] = {
        "path": str(audit_path.relative_to(repo)).replace("\\", "/"),
        "sha256": r1.sha256_file(audit_path),
    }
    receipt["l2_fresh_selection_prereg"] = prereg["preoutput_frozen_contracts"][
        "l2_prereg"
    ]
    receipt["l3_non_executable_lockbox_template"] = prereg[
        "preoutput_frozen_contracts"
    ]["l3_lockbox_template"]
    receipt["independent_review"] = {
        "path": str(review_path.relative_to(repo)).replace("\\", "/"),
        "sha256": r1.sha256_file(review_path),
        "status": "PASS",
    }
    r1.atomic_write_json(terminal_path, receipt)
    print(
        json.dumps(
            {
                "terminal_state": READY_STATE,
                "receipt": str(terminal_path),
                "sha256": r1.sha256_file(terminal_path),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

