#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import exploratory_profiles_r2_l1 as r1
from r2_l1x_l2p import (
    LEGAL_TERMINAL_STATES,
    MECHANISM_AUDIT_SCHEMA,
    OVERALL_SCHEMA,
    READY_STATE,
    build_context,
    load_json,
    output_root,
    verified_scope,
)


FORBIDDEN_KEY_FRAGMENTS = {
    "winner",
    "ranking",
    "best_candidate",
    "recommended_candidate",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise r1.ExecutionAborted(message)


def walk_forbidden(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                raise r1.ExecutionAborted(f"forbidden_selection_field:{path}.{key}")
            walk_forbidden(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_forbidden(child, f"{path}[{index}]")


def validate(
    repo: Path, config_path: Path, write_receipt: bool = True
) -> dict[str, Any]:
    prereg, context = build_context(repo, config_path)
    root = output_root(repo, prereg)
    root.mkdir(parents=True, exist_ok=True)
    gaps, ledgers, frames = verified_scope(
        context["groups"], root, context["route_map"]
    )
    terminal_path = root / "terminal-receipt-r2-l1x-l2p.json"
    result: dict[str, Any] = {
        "status": "VALID_R2_L1X_L2P_PREREG",
        "terminal_state": None,
        "verified_sequence_ledgers": ledgers,
        "verified_frames": frames,
        "expected_sequence_ledgers": 41,
        "expected_frames": 62229,
        "discontinuity_resets": len(context["resets"]),
        "r1_parent_preserved": True,
        "l2_l3_preoutput_freeze_valid": True,
    }
    if terminal_path.exists():
        terminal = load_json(terminal_path)
        require(terminal.get("schema") == OVERALL_SCHEMA, "terminal_schema_drift")
        require(terminal.get("stage") == "R2-L1X-L2P", "terminal_stage_drift")
        state = terminal.get("terminal_state")
        require(state in LEGAL_TERMINAL_STATES, "illegal_terminal_state")
        require(
            terminal.get("bindings", {}).get("r2_l1x_l2p_prereg_sha256")
            == context["bindings"]["r2_l1x_l2p_prereg_sha256"],
            "terminal_prereg_binding_drift",
        )
        require(
            terminal.get("immutable_r1_parent_summary") == context["parent_summary"],
            "terminal_r1_parent_summary_drift",
        )
        boundary = terminal.get("claim_boundary", {})
        require(boundary and all(value is False for value in boundary.values()), "authority_opened")
        if state == READY_STATE:
            require(ledgers == 41 and frames == 62229, "ready_input_coverage_incomplete")
            require(not any(row["missing_fields"] for row in gaps), "ready_gap_present")
            exploratory_binding = terminal.get("exploratory_profile_closure", {})
            exploratory_path = repo / exploratory_binding.get("path", "")
            require(exploratory_path.is_file(), "ready_exploratory_receipt_missing")
            require(
                r1.sha256_file(exploratory_path)
                == exploratory_binding.get("sha256"),
                "ready_exploratory_receipt_hash_drift",
            )
            exploratory = load_json(exploratory_path)
            require(
                exploratory.get("terminal_state") == "EXPLORATORY_PROFILES_COMPLETE",
                "ready_exploratory_not_complete",
            )
            require(
                exploratory["verified_scope"]["fully_input_verified_sequence_ledgers"]
                == 41,
                "ready_exploratory_ledger_coverage_incomplete",
            )
            require(
                exploratory["verified_scope"]["fully_input_verified_frames"] == 62229,
                "ready_exploratory_frame_coverage_incomplete",
            )
            require(
                len(exploratory.get("discontinuity_resets", [])) == 15,
                "ready_reset_coverage_incomplete",
            )
            require(
                exploratory["candidate_execution"]["authoritative_trace_count"]
                == 123,
                "ready_trace_coverage_incomplete",
            )
            require(len(exploratory.get("profiles", [])) == 3, "ready_profiles_missing")
            audit_binding = terminal.get("mechanism_gap_audit", {})
            audit_path = repo / audit_binding.get("path", "")
            require(audit_path.is_file(), "ready_mechanism_audit_missing")
            require(
                r1.sha256_file(audit_path) == audit_binding.get("sha256"),
                "ready_mechanism_audit_hash_drift",
            )
            audit = load_json(audit_path)
            require(audit.get("schema") == MECHANISM_AUDIT_SCHEMA, "audit_schema_drift")
            require(
                audit.get("claim_boundary")
                == {
                    "winner": False,
                    "rank": False,
                    "best_candidate": False,
                    "promotion": False,
                    "pooled_overrides_source_failure": False,
                    "not_evaluable_recoded_as_zero": False,
                },
                "audit_authority_drift",
            )
            review_binding = terminal.get("independent_review", {})
            review_path = repo / review_binding.get("path", "")
            require(review_path.is_file(), "independent_review_missing")
            require(
                r1.sha256_file(review_path) == review_binding.get("sha256"),
                "independent_review_hash_drift",
            )
            review = load_json(review_path)
            require(review.get("status") == "PASS", "independent_review_not_pass")
            result["status"] = "VALID_L2_FRESH_SELECTION_PREREG_READY"
        else:
            require(
                terminal["verified_scope"].get("first_blocker") is not None,
                "failure_terminal_without_blocker",
            )
            result["status"] = f"VALID_{state}"
        walk_forbidden(terminal)
        result["terminal_state"] = state
        result["terminal_receipt_sha256"] = r1.sha256_file(terminal_path)
    if write_receipt:
        r1.atomic_write_json(root / "validation-receipt-r2-l1x-l2p.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    result = validate(args.repo.resolve(), args.config, write_receipt=True)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

