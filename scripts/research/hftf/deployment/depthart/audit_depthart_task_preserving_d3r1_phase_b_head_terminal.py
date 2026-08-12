#!/usr/bin/env python3
"""Post-result consistency audit for the immutable D3R1 Phase-B HEAD result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_b_assets import (
    INCOMPLETE_TERMINAL,
    PASS_SUCCESSOR,
    PASS_TERMINAL,
    PROTOCOL_ID,
    RESULT_SCHEMA,
    UNAVAILABLE_TERMINAL,
    load_json,
    request_plan_sha256,
    requests_for,
    require,
    sha256_file,
    write_exclusive,
)


REPAIR_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_head_terminal_validator_repair_v1"
AUDIT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_head_post_result_audit_v1"


def verify_file(entry: dict[str, Any], label: str) -> Path:
    path = Path(entry["path"])
    require(path.is_file(), f"{label} missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"{label} bytes drift")
    require(sha256_file(path) == entry["sha256"], f"{label} SHA drift")
    return path


def independently_available(row: dict[str, Any]) -> bool:
    history = row["attempt_history"]
    final = history[-1]
    return bool(
        row["http_status"] == 200
        and final["http_status"] == 200
        and final["error"] is None
        and row["final_url"] == row["url"]
        and row["redirect_count"] == 0
        and isinstance(row["content_length_bytes"], int)
        and row["content_length_bytes"] > 0
        and row["etag"]
        and row["last_modified"]
        and row["response_body_bytes_read"] == 0
    )


def audit_result(
    result: dict[str, Any],
    protocol: dict[str, Any],
    scope: dict[str, Any],
    phase_a_result: dict[str, Any],
) -> dict[str, Any]:
    require(result.get("schema") == RESULT_SCHEMA, "result schema drift")
    require(result.get("protocol_id") == PROTOCOL_ID, "result protocol id drift")
    expected = requests_for(scope, phase_a_result, protocol["base_url"])
    actual = result["assets"]
    require(len(actual) == len(expected) == 64, "result row count drift")
    require(result["request_plan_sha256"] == request_plan_sha256(expected), "request plan SHA drift")
    available: list[bool] = []
    status_mismatch_count = 0
    redirect_mismatch_count = 0
    availability_mismatch_count = 0
    recovered_mismatch_count = 0
    for observed, planned in zip(actual, expected, strict=True):
        for key in (
            "selection_order",
            "pool_order",
            "visit_id",
            "video_id",
            "fold",
            "role",
            "asset",
            "url",
        ):
            require(observed[key] == planned[key], f"result request plan drift: {key}")
        history = observed["attempt_history"]
        require(isinstance(history, list) and history, "attempt history missing")
        require(observed["attempts"] == len(history), "attempt count drift")
        require(1 <= len(history) <= int(protocol["max_attempts"]), "attempt count range drift")
        require(
            [item["attempt"] for item in history] == list(range(1, len(history) + 1)),
            "attempt sequence drift",
        )
        require(all(item["method"] == "HEAD" for item in history), "non-HEAD attempt")
        for prior in history[:-1]:
            status = prior["http_status"]
            require(prior["error"] is not None, "successful attempt was retried")
            require(
                status is None or status in {408, 429} or 500 <= status < 600,
                "non-transient attempt was retried",
            )
        final = history[-1]
        final_status = final["http_status"]
        if final_status is None or final_status in {408, 429} or (
            isinstance(final_status, int) and 500 <= final_status < 600
        ):
            require(final["error"] is not None, "transient final attempt lacks error")
            require(len(history) == int(protocol["max_attempts"]), "transient retry stopped early")
        non_null_statuses = [item["http_status"] for item in history if item["http_status"] is not None]
        expected_row_status = non_null_statuses[-1] if non_null_statuses else None
        if observed["http_status"] != expected_row_status:
            status_mismatch_count += 1
        expected_redirect_count = int(
            isinstance(final_status, int) and 300 <= final_status < 400
        )
        if observed["redirect_count"] != expected_redirect_count:
            redirect_mismatch_count += 1
        is_available = independently_available(observed)
        available.append(is_available)
        if observed["unresolved_error"] is not (not is_available):
            availability_mismatch_count += 1
        expected_recovered = is_available and any(
            item["error"] is not None for item in history[:-1]
        )
        if observed["recovered_error"] is not expected_recovered:
            recovered_mismatch_count += 1
    require(status_mismatch_count == 0, "row/history status mismatch")
    require(redirect_mismatch_count == 0, "redirect evidence mismatch")
    require(availability_mismatch_count == 0, "availability flag mismatch")
    require(recovered_mismatch_count == 0, "recovered-error flag mismatch")
    final_statuses = [row["attempt_history"][-1]["http_status"] for row in actual]
    if any(status is None for status in final_statuses):
        terminal = INCOMPLETE_TERMINAL
    elif any(not value for value in available):
        terminal = UNAVAILABLE_TERMINAL
    else:
        terminal = PASS_TERMINAL
    require(result["terminal"] == terminal, "scientific terminal drift")
    expected_next_gate = PASS_SUCCESSOR if terminal == PASS_TERMINAL else None
    require(result.get("next_gate") == expected_next_gate, "result successor drift")
    require(result["asset_count"] == 64, "asset count drift")
    require(result["available_asset_count"] == sum(available), "available count drift")
    require(result["request_method"] == "HEAD", "request method drift")
    require(result["expected_request_count"] == 64, "expected request count drift")
    require(result["response_body_bytes_read"] == 0, "response body read summary")
    require(result["media_body_bytes_read"] == 0, "media body read")
    require(result["archive_member_read"] is False, "archive member read")
    require(result["depth_confidence_decoded"] is False, "depth/confidence decoded")
    require(result["source_truth_support_read"] is False, "source truth read")
    require(result["truth_or_model_output_read"] is False, "truth/model read")
    require(result["phase_b_selection_made"] is False, "Phase-B selection made")
    require(result["role_assignment_made"] is False, "role assignment made")
    require(result["r2_cohort_access"] == "NONE", "R2 access drift")
    total = sum(int(row["content_length_bytes"] or 0) for row in actual)
    require(result["total_content_length_bytes"] == total, "total length drift")
    return {
        "scientific_terminal": terminal,
        "asset_count": 64,
        "available_asset_count": sum(available),
        "attempt_count": sum(len(row["attempt_history"]) for row in actual),
        "head_method_count": sum(
            item["method"] == "HEAD" for row in actual for item in row["attempt_history"]
        ),
        "http_200_final_attempt_count": sum(status == 200 for status in final_statuses),
        "attempt_error_count": sum(
            item["error"] is not None for row in actual for item in row["attempt_history"]
        ),
        "redirect_count": sum(int(row["redirect_count"]) for row in actual),
        "response_body_bytes_read": sum(int(row["response_body_bytes_read"]) for row in actual),
        "etag_present_count": sum(bool(row["etag"]) for row in actual),
        "last_modified_present_count": sum(bool(row["last_modified"]) for row in actual),
        "declared_total_content_length_bytes": total,
        "row_history_status_mismatch_count": status_mismatch_count,
        "redirect_evidence_mismatch_count": redirect_mismatch_count,
        "availability_flag_mismatch_count": availability_mismatch_count,
        "recovered_error_flag_mismatch_count": recovered_mismatch_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repair = load_json(args.repair)
    require(repair.get("schema") == REPAIR_SCHEMA, "repair schema drift")
    require(repair.get("status") == "D3R1_PHASE_B_HEAD_VALIDATOR_REPAIR_FROZEN", "repair status drift")
    bindings = repair["bindings"]
    for key, label in (
        ("protocol", "protocol"),
        ("activation", "activation"),
        ("source_scope", "source scope"),
        ("phase_a_result", "Phase-A result"),
        ("head_result", "head result"),
        ("original_validation", "original validation"),
        ("auditor", "auditor"),
        ("tests", "tests"),
    ):
        verify_file(bindings[key], label)
    require(bindings["auditor"]["sha256"] == sha256_file(Path(__file__)), "auditor SHA drift")
    require(args.output.resolve() == Path(repair["output"]["path"]).resolve(), "output path drift")
    protocol = load_json(Path(bindings["protocol"]["path"]))
    scope = load_json(Path(bindings["source_scope"]["path"]))
    phase_a_result = load_json(Path(bindings["phase_a_result"]["path"]))
    result = load_json(Path(bindings["head_result"]["path"]))
    summary = audit_result(result, protocol, scope, phase_a_result)
    value = {
        "schema": AUDIT_SCHEMA,
        "status": "D3R1_PHASE_B_HEAD_POST_RESULT_AUDIT_PASS",
        "repair_sha256": sha256_file(args.repair),
        "result": bindings["head_result"],
        "original_validation": bindings["original_validation"],
        "summary": summary,
        "network_access": False,
        "head_requests_sent": 0,
        "get_requests_sent": 0,
        "response_body_bytes_read": 0,
        "media_body_bytes_read": 0,
        "next_gate": PASS_SUCCESSOR if summary["scientific_terminal"] == PASS_TERMINAL else None,
        "authority": (
            "Read-only consistency repair for the immutable HEAD result; no new transport, body, "
            "decode, source-truth-support, selection, role, outcome, R2 or deployment authority."
        ),
    }
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_exclusive(args.output, encoded)
    print(
        json.dumps(
            {
                "status": value["status"],
                **summary,
                "audit_bytes": len(encoded),
                "audit_sha256": hashlib.sha256(encoded).hexdigest().upper(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
