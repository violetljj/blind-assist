#!/usr/bin/env python3
"""Offline validation for a D3R1 exact-64 Phase-B HEAD result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_b_assets import (
    ASSETS,
    INCOMPLETE_TERMINAL,
    PASS_SUCCESSOR,
    PASS_TERMINAL,
    PROTOCOL_ID,
    RESULT_SCHEMA,
    UNAVAILABLE_TERMINAL,
    disposition,
    load_json,
    request_plan_sha256,
    requests_for,
    require,
    row_available,
    sha256_file,
    validate_bindings,
    write_exclusive,
)


VALIDATION_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_b_head_validation_v1"
ALLOWED_TERMINALS = {PASS_TERMINAL, UNAVAILABLE_TERMINAL, INCOMPLETE_TERMINAL}


def validate_result(
    result: dict[str, Any],
    protocol: dict[str, Any],
    scope: dict[str, Any],
    phase_a_result: dict[str, Any],
) -> dict[str, Any]:
    require(result.get("schema") == RESULT_SCHEMA, "result schema drift")
    require(result.get("protocol_id") == PROTOCOL_ID, "result protocol id drift")
    require(result.get("terminal") in ALLOWED_TERMINALS, "unknown scientific terminal")
    expected = requests_for(scope, phase_a_result, protocol["base_url"])
    actual = result["assets"]
    require(len(actual) == len(expected) == 64, "result row count drift")
    require(result["request_plan_sha256"] == request_plan_sha256(expected), "request plan SHA drift")
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
        require(1 <= observed["attempts"] <= int(protocol["max_attempts"]), "attempt count drift")
        require(isinstance(observed["attempt_history"], list), "attempt history schema drift")
        require(
            len(observed["attempt_history"]) == observed["attempts"]
            and all(item["method"] == "HEAD" for item in observed["attempt_history"]),
            "attempt history drift",
        )
        history = observed["attempt_history"]
        require(
            [item["attempt"] for item in history] == list(range(1, observed["attempts"] + 1)),
            "attempt sequence drift",
        )
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
            require(observed["attempts"] == int(protocol["max_attempts"]), "transient retry stopped early")
        if observed["redirect_count"]:
            require(
                isinstance(final_status, int)
                and 300 <= final_status < 400
                and final["error"] is not None,
                "redirect evidence drift",
            )
        require(observed["response_body_bytes_read"] == 0, "response body read")
        if row_available(observed):
            require(observed["attempt_history"][-1]["error"] is None, "available row final error")
            require(
                bool(observed["recovered_error"])
                == any(item["error"] is not None for item in observed["attempt_history"][:-1]),
                "recovered-error evidence drift",
            )
        else:
            require(observed["unresolved_error"] is True, "unavailable row not unresolved")
    recomputed_terminal = disposition(actual)
    require(result["terminal"] == recomputed_terminal, "scientific terminal drift")
    expected_next_gate = PASS_SUCCESSOR if recomputed_terminal == PASS_TERMINAL else None
    require(result.get("next_gate") == expected_next_gate, "result successor drift")
    require(result["request_method"] == "HEAD", "result method drift")
    require(result["expected_request_count"] == 64, "expected request drift")
    require(result["asset_count"] == 64, "asset count drift")
    require(
        result["available_asset_count"] == sum(row_available(row) for row in actual),
        "available count drift",
    )
    require(result["response_body_bytes_read"] == 0, "response body read summary")
    require(result["media_body_bytes_read"] == 0, "media body read")
    require(result["archive_member_read"] is False, "archive read")
    require(result["depth_confidence_decoded"] is False, "depth/confidence decoded")
    require(result["source_truth_support_read"] is False, "source truth read")
    require(result["truth_or_model_output_read"] is False, "truth/model read")
    require(result["phase_b_selection_made"] is False, "Phase-B selection made")
    require(result["role_assignment_made"] is False, "role assigned")
    require(result["r2_cohort_access"] == "NONE", "R2 access drift")
    require("HEAD availability and declared-size evidence only" in result["authority"], "authority drift")
    total = sum(int(row["content_length_bytes"] or 0) for row in actual)
    require(total == result["total_content_length_bytes"], "total length drift")
    by_asset = {
        asset: {
            "count": sum(row["asset"] == asset for row in actual),
            "available_count": sum(
                row["asset"] == asset and row_available(row) for row in actual
            ),
            "declared_content_length_bytes": sum(
                int(row["content_length_bytes"] or 0)
                for row in actual
                if row["asset"] == asset
            ),
        }
        for asset in ASSETS
    }
    require(all(item["count"] == 32 for item in by_asset.values()), "asset family count drift")
    return {
        "scientific_terminal": recomputed_terminal,
        "asset_count": 64,
        "available_asset_count": sum(row_available(row) for row in actual),
        "declared_total_content_length_bytes": total,
        "etag_present_count": sum(bool(row["etag"]) for row in actual),
        "last_modified_present_count": sum(bool(row["last_modified"]) for row in actual),
        "redirect_count": sum(int(row["redirect_count"]) for row in actual),
        "maximum_attempts": max(row["attempts"] for row in actual),
        "recovered_error_row_count": sum(bool(row["recovered_error"]) for row in actual),
        "unresolved_error_row_count": sum(bool(row["unresolved_error"]) for row in actual),
        "by_asset": by_asset,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--phase-a-result", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, scope, phase_a_result = validate_bindings(
        args.protocol, args.source_scope, args.phase_a_result, args.activation
    )
    result = load_json(args.result)
    require(
        args.result.resolve() == Path(protocol["output"]["artifact_path"]).resolve(),
        "result path drift",
    )
    require(
        args.output.resolve() == Path(protocol["output"]["validation_path"]).resolve(),
        "validation path drift",
    )
    require(result["protocol_sha256"] == sha256_file(args.protocol), "protocol result drift")
    require(result["source_scope_sha256"] == sha256_file(args.source_scope), "scope result drift")
    require(
        result["phase_a_result_sha256"] == sha256_file(args.phase_a_result),
        "Phase-A result drift",
    )
    require(result["activation_sha256"] == sha256_file(args.activation), "activation result drift")
    summary = validate_result(result, protocol, scope, phase_a_result)
    value = {
        "schema": VALIDATION_SCHEMA,
        "status": "D3R1_PHASE_B_HEAD_RESULT_OFFLINE_VALIDATION_PASS",
        "result": {
            "path": str(args.result.resolve()),
            "bytes": args.result.stat().st_size,
            "sha256": sha256_file(args.result),
        },
        "summary": summary,
        "response_body_bytes_read": 0,
        "media_body_bytes_read": 0,
        "next_gate": PASS_SUCCESSOR if summary["scientific_terminal"] == PASS_TERMINAL else None,
        "authority": (
            "Offline validation of exact-64 HEAD evidence only; no body, decode, source-truth-"
            "support, selection, roles, training, Development, R2, performance, production or "
            "safety authority."
        ),
    }
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_exclusive(args.output, encoded)
    print(
        json.dumps(
            {
                "status": value["status"],
                **summary,
                "result_bytes": len(encoded),
                "result_sha256": hashlib.sha256(encoded).hexdigest().upper(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
