#!/usr/bin/env python3
"""Offline validation for the D3R1 exact-254 HEAD preflight result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.research.hftf.deployment.depthart.preflight_depthart_task_preserving_d3r1_phase_a_assets import (
    ASSETS,
    PASS_TERMINAL,
    PASS_SUCCESSOR,
    PROTOCOL_ID,
    RESULT_SCHEMA,
    load_json,
    requests_for,
    require,
    row_available,
    sha256_file,
    validate_bindings,
    write_exclusive,
)


VALIDATION_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_a_head_validation_v1"


def validate_result(
    result: dict[str, Any], protocol: dict[str, Any], roster: dict[str, Any]
) -> dict[str, Any]:
    require(result.get("schema") == RESULT_SCHEMA, "result schema drift")
    require(result.get("protocol_id") == PROTOCOL_ID, "result protocol id drift")
    require(result.get("terminal") == PASS_TERMINAL, "HEAD result is not PASS")
    require(result.get("next_gate") == PASS_SUCCESSOR, "result successor drift")
    expected = requests_for(roster, protocol["base_url"])
    actual = result["assets"]
    require(len(actual) == len(expected) == 254, "result row count drift")
    for observed, planned in zip(actual, expected, strict=True):
        for key in ("pool_order", "visit_id", "video_id", "fold", "role", "asset", "url"):
            require(observed[key] == planned[key], f"result request plan drift: {key}")
        require(row_available(observed), "unavailable result row")
        require(observed["attempts"] >= 1, "invalid attempt count")
        require(observed["attempts"] <= int(protocol["max_attempts"]), "attempt count exceeds protocol")
        require(isinstance(observed["attempt_history"], list), "attempt history schema drift")
        require(
            len(observed["attempt_history"]) == observed["attempts"]
            and all(item["method"] == "HEAD" for item in observed["attempt_history"]),
            "attempt history drift",
        )
        require(
            observed["attempt_history"][-1]["error"] is None,
            "PASS row final attempt was not successful",
        )
        require(
            bool(observed["recovered_error"])
            == any(item["error"] is not None for item in observed["attempt_history"][:-1]),
            "recovered-error evidence drift",
        )
    require(result["request_method"] == "HEAD", "result method drift")
    require(result["expected_request_count"] == 254, "expected request drift")
    require(result["asset_count"] == 254, "asset count drift")
    require(result["available_asset_count"] == 254, "available count drift")
    require(result["response_body_bytes_read"] == 0, "response body read")
    require(result["media_body_bytes_read"] == 0, "media body read")
    require(result["archive_member_read"] is False, "archive read")
    require(result["pose_content_read"] is False, "pose content read")
    require(result["truth_or_model_output_read"] is False, "truth/model read")
    require(result["role_assignment_made"] is False, "role assigned")
    require(result["r2_cohort_access"] == "NONE", "R2 access drift")
    require("HEAD availability and declared-size evidence only" in result["authority"], "authority drift")
    total = sum(row["content_length_bytes"] for row in actual)
    require(total == result["total_content_length_bytes"] and total > 0, "total length drift")
    by_asset = {
        asset: {
            "count": sum(row["asset"] == asset for row in actual),
            "declared_content_length_bytes": sum(
                row["content_length_bytes"] for row in actual if row["asset"] == asset
            ),
        }
        for asset in ASSETS
    }
    require(all(item["count"] == 127 for item in by_asset.values()), "asset family count drift")
    return {
        "asset_count": 254,
        "available_asset_count": 254,
        "declared_total_content_length_bytes": total,
        "etag_present_count": sum(bool(row["etag"]) for row in actual),
        "last_modified_present_count": sum(bool(row["last_modified"]) for row in actual),
        "redirect_count": sum(bool(row["redirected"]) for row in actual),
        "maximum_attempts": max(row["attempts"] for row in actual),
        "recovered_error_row_count": sum(bool(row["recovered_error"]) for row in actual),
        "unresolved_error_row_count": sum(bool(row["unresolved_error"]) for row in actual),
        "by_asset": by_asset,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, roster = validate_bindings(
        args.protocol, args.roster, args.source_scope, args.activation
    )
    result = load_json(args.result)
    require(args.result.resolve() == Path(protocol["output"]["artifact_path"]).resolve(), "result path drift")
    require(args.output.resolve() == Path(protocol["output"]["validation_path"]).resolve(), "validation path drift")
    require(result["protocol_sha256"] == sha256_file(args.protocol), "protocol result drift")
    require(result["roster_sha256"] == sha256_file(args.roster), "roster result drift")
    require(result["source_scope_sha256"] == sha256_file(args.source_scope), "scope result drift")
    require(result["activation_sha256"] == sha256_file(args.activation), "activation result drift")
    summary = validate_result(result, protocol, roster)
    value = {
        "schema": VALIDATION_SCHEMA,
        "status": "D3R1_PHASE_A_HEAD_RESULT_OFFLINE_VALIDATION_PASS",
        "result": {
            "path": str(args.result.resolve()),
            "bytes": args.result.stat().st_size,
            "sha256": sha256_file(args.result),
        },
        "summary": summary,
        "media_body_bytes_read": 0,
        "next_gate": PASS_SUCCESSOR,
        "authority": "Offline validation of exact-254 HEAD evidence only; no body, continuity, truth, training, Development, R2, performance, production or safety authority.",
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
