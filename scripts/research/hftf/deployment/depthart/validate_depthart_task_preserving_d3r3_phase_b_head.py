#!/usr/bin/env python3
"""Independent offline validation for D3R3 exact-64 fresh HEAD evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = (
    "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_head_protocol_v1"
)
SCOPE_SCHEMA = (
    "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_scope_receipt_v1"
)
PHASE_A_RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_a_governed_result_v1"
ACTIVATION_SCHEMA = (
    "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_head_activation_v1"
)
RESULT_SCHEMA = (
    "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_head_preflight_v1"
)
VALIDATION_SCHEMA = (
    "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_head_validation_v1"
)
PROTOCOL_ID = (
    "DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_TRANSPORT_RECOVERY_HEAD_ONLY"
)
ASSETS = ("lowres_depth.zip", "confidence.zip")
PASS_TERMINAL = "D3R3_PHASE_B_EXACT64_FRESH_HEAD_PASS_MEDIA_BODY_UNOPENED"
UNAVAILABLE_TERMINAL = "D3R3_PHASE_B_FRESH_HEAD_ASSETS_NOT_AVAILABLE_MEDIA_BODY_UNOPENED"
INCOMPLETE_TERMINAL = "D3R3_PHASE_B_FRESH_HEAD_EXECUTION_INCOMPLETE_MEDIA_BODY_UNOPENED"
PASS_SUCCESSOR = "EXPLICIT_D3R3_PHASE_B_EXACT64_COVERAGE_ONLY_CENSUS_PROTOCOL_REGISTRATION"
ALLOWED_TERMINALS = {PASS_TERMINAL, UNAVAILABLE_TERMINAL, INCOMPLETE_TERMINAL}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def verify_file(entry: dict[str, Any], label: str) -> Path:
    path = Path(entry["path"])
    require(path.is_file(), f"{label} missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"{label} bytes drift")
    require(sha256_file(path) == entry["sha256"], f"{label} SHA drift")
    return path


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, "exclusive write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def selection_rows(
    source_scope: dict[str, Any], phase_a_result: dict[str, Any]
) -> list[dict[str, Any]]:
    require(source_scope.get("schema") == SCOPE_SCHEMA, "source-scope schema drift")
    require(
        source_scope.get("status")
        == "D3R3_PHASE_B_SOURCE_COVERAGE_TRANSPORT_RECOVERY_SCOPE_REGISTERED_MEDIA_UNOPENED",
        "source-scope status drift",
    )
    require(
        source_scope.get("next_gate")
        == "EXPLICIT_D3R3_PHASE_B_EXACT64_FRESH_HEAD_ONLY_PREFLIGHT_ACTIVATION",
        "source-scope next gate drift",
    )
    require(phase_a_result.get("schema") == PHASE_A_RESULT_SCHEMA, "Phase-A schema drift")
    require(
        phase_a_result.get("terminal")
        == "D3R1_PHASE_A_PORTRAIT_POSE_CONTINUITY_PASS_32_IDENTITIES_LOCKED",
        "Phase-A terminal drift",
    )
    scope_rows = source_scope["exact_phase_a_selection"]["identities"]
    result_rows = phase_a_result["selected_phase_a"]
    require(len(scope_rows) == len(result_rows) == 32, "expected exact 32 identities")
    rows: list[dict[str, Any]] = []
    encoded: list[str] = []
    for expected_order, (scope_row, result_row) in enumerate(
        zip(scope_rows, result_rows, strict=True), start=1
    ):
        for key in ("selection_order", "pool_order", "visit_id", "video_id"):
            require(scope_row[key] == result_row[key], f"Phase-A selection mismatch: {key}")
        require(scope_row["selection_order"] == expected_order, "selection order drift")
        require(
            isinstance(scope_row["visit_id"], str) and scope_row["visit_id"].isdigit(),
            "visit id drift",
        )
        require(
            isinstance(scope_row["video_id"], str) and scope_row["video_id"].isdigit(),
            "video id drift",
        )
        row = {
            "selection_order": expected_order,
            "pool_order": scope_row["pool_order"],
            "visit_id": scope_row["visit_id"],
            "video_id": scope_row["video_id"],
            "fold": "Training",
            "role": "D3R1_PHASE_A_SELECTED_IDENTITY_ONLY",
        }
        rows.append(row)
        encoded.append(
            f"{row['selection_order']}/{row['pool_order']}/{row['visit_id']}/{row['video_id']}"
        )
    require(len({row["visit_id"] for row in rows}) == 32, "visit overlap")
    require(len({row["video_id"] for row in rows}) == 32, "video overlap")
    digest = hashlib.sha256(("\n".join(encoded) + "\n").encode("ascii")).hexdigest().upper()
    require(
        digest == source_scope["exact_phase_a_selection"]["selection_sha256"],
        "selection SHA drift",
    )
    registered = source_scope["registered_asset_scope"]
    require(tuple(registered["assets"]) == ASSETS, "registered assets drift")
    require(registered["fresh_head_request_count"] == 64, "registered HEAD count drift")
    require(registered["future_body_asset_count"] == 64, "registered body count drift")
    require(registered["phase_c_rgb_registered"] is False, "RGB scope widened")
    return rows


def requests_for(
    source_scope: dict[str, Any], phase_a_result: dict[str, Any], base_url: str
) -> list[dict[str, Any]]:
    requests = [
        row
        | {
            "asset": asset,
            "url": f"{base_url}/{row['fold']}/{row['video_id']}/{asset}",
        }
        for row in selection_rows(source_scope, phase_a_result)
        for asset in ASSETS
    ]
    require(len(requests) == 64, "asset request count drift")
    require(len({row["url"] for row in requests}) == 64, "duplicate asset URL")
    return requests


def request_plan_sha256(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"{row['selection_order']}/{row['pool_order']}/{row['visit_id']}/"
        f"{row['video_id']}/{row['asset']}/{row['url']}"
        for row in rows
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest().upper()


def prior_head_lookup(
    prior_head_result: dict[str, Any], requests: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    require(
        prior_head_result.get("schema")
        == "blindassist_depthart_task_preserving_d3r1_phase_b_asset_header_preflight_v1",
        "prior HEAD schema drift",
    )
    prior_rows = prior_head_result.get("assets")
    require(isinstance(prior_rows, list) and len(prior_rows) == 64, "prior HEAD row count drift")
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in prior_rows:
        key = (row["video_id"], row["asset"])
        require(key not in lookup, "duplicate prior HEAD row")
        require(
            row["http_status"] == 200
            and row["final_url"] == row["url"]
            and row["redirect_count"] == 0
            and isinstance(row["content_length_bytes"], int)
            and row["content_length_bytes"] > 0
            and row["etag"]
            and row["last_modified"]
            and row["response_body_bytes_read"] == 0,
            "prior HEAD row not authoritative",
        )
        lookup[key] = row
    require(
        set(lookup) == {(row["video_id"], row["asset"]) for row in requests},
        "prior HEAD plan drift",
    )
    return lookup


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


def validate_result(
    result: dict[str, Any],
    protocol: dict[str, Any],
    scope: dict[str, Any],
    phase_a_result: dict[str, Any],
    prior_head_result: dict[str, Any],
) -> dict[str, Any]:
    require(result.get("schema") == RESULT_SCHEMA, "result schema drift")
    require(result.get("protocol_id") == PROTOCOL_ID, "result protocol id drift")
    require(result.get("head_terminal") in ALLOWED_TERMINALS, "unknown HEAD terminal")
    require(result.get("scientific_terminal") is None, "scientific terminal opened")
    require(result.get("selection_evaluated") is False, "selection evaluated")
    require(result.get("selected_phase_b") is None, "Phase-B selection published")
    expected = requests_for(scope, phase_a_result, protocol["base_url"])
    prior = prior_head_lookup(prior_head_result, expected)
    actual = result["assets"]
    require(len(actual) == len(expected) == 64, "result row count drift")
    require(result["request_plan_sha256"] == request_plan_sha256(expected), "request plan SHA drift")
    available: list[bool] = []
    drift_counts = {"content_length": 0, "etag": 0, "last_modified": 0}
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
        old = prior[(planned["video_id"], planned["asset"])]
        require(
            observed["prior_head"]
            == {
                "content_length_bytes": old["content_length_bytes"],
                "etag": old["etag"],
                "last_modified": old["last_modified"],
            },
            "prior HEAD binding drift",
        )
        history = observed["attempt_history"]
        require(isinstance(history, list) and history, "attempt history missing")
        require(observed["attempts"] == len(history), "attempt count drift")
        require(1 <= len(history) <= int(protocol["max_attempts"]), "attempt count range drift")
        require(
            [item["attempt"] for item in history] == list(range(1, len(history) + 1)),
            "attempt sequence drift",
        )
        require(all(item["method"] == "HEAD" for item in history), "non-HEAD attempt")
        for prior_attempt in history[:-1]:
            status = prior_attempt["http_status"]
            require(prior_attempt["error"] is not None, "successful attempt was retried")
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
        non_null = [item["http_status"] for item in history if item["http_status"] is not None]
        require(
            observed["http_status"] == (non_null[-1] if non_null else None),
            "row/history status mismatch",
        )
        expected_redirect = int(isinstance(final_status, int) and 300 <= final_status < 400)
        require(observed["redirect_count"] == expected_redirect, "redirect evidence mismatch")
        is_available = independently_available(observed)
        available.append(is_available)
        require(observed["unresolved_error"] is (not is_available), "availability flag mismatch")
        expected_recovered = is_available and any(item["error"] for item in history[:-1])
        require(observed["recovered_error"] is expected_recovered, "recovered flag mismatch")
        expected_drift = {
            "content_length": (
                observed["content_length_bytes"] != old["content_length_bytes"]
                if observed["content_length_bytes"] is not None
                else None
            ),
            "etag": observed["etag"] != old["etag"] if observed["etag"] is not None else None,
            "last_modified": (
                observed["last_modified"] != old["last_modified"]
                if observed["last_modified"] is not None
                else None
            ),
        }
        require(observed["header_drift_from_prior"] == expected_drift, "header drift evidence mismatch")
        for key, changed in expected_drift.items():
            drift_counts[key] += int(changed is True)
    final_statuses = [row["attempt_history"][-1]["http_status"] for row in actual]
    if any(status is None for status in final_statuses):
        terminal = INCOMPLETE_TERMINAL
    elif any(not value for value in available):
        terminal = UNAVAILABLE_TERMINAL
    else:
        terminal = PASS_TERMINAL
    require(result["head_terminal"] == terminal, "HEAD terminal drift")
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
    require(result["fresh_head_authority"] is (terminal == PASS_TERMINAL), "HEAD authority drift")
    total = sum(int(row["content_length_bytes"] or 0) for row in actual)
    require(result["total_content_length_bytes"] == total, "total length drift")
    by_asset = {
        asset: {
            "count": sum(row["asset"] == asset for row in actual),
            "available_count": sum(
                row["asset"] == asset and independently_available(row) for row in actual
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
        "head_terminal": terminal,
        "scientific_terminal": None,
        "asset_count": 64,
        "available_asset_count": sum(available),
        "declared_total_content_length_bytes": total,
        "etag_present_count": sum(bool(row["etag"]) for row in actual),
        "last_modified_present_count": sum(bool(row["last_modified"]) for row in actual),
        "redirect_count": sum(int(row["redirect_count"]) for row in actual),
        "maximum_attempts": max(row["attempts"] for row in actual),
        "recovered_error_row_count": sum(bool(row["recovered_error"]) for row in actual),
        "unresolved_error_row_count": sum(bool(row["unresolved_error"]) for row in actual),
        "header_drift_from_prior_counts": drift_counts,
        "by_asset": by_asset,
    }


def validate_bindings(
    protocol_path: Path,
    source_scope_path: Path,
    phase_a_result_path: Path,
    prior_head_result_path: Path,
    activation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    scope = load_json(source_scope_path)
    phase_a_result = load_json(phase_a_result_path)
    prior_head_result = load_json(prior_head_result_path)
    activation = load_json(activation_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id drift")
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    require(
        set(activation)
        == {
            "schema",
            "activation_id",
            "activated_at",
            "activated_by",
            "authorization_verbatim",
            "authorization_context",
            "status",
            "bindings",
            "execution",
            "authority",
            "forbidden",
            "next_action",
        },
        "activation top-level schema drift",
    )
    require(
        activation.get("status")
        == "D3R3_PHASE_B_EXACT64_FRESH_HEAD_ONLY_PREFLIGHT_ACTIVATED",
        "activation status drift",
    )
    for key, label in (
        ("owning_d3_router_protocol", "owning D3 router protocol"),
        ("phase_a_result", "Phase-A result"),
        ("prior_source_scope", "prior source scope"),
        ("d3r2_protocol", "D3R2 protocol"),
        ("d3r2_execution_stop", "D3R2 execution stop"),
        ("prior_head_result", "prior HEAD result"),
        ("prior_head_terminal_audit", "prior HEAD terminal audit"),
        ("source_scope", "source scope"),
        ("producer", "producer"),
        ("tests", "producer tests"),
        ("validator", "validator"),
        ("validator_tests", "validator tests"),
    ):
        verify_file(protocol[key], label)
    require(protocol["validator"]["sha256"] == sha256_file(Path(__file__)), "validator SHA drift")
    require(protocol["source_scope"]["sha256"] == sha256_file(source_scope_path), "scope SHA drift")
    require(protocol["phase_a_result"]["sha256"] == sha256_file(phase_a_result_path), "Phase-A SHA drift")
    require(
        protocol["prior_head_result"]["sha256"] == sha256_file(prior_head_result_path),
        "prior HEAD SHA drift",
    )
    require(
        activation["bindings"]["head_protocol"]["sha256"] == sha256_file(protocol_path),
        "activation protocol mismatch",
    )
    require(
        set(activation["bindings"])
        == {
            "head_protocol",
            "source_scope",
            "phase_a_result",
            "prior_head_result",
            "d3r2_execution_stop",
        },
        "activation binding schema drift",
    )
    require(
        set(activation["authority"])
        == {
            "fresh_exact64_head",
            "media_get",
            "media_body",
            "range_get",
            "redirect_following",
            "archive_member_read",
            "depth_confidence_decode",
            "source_truth_support",
            "phase_b_selection",
            "phase_c_rgb",
            "truth",
            "model_output",
            "role_assignment",
            "training",
            "development_outcome",
            "r2",
            "performance",
            "android_default",
            "production",
            "safety",
        },
        "activation authority schema drift",
    )
    require(
        set(activation["execution"])
        == {
            "one_shot",
            "method",
            "identity_count",
            "asset_count_per_identity",
            "expected_request_count",
            "assets",
            "request_plan_sha256",
            "fresh_output_root_required",
            "output_overwrite_forbidden",
            "redirect_following_forbidden",
            "response_body_bytes_read_required",
        },
        "activation execution schema drift",
    )
    execution = activation["execution"]
    require(
        execution["one_shot"] is True
        and execution["method"] == "HEAD"
        and execution["identity_count"] == 32
        and execution["asset_count_per_identity"] == 2
        and execution["expected_request_count"] == 64
        and tuple(execution["assets"]) == ASSETS
        and execution["request_plan_sha256"] == protocol["request_plan_sha256"]
        and execution["fresh_output_root_required"] is True
        and execution["output_overwrite_forbidden"] is True
        and execution["redirect_following_forbidden"] is True
        and execution["response_body_bytes_read_required"] == 0,
        "activation execution boundary drift",
    )
    require(
        activation["bindings"]["source_scope"]["sha256"]
        == protocol["source_scope"]["sha256"]
        and activation["bindings"]["phase_a_result"]["sha256"]
        == protocol["phase_a_result"]["sha256"]
        and activation["bindings"]["prior_head_result"]["sha256"]
        == protocol["prior_head_result"]["sha256"]
        and activation["bindings"]["d3r2_execution_stop"]["sha256"]
        == protocol["d3r2_execution_stop"]["sha256"],
        "activation binding mismatch",
    )
    require(activation["authority"]["fresh_exact64_head"] is True, "fresh HEAD not authorized")
    for key in (
        "media_get",
        "media_body",
        "range_get",
        "redirect_following",
        "archive_member_read",
        "depth_confidence_decode",
        "source_truth_support",
        "phase_b_selection",
        "phase_c_rgb",
        "truth",
        "model_output",
        "role_assignment",
        "training",
        "development_outcome",
        "r2",
        "performance",
        "android_default",
        "production",
        "safety",
    ):
        require(activation["authority"][key] is False, f"activation authority widened: {key}")
    requests = requests_for(scope, phase_a_result, protocol["base_url"])
    require(request_plan_sha256(requests) == protocol["request_plan_sha256"], "request plan drift")
    prior_head_lookup(prior_head_result, requests)
    return protocol, scope, phase_a_result, prior_head_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--phase-a-result", type=Path, required=True)
    parser.add_argument("--prior-head-result", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, scope, phase_a_result, prior_head_result = validate_bindings(
        args.protocol,
        args.source_scope,
        args.phase_a_result,
        args.prior_head_result,
        args.activation,
    )
    require(
        args.result.resolve() == Path(protocol["output"]["artifact_path"]).resolve(),
        "result path drift",
    )
    require(
        args.output.resolve() == Path(protocol["output"]["validation_path"]).resolve(),
        "validation path drift",
    )
    result = load_json(args.result)
    require(result["protocol_sha256"] == sha256_file(args.protocol), "protocol result drift")
    require(result["source_scope_sha256"] == sha256_file(args.source_scope), "scope result drift")
    require(result["phase_a_result_sha256"] == sha256_file(args.phase_a_result), "Phase-A drift")
    require(
        result["prior_head_result_sha256"] == sha256_file(args.prior_head_result),
        "prior HEAD drift",
    )
    require(result["activation_sha256"] == sha256_file(args.activation), "activation drift")
    summary = validate_result(result, protocol, scope, phase_a_result, prior_head_result)
    value = {
        "schema": VALIDATION_SCHEMA,
        "status": "D3R3_PHASE_B_FRESH_HEAD_OFFLINE_VALIDATION_PASS",
        "result": {
            "path": str(args.result.resolve()),
            "bytes": args.result.stat().st_size,
            "sha256": sha256_file(args.result),
        },
        "summary": summary,
        "response_body_bytes_read": 0,
        "media_body_bytes_read": 0,
        "scientific_terminal": None,
        "selection_evaluated": False,
        "selected_phase_b": None,
        "next_gate": PASS_SUCCESSOR if summary["head_terminal"] == PASS_TERMINAL else None,
        "authority": (
            "Independent validation of fresh exact-64 HEAD evidence only; no GET, body, archive, "
            "truth, selection, role, outcome, R2 or deployment authority."
        ),
    }
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_exclusive(args.output, encoded)
    print(
        json.dumps(
            {
                "status": value["status"],
                **summary,
                "validation_bytes": len(encoded),
                "validation_sha256": hashlib.sha256(encoded).hexdigest().upper(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
