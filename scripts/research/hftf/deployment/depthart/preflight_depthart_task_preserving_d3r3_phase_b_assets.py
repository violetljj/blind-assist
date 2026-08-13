#!/usr/bin/env python3
"""Fresh HEAD-only source snapshot for D3R3 Phase-B transport recovery."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


PROTOCOL_SCHEMA = (
    "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_head_protocol_v1"
)
SCOPE_SCHEMA = "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_scope_receipt_v1"
PHASE_A_RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_a_governed_result_v1"
ACTIVATION_SCHEMA = (
    "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_head_activation_v1"
)
RESULT_SCHEMA = (
    "blindassist_depthart_task_preserving_d3r3_phase_b_source_coverage_head_preflight_v1"
)
PROTOCOL_ID = (
    "DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_TRANSPORT_RECOVERY_HEAD_ONLY"
)
ASSETS = ("lowres_depth.zip", "confidence.zip")
PASS_TERMINAL = "D3R3_PHASE_B_EXACT64_FRESH_HEAD_PASS_MEDIA_BODY_UNOPENED"
UNAVAILABLE_TERMINAL = "D3R3_PHASE_B_FRESH_HEAD_ASSETS_NOT_AVAILABLE_MEDIA_BODY_UNOPENED"
INCOMPLETE_TERMINAL = "D3R3_PHASE_B_FRESH_HEAD_EXECUTION_INCOMPLETE_MEDIA_BODY_UNOPENED"
PASS_SUCCESSOR = "EXPLICIT_D3R3_PHASE_B_EXACT64_COVERAGE_ONLY_CENSUS_PROTOCOL_REGISTRATION"


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


def reserve_output(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
        0o644,
    )


def reserve_fresh_output_root(path: Path) -> int:
    """Create a new attempt root and reserve its result before transport."""
    path.parent.mkdir(parents=True, exist_ok=False)
    return reserve_output(path)


def write_reserved(descriptor: int, payload: bytes) -> None:
    try:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            require(written > 0, "reserved write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_file(entry: dict[str, Any], label: str) -> Path:
    path = Path(entry["path"])
    require(path.is_file(), f"{label} missing: {path}")
    require(path.stat().st_size == int(entry["bytes"]), f"{label} bytes drift")
    require(sha256_file(path) == entry["sha256"], f"{label} SHA drift")
    return path


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
    require(phase_a_result.get("schema") == PHASE_A_RESULT_SCHEMA, "Phase-A result schema drift")
    require(
        phase_a_result.get("terminal")
        == "D3R1_PHASE_A_PORTRAIT_POSE_CONTINUITY_PASS_32_IDENTITIES_LOCKED",
        "Phase-A result terminal drift",
    )
    scope_rows = source_scope["exact_phase_a_selection"]["identities"]
    result_rows = phase_a_result["selected_phase_a"]
    require(len(scope_rows) == len(result_rows) == 32, "expected exact 32 Phase-A identities")
    rows: list[dict[str, Any]] = []
    encoded: list[str] = []
    for expected_order, (scope_row, result_row) in enumerate(
        zip(scope_rows, result_rows, strict=True), start=1
    ):
        for key in ("selection_order", "pool_order", "visit_id", "video_id"):
            require(scope_row[key] == result_row[key], f"Phase-A selection mismatch: {key}")
        require(scope_row["selection_order"] == expected_order, "selection order drift")
        require(isinstance(scope_row["pool_order"], int) and scope_row["pool_order"] > 0, "pool order drift")
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
    require(tuple(registered["assets"]) == ASSETS, "registered asset family drift")
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


def prior_head_lookup(
    prior_head_result: dict[str, Any], requests: list[dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    require(
        prior_head_result.get("schema")
        == "blindassist_depthart_task_preserving_d3r1_phase_b_asset_header_preflight_v1",
        "prior HEAD result schema drift",
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


def attach_prior_snapshot(
    rows: list[dict[str, Any]], prior: dict[tuple[str, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    attached: list[dict[str, Any]] = []
    for row in rows:
        old = prior[(row["video_id"], row["asset"])]
        attached.append(
            row
            | {
                "prior_head": {
                    "content_length_bytes": old["content_length_bytes"],
                    "etag": old["etag"],
                    "last_modified": old["last_modified"],
                }
            }
        )
    return attached


def request_plan_sha256(rows: list[dict[str, Any]]) -> str:
    lines = [
        f"{row['selection_order']}/{row['pool_order']}/{row['visit_id']}/"
        f"{row['video_id']}/{row['asset']}/{row['url']}"
        for row in rows
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest().upper()


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> None:
        return None


def head(
    row: dict[str, Any],
    timeout: float,
    max_attempts: int,
    user_agent: str,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    history: list[dict[str, Any]] = []
    last_status: int | None = None
    last_final_url: str | None = None
    last_etag: str | None = None
    last_modified: str | None = None
    open_request = opener or urllib.request.build_opener(NoRedirectHandler()).open
    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            str(row["url"]), method="HEAD", headers={"User-Agent": user_agent}
        )
        try:
            with open_request(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                final_url = response.geturl()
                status = int(response.status)
                etag = response.headers.get("ETag")
                last_modified_value = response.headers.get("Last-Modified")
                try:
                    parsed_length = int(length) if length is not None else None
                except (TypeError, ValueError):
                    parsed_length = None
                history.append(
                    {
                        "attempt": attempt,
                        "method": request.get_method(),
                        "http_status": status,
                        "error": None,
                    }
                )
                valid = bool(
                    status == 200
                    and final_url == row["url"]
                    and isinstance(parsed_length, int)
                    and parsed_length > 0
                    and etag
                    and last_modified_value
                )
                return row | {
                    "attempts": attempt,
                    "http_status": status,
                    "final_url": final_url,
                    "redirect_count": int(final_url != row["url"]),
                    "content_length_bytes": parsed_length,
                    "etag": etag,
                    "last_modified": last_modified_value,
                    "response_body_bytes_read": 0,
                    "attempt_history": history,
                    "recovered_error": valid and any(item["error"] for item in history[:-1]),
                    "unresolved_error": not valid,
                    "header_drift_from_prior": {
                        "content_length": parsed_length
                        != row["prior_head"]["content_length_bytes"],
                        "etag": etag != row["prior_head"]["etag"],
                        "last_modified": last_modified_value
                        != row["prior_head"]["last_modified"],
                    },
                }
        except urllib.error.HTTPError as error:  # pragma: no cover - live transport
            last_status = int(error.code)
            last_final_url = error.geturl()
            last_etag = error.headers.get("ETag")
            last_modified = error.headers.get("Last-Modified")
            if 300 <= error.code < 400:
                history.append(
                    {
                        "attempt": attempt,
                        "method": request.get_method(),
                        "http_status": int(error.code),
                        "error": f"redirect: {error}",
                    }
                )
                return row | {
                    "attempts": attempt,
                    "http_status": int(error.code),
                    "final_url": error.headers.get("Location"),
                    "redirect_count": 1,
                    "content_length_bytes": None,
                    "etag": error.headers.get("ETag"),
                    "last_modified": error.headers.get("Last-Modified"),
                    "response_body_bytes_read": 0,
                    "attempt_history": history,
                    "recovered_error": False,
                    "unresolved_error": True,
                    "header_drift_from_prior": {
                        "content_length": None,
                        "etag": None,
                        "last_modified": None,
                    },
                }
            history.append(
                {
                    "attempt": attempt,
                    "method": request.get_method(),
                    "http_status": int(error.code),
                    "error": f"HTTPError: {error}",
                }
            )
            if error.code not in {408, 429} and not 500 <= error.code < 600:
                break
        except (urllib.error.URLError, OSError) as error:  # pragma: no cover - live transport
            history.append(
                {
                    "attempt": attempt,
                    "method": request.get_method(),
                    "http_status": None,
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return row | {
        "attempts": len(history),
        "http_status": last_status,
        "final_url": last_final_url,
        "redirect_count": 0,
        "content_length_bytes": None,
        "etag": last_etag,
        "last_modified": last_modified,
        "response_body_bytes_read": 0,
        "attempt_history": history,
        "recovered_error": False,
        "unresolved_error": True,
        "header_drift_from_prior": {
            "content_length": None,
            "etag": None,
            "last_modified": None,
        },
    }


def row_available(row: dict[str, Any]) -> bool:
    return bool(
        row["http_status"] == 200
        and row["redirect_count"] == 0
        and row["final_url"] == row["url"]
        and isinstance(row["content_length_bytes"], int)
        and row["content_length_bytes"] > 0
        and row["etag"]
        and row["last_modified"]
        and row["response_body_bytes_read"] == 0
        and not row["unresolved_error"]
    )


def disposition(rows: list[dict[str, Any]]) -> str:
    require(len(rows) == 64, "result row count drift")
    if any(
        not row["attempt_history"]
        or row["attempt_history"][-1]["http_status"] is None
        for row in rows
    ):
        return INCOMPLETE_TERMINAL
    if any(not row_available(row) for row in rows):
        return UNAVAILABLE_TERMINAL
    return PASS_TERMINAL


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
    for entry, label in (
        (protocol["owning_d3_router_protocol"], "owning D3 router protocol"),
        (protocol["phase_a_result"], "Phase-A result"),
        (protocol["prior_source_scope"], "prior source scope"),
        (protocol["d3r2_protocol"], "D3R2 protocol"),
        (protocol["d3r2_execution_stop"], "D3R2 execution stop"),
        (protocol["prior_head_result"], "prior HEAD result"),
        (protocol["prior_head_terminal_audit"], "prior HEAD terminal audit"),
        (protocol["source_scope"], "source scope"),
        (protocol["producer"], "producer"),
        (protocol["tests"], "tests"),
        (protocol["validator"], "validator"),
        (protocol["validator_tests"], "validator tests"),
    ):
        verify_file(entry, label)
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    require(protocol["source_scope"]["sha256"] == sha256_file(source_scope_path), "scope SHA drift")
    require(
        protocol["phase_a_result"]["sha256"] == sha256_file(phase_a_result_path),
        "Phase-A result SHA drift",
    )
    require(
        protocol["prior_head_result"]["sha256"] == sha256_file(prior_head_result_path),
        "prior HEAD result SHA drift",
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
        activation["bindings"]["source_scope"]["sha256"] == protocol["source_scope"]["sha256"]
        and activation["bindings"]["phase_a_result"]["sha256"]
        == protocol["phase_a_result"]["sha256"]
        and activation["bindings"]["prior_head_result"]["sha256"]
        == protocol["prior_head_result"]["sha256"]
        and activation["bindings"]["d3r2_execution_stop"]["sha256"]
        == protocol["d3r2_execution_stop"]["sha256"],
        "activation binding mismatch",
    )
    authority = activation["authority"]
    require(
        set(authority)
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
    require(authority["fresh_exact64_head"] is True, "fresh exact-64 HEAD not authorized")
    require(
        authority["media_body"] is False
        and authority["range_get"] is False
        and authority["redirect_following"] is False
        and authority["archive_member_read"] is False
        and authority["source_truth_support"] is False
        and authority["phase_b_selection"] is False,
        "activation authority widened",
    )
    require(tuple(protocol["assets_per_video"]) == ASSETS, "asset family drift")
    require(protocol["request_method"] == "HEAD", "request method drift")
    require(protocol["expected_parent_count"] == 32, "parent count drift")
    require(protocol["expected_request_count"] == 64, "request count drift")
    require(protocol["max_attempts"] == 3, "attempt policy drift")
    require(protocol["redirect_policy"]["follow_redirects"] is False, "redirect policy drift")
    require(protocol["output"]["overwrite_forbidden"] is True, "overwrite policy drift")
    require(activation["execution"]["one_shot"] is True, "one-shot policy drift")
    require(activation["execution"]["expected_request_count"] == 64, "activation count drift")
    execution = activation["execution"]
    require(
        set(execution)
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
    require(execution["method"] == "HEAD", "activation method drift")
    require(execution["identity_count"] == 32, "activation identity count drift")
    require(execution["asset_count_per_identity"] == 2, "activation asset count drift")
    require(tuple(execution["assets"]) == ASSETS, "activation assets drift")
    require(
        execution["request_plan_sha256"] == protocol["request_plan_sha256"],
        "activation request plan drift",
    )
    require(
        execution["fresh_output_root_required"] is True
        and execution["output_overwrite_forbidden"] is True
        and execution["redirect_following_forbidden"] is True
        and execution["response_body_bytes_read_required"] == 0,
        "activation execution boundary drift",
    )
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
        require(authority[key] is False, f"activation authority widened: {key}")
    requests = requests_for(scope, phase_a_result, protocol["base_url"])
    require(
        request_plan_sha256(requests) == protocol["request_plan_sha256"],
        "request plan SHA drift",
    )
    prior_head_lookup(prior_head_result, requests)
    return protocol, scope, phase_a_result, prior_head_result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--phase-a-result", type=Path, required=True)
    parser.add_argument("--prior-head-result", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    protocol, scope, phase_a_result, prior_head_result = validate_bindings(
        args.protocol,
        args.source_scope,
        args.phase_a_result,
        args.prior_head_result,
        args.activation,
    )
    requests = requests_for(scope, phase_a_result, protocol["base_url"])
    requests = attach_prior_snapshot(requests, prior_head_lookup(prior_head_result, requests))
    workers = args.workers or int(protocol["workers"])
    require(1 <= workers <= 16, "workers outside frozen safe range")
    expected_output = Path(protocol["output"]["artifact_path"])
    require(args.output.resolve() == expected_output.resolve(), "artifact output path drift")
    descriptor = reserve_fresh_output_root(args.output)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    lambda row: head(
                        row,
                        float(protocol["timeout_seconds"]),
                        int(protocol["max_attempts"]),
                        str(protocol["user_agent"]),
                    ),
                    requests,
                )
            )
        results.sort(key=lambda row: (row["selection_order"], ASSETS.index(row["asset"])))
        terminal = disposition(results)
        value = {
            "schema": RESULT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": sha256_file(args.protocol),
            "source_scope_sha256": sha256_file(args.source_scope),
            "phase_a_result_sha256": sha256_file(args.phase_a_result),
            "prior_head_result_sha256": sha256_file(args.prior_head_result),
            "activation_sha256": sha256_file(args.activation),
            "request_plan_sha256": request_plan_sha256(requests),
            "request_method": "HEAD",
            "expected_request_count": 64,
            "response_body_bytes_read": sum(row["response_body_bytes_read"] for row in results),
            "media_body_bytes_read": 0,
            "archive_member_read": False,
            "depth_confidence_decoded": False,
            "source_truth_support_read": False,
            "truth_or_model_output_read": False,
            "phase_b_selection_made": False,
            "role_assignment_made": False,
            "scientific_terminal": None,
            "selection_evaluated": False,
            "selected_phase_b": None,
            "r2_cohort_access": "NONE",
            "assets": results,
            "asset_count": len(results),
            "available_asset_count": sum(row_available(row) for row in results),
            "total_content_length_bytes": sum(
                int(row["content_length_bytes"] or 0) for row in results
            ),
            "head_terminal": terminal,
            "fresh_head_authority": terminal == PASS_TERMINAL,
            "next_gate": PASS_SUCCESSOR if terminal == PASS_TERMINAL else None,
            "authority": (
                "HEAD availability and declared-size evidence only; no body, integrity, decode, "
                "source-truth-support, selection, roles, training, Development, R2, performance, "
                "production or safety authority."
            ),
        }
        encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        write_reserved(descriptor, encoded)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise
    print(
        json.dumps(
            {
                **{key: item for key, item in value.items() if key != "assets"},
                "result_bytes": len(encoded),
                "result_sha256": hashlib.sha256(encoded).hexdigest().upper(),
            },
            indent=2,
        )
    )
    return 0 if terminal == PASS_TERMINAL else 1


if __name__ == "__main__":
    raise SystemExit(main())
