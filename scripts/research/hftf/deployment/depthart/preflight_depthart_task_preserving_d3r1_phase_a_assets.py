#!/usr/bin/env python3
"""HEAD-only preflight for the frozen DepthART D3R1 Phase-A pool."""

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


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_a_head_protocol_v1"
ROSTER_SCHEMA = "blindassist_depthart_task_preserving_d3r1_fresh_metadata_roster_lock_v1"
SCOPE_SCHEMA = "blindassist_depthart_task_preserving_d3r1_source_scope_receipt_v1"
ACTIVATION_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_a_head_activation_v1"
RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3r1_phase_a_asset_header_preflight_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D3R1_PHASE_A_INTRINSICS_TRAJECTORY_HEAD_ONLY_PREFLIGHT"
ASSETS = ("lowres_wide_intrinsics.zip", "lowres_wide.traj")
PASS_TERMINAL = "D3R1_PHASE_A_ASSET_HEADERS_254_OF_254_AVAILABLE_MEDIA_BODY_UNOPENED"
PASS_SUCCESSOR = (
    "EXPLICIT_D3R1_PHASE_A_INTRINSICS_TRAJECTORY_BODY_AND_LABEL_BLIND_CONTINUITY_ACTIVATION"
)


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
    """Create a new attempt root and reserve its result before any transport."""
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


def roster_rows(roster: dict[str, Any]) -> list[dict[str, Any]]:
    require(roster.get("schema") == ROSTER_SCHEMA, "roster schema drift")
    require(
        roster.get("status") == "D3R1_FRESH_METADATA_POOL_127_LOCKED_MEDIA_UNOPENED",
        "roster status drift",
    )
    pairs = roster["pool"]
    require(len(pairs) == 127, "expected exactly 127 D3R1 identities")
    rows: list[dict[str, Any]] = []
    for order, pair in enumerate(pairs, start=1):
        require(isinstance(pair, str) and pair.count("/") == 1, "malformed roster pair")
        visit_id, video_id = pair.split("/")
        require(visit_id.isdigit() and video_id.isdigit(), "non-numeric roster pair")
        rows.append(
            {
                "pool_order": order,
                "visit_id": visit_id,
                "video_id": video_id,
                "fold": "Training",
                "role": "D3R1_METADATA_CANDIDATE_POOL_ONLY",
            }
        )
    require(len({row["visit_id"] for row in rows}) == 127, "visit overlap")
    require(len({row["video_id"] for row in rows}) == 127, "video overlap")
    encoded = ("\n".join(pairs) + "\n").encode("utf-8")
    require(
        hashlib.sha256(encoded).hexdigest().upper() == roster["pool_pairs_sha256"],
        "pool pairs SHA drift",
    )
    return rows


def requests_for(roster: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    requests = [
        row
        | {
            "asset": asset,
            "url": f"{base_url}/{row['fold']}/{row['video_id']}/{asset}",
        }
        for row in roster_rows(roster)
        for asset in ASSETS
    ]
    require(len(requests) == 254, "asset request count drift")
    return requests


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


def head(
    row: dict[str, Any],
    timeout: float,
    max_attempts: int,
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
            str(row["url"]),
            method="HEAD",
            headers={"User-Agent": "BlindAssist-DepthART-D3R1-phase-a-head-only"},
        )
        try:
            with open_request(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                final_url = response.geturl()
                status = int(response.status)
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
                return row | {
                    "attempts": attempt,
                    "http_status": status,
                    "final_url": final_url,
                    "redirected": final_url != row["url"],
                    "content_length_bytes": parsed_length,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "response_body_bytes_read": 0,
                    "attempt_history": history,
                    "recovered_error": any(item["error"] for item in history),
                    "unresolved_error": False,
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
                    "redirected": True,
                    "content_length_bytes": None,
                    "etag": error.headers.get("ETag"),
                    "last_modified": error.headers.get("Last-Modified"),
                    "response_body_bytes_read": 0,
                    "attempt_history": history,
                    "recovered_error": False,
                    "unresolved_error": True,
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
        except (urllib.error.URLError, OSError) as error:  # pragma: no cover - live transport only
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
        "redirected": False,
        "content_length_bytes": None,
        "etag": last_etag,
        "last_modified": last_modified,
        "response_body_bytes_read": 0,
        "attempt_history": history,
        "recovered_error": False,
        "unresolved_error": True,
    }


def row_available(row: dict[str, Any]) -> bool:
    return bool(
        row["http_status"] == 200
        and not row["redirected"]
        and row["final_url"] == row["url"]
        and isinstance(row["content_length_bytes"], int)
        and row["content_length_bytes"] > 0
        and row["etag"]
        and row["last_modified"]
        and row["response_body_bytes_read"] == 0
        and not row["unresolved_error"]
    )


def disposition(rows: list[dict[str, Any]]) -> str:
    require(len(rows) == 254, "result row count drift")
    if any(row["http_status"] is None for row in rows):
        return "D3R1_PHASE_A_HEADER_PREFLIGHT_INCOMPLETE_MEDIA_BODY_UNOPENED"
    if any(not row_available(row) for row in rows):
        return "D3R1_PHASE_A_ASSETS_NOT_AVAILABLE_MEDIA_BODY_UNOPENED"
    return PASS_TERMINAL


def validate_bindings(
    protocol_path: Path,
    roster_path: Path,
    scope_path: Path,
    activation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    roster = load_json(roster_path)
    scope = load_json(scope_path)
    activation = load_json(activation_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol.get("protocol_id") == PROTOCOL_ID, "protocol id drift")
    require(scope.get("schema") == SCOPE_SCHEMA, "source-scope schema drift")
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    for entry, label in (
        (protocol["owning_protocol"], "owning protocol"),
        (protocol["roster"], "roster"),
        (protocol["source_scope"], "source scope"),
        (protocol["producer"], "producer"),
        (protocol["tests"], "tests"),
        (protocol["validator"], "validator"),
        (protocol["validator_tests"], "validator tests"),
    ):
        verify_file(entry, label)
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    require(protocol["roster"]["sha256"] == sha256_file(roster_path), "roster SHA drift")
    require(protocol["source_scope"]["sha256"] == sha256_file(scope_path), "scope SHA drift")
    require(
        activation["bindings"]["head_protocol"]["sha256"] == sha256_file(protocol_path),
        "activation protocol mismatch",
    )
    require(
        activation["bindings"]["roster"]["sha256"] == protocol["roster"]["sha256"]
        and activation["bindings"]["source_scope"]["sha256"]
        == protocol["source_scope"]["sha256"],
        "activation binding mismatch",
    )
    authority = activation["authority"]
    require(authority["phase_a_head"] is True, "Phase-A HEAD not authorized")
    require(
        authority["media_body"] is False
        and authority["range_get"] is False
        and authority["archive_member_read"] is False
        and authority["pose_content_read"] is False
        and authority["phase_a_selection"] is False,
        "activation authority widened",
    )
    require(tuple(protocol["assets_per_video"]) == ASSETS, "asset family drift")
    require(protocol["request_method"] == "HEAD", "request method drift")
    require(protocol["expected_parent_count"] == 127, "parent count drift")
    require(protocol["expected_request_count"] == 254, "request count drift")
    require(protocol["max_attempts"] == 3, "attempt policy drift")
    require(protocol["redirect_policy"]["follow_redirects"] is False, "redirect policy drift")
    require(protocol["output"]["overwrite_forbidden"] is True, "overwrite policy drift")
    require(activation["execution"]["one_shot"] is True, "one-shot policy drift")
    require(activation["execution"]["expected_request_count"] == 254, "activation count drift")
    require(scope["next_gate"] == "EXPLICIT_D3R1_PHASE_A_INTRINSICS_TRAJECTORY_HEAD_ONLY_PREFLIGHT_ACTIVATION", "scope gate drift")
    roster_rows(roster)
    return protocol, roster


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--source-scope", type=Path, required=True)
    parser.add_argument("--activation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    protocol, roster = validate_bindings(
        args.protocol, args.roster, args.source_scope, args.activation
    )
    requests = requests_for(roster, protocol["base_url"])
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
                    ),
                    requests,
                )
            )
        results.sort(key=lambda row: (row["pool_order"], ASSETS.index(row["asset"])))
        terminal = disposition(results)
        value = {
            "schema": RESULT_SCHEMA,
            "protocol_id": PROTOCOL_ID,
            "protocol_sha256": sha256_file(args.protocol),
            "roster_sha256": sha256_file(args.roster),
            "source_scope_sha256": sha256_file(args.source_scope),
            "activation_sha256": sha256_file(args.activation),
            "request_method": "HEAD",
            "expected_request_count": 254,
            "response_body_bytes_read": sum(row["response_body_bytes_read"] for row in results),
            "media_body_bytes_read": 0,
            "archive_member_read": False,
            "pose_content_read": False,
            "truth_or_model_output_read": False,
            "role_assignment_made": False,
            "r2_cohort_access": "NONE",
            "assets": results,
            "asset_count": len(results),
            "available_asset_count": sum(row_available(row) for row in results),
            "total_content_length_bytes": sum(
                int(row["content_length_bytes"] or 0) for row in results
            ),
            "terminal": terminal,
            "next_gate": PASS_SUCCESSOR if terminal == PASS_TERMINAL else None,
            "authority": (
                "HEAD availability and declared-size evidence only; no body, integrity, "
                "continuity, truth, training, Development, R2, performance, production or "
                "safety authority."
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
