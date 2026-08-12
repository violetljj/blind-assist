#!/usr/bin/env python3
"""HEAD-only preflight for the frozen DepthART D3 Phase-A metadata pool."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_head_protocol_v1"
ROSTER_SCHEMA = "blindassist_depthart_task_preserving_d3_fresh_metadata_roster_lock_v1"
SCOPE_SCHEMA = "blindassist_depthart_task_preserving_d3_source_scope_and_metadata_roster_receipt_v1"
ACTIVATION_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_head_activation_v1"
RESULT_SCHEMA = "blindassist_depthart_task_preserving_d3_phase_a_asset_header_preflight_v1"
ASSETS = ("lowres_wide_intrinsics.zip", "lowres_wide.traj")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def roster_rows(roster: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": str(row["fold"]),
            "role": str(row["role"]),
        }
        for row in roster["pool"]
    ]
    require(len(rows) == 48, "expected exactly 48 D3 metadata identities")
    require([row["pool_order"] for row in rows] == list(range(1, 49)), "pool order drift")
    require(len({row["visit_id"] for row in rows}) == 48, "visit overlap")
    require(len({row["video_id"] for row in rows}) == 48, "video overlap")
    require(all(row["fold"] == "Training" for row in rows), "fold drift")
    require(
        all(row["role"] == "D3_METADATA_CANDIDATE_POOL_ONLY" for row in rows),
        "role drift",
    )
    return rows


def requests_for(roster: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    requests = [
        row | {"asset": asset, "url": f"{base_url}/{row['fold']}/{row['video_id']}/{asset}"}
        for row in roster_rows(roster)
        for asset in ASSETS
    ]
    require(len(requests) == 96, "asset request count drift")
    return requests


def head(row: dict[str, Any], timeout: float, retries: int) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            str(row["url"]),
            method="HEAD",
            headers={"User-Agent": "BlindAssist-DepthART-D3-phase-a-head-only"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                return row | {
                    "attempts": attempt,
                    "http_status": int(response.status),
                    "content_length_bytes": int(length) if length else None,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "transport_errors": errors,
                }
        except Exception as error:  # pragma: no cover - live transport only
            errors.append(f"{type(error).__name__}: {error}")
    return row | {
        "attempts": retries,
        "http_status": None,
        "content_length_bytes": None,
        "etag": None,
        "last_modified": None,
        "transport_errors": errors,
    }


def disposition(rows: list[dict[str, Any]]) -> str:
    if any(row["http_status"] is None for row in rows):
        return "D3_PHASE_A_HEADER_PREFLIGHT_INCOMPLETE_MEDIA_BODY_UNOPENED"
    if any(row["http_status"] != 200 or not row["content_length_bytes"] for row in rows):
        return "D3_PHASE_A_ASSETS_NOT_AVAILABLE_MEDIA_BODY_UNOPENED"
    return "D3_PHASE_A_ASSET_HEADERS_96_OF_96_AVAILABLE_MEDIA_BODY_UNOPENED"


def validate_bindings(
    protocol_path: Path,
    roster_path: Path,
    scope_path: Path,
    activation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    scope = json.loads(scope_path.read_text(encoding="utf-8"))
    activation = json.loads(activation_path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(roster.get("schema") == ROSTER_SCHEMA, "roster schema drift")
    require(scope.get("schema") == SCOPE_SCHEMA, "source-scope schema drift")
    require(activation.get("schema") == ACTIVATION_SCHEMA, "activation schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    require(protocol["roster"]["sha256"] == sha256_file(roster_path), "roster SHA drift")
    require(protocol["source_scope"]["sha256"] == sha256_file(scope_path), "scope SHA drift")
    require(
        activation["bindings"]["head_protocol"]["sha256"] == sha256_file(protocol_path),
        "activation protocol mismatch",
    )
    require(
        activation["bindings"]["roster"]["sha256"] == protocol["roster"]["sha256"],
        "activation roster mismatch",
    )
    require(
        activation["bindings"]["source_scope"]["sha256"] == protocol["source_scope"]["sha256"],
        "activation scope mismatch",
    )
    require(activation["authority"]["phase_a_head"] is True, "Phase-A HEAD not authorized")
    require(activation["authority"]["media_body"] is False, "body authority must be false")
    require(activation["authority"]["archive_member_read"] is False, "archive authority must be false")
    require(tuple(protocol["assets_per_video"]) == ASSETS, "asset family drift")
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
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    protocol, roster = validate_bindings(
        args.protocol, args.roster, args.source_scope, args.activation
    )
    requests = requests_for(roster, protocol["base_url"])
    require(len(requests) == protocol["expected_request_count"], "request count drift")
    workers = args.workers or int(protocol["workers"])
    require(1 <= workers <= 16, "workers outside frozen safe range")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda row: head(
                    row,
                    float(protocol["timeout_seconds"]),
                    int(protocol["retries"]),
                ),
                requests,
            )
        )
    results.sort(key=lambda row: (row["pool_order"], row["asset"]))
    terminal = disposition(results)
    value = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "roster_sha256": sha256_file(args.roster),
        "source_scope_sha256": sha256_file(args.source_scope),
        "activation_sha256": sha256_file(args.activation),
        "request_method": "HEAD",
        "media_body_bytes_read": False,
        "archive_member_read": False,
        "truth_or_model_output_read": False,
        "role_assignment_made": False,
        "r2_cohort_access": "NONE",
        "assets": results,
        "asset_count": len(results),
        "available_asset_count": sum(
            row["http_status"] == 200 and bool(row["content_length_bytes"])
            for row in results
        ),
        "total_content_length_bytes": sum(
            int(row["content_length_bytes"] or 0) for row in results
        ),
        "terminal": terminal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    args.output.write_bytes(encoded)
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
    return 0 if terminal == "D3_PHASE_A_ASSET_HEADERS_96_OF_96_AVAILABLE_MEDIA_BODY_UNOPENED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
