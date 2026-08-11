#!/usr/bin/env python3
"""HEAD-only RGB preflight for the exact eight frozen D2 role identities."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_rgb_head_execution_protocol_v1"
SCOPE_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_rgb_head_scope_protocol_v1"
RESULT_LOCK_SCHEMA = "blindassist_depthart_task_preserving_d2r1_result_v1"
RECEIPT_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_rgb_head_scope_receipt_v1"
RESULT_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_c_rgb_head_result_v1"
ASSET = "lowres_wide.zip"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def exact_identities(scope: dict[str, Any], d2r1_result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "role": str(row["role"]),
            "role_order": index % 4 + 1,
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
        }
        for index, row in enumerate(scope["identity_scope"])
    ]
    require(len(rows) == 8, "identity count drift")
    require([row["role"] for row in rows] == ["D2_TRAIN"] * 4 + ["D2_DEVELOPMENT_SEALED"] * 4, "role order drift")
    require(len({row["visit_id"] for row in rows}) == 8 and len({row["video_id"] for row in rows}) == 8, "identity overlap")
    locked = [
        {key: str(row[key]) for key in ("role", "visit_id", "video_id")}
        for row in d2r1_result["role_assignments"]
    ]
    scoped = [{key: row[key] for key in ("role", "visit_id", "video_id")} for row in rows]
    require(scoped == locked, "D2R1 role binding drift")
    return rows


def requests_for(scope: dict[str, Any], d2r1_result: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    return [
        row | {"fold": "Training", "asset": ASSET, "url": f"{base_url}/Training/{row['video_id']}/{ASSET}"}
        for row in exact_identities(scope, d2r1_result)
    ]


def head(row: dict[str, Any], timeout: float, retries: int) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            str(row["url"]), method="HEAD", headers={"User-Agent": "BlindAssist-DepthART-D2-phase-c-rgb-head"}
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
        return "D2_PHASE_C_RGB_HEAD_INCOMPLETE_BODY_UNOPENED"
    if any(row["http_status"] != 200 or not row["content_length_bytes"] for row in rows):
        return "D2_PHASE_C_RGB_NOT_AVAILABLE_BODY_UNOPENED"
    return "D2_PHASE_C_RGB_HEADERS_AVAILABLE_BODY_UNOPENED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scope-protocol", type=Path, required=True)
    parser.add_argument("--d2r1-result", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    scope = json.loads(args.scope_protocol.read_text(encoding="utf-8"))
    d2r1_result = json.loads(args.d2r1_result.read_text(encoding="utf-8"))
    receipt = json.loads(args.license_receipt.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(scope.get("schema") == SCOPE_SCHEMA, "scope schema drift")
    require(d2r1_result.get("schema") == RESULT_LOCK_SCHEMA, "D2R1 result schema drift")
    require(receipt.get("schema") == RECEIPT_SCHEMA, "receipt schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    for name, path in (
        ("scope_protocol", args.scope_protocol),
        ("d2r1_result", args.d2r1_result),
        ("license_receipt", args.license_receipt),
    ):
        require(protocol[name]["sha256"] == sha256_file(path), f"{name} SHA drift")
    require(receipt["authority"]["phase_c_rgb_head"] is True, "RGB HEAD not authorized")
    require(receipt["authority"]["phase_c_rgb_body"] is False, "body authority drift")
    requests = requests_for(scope, d2r1_result, protocol["base_url"])
    require(len(requests) == 8, "request count drift")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(lambda row: head(row, float(protocol["timeout_seconds"]), int(protocol["retries"])), requests))
    rows.sort(key=lambda row: (["D2_TRAIN", "D2_DEVELOPMENT_SEALED"].index(row["role"]), row["role_order"]))
    terminal = disposition(rows)
    value = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "scope_protocol_sha256": sha256_file(args.scope_protocol),
        "d2r1_result_sha256": sha256_file(args.d2r1_result),
        "license_receipt_sha256": sha256_file(args.license_receipt),
        "request_method": "HEAD",
        "asset": ASSET,
        "identity_count": len(rows),
        "available_identity_count": sum(row["http_status"] == 200 and bool(row["content_length_bytes"]) for row in rows),
        "total_content_length_bytes": sum(int(row["content_length_bytes"] or 0) for row in rows),
        "assets": rows,
        "body_bytes_read": False,
        "model_output_read": False,
        "training_executed": False,
        "development_outcome_opened": False,
        "r2_cohort_access": "NONE",
        "terminal": terminal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in value.items() if key != "assets"}, indent=2))
    return 0 if terminal == "D2_PHASE_C_RGB_HEADERS_AVAILABLE_BODY_UNOPENED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
