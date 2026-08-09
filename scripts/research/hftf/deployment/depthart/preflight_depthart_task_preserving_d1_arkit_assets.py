#!/usr/bin/env python3
"""HEAD-only availability preflight for the frozen DepthART D1 ARKitScenes roster."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_media_preflight_protocol_v1"
ROSTER_SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_development_roster_lock_v1"
LICENSE_SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_license_scope_receipt_v1"
RESULT_SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_asset_header_preflight_v1"
ASSETS = (
    "lowres_wide.zip",
    "lowres_depth.zip",
    "confidence.zip",
    "lowres_wide_intrinsics.zip",
    "lowres_wide.traj",
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


def roster_rows(roster: dict[str, Any]) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    for role in ("primary", "reserve"):
        for order, parent in enumerate(roster[role], start=1):
            rows.append(
                {
                    "role": role.upper(),
                    "frozen_order": order,
                    "visit_id": str(parent["visit_id"]),
                    "video_id": str(parent["video_id"]),
                    "fold": str(parent["fold"]),
                }
            )
    require(len(rows) == 16, "expected exactly 8 primary and 8 reserve identities")
    require(len({row["visit_id"] for row in rows}) == 16, "visit overlap")
    require(len({row["video_id"] for row in rows}) == 16, "video overlap")
    return rows


def requests_for(roster: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for parent in roster_rows(roster):
        for asset in ASSETS:
            requests.append(
                parent
                | {
                    "asset": asset,
                    "url": f"{base_url}/{parent['fold']}/{parent['video_id']}/{asset}",
                }
            )
    require(len(requests) == 80, "asset request count drift")
    return requests


def head(row: dict[str, Any], timeout: float, retries: int) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            str(row["url"]),
            method="HEAD",
            headers={"User-Agent": "BlindAssist-DepthART-D1-media-preflight"},
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
        except Exception as error:  # pragma: no cover - exercised only by live transport
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
        return "D1_ARKIT_ASSET_HEADER_PREFLIGHT_INCOMPLETE_MEDIA_BODY_UNOPENED"
    if any(row["http_status"] != 200 or not row["content_length_bytes"] for row in rows):
        return "D1_ARKIT_ASSETS_NOT_AVAILABLE_MEDIA_BODY_UNOPENED"
    return "D1_ARKIT_ASSET_HEADERS_AVAILABLE_MEDIA_BODY_UNOPENED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    license_receipt = json.loads(args.license_receipt.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(roster.get("schema") == ROSTER_SCHEMA, "roster schema drift")
    require(license_receipt.get("schema") == LICENSE_SCHEMA, "license receipt schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    require(protocol["roster"]["sha256"] == sha256_file(args.roster), "roster SHA drift")
    require(
        protocol["license_receipt"]["sha256"] == sha256_file(args.license_receipt),
        "license receipt SHA drift",
    )
    require(license_receipt["media_preflight_authorized"] is True, "media preflight not authorized")
    require(
        license_receipt["roster"]["sha256"] == protocol["roster"]["sha256"],
        "license roster mismatch",
    )
    require(tuple(protocol["assets_per_video"]) == ASSETS, "asset family drift")
    rows = requests_for(roster, protocol["base_url"])
    require(len(rows) == protocol["expected_request_count"], "expected request count drift")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                lambda row: head(row, float(protocol["timeout_seconds"]), int(protocol["retries"])),
                rows,
            )
        )
    results.sort(key=lambda row: (row["role"], row["frozen_order"], row["asset"]))
    terminal = disposition(results)
    value = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "roster_sha256": sha256_file(args.roster),
        "license_receipt_sha256": sha256_file(args.license_receipt),
        "request_method": "HEAD",
        "media_body_bytes_read": False,
        "truth_model_or_task_outcome_read": False,
        "replacement_decision_made": False,
        "assets": results,
        "asset_count": len(results),
        "available_asset_count": sum(
            row["http_status"] == 200 and bool(row["content_length_bytes"]) for row in results
        ),
        "total_content_length_bytes": sum(int(row["content_length_bytes"] or 0) for row in results),
        "terminal": terminal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in value.items() if key != "assets"}, indent=2))
    return 0 if terminal == "D1_ARKIT_ASSET_HEADERS_AVAILABLE_MEDIA_BODY_UNOPENED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
