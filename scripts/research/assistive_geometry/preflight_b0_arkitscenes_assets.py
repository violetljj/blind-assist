#!/usr/bin/env python3
"""HEAD-only asset preflight for the frozen Assistive Geometry B0 rosters."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_media_preflight_protocol_v1"
ROSTER_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_rosters_v1"
RESULT_SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_asset_header_preflight_v1"
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


def requests_for(roster: dict[str, Any], base_url: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for role, parents in roster["roles"].items():
        for parent in parents:
            for asset in ASSETS:
                rows.append(
                    {
                        "role": role,
                        "visit_id": str(parent["visit_id"]),
                        "video_id": str(parent["video_id"]),
                        "official_fold": str(parent["official_fold"]),
                        "asset": asset,
                        "url": f"{base_url}/{parent['official_fold']}/{parent['video_id']}/{asset}",
                    }
                )
    expected = sum(len(parents) for parents in roster["roles"].values()) * len(ASSETS)
    require(len(rows) == expected, "asset request count drift")
    return rows


def head(row: dict[str, str], timeout: float, retries: int) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            row["url"],
            method="HEAD",
            headers={"User-Agent": "BlindAssist-Assistive-Geometry-B0-preflight"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return row | {
                    "attempts": attempt,
                    "http_status": int(response.status),
                    "content_length_bytes": int(response.headers["Content-Length"])
                    if response.headers.get("Content-Length")
                    else None,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "transport_errors": errors,
                }
        except Exception as error:
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
        return "B0_ARKIT_ASSET_HEADER_PREFLIGHT_INCOMPLETE_NO_REPLACEMENT"
    if any(row["http_status"] != 200 or row["content_length_bytes"] is None for row in rows):
        return "B0_ARKIT_ASSETS_NOT_AVAILABLE_NO_REPLACEMENT"
    return "B0_ARKIT_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    require(roster.get("schema") == ROSTER_SCHEMA, "roster schema drift")
    require(sha256_file(args.roster) == protocol["roster"]["sha256"], "roster SHA drift")
    rows = requests_for(roster, protocol["base_url"])
    require(len(rows) == protocol["expected_request_count"], "protocol request count drift")
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                lambda row: head(row, protocol["timeout_seconds"], protocol["retries"]),
                rows,
            )
        )
    results.sort(key=lambda row: (row["role"], row["visit_id"], row["video_id"], row["asset"]))
    terminal = disposition(results)
    value = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "roster_sha256": sha256_file(args.roster),
        "request_method": "HEAD",
        "media_body_bytes_read": False,
        "label_or_model_fields_read": False,
        "assets": results,
        "asset_count": len(results),
        "available_asset_count": sum(
            row["http_status"] == 200 and row["content_length_bytes"] is not None
            for row in results
        ),
        "total_content_length_bytes": sum(row["content_length_bytes"] or 0 for row in results),
        "replacement_allowed": False,
        "terminal": terminal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in value.items() if key != "assets"}, indent=2))
    return 0 if terminal.endswith("AVAILABLE_MEDIA_UNOPENED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
