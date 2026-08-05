#!/usr/bin/env python3
"""HEAD-only asset availability preflight for the locked P3 validation reserve."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_p3_r0_2_1_arkit_validation_media_preflight_protocol"
ROSTER_SCHEMA = "blindassist_p3_r0_2_1_arkit_validation_extension_roster"
RESULT_SCHEMA = "blindassist_p3_r0_2_1_arkit_validation_asset_header_preflight"
ASSETS = ("lowres_wide.zip", "lowres_depth.zip", "confidence.zip", "lowres_wide_intrinsics.zip", "lowres_wide.traj")


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
    rows = []
    for parent in roster["selected"]:
        for asset in ASSETS:
            rows.append({
                "visit_id": str(parent["visit_id"]), "video_id": str(parent["video_id"]), "asset": asset,
                "url": f"{base_url}/Validation/{parent['video_id']}/{asset}",
            })
    require(len(rows) == len(roster["selected"]) * len(ASSETS), "asset request count drift")
    return rows


def head(row: dict[str, str], timeout: float, retries: int) -> dict[str, Any]:
    errors = []
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(row["url"], method="HEAD", headers={"User-Agent": "BlindAssist-P3-R0.2.1-validation-preflight"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return row | {
                    "attempts": attempt, "http_status": int(response.status),
                    "content_length_bytes": int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None,
                    "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified"),
                    "transport_errors": errors,
                }
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    return row | {"attempts": retries, "http_status": None, "content_length_bytes": None, "etag": None, "last_modified": None, "transport_errors": errors}


def disposition(rows: list[dict[str, Any]]) -> str:
    if any(row["http_status"] is None for row in rows):
        return "P3_R0_2_1_ARKIT_VALIDATION_ASSET_HEADER_PREFLIGHT_INCOMPLETE_NO_REPLACEMENT"
    if any(row["http_status"] != 200 or row["content_length_bytes"] is None for row in rows):
        return "P3_R0_2_1_ARKIT_VALIDATION_ASSETS_NOT_AVAILABLE_NO_REPLACEMENT"
    return "P3_R0_2_1_ARKIT_VALIDATION_ASSET_HEADERS_AVAILABLE_MEDIA_UNOPENED"


def main() -> None:
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(lambda row: head(row, protocol["timeout_seconds"], protocol["retries"]), rows))
    results.sort(key=lambda row: (row["visit_id"], row["video_id"], row["asset"]))
    terminal = disposition(results)
    value = {
        "schema": RESULT_SCHEMA, "protocol_sha256": sha256_file(args.protocol), "roster_sha256": sha256_file(args.roster),
        "request_method": "HEAD", "media_body_bytes_read": False, "label_or_model_fields_read": False,
        "assets": results, "asset_count": len(results),
        "available_asset_count": sum(row["http_status"] == 200 and row["content_length_bytes"] is not None for row in results),
        "total_content_length_bytes": sum(row["content_length_bytes"] or 0 for row in results),
        "replacement_allowed": False, "terminal": terminal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in value.items() if key != "assets"}, indent=2))
    raise SystemExit(0 if terminal.endswith("AVAILABLE_MEDIA_UNOPENED") else 1)


if __name__ == "__main__":
    main()
