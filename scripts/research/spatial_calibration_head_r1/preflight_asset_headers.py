#!/usr/bin/env python3
"""HEAD-only availability/size preflight for the locked development assets."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import urllib.request
from pathlib import Path
from typing import Any

from validate_protocol import DEFAULT_PROTOCOL, sha256, validate

BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
ASSET_FILES = (
    "lowres_wide.zip",
    "lowres_depth.zip",
    "confidence.zip",
    "lowres_wide_intrinsics.zip",
    "lowres_wide.traj",
)


def cohort_header_requests(lock: dict[str, Any]) -> list[dict[str, str]]:
    requests = []
    for role in ("train", "validation"):
        for row in lock["roles"][role]:
            video_id = str(row["video_id"])
            for asset in ASSET_FILES:
                requests.append({
                    "role": role,
                    "visit_id": str(row["visit_id"]),
                    "video_id": video_id,
                    "asset": asset,
                    "url": f"{BASE_URL}/Training/{video_id}/{asset}",
                })
    for row in lock["roles"]["sealed"]:
        video_id = str(row["video_id"])
        requests.append({
            "role": "sealed_identity_only",
            "visit_id": str(row["visit_id"]),
            "video_id": video_id,
            "asset": "lowres_wide.zip",
            "url": f"{BASE_URL}/Validation/{video_id}/lowres_wide.zip",
        })
    if len(requests) != 104:
        raise ValueError("expected 100 development assets plus four sealed RGB identity assets")
    return requests


def head(row: dict[str, str], timeout: float, retries: int) -> dict[str, Any]:
    errors = []
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(row["url"], method="HEAD", headers={"User-Agent": "BlindAssist-R1-metadata-preflight"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return {
                    **row,
                    "attempts": attempt,
                    "http_status": int(response.status),
                    "content_length_bytes": int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "transport_errors": errors,
                }
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    return {**row, "attempts": retries, "http_status": None, "content_length_bytes": None, "etag": None, "last_modified": None, "transport_errors": errors}


def disposition(rows: list[dict[str, Any]]) -> str:
    if any(row["http_status"] is None for row in rows):
        return "SPATIAL_CALIBRATION_HEAD_R1_ASSET_HEADER_PREFLIGHT_INCOMPLETE"
    if any(row["http_status"] != 200 or row["content_length_bytes"] is None for row in rows):
        return "SPATIAL_CALIBRATION_HEAD_R1_COHORT_NOT_EVALUABLE_NO_REPLACEMENT"
    return "SPATIAL_CALIBRATION_HEAD_R1_ASSET_HEADERS_AVAILABLE"


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--roster-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    errors = validate(protocol)
    if errors:
        raise ValueError(f"protocol invalid: {errors}")
    lock = json.loads(args.roster_lock.read_text(encoding="utf-8"))
    if lock.get("protocol_sha256") != sha256(args.protocol):
        raise ValueError("roster lock protocol mismatch")
    requests = cohort_header_requests(lock)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(lambda row: head(row, args.timeout, args.retries), requests))
    rows.sort(key=lambda row: (row["role"], row["visit_id"], row["video_id"], row["asset"]))
    terminal = disposition(rows)
    passed = terminal == "SPATIAL_CALIBRATION_HEAD_R1_ASSET_HEADERS_AVAILABLE"
    total = sum(row["content_length_bytes"] or 0 for row in rows)
    result = {
        "schema": "blindassist_spatial_calibration_head_r1_asset_header_preflight",
        "protocol_sha256": sha256(args.protocol),
        "roster_lock_sha256": sha256(args.roster_lock),
        "request_method": "HEAD",
        "media_body_bytes_read": False,
        "sealed_header_urls_requested": True,
        "sealed_body_bytes_read": False,
        "assets": rows,
        "asset_count": len(rows),
        "available_asset_count": sum(row["http_status"] == 200 for row in rows),
        "total_content_length_bytes": total,
        "fixed_transport_retries": args.retries,
        "terminal": terminal,
    }
    write_json_new(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "assets"}, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
