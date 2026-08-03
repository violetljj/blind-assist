"""HEAD-only availability and size preflight for the frozen fresh cohort."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


BASE_URL = "https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw"
ASSET_FILES = (
    "lowres_wide.zip",
    "lowres_depth.zip",
    "confidence.zip",
    "lowres_wide_intrinsics.zip",
    "lowres_wide.traj",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def cohort_header_requests(lock: dict[str, Any]) -> list[dict[str, str]]:
    requests = []
    for row in lock["fresh_evaluation"]:
        video_id = str(row["video_id"])
        for asset in ASSET_FILES:
            requests.append(
                {
                    "role": "fresh_evaluation",
                    "visit_id": str(row["visit_id"]),
                    "video_id": video_id,
                    "asset": asset,
                    "url": f"{BASE_URL}/Validation/{video_id}/{asset}",
                }
            )
    expected = int(lock["selected_parent_count"]) * len(ASSET_FILES)
    if len(requests) != expected or expected != 20:
        raise ValueError("expected four parents and twenty assets")
    return requests


def head(row: dict[str, str], timeout: float, retries: int) -> dict[str, Any]:
    errors = []
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            row["url"],
            method="HEAD",
            headers={"User-Agent": "BlindAssist-known-height-r0-metadata-preflight"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return {
                    **row,
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
    return {
        **row,
        "attempts": retries,
        "http_status": None,
        "content_length_bytes": None,
        "etag": None,
        "last_modified": None,
        "transport_errors": errors,
    }


def write_json_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(
            descriptor,
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--roster-lock", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    arguments = parser.parse_args()
    lock = json.loads(arguments.roster_lock.read_text(encoding="utf-8"))
    if lock.get("protocol_sha256") != sha256(arguments.protocol):
        raise ValueError("roster lock protocol mismatch")
    requests = cohort_header_requests(lock)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=arguments.workers
    ) as executor:
        rows = list(
            executor.map(
                lambda row: head(row, arguments.timeout, arguments.retries), requests
            )
        )
    rows.sort(key=lambda row: (row["visit_id"], row["video_id"], row["asset"]))
    incomplete = any(row["http_status"] is None for row in rows)
    unavailable = any(
        row["http_status"] != 200 or row["content_length_bytes"] is None
        for row in rows
    )
    terminal = (
        "KNOWN_HEIGHT_R0_ASSET_HEADER_PREFLIGHT_INCOMPLETE"
        if incomplete
        else "KNOWN_HEIGHT_R0_COHORT_NOT_EVALUABLE_NO_REPLACEMENT"
        if unavailable
        else "KNOWN_HEIGHT_R0_ASSET_HEADERS_AVAILABLE"
    )
    result = {
        "schema": "blindassist_known_camera_height_ground_scale_r0_asset_header_preflight",
        "protocol_sha256": sha256(arguments.protocol),
        "roster_lock_sha256": sha256(arguments.roster_lock),
        "request_method": "HEAD",
        "media_body_bytes_read": False,
        "assets": rows,
        "asset_count": len(rows),
        "available_asset_count": sum(row["http_status"] == 200 for row in rows),
        "total_content_length_bytes": sum(
            row["content_length_bytes"] or 0 for row in rows
        ),
        "fixed_transport_retries": arguments.retries,
        "terminal": terminal,
    }
    write_json_new(arguments.output, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "assets"}, indent=2
        )
    )
    raise SystemExit(0 if terminal == "KNOWN_HEIGHT_R0_ASSET_HEADERS_AVAILABLE" else 1)


if __name__ == "__main__":
    main()
