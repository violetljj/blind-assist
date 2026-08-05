#!/usr/bin/env python3
"""HEAD-only ARKitScenes asset preflight for clearance fusion R0.1."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any, Callable


PROTOCOL_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_arkit_media_preflight_protocol"
ROSTER_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_arkit_roster"
LICENSE_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_arkit_license_scope_receipt"
RESULT_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_arkit_asset_header_preflight"
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


def load_bound(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = (root / binding["path"]).resolve()
    require(path.is_file(), f"bound file missing: {path}")
    require(sha256_file(path) == binding["sha256"], f"bound file SHA mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return path, value


def requests_for(roster: dict[str, Any], base_url: str) -> list[dict[str, str]]:
    rows = []
    for video in roster["selected"]:
        require(tuple(video["assets"]) == ASSETS, "authorized asset roster drift")
        for asset in ASSETS:
            rows.append({
                "visit_id": str(video["visit_id"]),
                "video_id": str(video["video_id"]),
                "asset": asset,
                "url": f"{base_url}/Validation/{video['video_id']}/{asset}",
            })
    return rows


def head(row: dict[str, str], timeout: float, retries: int) -> dict[str, Any]:
    errors = []
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(row["url"], method="HEAD", headers={"User-Agent": "BlindAssist-clearance-fusion-r0.1-preflight"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return row | {
                    "attempts": attempt,
                    "http_status": int(response.status),
                    "content_length_bytes": int(response.headers["Content-Length"]) if response.headers.get("Content-Length") else None,
                    "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified"),
                    "transport_errors": errors,
                }
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
    return row | {"attempts": retries, "http_status": None, "content_length_bytes": None, "etag": None, "last_modified": None, "transport_errors": errors}


def disposition(rows: list[dict[str, Any]]) -> str:
    if any(row["http_status"] is None for row in rows):
        return "QUALITY_GATED_CLEARANCE_FUSION_R0_1_ARKIT_PREFLIGHT_INCOMPLETE_NO_REPLACEMENT"
    if any(row["http_status"] != 200 or not row["content_length_bytes"] for row in rows):
        return "QUALITY_GATED_CLEARANCE_FUSION_R0_1_ARKIT_ASSET_UNAVAILABLE_NO_REPLACEMENT"
    return "QUALITY_GATED_CLEARANCE_FUSION_R0_1_ARKIT_HEADERS_AVAILABLE_MEDIA_UNOPENED"


def produce(root: Path, protocol_path: Path, output: Path, request: Callable[[dict[str, str], float, int], dict[str, Any]] = head, workers: int = 8) -> dict[str, Any]:
    require(not output.exists(), f"overwrite forbidden: {output}")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    _, roster = load_bound(root, protocol["roster"])
    _, license_receipt = load_bound(root, protocol["license_receipt"])
    require(roster.get("schema") == ROSTER_SCHEMA, "roster schema drift")
    require(license_receipt.get("schema") == LICENSE_SCHEMA, "license schema drift")
    require(license_receipt.get("media_download_authorized") is True, "media authorization missing")
    require(license_receipt["roster"]["sha256"] == protocol["roster"]["sha256"], "license roster mismatch")
    require(license_receipt.get("model_or_training_authorized") is False, "training authority drift")
    rows = requests_for(roster, protocol["base_url"])
    require(len(rows) == 15, "expected exactly 15 asset HEAD requests")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda row: request(row, protocol["timeout_seconds"], protocol["retries"]), rows))
    results.sort(key=lambda row: (row["visit_id"], row["video_id"], row["asset"]))
    terminal = disposition(results)
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "producer_sha256": sha256_file(Path(__file__)),
        "roster_sha256": protocol["roster"]["sha256"],
        "license_receipt_sha256": protocol["license_receipt"]["sha256"],
        "request_method": "HEAD",
        "media_body_bytes_read": False,
        "label_or_model_fields_read": False,
        "assets": results,
        "asset_count": len(results),
        "available_asset_count": sum(row["http_status"] == 200 and bool(row["content_length_bytes"]) for row in results),
        "total_content_length_bytes": sum(row["content_length_bytes"] or 0 for row in results),
        "replacement_allowed": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
        "terminal": terminal,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = produce(args.repo_root.resolve(), args.protocol.resolve(), args.output.resolve(), workers=args.workers)
    print(json.dumps({key: value for key, value in result.items() if key != "assets"}, indent=2))
    raise SystemExit(0 if result["terminal"].endswith("AVAILABLE_MEDIA_UNOPENED") else 1)


if __name__ == "__main__":
    main()
