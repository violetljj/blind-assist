#!/usr/bin/env python3
"""HEAD-only preflight for the exact D2R1 same-16-identity source scope."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_depthart_task_preserving_d2r1_head_protocol_v1"
RECOVERY_SCHEMA = "blindassist_depthart_task_preserving_d2r1_target_support_window_recovery_protocol_v1"
PHASE_A_SCHEMA = "blindassist_depthart_task_preserving_d2_phase_a_manifest_v1"
LICENSE_SCHEMA = "blindassist_depthart_task_preserving_d2r1_source_scope_receipt_v1"
RESULT_SCHEMA = "blindassist_depthart_task_preserving_d2r1_asset_header_preflight_v1"
ASSETS = ("lowres_wide_intrinsics.zip", "lowres_wide.traj", "lowres_depth.zip", "confidence.zip")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def selected_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        {
            "phase_a_order": index + 1,
            "pool_order": int(row["pool_order"]),
            "visit_id": str(row["visit_id"]),
            "video_id": str(row["video_id"]),
            "fold": str(row["fold"]),
        }
        for index, row in enumerate(manifest["selected_phase_b"])
    ]
    require(len(rows) == 16, "expected exact 16-identity Phase-A selection")
    require(len({row["visit_id"] for row in rows}) == 16, "visit overlap")
    require(len({row["video_id"] for row in rows}) == 16, "video overlap")
    require(all(row["fold"] == "Training" for row in rows), "fold drift")
    return rows


def requests_for(manifest: dict[str, Any], base_url: str) -> list[dict[str, Any]]:
    rows = [
        row | {"asset": asset, "url": f"{base_url}/{row['fold']}/{row['video_id']}/{asset}"}
        for row in selected_rows(manifest)
        for asset in ASSETS
    ]
    require(len(rows) == 64, "request count drift")
    return rows


def head(row: dict[str, Any], timeout: float, retries: int) -> dict[str, Any]:
    errors: list[str] = []
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            str(row["url"]), method="HEAD", headers={"User-Agent": "BlindAssist-DepthART-D2R1-preflight"}
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


def disposition(rows: list[dict[str, Any]], total_bytes: int, maximum_bytes: int) -> str:
    if any(row["http_status"] is None for row in rows):
        return "D2R1_HEADER_PREFLIGHT_INCOMPLETE_BODY_UNOPENED"
    if any(row["http_status"] != 200 or not row["content_length_bytes"] for row in rows):
        return "D2R1_ASSETS_NOT_AVAILABLE_BODY_UNOPENED"
    if total_bytes > maximum_bytes:
        return "D2R1_BODY_SIZE_EXCEEDS_AUTHORIZED_BOUND_BODY_UNOPENED"
    return "D2R1_ASSET_HEADERS_AVAILABLE_BODY_UNOPENED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--recovery-protocol", type=Path, required=True)
    parser.add_argument("--phase-a-manifest", type=Path, required=True)
    parser.add_argument("--license-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    recovery = json.loads(args.recovery_protocol.read_text(encoding="utf-8"))
    manifest = json.loads(args.phase_a_manifest.read_text(encoding="utf-8"))
    receipt = json.loads(args.license_receipt.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(recovery.get("schema") == RECOVERY_SCHEMA, "recovery protocol schema drift")
    require(manifest.get("schema") == PHASE_A_SCHEMA, "Phase-A schema drift")
    require(receipt.get("schema") == LICENSE_SCHEMA, "license schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(Path(__file__)), "producer SHA drift")
    for name, path in (
        ("recovery_protocol", args.recovery_protocol),
        ("phase_a_manifest", args.phase_a_manifest),
        ("license_receipt", args.license_receipt),
    ):
        require(protocol[name]["sha256"] == sha256_file(path), f"{name} SHA drift")
    require(receipt["authority"]["d2r1_head"] is True, "D2R1 HEAD not authorized")
    require(manifest["terminal"] == "D2_PHASE_A_PORTRAIT_CONTINUITY_PASS_16_IDENTITIES_LOCKED", "Phase-A terminal drift")
    require(tuple(protocol["assets_per_video"]) == ASSETS, "asset drift")
    requests = requests_for(manifest, protocol["base_url"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(
            executor.map(
                lambda row: head(row, float(protocol["timeout_seconds"]), int(protocol["retries"])),
                requests,
            )
        )
    results.sort(key=lambda row: (row["phase_a_order"], ASSETS.index(row["asset"])))
    total_bytes = sum(int(row["content_length_bytes"] or 0) for row in results)
    maximum_bytes = int(receipt["source_scope"]["maximum_total_body_bytes"])
    terminal = disposition(results, total_bytes, maximum_bytes)
    value = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol),
        "recovery_protocol_sha256": sha256_file(args.recovery_protocol),
        "phase_a_manifest_sha256": sha256_file(args.phase_a_manifest),
        "license_receipt_sha256": sha256_file(args.license_receipt),
        "request_method": "HEAD",
        "body_bytes_read": False,
        "rgb_read": False,
        "model_output_read": False,
        "r2_cohort_access": "NONE",
        "assets": results,
        "asset_count": len(results),
        "available_asset_count": sum(
            row["http_status"] == 200 and bool(row["content_length_bytes"]) for row in results
        ),
        "total_content_length_bytes": total_bytes,
        "authorized_maximum_body_bytes": maximum_bytes,
        "terminal": terminal,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in value.items() if key != "assets"}, indent=2))
    return 0 if terminal == "D2R1_ASSET_HEADERS_AVAILABLE_BODY_UNOPENED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
