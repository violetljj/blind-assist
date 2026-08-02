#!/usr/bin/env python3
"""Download open THÖR track/LiDAR files from Zenodo with checksums.

THÖR complementary video is not downloaded by this adapter.  The resulting
tracks/LiDAR files are source-native geometry references only and cannot be
treated as synchronized RGB event truth.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


def _get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "blindassist-hftf-d7/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _download(item: dict[str, Any], target: Path) -> dict[str, Any]:
    expected_size = int(item.get("size") or 0)
    checksum = str(item.get("checksum") or "")
    expected_md5 = checksum.split(":", 1)[-1] if checksum else ""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (expected_size <= 0 or target.stat().st_size == expected_size):
        md5 = hashlib.md5(usedforsecurity=False)
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                md5.update(chunk)
        if not expected_md5 or md5.hexdigest() == expected_md5:
            return {"key": item["key"], "local_path": str(target), "size": target.stat().st_size, "sha256": sha256_file(target), "md5": md5.hexdigest(), "status": "SKIPPED_EXISTING"}
    part = target.with_suffix(target.suffix + ".part")
    part.unlink(missing_ok=True)
    request = urllib.request.Request(str(item["links"]["self"]), headers={"User-Agent": "blindassist-hftf-d7/1.0"})
    md5 = hashlib.md5(usedforsecurity=False)
    sha = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=300) as response, part.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            md5.update(chunk)
            sha.update(chunk)
            size += len(chunk)
    if expected_size and size != expected_size:
        part.unlink(missing_ok=True)
        raise ContractError(f"size mismatch {item['key']}: {size} != {expected_size}")
    if expected_md5 and md5.hexdigest() != expected_md5:
        part.unlink(missing_ok=True)
        raise ContractError(f"MD5 mismatch {item['key']}: {md5.hexdigest()} != {expected_md5}")
    part.replace(target)
    return {"key": item["key"], "local_path": str(target), "size": size, "sha256": sha.hexdigest(), "md5": md5.hexdigest(), "status": "DOWNLOADED"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    record_url = f"https://zenodo.org/api/records/{args.record_id}"
    payload = _get_json(record_url)
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise ContractError("Zenodo record has no file list")
    patterns = [re.compile(value) for value in args.pattern]
    items = [item for item in payload["files"] if isinstance(item, dict) and any(pattern.search(str(item.get("key", ""))) for pattern in patterns)]
    items.sort(key=lambda item: str(item.get("key")))
    if args.max_bytes:
        chosen: list[dict[str, Any]] = []
        total = 0
        for item in items:
            size = int(item.get("size") or 0)
            if chosen and total + size > args.max_bytes:
                break
            chosen.append(item)
            total += size
        items = chosen
    raw_root = root / "raw" / f"thor-zenodo-{args.record_id}"
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    def worker(item: dict[str, Any]) -> dict[str, Any]:
        return _download(item, raw_root / Path(str(item["key"])).name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"thor={result['status']} key={result['key']} bytes={result['size']}")
            except Exception as exc:  # pragma: no cover - network-specific
                errors.append({"key": str(item.get("key")), "error": f"{type(exc).__name__}: {exc}"})
                print(f"thor=ERROR key={item.get('key')} error={type(exc).__name__}: {exc}")
    results.sort(key=lambda row: row["key"])
    receipt = {
        "schema": "hftf_d7_public_real_thor_zenodo_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "dataset_id": "THOR",
        "record_url": f"https://zenodo.org/records/{args.record_id}",
        "license": "CC-BY-4.0 (open tracks/LiDAR/eye-tracking record)",
        "access_status": "PUBLIC_TRACKS_LIDAR_DOWNLOADED_VIDEO_RESTRICTED",
        "event_truth_authority": False,
        "model_output_read": False,
        "selected_file_count": len(items),
        "materialized_file_count": len(results),
        "selected_bytes": sum(int(item.get("size") or 0) for item in items),
        "materialized_bytes": sum(int(item.get("size") or 0) for item in results),
        "errors": errors,
        "files": results,
        "status": "PUBLIC_TRACKS_LIDAR_DOWNLOADED" if not errors and len(results) == len(items) else "PARTIAL_DOWNLOAD",
        "raw_path": str(raw_root),
        "notes": [
            "Open track/LiDAR files do not provide synchronized RGB event truth.",
            "Complementary video was not accessed because it is restricted/different synchronization.",
            "No parent event was admitted from this receipt.",
        ],
    }
    receipt_dir = root / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"thor_zenodo_receipt_{args.run_id}.json"
    write_json(receipt_path, receipt)
    source_receipt_path = receipt_dir / "source_receipts.jsonl"
    if source_receipt_path.is_file():
        rows = load_jsonl(source_receipt_path)
        receipt_sha = sha256_file(receipt_path)
        for row in rows:
            if row.get("dataset_id") == "THOR":
                row.update({
                    "access_status": receipt["status"],
                    "retrieved_at_utc": receipt["generated_at_utc"],
                    "source_hash": receipt_sha,
                    "source_hash_kind": "MATERIALIZED_INTAKE_RECEIPT",
                    "local_evidence_paths": [str(raw_root), str(receipt_path)],
                    "receipt_kind": "zenodo_open_tracks_lidar_download",
                    "event_truth_authority": False,
                })
        write_jsonl(source_receipt_path, rows)
    return {key: value for key, value in receipt.items() if key != "files"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--record-id", default="3382145")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-bytes", type=int, default=0)
    parser.add_argument("--pattern", action="append", default=[r"_6D\.tsv$", r"_qualisys\.bag$"])
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
