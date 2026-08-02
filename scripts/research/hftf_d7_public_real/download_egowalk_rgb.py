#!/usr/bin/env python3
"""Download the public extracted EgoWalk RGB videos with hash receipts.

The raw-recordings repository remains gated.  This command only accesses the
MIT-licensed extracted ``EgoWalk/trajectories`` repository and never requests
credentials or bypasses a gate.  Downloaded RGB is still review input, not
event truth.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_jsonl, sha256_file, utc_now, write_json


API_URL = "https://huggingface.co/api/datasets/EgoWalk/trajectories/tree/main/video/rgb/?recursive=false&expand=false&limit=1000"
RAW_BASE = "https://huggingface.co/datasets/EgoWalk/trajectories/resolve/main/"
README_URL = "https://huggingface.co/datasets/EgoWalk/trajectories/raw/main/README.md"


def _fetch_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "blindassist-hftf-d7/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "blindassist-hftf-d7/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read().decode("utf-8")


def _download_one(item: dict[str, Any], target: Path) -> dict[str, Any]:
    expected_size = int(item.get("size") or 0)
    lfs = item.get("lfs") if isinstance(item.get("lfs"), dict) else {}
    expected_oid = str(lfs.get("oid") or "")
    relative = str(item.get("path") or "")
    if not relative.startswith("video/rgb/") or not relative.endswith(".mp4"):
        raise ContractError(f"unexpected RGB item path: {relative}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and (expected_size <= 0 or target.stat().st_size == expected_size):
        return {
            "path": relative,
            "local_path": str(target),
            "size": target.stat().st_size,
            "sha256": sha256_file(target),
            "provider_lfs_oid": expected_oid,
            "status": "SKIPPED_EXISTING_SIZE_MATCH",
            "url": RAW_BASE + relative,
        }
    temp = target.with_suffix(target.suffix + ".part")
    if temp.exists():
        temp.unlink()
    request = urllib.request.Request(RAW_BASE + relative, headers={"User-Agent": "blindassist-hftf-d7/1.0"})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=300) as response, temp.open("wb") as handle:
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if expected_size and size != expected_size:
        temp.unlink(missing_ok=True)
        raise ContractError(f"size mismatch for {relative}: got {size}, expected {expected_size}")
    temp.replace(target)
    return {
        "path": relative,
        "local_path": str(target),
        "size": size,
        "sha256": digest.hexdigest(),
        "provider_lfs_oid": expected_oid,
        "status": "DOWNLOADED",
        "url": RAW_BASE + relative,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    raw_root = root / "raw" / "egowalk-rgb"
    receipt_dir = root / "receipts"
    items = _fetch_json(API_URL)
    if not isinstance(items, list):
        raise ContractError("Hugging Face RGB tree response is not a list")
    items = [item for item in items if isinstance(item, dict) and str(item.get("path", "")).endswith(".mp4")]
    items.sort(key=lambda item: str(item.get("path")))
    if args.max_files:
        items = items[: args.max_files]
    if args.max_bytes:
        selected: list[dict[str, Any]] = []
        total = 0
        for item in items:
            size = int(item.get("size") or 0)
            if selected and total + size > args.max_bytes:
                break
            selected.append(item)
            total += size
        items = selected
    readme = _fetch_text(README_URL)
    readme_sha256 = hashlib.sha256(readme.encode("utf-8")).hexdigest()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    def worker(item: dict[str, Any]) -> dict[str, Any]:
        relative = str(item["path"])
        return _download_one(item, raw_root / Path(relative).name)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(worker, item): item for item in items}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f"egowalk_rgb={result['status']} path={result['path']} bytes={result['size']}")
            except Exception as exc:  # pragma: no cover - network-specific
                errors.append({"path": str(item.get("path")), "error": f"{type(exc).__name__}: {exc}"})
                print(f"egowalk_rgb=ERROR path={item.get('path')} error={type(exc).__name__}: {exc}")
    results.sort(key=lambda row: row["path"])
    expected_bytes = sum(int(item.get("size") or 0) for item in items)
    downloaded_bytes = sum(int(row.get("size") or 0) for row in results)
    receipt = {
        "schema": "hftf_d7_public_real_egowalk_rgb_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "dataset_id": "EgoWalk",
        "repository_url": "https://huggingface.co/datasets/EgoWalk/trajectories",
        "readme_url": README_URL,
        "readme_sha256": readme_sha256,
        "license_status": "MIT_EXTRACTED_TRAJECTORIES_REPOSITORY",
        "raw_recordings_status": "ACCESS_BLOCKED",
        "event_truth_authority": False,
        "model_output_read": False,
        "expected_file_count": len(items),
        "materialized_file_count": len(results),
        "expected_bytes": expected_bytes,
        "materialized_bytes": downloaded_bytes,
        "errors": errors,
        "files": results,
        "status": "PUBLIC_EXTRACTED_RGB_DOWNLOADED" if not errors and len(results) == len(items) else "PARTIAL_DOWNLOAD",
        "notes": [
            "RGB videos are review inputs only; no event labels are inferred by this downloader.",
            "The gated raw-recordings repository was not accessed.",
            "Frame/candidate registries retain the source receipt and model-blind selection contract.",
        ],
    }
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"egowalk_rgb_receipt_{args.run_id}.json"
    write_json(receipt_path, receipt)
    source_receipt_path = receipt_dir / "source_receipts.jsonl"
    if source_receipt_path.is_file():
        rows = load_jsonl(source_receipt_path)
        receipt_sha = sha256_file(receipt_path)
        for row in rows:
            if row.get("dataset_id") == "EgoWalk":
                row.update({
                    "access_status": receipt["status"],
                    "license": "MIT_EXTRACTED_TRAJECTORIES_REPOSITORY; raw recordings access blocked",
                    "retrieved_at_utc": receipt["generated_at_utc"],
                    "source_hash": receipt_sha,
                    "source_hash_kind": "MATERIALIZED_INTAKE_RECEIPT",
                    "local_evidence_paths": [str(raw_root), str(receipt_path)],
                    "rgb_media_count": len(results),
                    "rgb_media_bytes": downloaded_bytes,
                    "event_truth_authority": False,
                })
        from pipeline import write_jsonl
        write_jsonl(source_receipt_path, rows)
    return {key: value for key, value in receipt.items() if key != "files"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--max-bytes", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
