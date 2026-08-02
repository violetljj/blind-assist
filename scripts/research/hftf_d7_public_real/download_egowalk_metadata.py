#!/usr/bin/env python3
"""Download only the public EgoWalk trajectory metadata/parquet index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pipeline import ContractError, utc_now, write_json


API_ROOT = "https://huggingface.co/api/datasets/EgoWalk/trajectories/tree/main"
RAW_ROOT = "https://huggingface.co/datasets/EgoWalk/trajectories/resolve/main"
USER_AGENT = "BlindAssist-HFTF-D7/1.0 (public metadata intake)"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list(prefix: str, timeout: float) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"recursive": "false", "expand": "false"})
    request = urllib.request.Request(f"{API_ROOT}/{prefix}?{query}", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.load(response)
    except OSError as exc:
        raise ContractError(f"EgoWalk metadata listing failed for {prefix}: {exc}") from exc
    if not isinstance(value, list):
        raise ContractError(f"unexpected EgoWalk listing for {prefix}")
    return [item for item in value if isinstance(item, dict) and item.get("type") == "file"]


def _download(url: str, target: Path, timeout: float) -> tuple[int, str]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target.stat().st_size, sha256_file(target)
    temporary_name: str | None = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent)
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(descriptor, "wb") as handle:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary_name, target)
        temporary_name = None
        return size, digest.hexdigest()
    except OSError as exc:
        raise ContractError(f"EgoWalk metadata download failed: {url}: {exc}") from exc
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    payload_root = root / "raw" / "egowalk-trajectories"
    files = _list("meta", args.timeout) + _list("data", args.timeout)
    if not files:
        raise ContractError("EgoWalk returned no metadata files")
    downloaded: list[dict[str, Any]] = []
    for item in files:
        relative = str(item["path"])
        target = payload_root / Path(relative)
        size, sha = _download(f"{RAW_ROOT}/{urllib.parse.quote(relative, safe='/')}", target, args.timeout)
        expected = item.get("lfs", {}).get("oid") if isinstance(item.get("lfs"), dict) else None
        downloaded.append({"path": relative, "size": size, "sha256": sha, "provider_lfs_oid": expected, "hash_match": expected in (None, sha)})
    receipt = {
        "schema": "hftf_d7_public_real_egowalk_metadata_receipt_v1",
        "run_id": args.run_id,
        "dataset_id": "EgoWalk",
        "access_status": "PUBLIC_METADATA_DOWNLOADED",
        "license_status": "NOT_RESOLVED_BY_METADATA_ONLY_INTAKE",
        "source_url": "https://huggingface.co/datasets/EgoWalk/trajectories",
        "raw_source_url": "https://huggingface.co/datasets/EgoWalk/raw-recordings",
        "retrieved_at_utc": utc_now(),
        "payload_root": str(payload_root),
        "files": downloaded,
        "file_count": len(downloaded),
        "total_bytes": sum(int(item["size"]) for item in downloaded),
        "video_bytes_downloaded": 0,
        "event_truth_authority": False,
        "notes": [
            "This run downloads data/*.parquet and meta/*.json only; raw RGB/depth videos remain unopened.",
            "Parquet trajectories are source metadata/geometry candidates, not human event truth.",
            "Pose nulls and reinitialization boundaries must be cut before candidate generation.",
        ],
    }
    write_json(root / "receipts" / "egowalk_metadata_receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
