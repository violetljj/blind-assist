#!/usr/bin/env python3
"""Inventory public SANPO-Real GCS sessions without opening existing labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pipeline import ContractError, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
PREFIX = "sanpo_dataset/v0/sanpo-real/"


def _get(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": "blindassist-hftf-d7/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ContractError("SANPO GCS response is not an object")
    return payload


def _list_prefixes() -> list[str]:
    prefixes: list[str] = []
    token: str | None = None
    while True:
        params = {"prefix": PREFIX, "delimiter": "/", "maxResults": "1000"}
        if token:
            params["pageToken"] = token
        payload = _get(params)
        prefixes.extend(str(value) for value in payload.get("prefixes", []))
        token = payload.get("nextPageToken")
        if not token:
            return sorted(set(prefixes))


def _sample_session(session_prefix: str) -> dict[str, Any]:
    top = _get({"prefix": session_prefix, "delimiter": "/", "maxResults": "1000"})
    camera_prefixes = sorted(str(value) for value in top.get("prefixes", []))
    examples: list[dict[str, Any]] = []
    for camera_prefix in camera_prefixes:
        nested = _get({"prefix": camera_prefix, "delimiter": "/", "maxResults": "1000"})
        for value in nested.get("items", []):
            if isinstance(value, dict):
                examples.append({
                    "name": value.get("name"),
                    "size": value.get("size"),
                    "generation": value.get("generation"),
                    "md5Hash": value.get("md5Hash"),
                })
        lens_prefixes = sorted(str(value) for value in nested.get("prefixes", []))
        for lens_prefix in lens_prefixes:
            if not lens_prefix.endswith("/"):
                continue
            video_prefix = lens_prefix + "video_frames/"
            frame_page = _get({"prefix": video_prefix, "maxResults": "5"})
            for value in frame_page.get("items", []):
                if isinstance(value, dict):
                    examples.append({
                        "name": value.get("name"),
                        "size": value.get("size"),
                        "generation": value.get("generation"),
                        "md5Hash": value.get("md5Hash"),
                    })
    examples.sort(key=lambda value: str(value.get("name")))
    raw = json.dumps(examples, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "dataset_id": "SANPO-Real",
        "source_session_id": session_prefix.rstrip("/").split("/")[-1],
        "session_prefix": session_prefix,
        "camera_prefixes": camera_prefixes,
        "sample_object_count": len(examples),
        "sample_object_inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "sample_objects": examples,
        "candidate_selection_authority": "NOT_RUN",
        "event_truth_authority": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    prefixes = _list_prefixes()
    if args.max_sessions:
        prefixes = prefixes[: args.max_sessions]
    records = [_sample_session(prefix) for prefix in prefixes]
    inventory = {
        "schema": "hftf_d7_public_real_sanpo_public_gcs_inventory_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "official_url": "https://google-research-datasets.github.io/sanpo_dataset/",
        "gcs_api": API,
        "gcs_prefix": PREFIX,
        "license": "CC-BY-4.0",
        "session_count": len(records),
        "records": records,
        "event_truth_authority": False,
        "media_downloaded": False,
        "notes": [
            "This is public object metadata inventory only; it does not infer event labels.",
            "Existing local SANPO-derived/consumed cohorts were not relabeled or merged as fresh D7 truth.",
            "Each source session remains the split-isolation unit.",
        ],
    }
    raw_path = root / "raw" / "sanpo-gcs-inventory.json"
    receipt_path = root / "receipts" / f"sanpo_gcs_inventory_receipt_{args.run_id}.json"
    write_json(raw_path, inventory)
    receipt = {
        "schema": "hftf_d7_public_real_sanpo_gcs_inventory_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "dataset_id": "SANPO-Real",
        "access_status": "PUBLIC_GCS_OBJECT_METADATA_INVENTORIED",
        "license": "CC-BY-4.0",
        "session_count": len(records),
        "sample_object_count": sum(int(record.get("sample_object_count", 0)) for record in records),
        "inventory_path": str(raw_path),
        "inventory_sha256": sha256_file(raw_path),
        "media_downloaded": False,
        "event_truth_authority": False,
    }
    write_json(receipt_path, receipt)
    source_receipt_path = root / "receipts" / "source_receipts.jsonl"
    if source_receipt_path.is_file():
        rows = load_jsonl(source_receipt_path)
        for row in rows:
            if row.get("dataset_id") == "SANPO-Real":
                row.update({
                    "access_status": receipt["access_status"],
                    "retrieved_at_utc": receipt["generated_at_utc"],
                    "source_hash": receipt["inventory_sha256"],
                    "source_hash_kind": "MATERIALIZED_INTAKE_RECEIPT",
                    "local_evidence_paths": [str(raw_path), str(receipt_path)],
                    "receipt_kind": "public_gcs_metadata_inventory",
                    "event_truth_authority": False,
                })
        write_jsonl(source_receipt_path, rows)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-sessions", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
