#!/usr/bin/env python3
"""Inventory exact SANPO-Real frame spans without downloading media.

The public SANPO GCS inventory used by the first D7 canary intentionally kept
only five example frames per view.  That is sufficient to prove object access,
but it cannot support a reproducible continuous-window candidate surface.
This command enumerates provider metadata for one explicitly selected camera
and view across the public sessions, records the exact RGB/depth/mask frame
indices, and writes no event label or model-derived selection.

The resulting span inventory is still source-intake evidence.  A later bounded
media materializer must download selected objects and verify their provider
hashes before a reviewer can use them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

from pipeline import ContractError, canonical_json, load_json, load_jsonl, sha256_file, stable_id, utc_now, write_json, write_jsonl


API = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
PREFIX = "sanpo_dataset/v0/sanpo-real/"
DATASET = "SANPO-Real"
FRAME_RE = re.compile(r"/(?:video_frames|depth_maps|segmentation_masks)/(?P<index>\d{6})(?:\.float16\.gz|\.png)$")


def _get(params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{API}?{query}",
        headers={"User-Agent": "blindassist-hftf-d7/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
    except OSError as exc:
        raise ContractError(f"SANPO GCS metadata request failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("SANPO GCS response is not an object")
    return payload


def _list_objects(prefix: str, *, delimiter: str | None = None) -> list[dict[str, Any]]:
    """List all object metadata under a prefix, preserving provider fields."""

    items: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        params = {
            "prefix": prefix,
            "maxResults": "1000",
            "fields": "items(name,size,generation,md5Hash),nextPageToken",
        }
        if delimiter is not None:
            params["delimiter"] = delimiter
        if token:
            params["pageToken"] = token
        payload = _get(params)
        for item in payload.get("items", []):
            if isinstance(item, dict) and item.get("name"):
                items.append({
                    "name": str(item["name"]),
                    "size": int(item.get("size", 0)),
                    "generation": item.get("generation"),
                    "md5Hash": item.get("md5Hash"),
                })
        token = payload.get("nextPageToken")
        if not token:
            return items


def _frame_kind(name: str) -> str | None:
    lowered = name.lower()
    if "/video_frames/" in lowered and lowered.endswith(".png"):
        return "rgb"
    if "/depth_maps/" in lowered and lowered.endswith(".float16.gz"):
        return "depth"
    if "/segmentation_masks/" in lowered and lowered.endswith(".png"):
        return "mask"
    return None


def _frame_index(item: dict[str, Any]) -> tuple[str, int] | None:
    name = str(item.get("name", ""))
    match = FRAME_RE.search(name)
    kind = _frame_kind(name)
    if match is None or kind is None:
        return None
    return kind, int(match.group("index"))


def _contiguous_runs(indices: Iterable[int]) -> list[dict[str, int]]:
    ordered = sorted(set(int(value) for value in indices))
    if not ordered:
        return []
    runs: list[dict[str, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            runs.append({"start": start, "end": previous, "count": previous - start + 1})
            start = value
        previous = value
    runs.append({"start": start, "end": previous, "count": previous - start + 1})
    return runs


def _metadata_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _summarize_media(objects: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, list[dict[str, Any]]] = {"rgb": [], "depth": [], "mask": []}
    ignored: list[dict[str, Any]] = []
    for item in objects:
        parsed = _frame_index(item)
        if parsed is None:
            ignored.append(item)
            continue
        kind, _ = parsed
        by_kind[kind].append(item)
    indexed: dict[str, dict[int, dict[str, Any]]] = {kind: {} for kind in by_kind}
    for kind, items in by_kind.items():
        for item in items:
            parsed = _frame_index(item)
            assert parsed is not None
            _, index = parsed
            if index in indexed[kind]:
                raise ContractError(f"duplicate SANPO {kind} provider frame index: {index}")
            indexed[kind][index] = item
    frame_indices = {kind: sorted(values) for kind, values in indexed.items()}
    complete = sorted(set(frame_indices["rgb"]) & set(frame_indices["depth"]) & set(frame_indices["mask"]))
    return {
        "object_count": len(objects),
        "ignored_object_count": len(ignored),
        "frame_indices_by_kind": frame_indices,
        "complete_frame_indices": complete,
        "complete_frame_count": len(complete),
        "complete_frame_runs": _contiguous_runs(complete),
        "objects_by_kind": {
            kind: sorted(indexed[kind].values(), key=lambda item: str(item.get("name")))
            for kind in sorted(indexed)
        },
        "object_metadata_sha256": _metadata_digest(sorted(objects, key=lambda item: str(item.get("name")))),
    }


def _auxiliary_objects(session_prefix: str, camera: str) -> dict[str, list[dict[str, Any]]]:
    camera_prefix = f"{session_prefix}camera_{camera}/"
    camera_items = _list_objects(camera_prefix, delimiter="/")
    session_items = _list_objects(session_prefix, delimiter="/")
    return {
        "pose": sorted(
            [item for item in camera_items if str(item.get("name", "")).endswith(("camera_poses.csv", "fixed_camera_poses.csv"))],
            key=lambda item: str(item.get("name")),
        ),
        "intrinsics": sorted(
            [item for item in session_items if str(item.get("name", "")).endswith("/description.json")],
            key=lambda item: str(item.get("name")),
        ),
    }


def _inventory_one(record: dict[str, Any], *, camera: str, view: str) -> dict[str, Any]:
    session_id = str(record.get("source_session_id") or "")
    session_prefix = str(record.get("session_prefix") or f"{PREFIX}{session_id}/")
    if not session_id or not session_prefix.startswith(PREFIX):
        raise ContractError(f"invalid SANPO inventory session record: {record}")
    view_prefix = f"{session_prefix}camera_{camera}/{view}/"
    media_objects = _list_objects(view_prefix)
    media = _summarize_media(media_objects)
    auxiliary = _auxiliary_objects(session_prefix, camera)
    complete = media["complete_frame_indices"]
    return {
        "schema": "hftf_d7_public_real_sanpo_frame_span_v1",
        "dataset_id": DATASET,
        "source_session_id": session_id,
        "ancestry_group": stable_id("d7anc", DATASET, session_id),
        "session_prefix": session_prefix,
        "camera": camera,
        "view": view,
        "media_prefix": view_prefix,
        "source_license": "CC-BY-4.0",
        "provider_revision": "gcs-generation-metadata",
        "source_metadata": {
            "selection_authority": "MODEL_BLIND_SOURCE_FRAME_COVERAGE",
            "event_truth_authority": False,
            "rgb_depth_mask_binding": "INDEX_KEYED",
            "capture_timestamp_authoritative": False,
            "pose_row_binding": "NOT_EVALUABLE",
            "complete_frame_count": len(complete),
            "complete_frame_runs": media["complete_frame_runs"],
        },
        "media": media,
        "auxiliary": auxiliary,
        "status": "PUBLIC_FRAME_SPAN_METADATA_INVENTORIED" if complete else "NO_COMPLETE_RGB_DEPTH_MASK_SPAN",
    }


def _write_source_receipt(root: Path, receipt: dict[str, Any], raw_path: Path, receipt_path: Path) -> None:
    source_path = root / "receipts" / "source_receipts.jsonl"
    if not source_path.is_file():
        return
    rows = load_jsonl(source_path)
    found = False
    for row in rows:
        if row.get("dataset_id") != DATASET:
            continue
        found = True
        row.update({
            "access_status": receipt["access_status"],
            "retrieved_at_utc": receipt["generated_at_utc"],
            "source_hash": receipt["inventory_sha256"],
            "source_hash_kind": "FRAME_SPAN_METADATA_RECEIPT",
            "local_evidence_paths": [str(raw_path), str(receipt_path)],
            "receipt_kind": "public_gcs_frame_span_inventory",
            "event_truth_authority": False,
        })
    if found:
        write_jsonl(source_path, rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    inventory_path = root / "raw" / "sanpo-gcs-inventory.json"
    if not inventory_path.is_file():
        raise ContractError(f"SANPO object inventory is required: {inventory_path}")
    inventory = load_json(inventory_path)
    records = inventory.get("records") if isinstance(inventory, dict) else None
    if not isinstance(records, list) or not records:
        raise ContractError("SANPO object inventory has no records")
    requested = {str(value) for value in args.session_id}
    selected = [record for record in records if not requested or str(record.get("source_session_id")) in requested]
    if args.max_sessions:
        selected = selected[: args.max_sessions]
    if not selected:
        raise ContractError("SANPO frame-span selection is empty")
    if args.workers <= 0:
        raise ContractError("--workers must be positive")
    output_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_inventory_one, record, camera=args.camera, view=args.view): str(record.get("source_session_id"))
            for record in selected
        }
        for future in as_completed(futures):
            session_id = futures[future]
            try:
                output_records.append(future.result())
            except Exception as exc:  # keep an auditable per-session failure, do not hide it
                errors.append({"source_session_id": session_id, "error": f"{type(exc).__name__}: {exc}"})
    output_records.sort(key=lambda row: str(row.get("source_session_id")))
    errors.sort(key=lambda row: row["source_session_id"])
    raw_payload = {
        "schema": "hftf_d7_public_real_sanpo_frame_span_inventory_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "dataset_id": DATASET,
        "source_object_inventory": str(inventory_path),
        "source_object_inventory_sha256": sha256_file(inventory_path),
        "camera": args.camera,
        "view": args.view,
        "fps_for_later_nominal_time": args.fps,
        "selection_authority": "MODEL_BLIND_SOURCE_FRAME_COVERAGE",
        "event_truth_authority": False,
        "records": output_records,
        "errors": errors,
    }
    raw_path = root / "raw" / f"sanpo-frame-span-inventory-{args.run_id}.json"
    if raw_path.exists():
        raise ContractError(f"frame-span inventory already exists; refusing overwrite: {raw_path}")
    write_json(raw_path, raw_payload)
    receipt_path = root / "receipts" / f"sanpo_frame_span_receipt_{args.run_id}.json"
    if receipt_path.exists():
        raise ContractError(f"receipt already exists; refusing overwrite: {receipt_path}")
    complete_count = sum(int(row.get("media", {}).get("complete_frame_count", 0)) for row in output_records)
    status = "PUBLIC_GCS_FRAME_SPANS_INVENTORIED" if not errors else "PUBLIC_GCS_FRAME_SPANS_PARTIAL"
    receipt = {
        "schema": "hftf_d7_public_real_sanpo_frame_span_receipt_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "run_id": args.run_id,
        "generated_at_utc": raw_payload["generated_at_utc"],
        "dataset_id": DATASET,
        "camera": args.camera,
        "view": args.view,
        "access_status": status,
        "session_count_requested": len(selected),
        "session_count_materialized": len(output_records),
        "session_error_count": len(errors),
        "complete_frame_count": complete_count,
        "inventory_path": str(raw_path),
        "inventory_sha256": sha256_file(raw_path),
        "selection_authority": "MODEL_BLIND_SOURCE_FRAME_COVERAGE",
        "event_truth_authority": False,
        "license": "CC-BY-4.0",
    }
    write_json(receipt_path, receipt)
    _write_source_receipt(root, receipt, raw_path, receipt_path)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--camera", default="chest", choices=("chest", "head"))
    parser.add_argument("--view", default="left", choices=("left", "right"))
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--session-id", action="append", default=[])
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
