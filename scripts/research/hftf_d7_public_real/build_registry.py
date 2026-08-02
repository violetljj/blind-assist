#!/usr/bin/env python3
"""Build D7 source/session/frame/candidate registries from audited inputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline import (
    ContractError,
    canonical_sha256,
    dataset_id_for_ledger_row,
    load_json,
    load_jsonl,
    read_int,
    role_for_ledger_row,
    sha256_file,
    stable_id,
    utc_now,
    write_json,
    write_jsonl,
)


OUTPUT_DIRS = (
    "raw",
    "canonical",
    "candidates",
    "clips",
    "contact_sheets",
    "reviews",
    "adjudication",
    "manifests",
    "splits",
    "receipts",
    "reports",
)


def _resolve_input(path: Path, repo_root: Path) -> Path:
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _source_stats(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"ledger_rows": 0, "rgb_frames": 0, "mask_frames": 0, "depth_frames": 0, "pose_frames": 0, "media_rows": 0})
    for row in rows:
        dataset_id = dataset_id_for_ledger_row(row)
        if dataset_id is None:
            continue
        item = stats[dataset_id]
        item["ledger_rows"] += 1
        item["rgb_frames"] += read_int(row.get("rgb_count"))
        item["mask_frames"] += read_int(row.get("mask_count"))
        item["depth_frames"] += read_int(row.get("depth_count"))
        item["pose_frames"] += read_int(row.get("pose_count"))
        if (row.get("session_kind") or "") == "media_session":
            item["media_rows"] += 1
    return dict(stats)


def _build_sessions(rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sessions: dict[str, dict[str, Any]] = {}
    for row in rows:
        dataset_id = dataset_id_for_ledger_row(row)
        root = (row.get("session_root") or "").strip()
        if dataset_id is None or not root or (row.get("session_kind") or "") != "media_session":
            continue
        session_id = stable_id("d7sess", dataset_id, root.lower())
        existing = sessions.get(session_id)
        if existing is None:
            existing = {
                "schema": "hftf_d7_public_real_session_v1",
                "dataset_id": dataset_id,
                "source_session_id": session_id,
                "ancestry_group": stable_id("d7anc", dataset_id, str(Path(root).parent).lower()),
                "session_root": root,
                "data_role": role_for_ledger_row(row),
                "history_roles": sorted(filter(None, (row.get("history_roles") or "").split(";"))),
                "source_license_status": "UNKNOWN_UNTIL_SOURCE_RECEIPT" if dataset_id == "PublicVideo-Auxiliary" else "SEE_DATASET_REGISTRY",
                "rgb_count": 0,
                "mask_count": 0,
                "depth_count": 0,
                "pose_count": 0,
                "source_hashes": [],
            }
            sessions[session_id] = existing
        existing["rgb_count"] += read_int(row.get("rgb_count"))
        existing["mask_count"] += read_int(row.get("mask_count"))
        existing["depth_count"] += read_int(row.get("depth_count"))
        existing["pose_count"] += read_int(row.get("pose_count"))
        if row.get("sha256"):
            existing["source_hashes"].append(row["sha256"])
    ordered = sorted(sessions.values(), key=lambda item: item["source_session_id"])
    return ordered, sessions


def _candidate_rows(candidate_report: Path, sessions: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    report = load_json(candidate_report)
    candidates = report.get("candidates")
    if not isinstance(candidates, list):
        raise ContractError("candidate report must contain a candidates list")
    candidate_rows: list[dict[str, Any]] = []
    frame_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ContractError("candidate report contains non-object candidate")
        source_id = str(candidate.get("source_id", ""))
        raw_session_id = str(candidate.get("session_id", ""))
        source_session_id = stable_id("d7sess", "PublicVideo-Auxiliary", raw_session_id.lower())
        if source_session_id not in sessions:
            sessions[source_session_id] = {
                "schema": "hftf_d7_public_real_session_v1",
                "dataset_id": "PublicVideo-Auxiliary",
                "source_session_id": source_session_id,
                "ancestry_group": stable_id("d7anc", "PublicVideo-Auxiliary", source_id.lower()),
                "session_root": "candidate-report-only",
                "data_role": "THESIS_DEVELOPMENT_CONSUMED_DISCOVERY",
                "history_roles": ["candidate_event_mining_discovery"],
                "source_license_status": "SEE_F_SOURCE_RECORD",
                "rgb_count": 0,
                "mask_count": 0,
                "depth_count": 0,
                "pose_count": 0,
                "source_hashes": [],
            }
        session = sessions[source_session_id]
        candidate_rows.append({
            "schema": "hftf_d7_public_real_candidate_v1",
            "candidate_id": str(candidate.get("candidate_id", "")),
            "dataset_id": session["dataset_id"],
            "source_id": source_id,
            "source_session_id": source_session_id,
            "ancestry_group": session["ancestry_group"],
            "start_frame_index": read_int(candidate.get("start_frame_index")),
            "end_frame_index": read_int(candidate.get("end_frame_index")),
            "start_timestamp_ns": read_int(candidate.get("start_timestamp_ms")) * 1_000_000,
            "end_timestamp_ns": read_int(candidate.get("end_timestamp_ms")) * 1_000_000,
            "selection_role": "DEVELOPMENT_DISCOVERY_ONLY",
            "model_output_visible_to_selector": True,
            "model_hint": str(candidate.get("trigger_type", "")),
            "truth_status": "NOT_EVALUATED",
            "required_confirmation_selection": "MODEL_BLIND",
            "frame_count": len(candidate.get("frame_refs", [])) if isinstance(candidate.get("frame_refs"), list) else 0,
        })
        refs = candidate.get("frame_refs", [])
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            frame_index = read_int(ref.get("frame_index"))
            key = (source_session_id, frame_index)
            if key in frame_by_key:
                continue
            frame_by_key[key] = {
                "schema": "hftf_d7_public_real_frame_v1",
                "dataset_id": session["dataset_id"],
                "source_session_id": source_session_id,
                "ancestry_group": session["ancestry_group"],
                "frame_id": stable_id("d7frm", source_session_id, frame_index, ref.get("frame_sha256", "")),
                "frame_index": frame_index,
                "timestamp_ns": read_int(ref.get("timestamp_ms")) * 1_000_000,
                "rgb_path": str(ref.get("frame_ref", "")),
                "intrinsics_optional": None,
                "pose_optional": None,
                "depth_optional": None,
                "segmentation_optional": None,
                "tracks_optional": None,
                "source_metadata": {"candidate_report_only": True, "truth_blind": True},
                "source_license": "SEE_F_SOURCE_RECORD",
                "provider_revision": "candidate-event-mining-trace",
                "source_hash": str(ref.get("frame_sha256", "")),
            }
    return sorted(candidate_rows, key=lambda item: item["candidate_id"]), sorted(frame_by_key.values(), key=lambda item: (item["source_session_id"], item["frame_index"]))


def _write_reports(root: Path, catalog: dict[str, Any], stats: dict[str, dict[str, int]], session_count: int, candidate_count: int, frame_count: int) -> None:
    lines = [
        "# HFTF D7 source coverage report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "This is an intake snapshot. It does not grant event truth or Confirmation authority.",
        "",
        "| Dataset | Access status | Ledger rows | RGB frames | Mask frames | Depth frames | Pose frames |", 
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_id = {item.get("dataset_id"): item for item in catalog.get("sources", []) if isinstance(item, dict)}
    for dataset_id in sorted(by_id):
        item = by_id[dataset_id]
        values = stats.get(dataset_id, {})
        lines.append(f"| {dataset_id} | {item.get('access_status', 'UNKNOWN')} | {values.get('ledger_rows', 0)} | {values.get('rgb_frames', 0)} | {values.get('mask_frames', 0)} | {values.get('depth_frames', 0)} | {values.get('pose_frames', 0)} |")
    lines.extend([
        "",
        f"- Materialized session rows: `{session_count}`",
        f"- Candidate rows imported: `{candidate_count}` (discovery-only; model-selected)",
        f"- Frame rows imported: `{frame_count}`",
        "- A frame/window count is not a parent-event count.",
        "- Confirmation sources requiring a license agreement or credentials remain `ACCESS_BLOCKED` until the user has independently obtained and supplied lawful access.",
    ])
    (root / "reports" / "source_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    ledger_path = _resolve_input(Path(args.ledger), repo_root)
    catalog_path = _resolve_input(Path(args.source_catalog), repo_root)
    candidate_path = _resolve_input(Path(args.candidate_report), repo_root) if args.candidate_report else None
    output_root = Path(args.output_root).resolve()
    if not ledger_path.is_file():
        raise ContractError(f"ledger not found: {ledger_path}")
    if not catalog_path.is_file():
        raise ContractError(f"source catalog not found: {catalog_path}")
    if output_root.exists() and (output_root / "manifests" / "d7_intake_receipt.json").exists() and not args.allow_existing:
        raise ContractError(f"refusing to overwrite existing D7 run at {output_root}; use --allow-existing for a new run_id")
    for directory in OUTPUT_DIRS:
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    catalog = load_json(catalog_path)
    if not isinstance(catalog, dict) or not isinstance(catalog.get("sources"), list):
        raise ContractError("source catalog must contain sources list")
    with ledger_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    stats = _source_stats(rows)
    sessions, session_map = _build_sessions(rows)
    candidate_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    if candidate_path:
        candidate_rows, frame_rows = _candidate_rows(candidate_path, session_map)
        sessions = sorted(session_map.values(), key=lambda item: item["source_session_id"])
    dataset_registry = {
        "schema": "hftf_d7_public_real_dataset_registry_v1",
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "generated_at_utc": utc_now(),
        "catalog": catalog,
        "ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path), "row_count": len(rows)},
        "source_stats": stats,
        "candidate_discovery": {"path": str(candidate_path) if candidate_path else None, "sha256": sha256_file(candidate_path) if candidate_path else None, "candidate_count": len(candidate_rows), "frame_count": len(frame_rows)},
        "session_count": len(sessions),
        "data_role_policy": "source-session-first; consumed/burned remains Development only; gated sources are ACCESS_BLOCKED",
    }
    write_json(output_root / "manifests" / "dataset_registry.json", dataset_registry)
    write_jsonl(output_root / "manifests" / "session_registry.jsonl", sessions)
    write_jsonl(output_root / "canonical" / "frame_registry.jsonl", frame_rows)
    write_jsonl(output_root / "candidates" / "candidate_index.jsonl", candidate_rows)
    receipts = []
    for item in catalog["sources"]:
        if not isinstance(item, dict):
            continue
        receipts.append({
            "schema": "hftf_d7_public_real_source_receipt_v1",
            "dataset_id": item.get("dataset_id"),
            "official_url": item.get("official_url"),
            "download_url": item.get("download_url"),
            "license": item.get("license"),
            "access_status": item.get("access_status"),
            "retrieved_at_utc": None,
            "source_hash": None,
            "local_evidence_paths": item.get("local_evidence_paths", []),
            "receipt_kind": "catalog_and_local_ledger_snapshot",
            "event_truth_authority": False,
        })
    write_jsonl(output_root / "receipts" / "source_receipts.jsonl", receipts)
    _write_reports(output_root, catalog, stats, len(sessions), len(candidate_rows), len(frame_rows))
    receipt = {
        "schema": "hftf_d7_public_real_intake_receipt_v1",
        "run_id": args.run_id,
        "contract_id": "HFTF_D7_PUBLIC_REAL_R1",
        "generated_at_utc": utc_now(),
        "ledger": {"path": str(ledger_path), "sha256": sha256_file(ledger_path)},
        "catalog": {"path": str(catalog_path), "sha256": sha256_file(catalog_path)},
        "candidate_report": {"path": str(candidate_path), "sha256": sha256_file(candidate_path)} if candidate_path else None,
        "output_root": str(output_root),
        "counts": {"ledger_rows": len(rows), "sessions": len(sessions), "candidates": len(candidate_rows), "frames": len(frame_rows)},
        "authority": {"confirmation": False, "event_truth": False, "training": False, "production": False},
    }
    write_json(output_root / "manifests" / "d7_intake_receipt.json", receipt)
    write_json(output_root / "manifests" / "d7_intake_manifest.json", {"receipt_sha256": canonical_sha256(receipt), "required_registry_files": ["dataset_registry.json", "session_registry.jsonl", "frame_registry.jsonl", "source_receipts.jsonl"]})
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--candidate-report")
    parser.add_argument("--source-catalog", default=str(Path(__file__).with_name("source_catalog.json")))
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
