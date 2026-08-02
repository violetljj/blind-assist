#!/usr/bin/env python3
"""Refresh the D7 registry and source receipts from the frozen source catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import ContractError, canonical_sha256, load_json, load_jsonl, sha256_file, utc_now, write_json, write_jsonl


def _coverage_report(root: Path, catalog: dict[str, Any], registry: dict[str, Any]) -> None:
    stats = registry.get("source_stats", {})
    rows = [
        "# HFTF D7 source coverage report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "This is an intake snapshot. It does not grant event truth or Confirmation authority.",
        "",
        "| Dataset | Access status | Ledger rows | RGB frames | Mask frames | Depth frames | Pose frames | Candidate windows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in sorted(catalog.get("sources", []), key=lambda value: str(value.get("dataset_id"))):
        if not isinstance(item, dict):
            continue
        dataset_id = item.get("dataset_id")
        values = stats.get(dataset_id, {})
        rows.append(
            f"| {dataset_id} | {item.get('access_status', 'UNKNOWN')} | {values.get('ledger_rows', 0)} | "
            f"{values.get('rgb_frames', 0)} | {values.get('mask_frames', 0)} | {values.get('depth_frames', 0)} | "
            f"{values.get('pose_frames', 0)} | {values.get('candidate_windows', 0)} |"
        )
    discovery = registry.get("candidate_discovery", {})
    rows.extend([
        "",
        f"- Top-level session rows: `{registry.get('session_count', 0)}`",
        f"- Top-level candidate rows: `{discovery.get('total_candidate_count', discovery.get('candidate_count', 0))}`",
        f"- Top-level frame rows: `{discovery.get('total_frame_count', discovery.get('frame_count', 0))}`",
        "- Model-selected candidate reports remain Development discovery only.",
        "- EgoWalk extracted RGB/trajectory material remains review input; raw recordings remain ACCESS_BLOCKED.",
        "- Missing tracks, ancestry, event labels, or lawful source terms remain UNKNOWN/NOT_EVALUABLE.",
    ])
    (root / "reports" / "source_coverage_report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    catalog_path = Path(args.source_catalog).resolve()
    registry_path = root / "manifests" / "dataset_registry.json"
    receipt_path = root / "receipts" / "source_receipts.jsonl"
    intake_path = root / "manifests" / "d7_intake_receipt.json"
    if not catalog_path.is_file() or not registry_path.is_file() or not receipt_path.is_file():
        raise ContractError("D7 catalog refresh inputs are incomplete")
    catalog = load_json(catalog_path)
    registry = load_json(registry_path)
    if not isinstance(catalog, dict) or not isinstance(registry, dict):
        raise ContractError("catalog and registry must be objects")
    by_id = {str(item.get("dataset_id")): item for item in catalog.get("sources", []) if isinstance(item, dict)}
    receipt_rows = load_jsonl(receipt_path)
    refreshed = 0
    for row in receipt_rows:
        dataset_id = str(row.get("dataset_id"))
        item = by_id.get(dataset_id)
        if item is None:
            continue
        preserved = {
            key: row[key]
            for key in (
                "retrieved_at_utc", "source_hash", "rgb_media_count", "rgb_media_bytes",
                "raw_recording_status", "receipt_kind",
            ) if key in row and row[key] not in (None, "")
        }
        row.update({
            "official_url": item.get("official_url"),
            "download_url": item.get("download_url"),
            "license": item.get("license"),
            "access_status": item.get("access_status"),
            "local_evidence_paths": item.get("local_evidence_paths", []),
            "event_truth_authority": False,
            **preserved,
        })
        # A catalog-only source still needs a reproducible intake receipt.  This
        # hash identifies the frozen catalog/access snapshot; it must not be
        # interpreted as a hash of source media or event annotations.
        if not row.get("source_hash"):
            row["source_hash"] = canonical_sha256({
                "dataset_id": dataset_id,
                "official_url": row.get("official_url"),
                "download_url": row.get("download_url"),
                "license": row.get("license"),
                "access_status": row.get("access_status"),
                "local_evidence_paths": row.get("local_evidence_paths", []),
                "receipt_kind": row.get("receipt_kind"),
            })
            row["source_hash_kind"] = "CATALOG_ACCESS_SNAPSHOT"
            row["source_hash_note"] = "Hash covers the catalog/access receipt only; source media and event truth remain unmaterialized."
        else:
            row.setdefault("source_hash_kind", "MATERIALIZED_INTAKE_RECEIPT")
        refreshed += 1
    write_jsonl(receipt_path, receipt_rows)
    registry["catalog"] = catalog
    registry["catalog_source_path"] = str(catalog_path)
    registry["catalog_sha256"] = sha256_file(catalog_path)
    registry["generated_at_utc"] = utc_now()
    write_json(registry_path, registry)
    if intake_path.is_file():
        intake = load_json(intake_path)
        if isinstance(intake, dict):
            intake["catalog"] = {"path": str(catalog_path), "sha256": sha256_file(catalog_path)}
            intake["catalog_refresh_run_id"] = args.run_id
            intake["generated_at_utc"] = utc_now()
            write_json(intake_path, intake)
    _coverage_report(root, catalog, registry)
    return {
        "schema": "hftf_d7_public_real_catalog_refresh_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "catalog_path": str(catalog_path),
        "catalog_sha256": sha256_file(catalog_path),
        "source_receipts_refreshed": refreshed,
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--source-catalog", default=str(Path(__file__).with_name("source_catalog.json")))
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
