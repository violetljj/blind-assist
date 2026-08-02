#!/usr/bin/env python3
"""Merge the model-blind EgoWalk intake into the D7 top-level registries.

The merge is append-only at the logical-record level: existing records are
validated and retained, and duplicate IDs fail closed.  The generated D7
registry still describes all imported windows as development discovery; it
does not create event truth or confirmation authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import (
    ContractError,
    load_json,
    load_jsonl,
    sha256_file,
    utc_now,
    write_json,
    write_jsonl,
)


def _merge_jsonl(
    target: Path,
    addition: Path,
    *,
    id_field: str,
) -> tuple[int, int]:
    if not target.is_file() or not addition.is_file():
        raise ContractError(f"merge input missing: {target} / {addition}")
    seen: set[str] = set()
    before = 0
    after = 0
    temp = target.with_suffix(target.suffix + ".tmp")
    with target.open("r", encoding="utf-8") as source, temp.open("w", encoding="utf-8", newline="\n") as dest:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ContractError(f"invalid existing JSONL {target}:{line_number}: {exc}") from exc
            if not isinstance(row, dict) or not row.get(id_field):
                raise ContractError(f"existing JSONL row missing {id_field}: {target}:{line_number}")
            record_id = str(row[id_field])
            if record_id in seen:
                raise ContractError(f"duplicate {id_field} in existing registry: {record_id}")
            seen.add(record_id)
            dest.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            before += 1
        with addition.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"invalid added JSONL {addition}:{line_number}: {exc}") from exc
                if not isinstance(row, dict) or not row.get(id_field):
                    raise ContractError(f"added JSONL row missing {id_field}: {addition}:{line_number}")
                record_id = str(row[id_field])
                if record_id in seen:
                    raise ContractError(f"duplicate {id_field} across registries: {record_id}")
                seen.add(record_id)
                dest.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                after += 1
    temp.replace(target)
    return before, after


def _merge_sessions(target: Path, trajectory_manifest: dict[str, Any]) -> tuple[int, int]:
    existing = load_jsonl(target)
    seen = {str(row.get("source_session_id")) for row in existing}
    added: list[dict[str, Any]] = []
    for item in trajectory_manifest.get("records", []):
        if not isinstance(item, dict):
            raise ContractError("trajectory manifest contains non-object record")
        session_id = str(item.get("source_session_id", ""))
        if not session_id or session_id in seen:
            raise ContractError(f"duplicate or missing EgoWalk source session: {session_id}")
        seen.add(session_id)
        added.append({
            "schema": "hftf_d7_public_real_session_v1",
            "dataset_id": "EgoWalk",
            "source_session_id": session_id,
            "ancestry_group": item.get("ancestry_group"),
            "session_root": item.get("source_parquet"),
            "data_role": "DEVELOPMENT_CANDIDATE_DISCOVERY",
            "history_roles": ["d7_uniform_coverage_metadata"],
            "source_license_status": "TERMS_NOT_RESOLVED",
            "rgb_count": 0,
            "mask_count": 0,
            "depth_count": 0,
            "pose_count": int(item.get("valid_pose_row_count", 0)),
            "source_hashes": [item.get("source_hash")],
            "source_truth_status": "METADATA_GEOMETRY_ONLY",
            "candidate_count": int(item.get("candidate_count", 0)),
        })
    write_jsonl(target, existing + sorted(added, key=lambda row: row["source_session_id"]))
    return len(existing), len(added)


def _coverage_report(root: Path, catalog: dict[str, Any], registry: dict[str, Any], *, egowalk_receipt: dict[str, Any]) -> None:
    stats = registry.get("source_stats", {})
    lines = [
        "# HFTF D7 source coverage report",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "This is an intake snapshot. It does not grant event truth or Confirmation authority.",
        "",
        "| Dataset | Access status | Ledger rows | RGB frames | Mask frames | Depth frames | Pose frames | Candidate windows |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    catalog_by_id = {item.get("dataset_id"): item for item in catalog.get("sources", []) if isinstance(item, dict)}
    for dataset_id in sorted(catalog_by_id):
        item = catalog_by_id[dataset_id]
        values = stats.get(dataset_id, {})
        candidates = values.get("candidate_windows", 0)
        if dataset_id == "EgoWalk":
            candidates = egowalk_receipt.get("counts", {}).get("candidate_windows", candidates)
        lines.append(
            f"| {dataset_id} | {item.get('access_status', 'UNKNOWN')} | {values.get('ledger_rows', 0)} | "
            f"{values.get('rgb_frames', 0)} | {values.get('mask_frames', 0)} | {values.get('depth_frames', 0)} | "
            f"{values.get('pose_frames', 0)} | {candidates} |"
        )
    counts = egowalk_receipt.get("counts", {})
    lines.extend([
        "",
        f"- Top-level session rows after merge: `{registry.get('session_count', 0)}`",
        f"- Top-level candidate rows after merge: `{registry.get('candidate_discovery', {}).get('total_candidate_count', 'UNKNOWN')}`",
        f"- EgoWalk model-blind coverage candidates: `{counts.get('candidate_windows', 0)}`",
        f"- EgoWalk parent events admitted: `{counts.get('parent_events_admitted', 0)}`",
        "- EgoWalk extracted pose/video metadata is not event truth; raw recordings remain access-gated.",
        "- A frame/window count is not a parent-event count.",
        "- Missing tracks, ancestry, event labels, or lawful source terms remain `UNKNOWN`/`NOT_EVALUABLE`.",
    ])
    (root / "reports" / "source_coverage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).resolve()
    if not (root / "manifests" / "dataset_registry.json").is_file():
        raise ContractError(f"D7 registry not found: {root}")
    trajectory_manifest_path = root / "manifests" / "egowalk_trajectory_manifest.json"
    ingest_receipt_paths = sorted((root / "receipts").glob("egowalk_metadata_ingest_receipt_*.json"))
    if not trajectory_manifest_path.is_file() or len(ingest_receipt_paths) != 1:
        raise ContractError("expected exactly one EgoWalk trajectory manifest and ingest receipt")
    trajectory_manifest = load_json(trajectory_manifest_path)
    egowalk_receipt = load_json(ingest_receipt_paths[0])
    registry_path = root / "manifests" / "dataset_registry.json"
    registry = load_json(registry_path)
    catalog = registry.get("catalog", {})
    if not isinstance(registry, dict) or not isinstance(catalog, dict):
        raise ContractError("dataset registry/catalog malformed")

    frame_before, frame_added = _merge_jsonl(
        root / "canonical" / "frame_registry.jsonl",
        root / "canonical" / "egowalk_frame_registry.jsonl",
        id_field="frame_id",
    )
    candidate_before, candidate_added = _merge_jsonl(
        root / "candidates" / "candidate_index.jsonl",
        root / "candidates" / "egowalk_candidate_index.jsonl",
        id_field="candidate_id",
    )
    session_before, session_added = _merge_sessions(root / "manifests" / "session_registry.jsonl", trajectory_manifest)

    source_stats = registry.setdefault("source_stats", {})
    counts = egowalk_receipt.get("counts", {})
    source_stats["EgoWalk"] = {
        "ledger_rows": len(trajectory_manifest.get("records", [])),
        "rgb_frames": 0,
        "mask_frames": 0,
        "depth_frames": 0,
        "pose_frames": counts.get("frame_rows", 0),
        "media_rows": len(trajectory_manifest.get("records", [])),
        "candidate_windows": counts.get("candidate_windows", 0),
        "metadata_rows": sum(int(item.get("row_count", 0)) for item in trajectory_manifest.get("records", []) if isinstance(item, dict)),
    }
    old_candidate_discovery = registry.get("candidate_discovery", {})
    if not isinstance(old_candidate_discovery, dict):
        old_candidate_discovery = {}
    old_count = int(old_candidate_discovery.get("candidate_count", 0))
    old_frames = int(old_candidate_discovery.get("frame_count", 0))
    registry["candidate_discovery"] = {
        **old_candidate_discovery,
        "candidate_count": old_count,
        "frame_count": old_frames,
        "total_candidate_count": candidate_before + candidate_added,
        "total_frame_count": frame_before + frame_added,
        "imports": [
            {
                "source": "candidate-event-mining",
                "path": old_candidate_discovery.get("path"),
                "candidate_count": old_count,
                "frame_count": old_frames,
                "model_blind": False,
                "authority": "DEVELOPMENT_DISCOVERY_ONLY",
            },
            {
                "source": "EgoWalk/trajectories",
                "path": str(root / "candidates" / "egowalk_candidate_index.jsonl"),
                "candidate_count": candidate_added,
                "frame_count": frame_added,
                "model_blind": True,
                "authority": "DEVELOPMENT_DISCOVERY_ONLY",
                "receipt_sha256": sha256_file(ingest_receipt_paths[0]),
            },
        ],
    }
    registry["session_count"] = session_before + session_added
    registry["generated_at_utc"] = utc_now()
    registry["data_role_policy"] = "source-session-first; EgoWalk extracted trajectory is model-blind Development discovery only; raw media and event truth remain gated/unknown"
    write_json(registry_path, registry)

    receipt_rows = load_jsonl(root / "receipts" / "source_receipts.jsonl")
    for row in receipt_rows:
        if row.get("dataset_id") == "EgoWalk":
            row.update({
                "access_status": "PUBLIC_EXTRACTED_METADATA_DOWNLOADED_RAW_RECORDINGS_ACCESS_BLOCKED",
                "retrieved_at_utc": egowalk_receipt.get("generated_at_utc"),
                "source_hash": egowalk_receipt.get("trajectory_manifest_sha256"),
                "source_hash_kind": "MATERIALIZED_INTAKE_RECEIPT",
                "local_evidence_paths": [
                    "F:/ba-data/hftf-d7-public-real/raw/egowalk-trajectories",
                    "F:/ba-data/hftf-d7-public-real/manifests/egowalk_trajectory_manifest.json",
                ],
                "receipt_kind": "metadata_download_and_model_blind_coverage_ingest",
                "event_truth_authority": False,
                "raw_recording_status": "ACCESS_BLOCKED",
            })
    write_jsonl(root / "receipts" / "source_receipts.jsonl", receipt_rows)
    _coverage_report(root, catalog, registry, egowalk_receipt=egowalk_receipt)

    merge_receipt = {
        "schema": "hftf_d7_public_real_egowalk_registry_merge_receipt_v1",
        "run_id": args.run_id,
        "generated_at_utc": utc_now(),
        "egowalk_ingest_receipt": str(ingest_receipt_paths[0]),
        "egowalk_ingest_receipt_sha256": sha256_file(ingest_receipt_paths[0]),
        "counts": {
            "existing_frames": frame_before,
            "added_frames": frame_added,
            "existing_candidates": candidate_before,
            "added_candidates": candidate_added,
            "existing_sessions": session_before,
            "added_sessions": session_added,
            "admitted_parent_events": 0,
        },
        "authority": {"event_truth": False, "training": False, "confirmation": False, "production": False},
        "invariants": [
            "All EgoWalk windows use model-blind uniform coverage selection.",
            "No raw recording, detector output, HFTF output, segmentation output, or event label was read.",
            "All imported EgoWalk windows remain NOT_EVALUABLE until independent RGB/geometry/counterexample review.",
        ],
    }
    write_json(root / "manifests" / "egowalk_registry_merge_receipt.json", merge_receipt)
    return merge_receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=r"F:\ba-data\hftf-d7-public-real")
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, sort_keys=True))
