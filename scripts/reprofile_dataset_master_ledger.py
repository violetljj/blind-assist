#!/usr/bin/env python3
"""Refresh decode/profile evidence in an existing master ledger.

The initial full pass may have already paid the SHA-256/MD5 cost.  This helper
keeps that hash snapshot and re-runs only media/structured profiling with the
project's dependency-rich Python environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from audit_dataset_master_ledger import (
    FileCandidate,
    aggregate_timestamp_info,
    alignment_summary,
    build_conflict_document,
    build_gap_document,
    build_role_conflicts,
    discover_ffprobe,
    frame_key_summary,
    infer_roles,
    profile_file,
    read_roles_from_metadata,
    supported_questions,
    write_outputs,
)


EXCLUSIVE_ROLES = {"consumed", "burned", "fresh", "reserved"}


def iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def refresh_record(record: dict[str, Any], ffprobe: Path | None, progress: dict[str, int]) -> dict[str, Any]:
    ordered = sorted(record.get("files", []), key=lambda item: str(item.get("relative_path", "")).lower())
    damaged: list[str] = []
    mismatches: list[str] = []
    for file_record in ordered:
        path = Path(str(file_record.get("path", "")))
        classification = {
            "asset_kind": str(file_record.get("asset_kind", "structured")),
            "modality": str(file_record.get("modality", "metadata")),
        }
        try:
            current_size = path.stat().st_size
        except OSError as exc:
            profile = {"status": "not_readable", "error": f"{type(exc).__name__}:{exc}"}
        else:
            if file_record.get("bytes") is not None and current_size != int(file_record["bytes"]):
                mismatches.append(str(file_record.get("relative_path", path)))
            profile = profile_file(path, classification, ffprobe, decode=True)
        file_record["profile"] = profile
        file_record["integrity_status"] = profile.get("status", "not_evaluable")
        file_record["integrity_error"] = profile.get("error")
        if file_record["integrity_status"].startswith("corrupt") or file_record["integrity_status"] in {"not_readable", "empty"}:
            damaged.append(str(file_record.get("relative_path", path)))
        progress["files"] += 1
        if progress["files"] % 500 == 0:
            print(f"[dataset-audit] reprofile_files={progress['files']}", flush=True)

    dataset = str(record.get("dataset") or "Unknown")
    path_text = f"{record.get('scan_root_id', '')} {record.get('session_root', '')} " + " ".join(
        str(item.get("relative_path", "")) for item in ordered[:32]
    )
    path_roles, path_flags = infer_roles(path_text, dataset)
    metadata_roles, metadata_flags = read_roles_from_metadata(ordered)
    history_roles = set(record.get("history_roles", [])) | set(path_roles) | metadata_roles
    flags = {role: bool(record.get("role_flags", {}).get(f"is_{role}") is True) for role in EXCLUSIVE_ROLES}
    for role in set(record.get("history_roles", [])) & EXCLUSIVE_ROLES:
        flags[role] = True
    for role, value in path_flags.items():
        if value:
            flags[role] = True
    for role, value in metadata_flags.items():
        if value:
            flags[role] = True
            history_roles.add(role)
    for role, value in flags.items():
        if value:
            history_roles.add(role)

    modality_counts: Counter[str] = Counter(str(item.get("modality", "unknown")) for item in ordered)
    modality_set = set(modality for modality in modality_counts if modality not in {"metadata", "archive", "unknown"})
    fake_candidates = [
        FileCandidate(
            path=Path(str(item.get("path", ""))),
            root_id=str(record.get("scan_root_id", "")),
            root=Path("."),
            rel_path=str(item.get("relative_path", "")),
            classification={"asset_kind": str(item.get("asset_kind", "structured")), "modality": str(item.get("modality", "metadata"))},
            group_rel="",
            video_base=None,
            anchor=(),
        )
        for item in ordered
    ]
    temporal = aggregate_timestamp_info(ordered, fake_candidates)
    frame_summary = frame_key_summary(ordered)
    alignment = alignment_summary(ordered)
    declared_counts: dict[str, int] = {}
    for item in ordered:
        for key, value in (item.get("profile") or {}).get("declared_counts", {}).items():
            declared_counts[key] = max(declared_counts.get(key, 0), int(value))
    counts_frames: dict[str, int | None] = {}
    for modality in ("rgb", "mask", "depth", "pose"):
        values = [
            (item.get("profile") or {}).get("frame_count")
            for item in ordered
            if item.get("modality") == modality and (item.get("profile") or {}).get("frame_count")
        ]
        counts_frames[modality] = sum(int(value) for value in values) if values else None
    counts_files = {modality: modality_counts.get(modality, 0) for modality in ("rgb", "mask", "depth", "pose")}
    counts = {
        "files": counts_files,
        "frames_or_records": counts_frames,
        "rgb_count": counts_frames["rgb"] if counts_frames["rgb"] is not None else counts_files["rgb"],
        "mask_count": counts_frames["mask"] if counts_frames["mask"] is not None else counts_files["mask"],
        "depth_count": counts_frames["depth"] if counts_frames["depth"] is not None else counts_files["depth"],
        "pose_count": counts_frames["pose"] if counts_frames["pose"] is not None else counts_files["pose"],
    }
    hash_errors = list(record.get("hash_errors", []))
    record.update(
        {
            "session_kind": "media_session" if modality_set else "manifest_only",
            "media_types": sorted(set(str(item.get("asset_kind", "structured")) for item in ordered) | modality_set),
            "file_count": len(ordered),
            "file_size_bytes": sum(int(item.get("bytes") or 0) for item in ordered),
            "counts": counts,
            "timestamp_range": temporal["timestamp_range"],
            "fps": temporal["fps"],
            "fps_basis": temporal["fps_basis"],
            "resolution": temporal["resolution"],
            "video_frame_counts": temporal["video_frame_counts"],
            "missing_frames": frame_summary,
            "duplicate_frames": {
                modality: details["duplicate_frame_keys"]
                for modality, details in frame_summary.items()
                if details.get("duplicate_frame_keys")
            },
            "corrupt_frames": damaged,
            "hash_errors": hash_errors,
            "hash_snapshot_mismatches": mismatches,
            "decodability": {
                "status": "all_profiled_readable" if not damaged and not hash_errors else "partial_or_failed",
                "corrupt_or_unreadable_count": len(damaged),
                "hash_error_count": len(hash_errors),
                "non_evaluable_dependency_count": sum(
                    1 for item in ordered if str(item.get("integrity_status", "")).startswith("not_evaluable")
                ),
            },
            "rgb_mask_depth_pose_alignment": alignment,
            "history_roles": sorted(history_roles),
            "role_flags": {
                "is_consumed": True if flags["consumed"] else None,
                "is_burned": True if flags["burned"] else None,
                "is_fresh": True if flags["fresh"] else None,
                "is_reserved": True if flags["reserved"] else None,
                "unknown_means_no_local_evidence_not_proof_of_absence": True,
            },
            "metadata_declared_counts": declared_counts,
            "research_questions_supported": supported_questions(dataset, modality_set, history_roles, alignment),
            "files": ordered,
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    with args.ledger.open("r", encoding="utf-8") as handle:
        ledger = json.load(handle)
    ffprobe = discover_ffprobe()
    progress = {"files": 0}
    for record in ledger.get("sessions", []):
        refresh_record(record, ffprobe, progress)
    duplicate_groups = ledger.get("duplicate_content_groups", [])
    conflicts = build_role_conflicts(ledger.get("sessions", []), duplicate_groups)
    ledger["role_conflicts"] = conflicts
    ledger.setdefault("scan_policy", {})["decode_reprofiled_from_existing_hash_snapshot"] = True
    ledger["scan_policy"]["reprofile_runtime"] = sys.executable
    ledger["profile_refreshed_at"] = iso_now()
    sessions = ledger.get("sessions", [])
    ledger.setdefault("summary", {}).update(
        {
            "session_count": len(sessions),
            "media_session_count": sum(item.get("session_kind") == "media_session" for item in sessions),
            "manifest_only_count": sum(item.get("session_kind") == "manifest_only" for item in sessions),
            "file_count": sum(int(item.get("file_count") or 0) for item in sessions),
            "file_size_bytes": sum(int(item.get("file_size_bytes") or 0) for item in sessions),
            "corrupt_session_count": sum(bool(item.get("corrupt_frames")) for item in sessions),
            "duplicate_content_group_count": len(duplicate_groups),
            "role_conflict_count": len(conflicts),
            "reprofile_elapsed_seconds": round(time.time() - started, 3),
        }
    )
    roots = [(item["root_id"], Path(item["path"])) for item in ledger.get("scan_roots", [])]
    gaps = build_gap_document(sessions, duplicate_groups, conflicts, ledger.get("root_stats", {}), roots, ledger["generated_at"])
    conflict_text = build_conflict_document(conflicts, duplicate_groups, ledger["generated_at"])
    write_outputs(args.output_dir.resolve(), ledger, gaps, conflict_text)
    print(
        f"[dataset-audit] reprofile_complete sessions={len(sessions)} files={progress['files']} "
        f"corrupt_sessions={ledger['summary']['corrupt_session_count']} elapsed={ledger['summary']['reprofile_elapsed_seconds']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
