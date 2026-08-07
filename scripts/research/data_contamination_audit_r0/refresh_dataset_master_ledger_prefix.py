"""Refresh one canonical artifacts.local prefix in an existing master ledger.

This preserves previously hashed sessions outside the requested prefix while
fully rediscovering, decoding, and hashing every asset inside it.  Global
duplicate groups, role conflicts, summaries, and rendered outputs are then
recomputed from the merged session set.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

try:
    from . import audit_dataset_master_ledger as ledger
except ModuleNotFoundError:  # Imported as scripts.refresh_dataset_master_ledger_prefix.
    from scripts.research.data_contamination_audit_r0 import audit_dataset_master_ledger as ledger


def discover_prefix(root_id: str, root: Path, prefix: Path) -> list[ledger.FileCandidate]:
    candidates: list[ledger.FileCandidate] = []
    for path in ledger.iter_asset_files(prefix):
        classification = ledger.classify_file(path)
        group_rel, video_stem, anchor = ledger.group_rel_for_file(path, root, classification)
        candidates.append(
            ledger.FileCandidate(
                path=path,
                root_id=root_id,
                root=root,
                rel_path=path.relative_to(root).as_posix(),
                classification=classification,
                group_rel=group_rel,
                video_base=video_stem,
                anchor=anchor,
            )
        )
    return candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ledger.DEFAULT_REPO_ROOT)
    parser.add_argument("--ledger", type=Path, default=Path("DATASET_MASTER_LEDGER.json"))
    parser.add_argument("--relative-prefix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    canonical_root = repo_root / "artifacts.local"
    prefix = Path(os.path.abspath(canonical_root / args.relative_prefix))
    try:
        relative_prefix = prefix.relative_to(canonical_root).as_posix().rstrip("/")
    except ValueError:
        raise SystemExit("relative prefix escapes canonical artifacts.local")
    if not prefix.is_dir():
        raise SystemExit(f"prefix is not a directory: {prefix}")

    ledger_path = args.ledger if args.ledger.is_absolute() else repo_root / args.ledger
    prior = json.loads(ledger_path.read_text(encoding="utf-8"))
    if prior.get("schema_version") != ledger.SCHEMA_VERSION:
        raise SystemExit("existing ledger schema mismatch")

    started = time.time()
    candidates = discover_prefix("checkout_artifacts.local", canonical_root, prefix)
    sessions = ledger.build_sessions(candidates)
    progress = {"files_profiled": 0, "bytes_profiled": 0, "sessions_done": 0}
    refreshed: list[dict] = []
    for session in sessions:
        refreshed.append(
            ledger.profile_session(
                session,
                ledger.discover_ffprobe(),
                do_hash=True,
                decode=True,
                progress=progress,
            )
        )
        progress["sessions_done"] += 1

    prefix_native = str(prefix).lower().rstrip("\\/")
    retained = [
        record
        for record in prior.get("sessions", [])
        if not str(record.get("session_root", "")).lower().startswith(prefix_native)
    ]
    records = sorted(retained + refreshed, key=lambda record: record["source_id"])
    duplicate_groups = ledger.add_global_duplicate_groups(records)
    conflicts = ledger.build_role_conflicts(records, duplicate_groups)
    generated_at = ledger.iso_now()

    prior["generated_at"] = generated_at
    prior["incremental_refresh"] = {
        "method": "full_hash_and_decode_of_prefix_with_global_derived_fields_recomputed",
        "relative_prefix": relative_prefix,
        "replaced_prior_session_count": len(prior.get("sessions", [])) - len(retained),
        "refreshed_session_count": len(refreshed),
        "refreshed_file_count": sum(record.get("file_count", 0) for record in refreshed),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    prior["duplicate_content_groups"] = duplicate_groups
    prior["role_conflicts"] = conflicts
    prior["sessions"] = records
    prior["summary"] = {
        "session_count": len(records),
        "media_session_count": sum(record.get("session_kind") == "media_session" for record in records),
        "manifest_only_count": sum(record.get("session_kind") == "manifest_only" for record in records),
        "file_count": sum(record.get("file_count", 0) for record in records),
        "file_size_bytes": sum(record.get("file_size_bytes", 0) or 0 for record in records),
        "corrupt_session_count": sum(bool(record.get("corrupt_frames")) for record in records),
        "duplicate_content_group_count": len(duplicate_groups),
        "role_conflict_count": len(conflicts),
        "elapsed_seconds": round(time.time() - started, 3),
    }

    roots = [(item["root_id"], Path(item["path"])) for item in prior.get("scan_roots", [])]
    gap_text = ledger.build_gap_document(records, duplicate_groups, conflicts, prior.get("root_stats", {}), roots, generated_at)
    conflict_text = ledger.build_conflict_document(conflicts, duplicate_groups, generated_at)
    ledger.write_outputs(args.output_dir.resolve(), prior, gap_text, conflict_text)
    print(
        f"[dataset-audit-prefix] complete prefix={relative_prefix} "
        f"sessions={len(refreshed)} files={sum(record.get('file_count', 0) for record in refreshed)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
