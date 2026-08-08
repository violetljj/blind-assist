#!/usr/bin/env python3
"""Freeze an ARKitScenes metadata-only R2 roster without opening media."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "blindassist_depthart_task_preserving_r2_arkit_roster_v1"
ID_PATTERN = re.compile(r"(?<!\d)\d{6,8}(?!\d)")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _tracked_exclusions(repo: Path, commit: str, known_ids: set[str]) -> tuple[set[str], int]:
    names = [
        name for name in _git(
            repo, "ls-tree", "-r", "--name-only", commit, "--", "docs/research/hftf"
        ).splitlines()
        if name.endswith((".json", ".md"))
    ]
    excluded: set[str] = set()
    for name in names:
        text = _git(repo, "show", f"{commit}:{name}")
        excluded.update(value for value in ID_PATTERN.findall(text) if value in known_ids)
    return excluded, len(names)


def select(rows: Iterable[dict[str, str]], excluded: set[str], count: int) -> list[dict[str, str]]:
    eligible = [
        row for row in rows
        if row["fold"] == "Validation"
        and row["visit_id"] != "NA"
        and row["visit_id"] not in excluded
        and row["video_id"] not in excluded
    ]
    eligible.sort(
        key=lambda row: hashlib.sha256(
            f"{row['visit_id']}:{row['video_id']}".encode("ascii")
        ).hexdigest()
    )
    selected: list[dict[str, str]] = []
    visits: set[str] = set()
    for row in eligible:
        if row["visit_id"] in visits:
            continue
        selected.append({"visit_id": row["visit_id"], "video_id": row["video_id"], "fold": row["fold"]})
        visits.add(row["visit_id"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} unique eligible visits for requested {count}")
    return selected


def plan(metadata: Path, repo: Path, exclusion_commit: str, count: int) -> dict[str, Any]:
    with metadata.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"video_id", "visit_id", "fold"}:
        raise ValueError("unexpected ARKitScenes split CSV schema")
    known_ids = {
        value for row in rows for value in (row["video_id"], row["visit_id"]) if value != "NA"
    }
    excluded, scanned_files = _tracked_exclusions(repo, exclusion_commit, known_ids)
    selected = select(rows, excluded, count)
    return {
        "schema": SCHEMA,
        "protocol_id": "DEPTHART_TASK_PRESERVING_DEPLOYMENT_R2",
        "status": f"METADATA_ROSTER_{count}_LOCKED_MEDIA_UNOPENED",
        "source": {
            "repository": "https://github.com/apple/ARKitScenes.git",
            "repository_commit": "7283761bf26c27570ec59a5dc0f8686fbff07726",
            "metadata_path": "threedod/3dod_train_val_splits.csv",
            "metadata_bytes": metadata.stat().st_size,
            "metadata_sha256": _sha256(metadata),
            "official_rows": len(rows),
        },
        "exclusion_snapshot": {
            "repository_commit": exclusion_commit,
            "scope": "all tracked .json/.md under docs/research/hftf",
            "scanned_file_count": scanned_files,
            "matched_official_identity_count": len(excluded),
            "matched_official_identities": sorted(excluded),
        },
        "selection": {
            "rule": "validation rows, no tracked HFTF identity match, unique visit, ascending sha256(visit_id:video_id)",
            "requested_parent_count": count,
            "selected": selected,
        },
        "invariants": {
            "parent_count": len({row["visit_id"] for row in selected}),
            "session_count": len({row["video_id"] for row in selected}),
            "parent_session_overlap_with_exclusion_snapshot": 0,
            "media_body_bytes_read": False,
            "depth_or_rgb_opened": False,
            "model_outputs_read": False,
            "outcome_access": "NONE",
            "replacement_allowed": False,
        },
        "next_gate": "download only the locked RGB/depth/intrinsics/confidence assets, then run label-blind integrity and task-support coverage preflight",
        "authority": "Metadata roster only; not SEALED_UNSEEN activation, task quality, performance, DA2 replacement, Android default, production or safety authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--exclusion-commit", required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = plan(args.metadata, args.repo, args.exclusion_commit, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
