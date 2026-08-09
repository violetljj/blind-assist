#!/usr/bin/env python3
"""Freeze a metadata-only ARKitScenes roster for the D1 Development screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "blindassist_depthart_task_preserving_d1_arkit_development_roster_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D1_FIXED_MIXED_DEVELOPMENT_SCREEN"
ID_PATTERN = re.compile(r"(?<!\d)\d{6,8}(?!\d)")
EXCLUSION_SCOPES = (
    "docs/research/hftf",
    "docs/research/assistive-geometry",
)


SOURCE_COMMIT = "7283761bf26c27570ec59a5dc0f8686fbff07726"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def _metadata_blob(metadata: Path) -> bytes:
    repository = Path(_git(metadata.parent, "rev-parse", "--show-toplevel").strip())
    relative = metadata.resolve().relative_to(repository.resolve()).as_posix()
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{SOURCE_COMMIT}:{relative}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def _tracked_exclusions(
    repo: Path, commit: str, known_ids: set[str]
) -> tuple[set[str], int]:
    names = [
        name
        for name in _git(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            *EXCLUSION_SCOPES,
        ).splitlines()
        if name.endswith((".json", ".md"))
    ]
    excluded: set[str] = set()
    for name in names:
        text = _git(repo, "show", f"{commit}:{name}")
        excluded.update(value for value in ID_PATTERN.findall(text) if value in known_ids)
    return excluded, len(names)


def select(
    rows: Iterable[dict[str, str]],
    excluded: set[str],
    primary_count: int,
    reserve_count: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if primary_count <= 0 or reserve_count < 0:
        raise ValueError("primary_count must be positive and reserve_count non-negative")
    eligible = [
        row
        for row in rows
        if row["fold"] == "Training"
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
    required = primary_count + reserve_count
    for row in eligible:
        if row["visit_id"] in visits:
            continue
        selected.append(
            {
                "visit_id": row["visit_id"],
                "video_id": row["video_id"],
                "fold": row["fold"],
            }
        )
        visits.add(row["visit_id"])
        if len(selected) == required:
            break
    if len(selected) != required:
        raise ValueError(
            f"only {len(selected)} unique eligible visits for requested {required}"
        )
    return selected[:primary_count], selected[primary_count:]


def plan(
    metadata: Path,
    repo: Path,
    exclusion_commit: str,
    primary_count: int,
    reserve_count: int,
) -> dict[str, Any]:
    metadata_blob = _metadata_blob(metadata)
    rows = list(csv.DictReader(io.StringIO(metadata_blob.decode("utf-8"))))
    if not rows or set(rows[0]) != {"video_id", "visit_id", "fold"}:
        raise ValueError("unexpected ARKitScenes split CSV schema")
    known_ids = {
        value
        for row in rows
        for value in (row["video_id"], row["visit_id"])
        if value != "NA"
    }
    excluded, scanned_files = _tracked_exclusions(repo, exclusion_commit, known_ids)
    primary, reserve = select(rows, excluded, primary_count, reserve_count)
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "METADATA_ROSTER_LOCKED_MEDIA_UNOPENED",
        "source": {
            "repository": "https://github.com/apple/ARKitScenes.git",
            "repository_commit": SOURCE_COMMIT,
            "metadata_path": "threedod/3dod_train_val_splits.csv",
            "metadata_bytes": len(metadata_blob),
            "metadata_sha256": hashlib.sha256(metadata_blob).hexdigest().upper(),
            "working_tree_line_endings_ignored": True,
            "official_rows": len(rows),
        },
        "exclusion_snapshot": {
            "repository_commit": exclusion_commit,
            "scopes": list(EXCLUSION_SCOPES),
            "scanned_file_count": scanned_files,
            "matched_official_identity_count": len(excluded),
            "matched_official_identities": sorted(excluded),
        },
        "selection": {
            "rule": (
                "Training rows; exclude every official visit_id/video_id named in the "
                "frozen HFTF and Assistive Geometry snapshots; require unique visit; "
                "sort by sha256(visit_id:video_id); take primary then reserve without replacement"
            ),
            "primary": primary,
            "reserve": reserve,
            "reserve_use_rule": (
                "after label-blind media preflight, replace an ineligible primary with the "
                "first eligible reserve in frozen order; model/truth outcome cannot affect replacement"
            ),
        },
        "invariants": {
            "primary_parent_count": len({row["visit_id"] for row in primary}),
            "primary_session_count": len({row["video_id"] for row in primary}),
            "reserve_parent_count": len({row["visit_id"] for row in reserve}),
            "reserve_session_count": len({row["video_id"] for row in reserve}),
            "parent_session_overlap_with_exclusion_snapshot": 0,
            "media_body_bytes_read": False,
            "depth_or_rgb_opened": False,
            "model_outputs_read": False,
            "outcome_access": "NONE",
            "download_authorized": False,
            "replacement_by_outcome_allowed": False,
        },
        "next_gate": (
            "extend the reviewed ARKitScenes use scope to the locked identities, then run "
            "label-blind portrait/pose/RGB-D continuity preflight"
        ),
        "authority": (
            "Metadata-only Development roster lock. No media, truth/model outcome, candidate "
            "selection, R2 activation, performance, Android default, production or safety authority."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--exclusion-commit", required=True)
    parser.add_argument("--primary-count", type=int, default=8)
    parser.add_argument("--reserve-count", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = plan(
        args.metadata,
        args.repo,
        args.exclusion_commit,
        args.primary_count,
        args.reserve_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
