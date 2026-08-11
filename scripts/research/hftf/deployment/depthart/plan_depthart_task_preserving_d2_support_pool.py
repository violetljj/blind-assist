#!/usr/bin/env python3
"""Freeze a metadata-only ARKitScenes pool for D2 source-support admission."""

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


SCHEMA = "blindassist_depthart_task_preserving_d2_source_support_pool_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D2_TASK_EVIDENCE_HEAD"
SOURCE_COMMIT = "7283761bf26c27570ec59a5dc0f8686fbff07726"
ID_PATTERN = re.compile(r"(?<!\d)\d{6,8}(?!\d)")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def metadata_blob(metadata: Path) -> bytes:
    repository = Path(_git(metadata.parent, "rev-parse", "--show-toplevel").strip())
    relative = metadata.resolve().relative_to(repository.resolve()).as_posix()
    result = subprocess.run(
        ["git", "-C", str(repository), "show", f"{SOURCE_COMMIT}:{relative}"],
        check=True,
        capture_output=True,
    )
    return result.stdout


def tracked_exclusions(repo: Path, commit: str, known_ids: set[str]) -> tuple[set[str], int]:
    names = [
        name
        for name in _git(repo, "ls-tree", "-r", "--name-only", commit, "--", "docs/research").splitlines()
        if name.endswith((".json", ".md"))
    ]
    grep = subprocess.run(
        ["git", "-C", str(repo), "grep", "-h", "-E", r"[0-9]{6,8}", commit, "--", "docs/research"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if grep.returncode not in (0, 1):
        raise subprocess.CalledProcessError(grep.returncode, grep.args, grep.stdout, grep.stderr)
    excluded = {value for value in ID_PATTERN.findall(grep.stdout) if value in known_ids}
    return excluded, len(names)


def select(rows: Iterable[dict[str, str]], excluded: set[str], pool_count: int) -> list[dict[str, str]]:
    if pool_count <= 0:
        raise ValueError("pool_count must be positive")
    eligible = [
        row
        for row in rows
        if row["fold"] == "Training"
        and row["visit_id"] != "NA"
        and row["visit_id"] not in excluded
        and row["video_id"] not in excluded
    ]
    eligible.sort(
        key=lambda row: hashlib.sha256(f"{row['visit_id']}:{row['video_id']}".encode("ascii")).hexdigest()
    )
    selected: list[dict[str, str]] = []
    visits: set[str] = set()
    for row in eligible:
        if row["visit_id"] in visits:
            continue
        selected.append(
            {
                "pool_order": len(selected) + 1,
                "visit_id": row["visit_id"],
                "video_id": row["video_id"],
                "fold": row["fold"],
                "role": "D2_SOURCE_SUPPORT_POOL_ONLY",
            }
        )
        visits.add(row["visit_id"])
        if len(selected) == pool_count:
            break
    if len(selected) != pool_count:
        raise ValueError(f"only {len(selected)} unique eligible visits for requested {pool_count}")
    return selected


def plan(metadata: Path, repo: Path, exclusion_commit: str, pool_count: int) -> dict[str, Any]:
    blob = metadata_blob(metadata)
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8"))))
    if not rows or set(rows[0]) != {"video_id", "visit_id", "fold"}:
        raise ValueError("unexpected ARKitScenes split CSV schema")
    known_ids = {
        value for row in rows for value in (row["video_id"], row["visit_id"]) if value != "NA"
    }
    excluded, scanned_files = tracked_exclusions(repo, exclusion_commit, known_ids)
    selected = select(rows, excluded, pool_count)
    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "METADATA_SUPPORT_POOL_LOCKED_MEDIA_UNOPENED",
        "source": {
            "repository": "https://github.com/apple/ARKitScenes.git",
            "repository_commit": SOURCE_COMMIT,
            "metadata_path": "threedod/3dod_train_val_splits.csv",
            "metadata_bytes": len(blob),
            "metadata_sha256": hashlib.sha256(blob).hexdigest().upper(),
            "official_rows": len(rows),
        },
        "exclusion_snapshot": {
            "repository_commit": exclusion_commit,
            "scope": "all tracked JSON/Markdown under docs/research",
            "scanned_file_count": scanned_files,
            "matched_official_identity_count": len(excluded),
            "matched_official_identities": sorted(excluded),
        },
        "selection": {
            "rule": (
                "Training rows; exclude every official visit_id/video_id named in the frozen "
                "docs/research snapshot; require unique visit; sort by sha256(visit_id:video_id); "
                "take the first pool_count without replacement"
            ),
            "pool": selected,
        },
        "staged_admission": {
            "phase_a": "trajectory-only portrait/pose continuity; no RGB/depth/confidence body",
            "phase_b": "geometry-only depth/confidence/intrinsics truth-support audit; no model output",
            "phase_c": "RGB only for the first eight support-qualified identities",
            "role_assignment": "first four qualified by pool order become D2 TRAIN; next four become D2 DEVELOPMENT",
            "minimum_qualified": 8,
        },
        "invariants": {
            "pool_count": len(selected),
            "unique_parent_count": len({row["visit_id"] for row in selected}),
            "media_body_bytes_read": False,
            "truth_read": False,
            "model_outputs_read": False,
            "outcome_access": "NONE",
            "r2_cohort_access": "NONE",
            "download_authorized": False,
        },
        "next_gate": "bind an exact D2 source-use receipt before any HEAD or media request",
        "authority": "Metadata-only D2 source-support pool; no training, task outcome, R2, device, performance, production or safety authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--exclusion-commit", required=True)
    parser.add_argument("--pool-count", type=int, default=32)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = plan(args.metadata, args.repo, args.exclusion_commit, args.pool_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
