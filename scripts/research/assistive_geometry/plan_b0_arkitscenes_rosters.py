#!/usr/bin/env python3
"""Freeze disjoint metadata-only ARKitScenes rosters for Assistive Geometry B0."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "blindassist_assistive_geometry_b0_arkitscenes_rosters_v1"
ID_PATTERN = re.compile(r"(?<!\d)\d{6,8}(?!\d)")
ROLE_SPECS = (
    ("TRAIN", "Training", 16),
    ("DEVELOPMENT", "Training", 8),
    ("CONFIRMATION", "Validation", 8),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout


def tracked_exclusions(
    repo: Path,
    commit: str,
    known_ids: set[str],
) -> tuple[set[str], int]:
    names = [
        name
        for name in git(
            repo,
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            "docs/research/hftf",
            "docs/research/assistive-geometry",
        ).splitlines()
        if name.endswith((".json", ".md"))
    ]
    excluded: set[str] = set()
    for name in names:
        text = git(repo, "show", f"{commit}:{name}")
        excluded.update(value for value in ID_PATTERN.findall(text) if value in known_ids)
    return excluded, len(names)


def _rank(role: str, row: dict[str, str]) -> str:
    value = f"ASSISTIVE_GEOMETRY_B0:{role}:{row['visit_id']}:{row['video_id']}"
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def select_rosters(
    rows: Iterable[dict[str, str]],
    excluded_ids: set[str],
    role_specs: tuple[tuple[str, str, int], ...] = ROLE_SPECS,
) -> dict[str, list[dict[str, str]]]:
    source_rows = list(rows)
    used_visits: set[str] = set()
    used_videos: set[str] = set()
    selected: dict[str, list[dict[str, str]]] = {}
    for role, official_fold, count in role_specs:
        eligible = [
            row
            for row in source_rows
            if row["fold"] == official_fold
            and row["visit_id"] != "NA"
            and row["visit_id"] not in excluded_ids
            and row["video_id"] not in excluded_ids
            and row["visit_id"] not in used_visits
            and row["video_id"] not in used_videos
        ]
        eligible.sort(key=lambda row: _rank(role, row))
        role_rows: list[dict[str, str]] = []
        role_visits: set[str] = set()
        for row in eligible:
            if row["visit_id"] in role_visits:
                continue
            role_rows.append(
                {
                    "visit_id": row["visit_id"],
                    "video_id": row["video_id"],
                    "official_fold": row["fold"],
                    "selection_rank_sha256": _rank(role, row).upper(),
                }
            )
            role_visits.add(row["visit_id"])
            if len(role_rows) == count:
                break
        if len(role_rows) != count:
            raise ValueError(
                f"{role}: only {len(role_rows)} unique eligible visits for requested {count}"
            )
        selected[role] = role_rows
        used_visits.update(row["visit_id"] for row in role_rows)
        used_videos.update(row["video_id"] for row in role_rows)
    return selected


def plan(
    metadata: Path,
    repo: Path,
    exclusion_commit: str,
    additional_excluded_ids: set[str] | None = None,
) -> dict[str, Any]:
    with metadata.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"video_id", "visit_id", "fold"}:
        raise ValueError("unexpected ARKitScenes split CSV schema")
    known_ids = {
        value
        for row in rows
        for value in (row["video_id"], row["visit_id"])
        if value != "NA"
    }
    excluded_ids, scanned_files = tracked_exclusions(repo, exclusion_commit, known_ids)
    explicit_exclusions = set(additional_excluded_ids or set())
    unknown_exclusions = explicit_exclusions - known_ids
    if unknown_exclusions:
        raise ValueError(f"additional exclusion IDs not found in metadata: {sorted(unknown_exclusions)}")
    excluded_ids.update(explicit_exclusions)
    roles = select_rosters(rows, excluded_ids)
    all_rows = [row for role_rows in roles.values() for row in role_rows]
    visits = [row["visit_id"] for row in all_rows]
    videos = [row["video_id"] for row in all_rows]
    overlap = sorted((set(visits) | set(videos)) & excluded_ids)
    return {
        "schema": SCHEMA,
        "protocol_id": "BLINDASSIST_ASSISTIVE_GEOMETRY_B0",
        "status": "METADATA_ROSTERS_16_8_8_LOCKED_MEDIA_UNOPENED_LICENSE_SCOPE_REQUIRED",
        "source": {
            "repository": "https://github.com/apple/ARKitScenes.git",
            "repository_commit": "7283761bf26c27570ec59a5dc0f8686fbff07726",
            "metadata_path": "threedod/3dod_train_val_splits.csv",
            "local_metadata_bytes": metadata.stat().st_size,
            "local_metadata_sha256": sha256_file(metadata),
            "official_rows": len(rows),
        },
        "exclusion_snapshot": {
            "repository_commit": exclusion_commit,
            "scope": [
                "all tracked .json/.md under docs/research/hftf",
                "all tracked .json/.md under docs/research/assistive-geometry",
            ],
            "scanned_file_count": scanned_files,
            "matched_official_identity_count": len(excluded_ids),
            "matched_official_identities": sorted(excluded_ids),
            "additional_source_availability_exclusions": sorted(explicit_exclusions),
        },
        "selection_rule": {
            "TRAIN": "official Training; unique visit; first 16 by salted SHA-256 rank",
            "DEVELOPMENT": "official Training; disjoint from TRAIN; unique visit; first 8 by salted SHA-256 rank",
            "CONFIRMATION": "official Validation; disjoint from all earlier roles; unique visit; first 8 by salted SHA-256 rank",
            "replacement_allowed": False,
            "role_reassignment_allowed": False,
            "budget_expansion_allowed": False,
        },
        "roles": roles,
        "invariants": {
            "selected_session_count": len(all_rows),
            "selected_parent_count": len(set(visits)),
            "unique_video_count": len(set(videos)),
            "role_visit_overlap": len(visits) != len(set(visits)),
            "role_video_overlap": len(videos) != len(set(videos)),
            "overlap_with_exclusion_snapshot": overlap,
            "media_body_bytes_read": False,
            "depth_or_rgb_opened": False,
            "model_outputs_read": False,
            "outcome_access": "NONE",
        },
        "required_assets_per_video": [
            "lowres_wide.zip",
            "lowres_depth.zip",
            "confidence.zip",
            "lowres_wide_intrinsics.zip",
            "lowres_wide.traj",
        ],
        "next_gate": "explicit user extension of the reviewed ARKitScenes license scope to this exact B0 16/8/8 roster and use; then label-blind HEAD/integrity preflight",
        "authority": "Metadata identity lock only. No media download, training, task quality, confirmation opening, deployment, product, production, or safety authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--exclusion-commit", required=True)
    parser.add_argument("--additional-exclusion-id", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = plan(
        args.metadata,
        args.repo,
        args.exclusion_commit,
        set(args.additional_exclusion_id),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
