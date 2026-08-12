#!/usr/bin/env python3
"""Freeze the metadata-only, parent-disjoint ARKitScenes pool for DepthART D3."""

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


SCHEMA = "blindassist_depthart_task_preserving_d3_fresh_metadata_roster_v1"
PROTOCOL_ID = "DEPTHART_TASK_PRESERVING_D3_BIDIRECTIONAL_ERROR_CERTIFICATE_ROUTER"
SOURCE_COMMIT = "7283761bf26c27570ec59a5dc0f8686fbff07726"
ID_PATTERN = re.compile(r"(?<!\d)\d{6,8}(?!\d)")
ANCHOR_ROSTERS = {
    "d1": "docs/research/hftf/DEPTHART_TASK_PRESERVING_D1_ARKIT_DEVELOPMENT_ROSTER_LOCK_2026-08-09.json",
    "d2": "docs/research/hftf/DEPTHART_TASK_PRESERVING_D2_SOURCE_SUPPORT_POOL_LOCK_2026-08-11.json",
    "r2": "docs/research/hftf/DEPTHART_TASK_PRESERVING_R2_ARKIT_ROSTER_LOCK_2026-08-09.json",
}


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


def _official_ids(data: bytes, known_ids: set[str]) -> list[str]:
    text = data.decode("utf-8", errors="ignore")
    return sorted({value for value in ID_PATTERN.findall(text) if value in known_ids})


def scan_workspace_exclusions(repo: Path, known_ids: set[str]) -> dict[str, Any]:
    root = repo / "docs" / "research"
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".md"}
    )
    receipts: list[dict[str, Any]] = []
    excluded: set[str] = set()
    for path in paths:
        data = path.read_bytes()
        matches = _official_ids(data, known_ids)
        if not matches:
            continue
        excluded.update(matches)
        receipts.append(
            {
                "path": path.relative_to(repo).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest().upper(),
                "matched_official_identities": matches,
            }
        )
    manifest_bytes = json.dumps(
        receipts, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "scope": "current workspace docs/research JSON and Markdown, including untracked route receipts",
        "scanned_file_count": len(paths),
        "matched_file_count": len(receipts),
        "matched_official_identity_count": len(excluded),
        "matched_official_identities": sorted(excluded),
        "file_receipts": receipts,
        "file_receipts_sha256": hashlib.sha256(manifest_bytes).hexdigest().upper(),
    }


def select(
    rows: Iterable[dict[str, str]], excluded: set[str], pool_count: int
) -> list[dict[str, Any]]:
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
        key=lambda row: hashlib.sha256(
            f"{row['visit_id']}:{row['video_id']}".encode("ascii")
        ).hexdigest()
    )
    selected: list[dict[str, Any]] = []
    visits: set[str] = set()
    for row in eligible:
        if row["visit_id"] in visits:
            continue
        selection_digest = hashlib.sha256(
            f"{row['visit_id']}:{row['video_id']}".encode("ascii")
        ).hexdigest().upper()
        selected.append(
            {
                "pool_order": len(selected) + 1,
                "visit_id": row["visit_id"],
                "video_id": row["video_id"],
                "fold": row["fold"],
                "role": "D3_METADATA_CANDIDATE_POOL_ONLY",
                "selection_sha256": selection_digest,
            }
        )
        visits.add(row["visit_id"])
        if len(selected) == pool_count:
            break
    if len(selected) != pool_count:
        raise ValueError(f"only {len(selected)} unique eligible visits for requested {pool_count}")
    return selected


def plan(metadata: Path, repo: Path, pool_count: int) -> dict[str, Any]:
    blob = metadata_blob(metadata)
    rows = list(csv.DictReader(io.StringIO(blob.decode("utf-8"))))
    if not rows or set(rows[0]) != {"video_id", "visit_id", "fold"}:
        raise ValueError("unexpected ARKitScenes split CSV schema")
    known_ids = {
        value for row in rows for value in (row["video_id"], row["visit_id"]) if value != "NA"
    }
    snapshot = scan_workspace_exclusions(repo, known_ids)
    repeated_snapshot = scan_workspace_exclusions(repo, known_ids)
    if snapshot["file_receipts_sha256"] != repeated_snapshot["file_receipts_sha256"]:
        raise RuntimeError("research exclusion snapshot changed during planning")
    excluded = set(snapshot["matched_official_identities"])
    selected = select(rows, excluded, pool_count)

    anchor_firewall: dict[str, Any] = {}
    selected_ids = {
        value for row in selected for value in (row["visit_id"], row["video_id"])
    }
    for name, relative in ANCHOR_ROSTERS.items():
        path = repo / relative
        data = path.read_bytes()
        anchor_ids = set(_official_ids(data, known_ids))
        anchor_firewall[name] = {
            "path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest().upper(),
            "official_identity_count": len(anchor_ids),
            "selection_overlap_count": len(selected_ids & anchor_ids),
        }
        if selected_ids & anchor_ids:
            raise ValueError(f"D3 selection overlaps frozen {name} identities")

    return {
        "schema": SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "status": "D3_FRESH_METADATA_POOL_48_LOCKED_MEDIA_UNOPENED",
        "source": {
            "repository": "https://github.com/apple/ARKitScenes.git",
            "repository_commit": SOURCE_COMMIT,
            "metadata_path": "threedod/3dod_train_val_splits.csv",
            "metadata_bytes": len(blob),
            "metadata_sha256": hashlib.sha256(blob).hexdigest().upper(),
            "official_rows": len(rows),
        },
        "workspace_snapshot": {
            "repository_head": _git(repo, "rev-parse", "HEAD").strip(),
            **snapshot,
        },
        "anchor_firewall": anchor_firewall,
        "selection": {
            "rule": (
                "official Training rows; exclude every official visit_id/video_id named in the "
                "byte-frozen current docs/research snapshot; require unique visit; sort by "
                "sha256(visit_id:video_id); take the first 48 without replacement"
            ),
            "pool": selected,
        },
        "future_stages_not_activated": {
            "phase_a": "take the first 32 label-blind portrait/pose-qualified identities",
            "phase_b": "take the first 16 source-truth-support-qualified identities",
            "role_assignment": "first 8 qualified become D3 TRAIN; next 8 become sealed D3 DEVELOPMENT",
            "frames_per_identity": 300,
        },
        "invariants": {
            "pool_count": len(selected),
            "unique_parent_count": len({row["visit_id"] for row in selected}),
            "unique_session_count": len({row["video_id"] for row in selected}),
            "selection_overlap_with_workspace_snapshot": len(selected_ids & excluded),
            "media_head_requests": 0,
            "media_body_bytes_read": 0,
            "truth_read": False,
            "model_outputs_read": False,
            "training": False,
            "development_outcome_access": "NONE",
            "r2_cohort_access": "NONE",
            "download_authorized": False,
        },
        "next_gate": "EXPLICIT_D3_PHASE_A_INTRINSICS_TRAJECTORY_HEAD_ONLY_PREFLIGHT_ACTIVATION",
        "authority": "Metadata-only D3 source pool and identity firewall; no media, truth, training, Development, R2, performance, production or safety authority.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--pool-count", type=int, default=48)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = plan(args.metadata, args.repo, args.pool_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(result, indent=2) + "\n").encode("utf-8")
    args.output.write_bytes(encoded)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pool_count": result["invariants"]["pool_count"],
                "excluded_identity_count": result["workspace_snapshot"][
                    "matched_official_identity_count"
                ],
                "sha256": hashlib.sha256(encoded).hexdigest().upper(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
