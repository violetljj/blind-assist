#!/usr/bin/env python3
"""Create the metadata-only, parent-disjoint ARKitScenes R1 roster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from validate_protocol import DEFAULT_PROTOCOL, REPO_ROOT, sha256, validate

DEFAULT_SOURCE_ROOT = REPO_ROOT / "artifacts.local/downloads/ARKitScenes-7283761"
DEFAULT_OUTPUT = REPO_ROOT / "artifacts.local/evidence/hftf/spatial-calibration-head-r1-roster-20260804/roster.json"


def rank(protocol_id: str, role: str, visit_id: str) -> str:
    return hashlib.sha256(f"{protocol_id}|{role}|{visit_id}".encode()).hexdigest()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"video_id", "visit_id", "fold"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("ARKitScenes split CSV schema mismatch")
    return rows


def build_roster(rows: list[dict[str, str]], protocol: dict[str, Any]) -> dict[str, Any]:
    by_fold_visit: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    visit_folds: dict[str, set[str]] = defaultdict(set)
    na_rows = 0
    for row in rows:
        visit = row["visit_id"].strip()
        fold = row["fold"].strip()
        video = row["video_id"].strip()
        if not visit or visit == "NA":
            na_rows += 1
            continue
        visit_folds[visit].add(fold)
        by_fold_visit[fold][visit].append(video)
    cross_fold = sorted(visit for visit, folds in visit_folds.items() if len(folds) > 1)
    excluded = set(cross_fold)

    protocol_id = protocol["protocol_id"]
    selected_visits: set[str] = set()
    selections: dict[str, list[dict[str, Any]]] = {}
    role_specs = (("train", "Training", 16), ("validation", "Training", 4), ("sealed", "Validation", 4))
    for role, fold, quota in role_specs:
        candidates = [
            visit for visit in by_fold_visit[fold]
            if visit not in excluded and visit not in selected_visits
        ]
        ordered = sorted(candidates, key=lambda visit: (rank(protocol_id, role, visit), visit))
        if len(ordered) < quota:
            raise ValueError(f"insufficient metadata candidates for {role}")
        chosen = []
        for visit in ordered[:quota]:
            videos = sorted(set(by_fold_visit[fold][visit]))
            chosen.append({
                "role": role,
                "official_fold": fold,
                "visit_id": visit,
                "video_id": videos[0],
                "available_video_ids": videos,
                "selection_rank_sha256": rank(protocol_id, role, visit),
                "media_status": "UNOPENED_PENDING_ASSET_QUALIFICATION",
            })
            selected_visits.add(visit)
        selections[role] = chosen

    cv_order = sorted(
        selections["train"],
        key=lambda row: (rank(protocol_id, "cv", row["visit_id"]), row["visit_id"]),
    )
    for index, row in enumerate(cv_order):
        row["cv_fold"] = index // 4
    cv_by_visit = {row["visit_id"]: row["cv_fold"] for row in cv_order}
    for row in selections["train"]:
        row["cv_fold"] = cv_by_visit[row["visit_id"]]

    role_sets = {role: {row["visit_id"] for row in values} for role, values in selections.items()}
    overlap = sorted(
        (role_a, role_b, sorted(role_sets[role_a] & role_sets[role_b]))
        for index, role_a in enumerate(role_sets)
        for role_b in list(role_sets)[index + 1:]
        if role_sets[role_a] & role_sets[role_b]
    )
    return {
        "source_inventory": {
            "rows": len(rows),
            "na_visit_rows_excluded": na_rows,
            "cross_official_fold_visits_excluded": cross_fold,
            "official_fold_rows": {fold: sum(row["fold"] == fold for row in rows) for fold in sorted(by_fold_visit)},
            "eligible_unique_visits": {
                fold: len([visit for visit in visits if visit not in excluded])
                for fold, visits in sorted(by_fold_visit.items())
            },
        },
        "roles": selections,
        "role_visit_overlap": overlap,
        "selected_parent_count": len(selected_visits),
        "terminal": "SPATIAL_CALIBRATION_HEAD_R1_METADATA_ROSTER_24_LOCKED_MEDIA_UNOPENED",
    }


def write_new(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    errors = validate(protocol)
    if errors:
        raise ValueError(f"protocol invalid: {errors}")
    commit = subprocess.run(
        ["git", "-C", str(args.source_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if commit != protocol["data"]["repository_commit"]:
        raise ValueError("ARKitScenes repository commit mismatch")
    metadata = args.source_root / protocol["data"]["metadata_path"]
    if sha256(metadata) != protocol["data"]["metadata_sha256"]:
        raise ValueError("ARKitScenes metadata hash mismatch")
    result = build_roster(load_rows(metadata), protocol)
    if result["source_inventory"]["cross_official_fold_visits_excluded"] != ["381879"]:
        raise ValueError("pinned CSV cross-fold exclusion changed")
    if result["role_visit_overlap"] or result["selected_parent_count"] != 24:
        raise ValueError("roster is not 24-parent disjoint")
    result.update({
        "schema": "blindassist_spatial_calibration_head_r1_metadata_roster",
        "protocol_sha256": sha256(args.protocol),
        "source": {
            "repository_commit": commit,
            "metadata_path": str(metadata.resolve()),
            "metadata_sha256": sha256(metadata),
            "data_document_sha256": sha256(args.source_root / "DATA.md"),
            "license_sha256": sha256(args.source_root / "LICENSE"),
            "download_script_sha256": sha256(args.source_root / "download_data.py"),
        },
        "license_receipt": {
            "license_text_bound": True,
            "media_download_authorized": False,
            "status": "SOURCE_LICENSE_TEXT_BOUND_USER_REVIEW_REQUIRED_BEFORE_MEDIA_DOWNLOAD",
        },
        "media_bytes_read": False,
        "sealed_media_opened": False,
    })
    write_new(args.output, result)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "selected_parent_count": result["selected_parent_count"],
        "cross_official_fold_visits_excluded": result["source_inventory"]["cross_official_fold_visits_excluded"],
        "terminal": result["terminal"],
        "license_status": result["license_receipt"]["status"],
    }, indent=2))


if __name__ == "__main__":
    main()
