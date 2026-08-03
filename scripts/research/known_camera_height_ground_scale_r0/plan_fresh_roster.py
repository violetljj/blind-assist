"""Lock a metadata-only, parent-disjoint ARKitScenes mechanism cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def rank(protocol_id: str, visit_id: str) -> str:
    return hashlib.sha256(
        f"{protocol_id}|fresh_evaluation|{visit_id}".encode("utf-8")
    ).hexdigest()


def plan(
    metadata_path: Path,
    predecessor_roster_path: Path,
    protocol_path: Path,
    parent_count: int = 4,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "KNOWN_CAMERA_HEIGHT_GROUND_SCALE_R0":
        raise ValueError("unexpected protocol")
    if parent_count != int(
        protocol["fresh_evaluation_gates"]["minimum_parent_count"]
    ):
        raise ValueError("parent count must equal the frozen minimum")

    predecessor = json.loads(predecessor_roster_path.read_text(encoding="utf-8"))
    excluded = {
        str(row["visit_id"])
        for rows in predecessor["roles"].values()
        for row in rows
    }
    visits: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != {"video_id", "visit_id", "fold"}:
            raise ValueError("unexpected ARKitScenes metadata columns")
        for row in reader:
            visit_id = row["visit_id"].strip()
            if not visit_id or visit_id == "NA":
                continue
            visits[visit_id][row["fold"].strip()].append(row["video_id"].strip())

    cross_fold = sorted(visit for visit, folds in visits.items() if len(folds) != 1)
    candidates = []
    for visit_id, folds in visits.items():
        if visit_id in excluded or visit_id in cross_fold:
            continue
        if set(folds) != {"Validation"}:
            continue
        videos = sorted(set(folds["Validation"]), key=lambda value: (int(value), value))
        candidates.append(
            {
                "visit_id": visit_id,
                "video_id": videos[0],
                "available_video_ids": videos,
                "selection_rank_sha256": rank(protocol["protocol_id"], visit_id),
                "official_fold": "Validation",
                "role": "fresh_evaluation",
                "height_authority": "SOURCE_TRUTH_DERIVED_PER_FRAME_ORACLE",
                "claim_ceiling": "ORACLE_HEIGHT_MECHANISM_ONLY",
                "media_status": "UNOPENED_PENDING_ASSET_QUALIFICATION"
            }
        )
    selected = sorted(
        candidates, key=lambda row: (row["selection_rank_sha256"], row["visit_id"])
    )[:parent_count]
    if len(selected) != parent_count:
        raise ValueError("insufficient eligible fresh parents")
    if excluded & {str(row["visit_id"]) for row in selected}:
        raise AssertionError("predecessor parent overlap")
    return {
        "schema": "blindassist_known_camera_height_ground_scale_r0_metadata_roster",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
        "source": {
            "dataset": "ARKitScenes raw",
            "metadata_path": str(metadata_path.resolve()),
            "metadata_sha256": sha256(metadata_path),
            "repository_commit": predecessor["source"]["repository_commit"],
        },
        "predecessor_exclusion": {
            "roster_path": str(predecessor_roster_path.resolve()),
            "roster_sha256": sha256(predecessor_roster_path),
            "excluded_parent_count": len(excluded),
            "excluded_visit_ids": sorted(excluded),
        },
        "source_inventory": {
            "cross_official_fold_visits_excluded": cross_fold,
            "eligible_validation_visits_after_exclusion": len(candidates),
        },
        "fresh_evaluation": selected,
        "selected_parent_count": len(selected),
        "media_bytes_read": False,
        "metric_truth_opened": False,
        "outcomes_opened": False,
        "replacement_allowed": False,
        "wearable_height_confirmation": "NOT_EVALUABLE_NO_INDEPENDENT_FIXED_MOUNT_HEIGHT_RECEIPT",
        "terminal": "KNOWN_CAMERA_HEIGHT_GROUND_SCALE_R0_FRESH_ROSTER_LOCKED_MEDIA_UNOPENED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--predecessor-roster", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    result = plan(
        arguments.metadata,
        arguments.predecessor_roster,
        arguments.protocol,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(arguments.output), **result}, indent=2))


if __name__ == "__main__":
    main()
