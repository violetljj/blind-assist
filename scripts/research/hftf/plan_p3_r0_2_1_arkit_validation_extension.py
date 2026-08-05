#!/usr/bin/env python3
"""Select a finite label-blind ARKitScenes validation reserve roster."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_p3_r0_2_1_arkit_validation_extension_protocol"
ROSTER_SCHEMA = "blindassist_p3_r0_2_1_arkit_validation_extension_roster"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), "JSON object required")
    return value


def bound_file(root: Path, binding: dict[str, Any]) -> Path:
    require(set(binding) == {"path", "sha256"}, "binding field drift")
    path = (root / binding["path"]).resolve()
    require(path.is_file(), f"bound file missing: {path}")
    require(sha256_file(path) == binding["sha256"], f"bound SHA mismatch: {path}")
    return path


def selection_rank(protocol_id: str, visit_id: str) -> str:
    return hashlib.sha256(f"{protocol_id}|validation_extension|{visit_id}".encode("utf-8")).hexdigest()


def metadata_visits(path: Path) -> tuple[dict[str, list[str]], set[str]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        require(set(reader.fieldnames or ()) == {"video_id", "visit_id", "fold"}, "metadata columns drift")
        for row in reader:
            visit = row["visit_id"].strip()
            if not visit or visit == "NA":
                continue
            grouped[visit][row["fold"].strip()].append(row["video_id"].strip())
    cross_fold = {visit for visit, folds in grouped.items() if len(folds) != 1}
    validation = {
        visit: sorted(folds["Validation"], key=lambda value: int(value))
        for visit, folds in grouped.items()
        if visit not in cross_fold and set(folds) == {"Validation"}
    }
    return validation, cross_fold


def plan(root: Path, protocol_path: Path, source_path: Path) -> dict[str, Any]:
    root = root.resolve()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "protocol schema drift")
    require(protocol["producer"]["sha256"] == sha256_file(source_path), "producer SHA drift")
    metadata_path = bound_file(root, protocol["sources"]["official_raw_split_csv"])
    scoped = load_json(bound_file(root, protocol["sources"]["existing_scoped_manifest"]))
    known = load_json(bound_file(root, protocol["sources"]["known_height_roster"]))
    bound_file(root, protocol["sources"]["invalid_role_freeze_result"])
    excluded = {str(video["visit_id"]) for video in scoped["videos"]}
    excluded.update(str(row["visit_id"]) for row in known["fresh_evaluation"])
    require(excluded == set(protocol["excluded_visit_ids"]), "global ARKit exclusion roster drift")
    visits, cross_fold = metadata_visits(metadata_path)
    require(set(protocol["cross_fold_visit_ids"]) == cross_fold, "cross-fold exclusion drift")
    candidates = []
    for visit, videos in visits.items():
        if visit in excluded:
            continue
        candidates.append({
            "visit_id": visit,
            "video_id": videos[0],
            "available_video_ids": videos,
            "official_fold": "Validation",
            "selection_rank_sha256": selection_rank(protocol["protocol_id"], visit),
        })
    candidates.sort(key=lambda row: (row["selection_rank_sha256"], row["visit_id"]))
    count = int(protocol["reserve_parent_count"])
    require(len(candidates) >= count, "candidate capacity insufficient")
    selected = candidates[:count]
    return {
        "schema": ROSTER_SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "metadata_sha256": protocol["sources"]["official_raw_split_csv"]["sha256"],
        "selection_method": "ascending SHA256(protocol_id|validation_extension|visit_id), then visit_id; lowest numeric video_id per visit",
        "excluded_visit_count": len(excluded),
        "eligible_validation_visit_count": len(candidates),
        "selected_parent_count": len(selected),
        "selected": selected,
        "rgb_or_metric_payload_read": False,
        "label_or_transition_fields_read": False,
        "model_outputs_read": False,
        "replacement_allowed": False,
        "terminal": "P3_R0_2_1_ARKIT_VALIDATION_EXTENSION_ROSTER_LOCKED_MEDIA_UNOPENED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"overwrite forbidden: {args.output}")
    value = plan(args.repo_root, args.protocol.resolve(), Path(__file__).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
