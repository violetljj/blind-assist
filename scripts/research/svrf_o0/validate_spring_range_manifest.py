#!/usr/bin/env python3
"""Validate the outcome-blind Spring locked-parent central-directory manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


EXPECTED_COUNTS = {
    "train_cam_data.zip": {"0005": 3, "0014": 3, "0016": 3, "0018": 3, "0047": 3},
    "train_frame_left.zip": {"0005": 116, "0014": 158, "0016": 282, "0018": 111, "0047": 268},
    "train_disp1_left.zip": {"0005": 116, "0014": 158, "0016": 282, "0018": 111, "0047": 268},
    "train_flow_FW_left.zip": {"0005": 115, "0014": 157, "0016": 281, "0018": 110, "0047": 267},
    "train_maps.zip": {"0005": 3023, "0014": 4115, "0016": 7339, "0018": 2893, "0047": 6975},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(lock_path: Path, manifest_path: Path, receipt_path: Path) -> None:
    lock_bytes = lock_path.read_bytes()
    lock = json.loads(lock_bytes)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema") == "blindassist.svrf_o0.spring_range_manifest_receipt.v1", "Spring receipt schema drift")
    require(receipt.get("status") == "SPRING_LOCKED_PARENT_RANGE_MANIFEST_COMPLETE", "Spring receipt terminal drift")
    require(receipt.get("source_lock_sha256") == hashlib.sha256(lock_bytes).hexdigest(), "Spring source-lock drift")
    require(receipt.get("manifest_sha256") == sha256(manifest_path), "Spring manifest hash drift")
    require(receipt.get("payload_materialized_count") == 0, "Spring payload was materialized")
    require(receipt.get("media_tensor_or_label_decode_count") == 0, "Spring content semantics were decoded")
    require(receipt.get("candidate_run_count") == 0 and receipt.get("outcome_accessed") is False, "Spring opened candidate outcome")

    source = next(item for item in lock["sources"] if item["source_id"] == "SPRING_V2")
    bindings = source["archive_bindings"]
    locked_parents = {parent["parent_id"] for parent in source["parents"]}
    require(set(bindings) == set(EXPECTED_COUNTS), "Spring archive roster drift")
    require(locked_parents == set(next(iter(EXPECTED_COUNTS.values()))), "Spring parent roster drift")

    counts: Counter[tuple[str, str]] = Counter()
    rows = 0
    with manifest_path.open(encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            row = json.loads(line)
            archive = row.get("source_archive")
            parent = row.get("parent_id")
            require(row.get("schema") == "blindassist.svrf_o0.materialized_member.v1", f"row {line_number}: schema drift")
            require(row.get("source_id") == "SPRING_V2", f"row {line_number}: source drift")
            require(archive in bindings, f"row {line_number}: archive outside lock")
            require(parent in locked_parents, f"row {line_number}: parent outside lock")
            require(row.get("datafile_id") == bindings[archive]["datafile_id"], f"row {line_number}: datafile drift")
            require(row.get("archive_member_path", "").startswith(f"spring/train/{parent}/"), f"row {line_number}: member-parent drift")
            require(row.get("payload_materialized") is False, f"row {line_number}: payload materialized")
            require(row.get("semantic_content_inspected") is False, f"row {line_number}: semantic inspection opened")
            candidate = row.get("candidate_visible")
            truth = row.get("truth_visible")
            require(isinstance(candidate, bool) and isinstance(truth, bool) and candidate != truth, f"row {line_number}: visibility is not role-disjoint")
            require(candidate is (archive == "train_frame_left.zip"), f"row {line_number}: candidate visibility drift")
            require(truth is (archive != "train_frame_left.zip"), f"row {line_number}: truth visibility drift")
            require(row.get("compressed_bytes", -1) >= 0 and row.get("uncompressed_bytes", -1) >= 0, f"row {line_number}: member size drift")
            require(row.get("local_header_offset", -1) >= 0, f"row {line_number}: local-header offset drift")
            counts[(archive, parent)] += 1
            rows += 1

    expected_rows = sum(sum(parent_counts.values()) for parent_counts in EXPECTED_COUNTS.values())
    require(rows == expected_rows == 27160, "Spring manifest row denominator drift")
    require(receipt.get("manifest_rows") == rows, "Spring receipt row denominator drift")
    archives = {item["archive"]: item for item in receipt.get("archives", [])}
    require(set(archives) == set(EXPECTED_COUNTS), "Spring receipt archive roster drift")
    for archive, parent_counts in EXPECTED_COUNTS.items():
        actual = {parent: counts[(archive, parent)] for parent in sorted(locked_parents)}
        require(actual == parent_counts, f"Spring manifest parent counts drift for {archive}")
        require(archives[archive].get("locked_parent_counts") == parent_counts, f"Spring receipt parent counts drift for {archive}")
        require(archives[archive].get("locked_parent_member_count") == sum(parent_counts.values()), f"Spring receipt archive denominator drift for {archive}")
        require(archives[archive].get("archive_bytes") == bindings[archive]["bytes"], f"Spring receipt archive bytes drift for {archive}")
        require(archives[archive].get("range_bytes_received", 0) > 0, f"Spring receipt range byte count absent for {archive}")
        require(archives[archive].get("central_directory_bytes", 0) > 0, f"Spring central-directory byte count absent for {archive}")
    require(receipt.get("range_bytes_received") == sum(item["range_bytes_received"] for item in archives.values()), "Spring total range byte count drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    validate(args.lock, args.manifest, args.receipt)
    print("SVRF_O0_SPRING_RANGE_MANIFEST_VALID")


if __name__ == "__main__":
    main()
