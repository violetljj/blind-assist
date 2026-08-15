#!/usr/bin/env python3
"""Validate the outcome-blind SVRF-O0 archive-access capability receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_STATUS = "PARTIAL_PASS_SPRING_RANDOM_ACCESS_A2D2_STREAM_PLAN_REQUIRED"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(lock_path: Path, receipt_path: Path, capability_lock_path: Path | None = None) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("schema") == "blindassist.svrf_o0.archive_access_capability.v1", "capability schema drift")
    require(receipt.get("status") == EXPECTED_STATUS, "capability terminal drift")
    require(receipt.get("source_lock_sha256") == file_hash(lock_path), "source-lock hash drift")
    require(receipt.get("outcome_accessed") is False, "outcome was accessed during capability probe")
    require(receipt.get("candidate_run_count") == 0, "candidate ran during capability probe")
    require(receipt.get("archive_member_crc_validation_count") == 5, "archive CRC validation count drift")
    require(receipt.get("media_tensor_or_label_decode_count") == 0, "source media/tensor/label was decoded")

    sources = {source["source_id"]: source for source in lock["sources"]}
    expected_a2d2 = {
        archive["name"]: int(archive["bytes"])
        for parent in sources["A2D2_SENSOR_FUSION"]["parents"]
        for archive in parent["archives"].values()
    }
    observed_a2d2 = {item["archive"]: item for item in receipt["a2d2"]["archives"]}
    require(set(observed_a2d2) == set(expected_a2d2), "A2D2 archive roster drift")
    for name, size in expected_a2d2.items():
        item = observed_a2d2[name]
        require(item["bytes"] == size and item["range_supported"] is True, f"A2D2 range/size drift: {name}")
        require(item["central_member_index"] is False, f"A2D2 TAR cannot claim a central index: {name}")
        require(item["prefix_members"], f"A2D2 TAR header probe missing: {name}")
    inventory = receipt["a2d2"]["bucket_inventory"]
    require(inventory["listing_truncated"] is False, "A2D2 bucket inventory is incomplete")
    require(inventory["individual_sensor_object_count"] == 0, "A2D2 individual-object finding drift")
    require(receipt["a2d2"]["random_member_index_supported"] is False, "A2D2 random index claim drift")

    expected_spring = sources["SPRING_V2"]["archive_bindings"]
    observed_spring = {item["archive"]: item for item in receipt["spring"]["archives"]}
    require(set(observed_spring) == set(expected_spring), "Spring archive roster drift")
    locked_parents = {parent["parent_id"] for parent in sources["SPRING_V2"]["parents"]}
    for name, binding in expected_spring.items():
        item = observed_spring[name]
        require(item["bytes"] == binding["bytes"] and item["range_supported"] is True, f"Spring range/size drift: {name}")
        require(item["central_member_index"] is True, f"Spring central index missing: {name}")
        require(item["locked_parent_member_count"] > 0, f"Spring locked members missing: {name}")
        require(set(item["locked_parent_counts"]) == locked_parents, f"Spring parent-count roster drift: {name}")
        require(all(value > 0 for value in item["locked_parent_counts"].values()), f"Spring parent support missing: {name}")
        require(
            item["selective_crc_sample"]["status"] == "CRC_VALID_WITHOUT_CONTENT_DECODE",
            f"Spring selective CRC probe failed: {name}",
        )
    require(receipt["spring"]["random_member_index_supported"] is True, "Spring random index finding drift")

    expected_total = sum(expected_a2d2.values()) + sum(int(value["bytes"]) for value in expected_spring.values())
    storage = receipt["materialization_storage"]
    require(storage["all_bound_archives_bytes"] == expected_total, "full archive denominator drift")
    require(storage["physical_drive"].upper() == "F:", "large-data physical drive drift")
    require(
        storage["physical_root"].lower().startswith(r"f:\ba-data\blindassist-artifacts-20260805"),
        "canonical F-drive artifact root drift",
    )
    require(storage["all_bound_archives_fit_raw"] is True, "raw F-drive archive fit finding drift")
    require(
        storage["all_bound_archives_fit_with_safety_reserve"] is False,
        "64-GiB safety-reserve finding drift",
    )
    require(storage["spring_archives_fit_with_safety_reserve"] is True, "Spring storage fit finding drift")
    require(storage["largest_single_archive_fits_with_safety_reserve"] is True, "single-archive fit finding drift")
    require(receipt.get("bulk_download_authorized") is False, "bulk download authority opened")
    require(receipt.get("candidate_run_authorized") is False, "candidate authority opened")
    require(receipt.get("outcome_access_authorized") is False, "outcome authority opened")

    if capability_lock_path is not None:
        capability = json.loads(capability_lock_path.read_text(encoding="utf-8"))
        require(capability.get("schema") == "blindassist.svrf_o0.archive_access_capability_lock.v1", "capability-lock schema drift")
        require(capability.get("receipt_sha256") == file_hash(receipt_path), "capability receipt hash drift")
        require(capability.get("status") == EXPECTED_STATUS, "capability-lock terminal drift")
        require(capability.get("source_lock_sha256") == file_hash(lock_path), "capability-lock source hash drift")
        require(capability["activation"] == {
            "bulk_download_authorized": False,
            "stream_index_execution_authorized": False,
            "candidate_run_authorized": False,
            "outcome_access_authorized": False,
        }, "capability-lock activation drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--capability-lock", type=Path)
    args = parser.parse_args()
    validate(args.lock, args.receipt, args.capability_lock)
    print("SVRF_O0_ARCHIVE_ACCESS_CAPABILITY_VALID")


if __name__ == "__main__":
    main()
