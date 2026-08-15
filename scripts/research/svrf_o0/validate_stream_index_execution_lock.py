#!/usr/bin/env python3
"""Validate the user-authorized SVRF-O0 archive-index execution boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(lock_path: Path, repo_root: Path, artifact_root: Path) -> None:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == "blindassist.svrf_o0.stream_index_execution_lock.v1", "execution-lock schema drift")
    require(lock.get("status") == "AUTHORIZED_OUTCOME_BLIND_ARCHIVE_INDEX_EXECUTION", "execution authority drift")
    source_path = repo_root / "docs/research/svrf" / lock["source_lock"]
    capability_path = repo_root / "docs/research/svrf" / lock["capability_lock"]
    require(sha256(source_path) == lock["source_lock_sha256"], "execution source-lock hash drift")
    require(sha256(capability_path) == lock["capability_lock_sha256"], "execution capability-lock hash drift")
    implementation = lock["implementation"]
    for path_key, hash_key in (
        ("a2d2_stream_indexer", "a2d2_stream_indexer_sha256"),
        ("spring_range_manifest_builder", "spring_range_manifest_builder_sha256"),
    ):
        require(sha256(repo_root / implementation[path_key]) == implementation[hash_key], f"implementation hash drift: {path_key}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    a2d2 = next(item for item in source["sources"] if item["source_id"] == "A2D2_SENSOR_FUSION")
    archives = [archive["name"] for parent in a2d2["parents"] for archive in parent["archives"].values()]
    require(set(archives) == set(lock["a2d2_execution"]["archive_order"]), "execution archive roster drift")
    expected_bytes = sum(
        int(archive["bytes"])
        for parent in a2d2["parents"]
        for archive in parent["archives"].values()
    )
    require(expected_bytes == lock["a2d2_execution"]["maximum_network_stream_bytes_before_retries"], "network denominator drift")
    physical_root = artifact_root.resolve()
    require(physical_root.drive.upper() == "F:", "execution artifact root is not on F drive")
    require(str(physical_root).lower().startswith(r"f:\ba-data\blindassist-artifacts-20260805"), "execution F-drive root drift")
    reserve = int(lock["storage"]["minimum_free_space_reserve_bytes"])
    require(shutil.disk_usage(physical_root).free >= reserve, "execution F-drive reserve unavailable")
    require(lock["activation"] == {
        "a2d2_stream_index_execution_authorized": True,
        "spring_range_manifest_execution_authorized": True,
        "bulk_archive_retention_authorized": False,
        "selected_payload_materialization_authorized": False,
        "truth_writer_execution_authorized": False,
        "candidate_run_authorized": False,
        "outcome_access_authorized": False,
    }, "execution activation drift")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts.local"))
    args = parser.parse_args()
    validate(args.lock, args.repo_root, args.artifact_root)
    print("SVRF_O0_STREAM_INDEX_EXECUTION_LOCK_VALID")


if __name__ == "__main__":
    main()
