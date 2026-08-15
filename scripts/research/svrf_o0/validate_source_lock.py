#!/usr/bin/env python3
"""Validate the metadata-only A2D2 + Spring SVRF-O0 source lock."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile


SCHEMA = "blindassist.svrf_o0.source_lock.v1"
SPRING_PATH = re.compile(r"^spring/train/(?P<id>\d{4})/cam_data/extrinsics\.txt$")


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_a2d2_checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{32})\s+(.+)", line.strip())
        if match:
            result[match.group(2)] = match.group(1)
    return result


def spring_sequence_rows(path: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            match = SPRING_PATH.fullmatch(info.filename)
            if not match:
                continue
            content = archive.read(info).decode("utf-8")
            rows[match.group("id")] = sum(bool(line.strip()) for line in content.splitlines())
    return rows


def spring_rank(salt: str, sequence_id: str) -> str:
    return hashlib.sha256(f"{salt}|{sequence_id}".encode("utf-8")).hexdigest()


def select_spring(rows: dict[str, int], salt: str, minimum_rows: int, count: int) -> list[dict[str, object]]:
    eligible = [
        {"parent_id": sequence_id, "camera_rows": row_count, "selection_rank_sha256": spring_rank(salt, sequence_id)}
        for sequence_id, row_count in rows.items()
        if row_count >= minimum_rows
    ]
    return sorted(eligible, key=lambda item: str(item["selection_rank_sha256"]))[:count]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(lock_path: Path, metadata_root: Path, ledger_path: Path, repo_root: Path) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    require(lock.get("schema") == SCHEMA, "SVRF source-lock schema mismatch")
    require(lock.get("status") == "FROZEN_METADATA_ONLY_PAYLOAD_NOT_MATERIALIZED", "SVRF source-lock status drift")
    require(lock.get("parent_count") == 8 and lock.get("source_count") == 2, "SVRF source/parent denominator drift")
    sources = {item["source_id"]: item for item in lock["sources"]}
    require(set(sources) == {"A2D2_SENSOR_FUSION", "SPRING_V2"}, "SVRF source identity drift")

    files = {
        "a2d2_checksums_sha256": metadata_root / "a2d2-Checksums.txt",
        "a2d2_license_sha256": metadata_root / "a2d2-LICENSE.txt",
        "a2d2_sensor_fusion_readme_sha256": metadata_root / "a2d2-README-SensorFusion.txt",
        "spring_darus_api_sha256": metadata_root / "spring-darus-3376-v2.0.json",
        "spring_cam_data_zip_sha256": metadata_root / "spring-train_cam_data-v2.0.zip",
    }
    for key, path in files.items():
        require(path.is_file(), f"missing metadata input: {path}")
        require(file_hash(path) == lock["metadata_inputs"][key], f"metadata hash drift: {path.name}")

    a2d2 = sources["A2D2_SENSOR_FUSION"]
    require("CC BY-ND 4.0" in files["a2d2_license_sha256"].read_text(encoding="utf-8"), "A2D2 license drift")
    checksums = parse_a2d2_checksums(files["a2d2_checksums_sha256"])
    require(len(a2d2["parents"]) == 3, "A2D2 parent count drift")
    for parent in a2d2["parents"]:
        for archive in parent["archives"].values():
            require(checksums.get(archive["name"]) == archive["md5"], f"A2D2 checksum drift: {archive['name']}")

    spring = sources["SPRING_V2"]
    api = json.loads(files["spring_darus_api_sha256"].read_text(encoding="utf-8"))
    version = api["data"]["latestVersion"]
    require(f"{version['versionNumber']}.{version['versionMinorNumber']}" == spring["version"], "Spring version drift")
    require(version["license"]["name"] == spring["license"], "Spring license drift")
    by_name = {item["label"]: item["dataFile"] for item in version["files"]}
    for name, binding in spring["archive_bindings"].items():
        datafile = by_name.get(name)
        require(datafile is not None, f"Spring archive missing: {name}")
        require(datafile["id"] == binding["datafile_id"], f"Spring datafile id drift: {name}")
        require(datafile["filesize"] == binding["bytes"], f"Spring byte size drift: {name}")
        require(datafile["checksum"]["value"].lower() == binding["md5"], f"Spring checksum drift: {name}")
    camera_zip = files["spring_cam_data_zip_sha256"]
    require(file_hash(camera_zip, "md5") == spring["archive_bindings"]["train_cam_data.zip"]["md5"], "Spring camera ZIP MD5 drift")
    expected = select_spring(spring_sequence_rows(camera_zip), lock["selection_salt"], spring["minimum_camera_rows"], 5)
    require(expected == spring["parents"], "Spring deterministic parent selection drift")

    prior = lock["prior_use_exclusion"]
    command = ["git", "grep", "-I", "-i", "-E", prior["patterns"], prior["repository_commit"], "--", *prior["scope"]]
    scan = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)
    require(scan.returncode in {0, 1}, f"git grep failed: {scan.stderr.strip()}")
    matches = [line for line in scan.stdout.splitlines() if line.strip()]
    require(len(matches) == prior["matched_tracked_paths"] == 0, "pre-lock tracked prior-use scan drift")
    with ledger_path.open("r", encoding="utf-8-sig", newline="") as source:
        ledger_matches = [
            row
            for row in csv.DictReader(source)
            if row.get("dataset", "").lower() in {"a2d2", "spring"}
            or re.search(r"(^|[\\/_-])(a2d2|spring)([\\/_-]|$)", row.get("session_root", ""), re.I)
        ]
    require(len(ledger_matches) == prior["master_ledger_matching_rows"] == 0, "master-ledger prior-use scan drift")
    require(lock["activation"] == {
        "source_roster_frozen": True,
        "payload_materialized": False,
        "truth_writer_frozen": False,
        "candidate_run_authorized": False,
        "outcome_access_authorized": False,
    }, "SVRF source-lock activation boundary drift")
    require(lock["freshness_boundary"] == {
        "project_level_parent_freshness": True,
        "upstream_foundation_or_teacher_pretraining_exposure_excluded": False,
        "interpretation": "zero tracked-repository and master-ledger matches only establishes BlindAssist project-level prior-use exclusion; it is not a claim that DepthART or any upstream foundation model never saw A2D2 or Spring",
    }, "SVRF source-lock freshness boundary drift")
    return {
        "schema": "blindassist.svrf_o0.source_lock_validation.v1",
        "status": "SVRF_O0_FRESH_PARENT_SOURCE_LOCK_VALID",
        "lock_id": lock["lock_id"],
        "source_count": 2,
        "parent_count": 8,
        "parents": {source_id: [item["parent_id"] for item in source["parents"]] for source_id, source in sources.items()},
        "candidate_run_authorized": False,
        "outcome_access_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=Path("DATASET_MASTER_LEDGER.csv"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    result = validate(args.lock, args.metadata_root, args.ledger, args.repo_root)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["status"])


if __name__ == "__main__":
    main()
