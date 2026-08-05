"""Label-blind source admission for quality-gated clearance fusion R0.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tarfile
from pathlib import Path


SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_1_source_admission_result"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def required_tar_members(path: Path, required: list[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            name = Path(member.name).name
            if name in required and member.isfile():
                found[name] = member.name
    if set(found) != set(required):
        raise ValueError(f"required TUM metadata members missing: {sorted(set(required) - set(found))}")
    return found


def arkit_selection(split: Path, excluded: set[str]) -> tuple[str, list[str], int]:
    with split.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    visits: dict[str, list[str]] = {}
    for row in rows:
        visit = row["visit_id"]
        if row["fold"] == "Validation" and visit != "NA" and visit not in excluded:
            visits.setdefault(visit, []).append(row["video_id"])
    if not visits:
        raise ValueError("no eligible ARKit Validation visit")
    visit = sorted(visits)[0]
    return visit, sorted(visits[visit]), len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("overwrite forbidden")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha256_file(Path(__file__)) != protocol["implementation"]["producer_sha256"]:
        raise ValueError("producer SHA mismatch")
    split = Path(protocol["arkitscenes_split"]["path"])
    expected_split_sha = protocol["arkitscenes_split"]["sha256"]
    if expected_split_sha and sha256_file(split) != expected_split_sha:
        raise ValueError("ARKit split SHA mismatch")
    excluded = set().union(*[set(values) for values in protocol["ancestry_exclusions"].values()])
    visit, videos, row_count = arkit_selection(split, excluded)
    expected_arkit = next(row for row in protocol["sources"] if row["dataset"] == "ARKitScenes")
    if visit != expected_arkit["parent_id"] or videos != expected_arkit["video_ids"] or row_count != protocol["arkitscenes_split"]["row_count"]:
        raise ValueError("ARKit identity selection drift")
    sources = []
    for source in protocol["sources"]:
        if source["dataset"] == "TUM_RGBD":
            archive = args.tum_root / f"{source['parent_id']}.tgz"
            if not archive.is_file():
                sources.append({"dataset": "TUM_RGBD", "parent_id": source["parent_id"], "available": False})
                continue
            members = required_tar_members(archive, source["required_members"])
            sources.append({"dataset": "TUM_RGBD", "parent_id": source["parent_id"], "available": True, "archive_sha256": sha256_file(archive), "bytes": archive.stat().st_size, "metadata_members": members})
        else:
            sources.append({"dataset": "ARKitScenes", "parent_id": visit, "video_ids": videos, "available": False, "license_scope_extension_required": True})
    available = sum(bool(row["available"]) for row in sources)
    terminal = protocol["terminals"]["license_blocked"] if available == 2 else protocol["terminals"]["insufficient"]
    result = {"schema": SCHEMA, "protocol_sha256": sha256_file(args.protocol), "producer_sha256": sha256_file(Path(__file__)), "sources": sources, "available_new_parent_count": available, "minimum_new_parents": protocol["minimum_new_parents"], "rgb_or_depth_image_body_read": False, "labels_read": False, "model_loaded": False, "optimizer_constructed": False, "training_started": False, "terminal": terminal}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(args.output, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush(); os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
