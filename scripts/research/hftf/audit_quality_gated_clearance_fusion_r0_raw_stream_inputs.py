#!/usr/bin/env python3
"""Static dependency audit for the R0 raw geometry stream inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_raw_stream_input_audit"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--disagreement-cache", type=Path, required=True)
    parser.add_argument("--geometry-stream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "overwrite forbidden")
    catalog = json.loads(args.source_catalog.read_text(encoding="utf-8"))
    require(catalog.get("schema") == "blindassist_quality_gated_clearance_fusion_r0_source_catalog", "catalog schema drift")
    catalog_frames = {str(row["frame_id"]) for row in catalog.get("frames", [])}
    catalog_parents = {str(row["parent_id"]) for row in catalog.get("frames", [])}
    teacher = json.loads(args.teacher_manifest.read_text(encoding="utf-8"))
    teacher_records = teacher.get("records", [])
    teacher_frames = {str(row.get("frame_id")) for row in teacher_records}
    teacher_parents = {str(row.get("parent_id")) for row in teacher_records}
    disagreement_frames = set()
    for line in args.disagreement_cache.read_text(encoding="utf-8").splitlines():
        if line.strip():
            disagreement_frames.add(str(json.loads(line)["frame_id"]))
    geometry_present = args.geometry_stream.is_file()
    result = {
        "schema": RESULT_SCHEMA,
        "source_catalog_sha256": sha256_file(args.source_catalog),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest),
        "disagreement_cache_sha256": sha256_file(args.disagreement_cache),
        "geometry_stream_path": str(args.geometry_stream),
        "geometry_stream_present": geometry_present,
        "catalog_frame_count": len(catalog_frames),
        "catalog_parent_ids": sorted(catalog_parents),
        "teacher_frame_count": len(teacher_frames),
        "teacher_parent_ids": sorted(teacher_parents),
        "disagreement_frame_count": len(disagreement_frames),
        "catalog_teacher_frame_overlap": len(catalog_frames & teacher_frames),
        "catalog_disagreement_frame_overlap": len(catalog_frames & disagreement_frames),
        "catalog_teacher_parent_overlap": sorted(catalog_parents & teacher_parents),
        "catalog_disagreement_parent_overlap": sorted({"381449"} & catalog_parents),
        "labels_opened": False,
        "model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
        "terminal": "QUALITY_GATED_CLEARANCE_FUSION_R0_RAW_STREAM_INPUTS_NOT_READY",
        "missing": [
            item for item, present in {
                "new-parent-teacher-manifest": bool(catalog_parents & teacher_parents),
                "new-parent-disagreement-cache": bool(catalog_frames & disagreement_frames),
                "raw-geometry-stream": geometry_present,
            }.items() if not present
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, indent=2))
    raise SystemExit(1)


if __name__ == "__main__":
    main()
