#!/usr/bin/env python3
"""Static, fail-closed preflight for the R0 raw geometry stream.

This command only validates schemas, paths and SHA-256 bindings. It never
imports an ML runtime, deserializes a checkpoint, decodes media, or opens
labels/outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


RESULT_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_raw_stream_preflight_result"
CATALOG_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_source_catalog"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def bind(root: Path, value: dict[str, str], label: str) -> Path:
    require(set(value) == {"path", "sha256"}, f"{label} binding fields drift")
    path = Path(value["path"])
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    require(path.is_file(), f"{label} missing: {path}")
    require(sha256_file(path) == str(value["sha256"]).upper(), f"{label} SHA mismatch")
    return path


def check_json(path: Path, expected_schema: str, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema") == expected_schema, f"{label} schema drift")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--a2-receipt", type=Path, required=True)
    parser.add_argument("--a2-checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--disagreement-cache", type=Path, required=True)
    parser.add_argument("--geometry-stream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "overwrite forbidden")
    root = args.repo_root.resolve()
    protocol = check_json(args.protocol.resolve(), "blindassist_quality_gated_clearance_fusion_r0_development_protocol", "protocol")
    catalog = check_json(args.source_catalog.resolve(), CATALOG_SCHEMA, "source catalog")
    require(catalog.get("labels_opened") is False and catalog.get("image_or_depth_bytes_decoded") is False, "catalog boundary violated")
    required_parents = {"rgbd_dataset_freiburg3_sitting_halfsphere", "rgbd_dataset_freiburg3_sitting_rpy", "381644"}
    require(set(catalog.get("parents", [])) == required_parents, "source parent set drift")
    require(int(catalog.get("clip_count", 0)) >= 24, "source clip capacity unexpectedly low")

    a2_receipt = json.loads(args.a2_receipt.resolve().read_text(encoding="utf-8"))
    require(a2_receipt.get("schema") == "blindassist_dav2_392_distillation_a2_r0_training_result", "A2 receipt schema drift")
    checkpoint = args.a2_checkpoint.resolve()
    require(checkpoint.is_file(), "A2 checkpoint missing")
    require(sha256_file(checkpoint) == a2_receipt.get("checkpoint", {}).get("sha256", "").upper(), "A2 checkpoint SHA mismatch")
    teacher = json.loads(args.teacher_manifest.resolve().read_text(encoding="utf-8"))
    require(teacher.get("schema") == "blindassist_dav2_distillation_teacher_r0", "teacher manifest schema drift")
    disagreement = args.disagreement_cache.resolve()
    require(disagreement.is_file(), "frozen disagreement cache missing")
    require(args.geometry_stream.resolve().is_file(), "raw geometry stream missing")
    result = {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(args.protocol.resolve()),
        "source_catalog_sha256": sha256_file(args.source_catalog.resolve()),
        "a2_receipt_sha256": sha256_file(args.a2_receipt.resolve()),
        "a2_checkpoint_sha256": sha256_file(checkpoint),
        "teacher_manifest_sha256": sha256_file(args.teacher_manifest.resolve()),
        "disagreement_cache_sha256": sha256_file(disagreement),
        "geometry_stream_present": args.geometry_stream.resolve().is_file(),
        "geometry_stream_sha256": sha256_file(args.geometry_stream.resolve()) if args.geometry_stream.resolve().is_file() else None,
        "model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
        "labels_opened": False,
        "holdout_outcomes_opened": False,
        "terminal": "QUALITY_GATED_CLEARANCE_FUSION_R0_RAW_STREAM_INPUTS_READY",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
