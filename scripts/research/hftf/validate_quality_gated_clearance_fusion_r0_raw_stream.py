#!/usr/bin/env python3
"""Static, fail-closed validation for the frozen R0 raw geometry stream.

This validator uses only the Python standard library. It never imports an ML
runtime, loads a checkpoint, decodes media, or reads outcome labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


PROTOCOL_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_raw_stream_validation_protocol"
CATALOG_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_source_catalog"
STREAM_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_raw_geometry_stream"
RESULT_SCHEMA = "blindassist_quality_gated_clearance_fusion_r0_raw_stream_validation_result"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def verify_binding(root: Path, binding: dict[str, Any], label: str) -> Path:
    require(set(binding) == {"path", "sha256"}, f"{label} binding fields drift")
    path = resolve(root, str(binding["path"]))
    require(path.is_file(), f"{label} missing: {path}")
    require(sha256_file(path) == str(binding["sha256"]).upper(), f"{label} SHA mismatch")
    return path


def expected_state(value: Any, valid: bool, threshold: float) -> str:
    if not valid:
        return "UNKNOWN"
    require(value is not None and math.isfinite(float(value)), "valid clearance must be finite")
    return "OCCUPIED" if float(value) <= threshold else "CLEAR"


def validate(root: Path, protocol_path: Path, catalog_path: Path, stream_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "validation protocol schema drift")
    for label in (
        "parent_development_protocol",
        "materialization_protocol",
        "materializer_producer",
        "source_catalog",
        "a2_training_receipt",
        "a2_checkpoint",
    ):
        verify_binding(root, protocol[label], label)
    require(sha256_file(catalog_path) == protocol["source_catalog"]["sha256"].upper(), "catalog argument drift")
    require(sha256_file(stream_path) == protocol["raw_stream"]["sha256"].upper(), "raw stream SHA mismatch")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    require(catalog.get("schema") == CATALOG_SCHEMA, "source catalog schema drift")
    require(catalog.get("labels_opened") is False, "catalog labels boundary violated")
    require(catalog.get("image_or_depth_bytes_decoded") is False, "catalog decode boundary violated")
    catalog_rows = {str(row["frame_id"]): row for row in catalog.get("frames", [])}
    require(len(catalog_rows) == int(protocol["expected_frame_count"]), "catalog frame count drift")

    required_fields = set(protocol["required_frame_fields"])
    threshold = float(protocol["clearance_threshold_m"])
    rows: list[dict[str, Any]] = []
    with stream_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            require(line.endswith("\n"), f"unterminated JSONL row: {line_number}")
            row = json.loads(line)
            require(row.get("schema") == STREAM_SCHEMA, f"stream schema drift at row {line_number}")
            require(not (required_fields - set(row)), f"missing fields at row {line_number}")
            frame_id = str(row["frame_id"])
            require(frame_id in catalog_rows, f"unknown frame identity: {frame_id}")
            source = catalog_rows[frame_id]
            for field in ("parent_id", "video_id", "timestamp_ns"):
                require(row[field] == source[field], f"identity mismatch for {frame_id}: {field}")
            clearances = row["raw_clearance_m"]
            valids = row["raw_geometry_valid"]
            states = row["raw_geometry_state"]
            require(len(clearances) == len(valids) == len(states) == 3, f"geometry shape drift: {frame_id}")
            for value, valid, state in zip(clearances, valids, states):
                require(isinstance(valid, bool), f"geometry validity type drift: {frame_id}")
                require(state == expected_state(value, valid, threshold), f"geometry state mismatch: {frame_id}")
            require(isinstance(row["tof_valid"], bool), f"sensor validity type drift: {frame_id}")
            require(float(row["teacher_age_s"]) == 0.0, f"teacher age drift: {frame_id}")
            disagreement = float(row["frozen_a2_disagreement"])
            require(math.isfinite(disagreement) and disagreement >= 0.0, f"invalid disagreement: {frame_id}")
            for field in ("rgb_sha256", "metric_depth_sha256"):
                value = str(row[field])
                require(len(value) == 64 and all(char in "0123456789ABCDEF" for char in value), f"invalid {field}: {frame_id}")
            rows.append(row)

    frame_ids = [str(row["frame_id"]) for row in rows]
    require(len(frame_ids) == len(set(frame_ids)), "duplicate raw stream frame identity")
    require(set(frame_ids) == set(catalog_rows), "raw stream/catalog frame set mismatch")
    require(len(rows) == int(protocol["expected_frame_count"]), "raw stream frame count drift")
    parent_counts = Counter(str(row["parent_id"]) for row in rows)
    require(dict(sorted(parent_counts.items())) == protocol["expected_parent_frame_counts"], "parent frame counts drift")

    return {
        "schema": RESULT_SCHEMA,
        "protocol_sha256": sha256_file(protocol_path),
        "source_catalog_sha256": sha256_file(catalog_path),
        "raw_stream_sha256": sha256_file(stream_path),
        "frame_count": len(rows),
        "parent_frame_counts": dict(sorted(parent_counts.items())),
        "geometry_status_counts": dict(sorted(Counter(str(row.get("geometry_status")) for row in rows).items())),
        "model_loaded": False,
        "optimizer_constructed": False,
        "training_started": False,
        "labels_opened": False,
        "holdout_outcomes_opened": False,
        "terminal": "QUALITY_GATED_CLEARANCE_FUSION_R0_RAW_STREAM_VALIDATED_MODEL_UNLOADED",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-catalog", type=Path, required=True)
    parser.add_argument("--raw-stream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "overwrite forbidden")
    result = validate(
        args.repo_root.resolve(),
        args.protocol.resolve(),
        args.source_catalog.resolve(),
        args.raw_stream.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
