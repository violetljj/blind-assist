#!/usr/bin/env python3
"""Freeze exact RGB members and crops without reading image payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
import zipfile


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-instance-head-rgb-manifest-freeze-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-instance-head-rgb-manifest-v1"


def _split_families(families: list[str], seed: str, validation_count: int) -> tuple[set[str], set[str]]:
    ordered = sorted(families, key=lambda value: (hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(), value))
    validation = set(ordered[:validation_count])
    return set(ordered[validation_count:]), validation


def _sample(
    identity_key: str,
    role: str,
    scan_id: str,
    zip_member: str,
    bbox_xyxy: list[float],
    color_size: list[int],
    instance_id: int,
    label: str,
    ordinal: int,
) -> dict[str, Any]:
    sample_key = f"{identity_key}|{role}|{scan_id}|{zip_member}|{instance_id}|{ordinal}"
    return {
        "sample_id": hashlib.sha256(sample_key.encode()).hexdigest()[:24],
        "identity_key": identity_key,
        "role": role,
        "scan_id": scan_id,
        "zip_member": zip_member,
        "bbox_xyxy": [float(value) for value in bbox_xyxy],
        "color_size": [int(value) for value in color_size],
        "instance_id": int(instance_id),
        "label": label,
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    source_row = protocol["geometry_source"]
    source_path = HERE / source_row["path"]
    pixel.require(pixel.sha256(source_path) == source_row["sha256"], "GEOMETRY_SOURCE_HASH")
    source = pixel.load_json(source_path)
    pixel.require(source["schema"] == source_row["required_schema"], "GEOMETRY_SOURCE_SCHEMA")

    families = sorted({str(row["reference_family"]) for row in source["training_identities"]})
    train_families, validation_families = _split_families(
        families,
        str(protocol["split"]["seed"]),
        int(protocol["split"]["validation_family_count"]),
    )
    samples: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    for identity in source["training_identities"]:
        identity_key = str(identity["identity_key"])
        family = str(identity["reference_family"])
        split = "train" if family in train_families else "validation"
        sample_ids: list[str] = []
        for ordinal, positive in enumerate(identity["positive_images"]):
            row = _sample(
                identity_key,
                "positive",
                str(positive["scan_id"]),
                str(positive["zip_member"]),
                positive["bbox_xyxy"],
                positive.get("color_size", protocol["source"]["default_color_size"]),
                int(identity["target_instance_id"]),
                str(positive["target_label"]),
                ordinal,
            )
            samples.append(row)
            sample_ids.append(row["sample_id"])
        for role, negative in (
            ("same_scene_hard_negative", identity["same_scene_hard_negative"]),
            ("ordinary_negative", identity["ordinary_negative"]),
        ):
            selected = negative["selected"]
            row = _sample(
                identity_key,
                role,
                str(identity["negative_scan_id"]),
                str(selected["zip_member"]),
                selected["bbox_xyxy"],
                selected["color_size"],
                int(negative["instance_id"]),
                str(negative["label"]),
                0,
            )
            samples.append(row)
            sample_ids.append(row["sample_id"])
        identities.append(
            {
                "identity_key": identity_key,
                "reference_family": family,
                "split": split,
                "sample_ids": sample_ids,
            }
        )

    pixel.require(len({row["sample_id"] for row in samples}) == len(samples), "DUPLICATE_SAMPLE_ID")
    split_by_identity = {row["identity_key"]: row["split"] for row in identities}
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    zip_receipts: dict[str, dict[str, Any]] = {}
    members_by_scan: dict[str, set[str]] = {}
    for sample in samples:
        members_by_scan.setdefault(sample["scan_id"], set()).add(sample["zip_member"])
    for scan_id in sorted(members_by_scan):
        archive_path = artifact_root / "datasets/3rscan" / scan_id / "sequence.zip"
        pixel.require(archive_path.is_file(), f"MISSING_ZIP:{scan_id}")
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
        missing = sorted(members_by_scan[scan_id] - names)
        pixel.require(not missing, f"MISSING_RGB_MEMBER:{scan_id}:{missing}")
        zip_receipts[scan_id] = {
            "path": archive_path.resolve().relative_to(artifact_root.resolve()).as_posix(),
            "bytes": archive_path.stat().st_size,
            "sha256": pixel.sha256(archive_path),
            "selected_member_count": len(members_by_scan[scan_id]),
        }

    train_identity_count = sum(row["split"] == "train" for row in identities)
    validation_identity_count = sum(row["split"] == "validation" for row in identities)
    pixel.require(train_identity_count >= int(protocol["gate"]["minimum_train_identities"]), "INSUFFICIENT_TRAIN_IDENTITIES")
    pixel.require(validation_identity_count >= int(protocol["gate"]["minimum_validation_identities"]), "INSUFFICIENT_VALIDATION_IDENTITIES")
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "PRE_RGB_PRE_FEATURE_EXACT_MEMBER_AND_CROP_MANIFEST",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "geometry_source": source_row,
        "split": {
            "seed": protocol["split"]["seed"],
            "train_reference_families": sorted(train_families),
            "validation_reference_families": sorted(validation_families),
            "heldout_reference_families": source["heldout_reference_families"],
            "train_identity_count": train_identity_count,
            "validation_identity_count": validation_identity_count,
        },
        "crop": protocol["crop"],
        "identities": identities,
        "samples": samples,
        "sample_count": len(samples),
        "zip_receipts": zip_receipts,
        "opened_rgb_members": 0,
        "feature_calls": 0,
        "model_calls": 0,
        "split_by_identity": split_by_identity,
        "next_action": protocol["next_action"],
        "conclusion": "L10_3RSCAN_INSTANCE_HEAD_RGB_MANIFEST_EVALUABLE",
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
