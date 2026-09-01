#!/usr/bin/env python3
"""Freeze target-disjoint multi-view identities and negative-source queues."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-instance-head-manifest-freeze-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-instance-head-manifest-v1"


def _scan_maps(metadata: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    family_by_scan: dict[str, str] = {}
    role_by_scan: dict[str, str] = {}
    for family in metadata:
        reference = str(family["reference"])
        family_by_scan[reference] = reference
        role_by_scan[reference] = "reference"
        for scan in family.get("scans", []):
            scan_id = str(scan["reference"])
            family_by_scan[scan_id] = reference
            role_by_scan[scan_id] = "rescan"
    return family_by_scan, role_by_scan


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["metadata"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    metadata = pixel.load_json(artifact_root / protocol["source"]["metadata"][0]["path"])
    objects_payload = pixel.load_json(artifact_root / protocol["source"]["metadata"][1]["path"])
    family_by_scan, role_by_scan = _scan_maps(metadata)
    objects_by_scan = {
        str(row["scan"]): row.get("objects", []) for row in objects_payload["scans"]
    }
    heldout = set(protocol["split"]["heldout_reference_families"])

    identities: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    cohort_inputs: list[dict[str, Any]] = []
    for path in sorted(HERE.glob(protocol["source"]["cohort_glob"])):
        try:
            cohort = pixel.load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        images = cohort.get("images")
        if not isinstance(images, dict):
            continue
        used = 0
        for image_key, row in sorted(images.items()):
            scan_id = str(row.get("scan_id", ""))
            if scan_id not in family_by_scan or "door" not in str(row.get("target_label", "")).lower():
                continue
            target_id = int(row["target_instance_id"])
            family = family_by_scan[scan_id]
            identity_key = f"{family}::{target_id}"
            bbox = tuple(float(value) for value in row["bbox_xyxy"])
            dedupe = (scan_id, str(row["zip_member"]), bbox, target_id)
            identities[identity_key][dedupe] = {
                "image_key": str(image_key),
                "cohort_path": path.name,
                "scan_id": scan_id,
                "scan_role": role_by_scan[scan_id],
                "frame": int(row["frame"]),
                "zip_member": str(row["zip_member"]),
                "bbox_xyxy": list(bbox),
                "target_instance_id": target_id,
                "target_label": str(row["target_label"]),
            }
            used += 1
        if used:
            cohort_inputs.append(
                {"path": path.name, "sha256": pixel.sha256(path), "admitted_rows_before_dedupe": used}
            )

    minimum_images = int(protocol["selection"]["minimum_positive_images"])
    minimum_scans = int(protocol["selection"]["minimum_distinct_scans"])
    maximum_images = int(protocol["selection"]["maximum_positive_images_per_identity"])
    label_priority = protocol["selection"]["ordinary_negative_label_priority"]
    priority = {label: index for index, label in enumerate(label_priority)}
    records: list[dict[str, Any]] = []
    heldout_records: list[dict[str, Any]] = []
    source_root = artifact_root / "datasets/3rscan"
    for identity_key in sorted(identities):
        family, target_text = identity_key.split("::", maxsplit=1)
        target_id = int(target_text)
        images = sorted(
            identities[identity_key].values(),
            key=lambda row: (row["scan_role"] != "reference", row["scan_id"], row["frame"], row["image_key"]),
        )
        scans = sorted({row["scan_id"] for row in images})
        if len(images) < minimum_images or len(scans) < minimum_scans:
            continue
        selected_images = images[:maximum_images]
        base_record = {
            "identity_key": identity_key,
            "reference_family": family,
            "target_instance_id": target_id,
            "positive_images": selected_images,
            "positive_image_count_before_cap": len(images),
            "distinct_positive_scans": scans,
        }
        if family in heldout:
            heldout_records.append(base_record)
            continue
        query_scans = sorted(
            {row["scan_id"] for row in images if row["scan_role"] == "rescan"},
        )
        if not query_scans:
            continue
        negative_source = None
        for query_scan in query_scans:
            required = ("semseg.v2.json", "labels.instances.annotated.v2.ply", "sequence.zip")
            if not all((source_root / query_scan / name).is_file() for name in required):
                continue
            objects = objects_by_scan.get(query_scan, [])
            hard_ids = sorted(
                int(row["id"])
                for row in objects
                if int(row["id"]) != target_id and "door" in str(row.get("label", "")).lower()
            )
            ordinary = [
                {"instance_id": int(row["id"]), "label": str(row.get("label", ""))}
                for row in objects
                if int(row["id"]) != target_id
                and "door" not in str(row.get("label", "")).lower()
                and str(row.get("label", "")).lower() not in {"wall", "floor", "ceiling"}
            ]
            ordinary.sort(
                key=lambda row: (
                    priority.get(row["label"].lower(), len(priority)),
                    row["label"].lower(),
                    row["instance_id"],
                )
            )
            if hard_ids and ordinary:
                negative_source = {
                    "scan_id": query_scan,
                    "same_scene_door_instance_queue": hard_ids,
                    "ordinary_different_class_queue": ordinary[: int(protocol["selection"]["maximum_ordinary_negative_queue"])],
                    "selection": "For each queue, take the first source-admitted instance under the frozen pose/depth view rule before RGB/model access.",
                }
                break
        if negative_source is None:
            continue
        records.append({**base_record, "negative_source": negative_source})

    pixel.require(len(records) >= int(protocol["selection"]["minimum_training_identities"]), "INSUFFICIENT_TRAINING_IDENTITIES")
    heldout_by_key = {row["identity_key"]: row for row in heldout_records}
    for required in protocol["split"]["required_heldout_identities"]:
        pixel.require(required in heldout_by_key, f"HELDOUT_IDENTITY_MISSING:{required}")
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "PRE_FEATURE_PRE_RGB_MODEL_TARGET_DISJOINT_INSTANCE_HEAD_MANIFEST",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "source": {
            "metadata": protocol["source"]["metadata"],
            "cohort_inputs": cohort_inputs,
            "cohort_input_count": len(cohort_inputs),
        },
        "training": {
            "identity_count": len(records),
            "identities": records,
            "opened_pose_members": 0,
            "opened_depth_members": 0,
            "opened_rgb_members": 0,
            "feature_calls": 0,
            "model_calls": 0,
        },
        "heldout": {
            "reference_families": sorted(heldout),
            "required_identities": [heldout_by_key[key] for key in protocol["split"]["required_heldout_identities"]],
        },
        "head": protocol["head"],
        "literature_motivation": protocol["literature_motivation"],
        "next_action": protocol["next_action"],
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
