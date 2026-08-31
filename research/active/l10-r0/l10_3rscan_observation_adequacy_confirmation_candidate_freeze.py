#!/usr/bin/env python3
"""Freeze one metadata-only 3RScan family for observation-adequate confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_registered_extent_ceiling as extent  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-observation-adequacy-confirmation-candidate-freeze-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-observation-adequacy-confirmation-candidate-v1"


def freeze(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("candidate_protocol", "predecessor", "exclusion_authority"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    for row in protocol["source"]["metadata"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")

    exclusion_protocol = pixel.load_json(HERE / protocol["exclusion_authority"]["path"])
    consumed: set[tuple[str, int]] = set()
    used_references: set[str] = set()
    for source in exclusion_protocol["physical_target_exclusions"]:
        path = HERE / source["path"]
        pixel.require(pixel.sha256(path) == source["sha256"], f"EXCLUSION_HASH:{source['path']}")
        cohort = pixel.load_json(path)
        for episode in cohort.get("episodes", []):
            if "reference_scan_id" in episode and "target_instance_id" in episode:
                reference = str(episode["reference_scan_id"])
                consumed.add((reference, int(episode["target_instance_id"])))
                used_references.add(reference)

    candidate_protocol = pixel.load_json(HERE / protocol["candidate_protocol"]["path"])
    candidates = extent.candidate_rows(candidate_protocol, data_root, require_geometry=False)
    considered = 0
    selected: dict[str, Any] | None = None
    for candidate in candidates:
        reference = str(candidate["reference_scan_id"])
        query = str(candidate["rescan_id"])
        target = int(candidate["target_instance_id"])
        if reference in used_references or (reference, target) in consumed:
            continue
        considered += 1
        required_names = ("semseg.v2.json", "labels.instances.annotated.v2.ply", "sequence.zip")
        if any((data_root / scan_id / name).exists() for scan_id in (reference, query) for name in required_names):
            continue
        selected = candidate
        break
    pixel.require(selected is not None, "NO_METADATA_ONLY_FRESH_SCAN_FAMILY")

    reference = str(selected["reference_scan_id"])
    query = str(selected["rescan_id"])
    download_plan = [
        {"scan_id": scan_id, "file": name}
        for scan_id in (reference, query)
        for name in ("semseg.v2.json", "labels.instances.annotated.v2.ply", "sequence.zip")
    ]
    result = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_DOWNLOAD_PRE_RGB_PRE_MODEL_SCAN_FAMILY_CANDIDATE",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "selection": {
            "ranking": protocol["selection"]["ranking"],
            "metadata_candidates": len(candidates),
            "eligible_candidates_considered_before_selection": considered,
            "excluded_physical_targets": len(consumed),
            "excluded_reference_scan_families": len(used_references),
            "opened_rgb_members": 0,
            "model_calls": 0,
        },
        "candidate": selected,
        "download_plan": download_plan,
        "post_download_gate": protocol["post_download_gate"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
