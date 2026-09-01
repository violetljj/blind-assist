#!/usr/bin/env python3
"""Materialize exact instance-head negatives using geometry/depth only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_joint_covisibility_selector_posthoc as views  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-instance-head-negative-geometry-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-instance-head-negative-geometry-v1"


def _best(candidates: list[tuple[dict[str, Any], Any, Any]]) -> dict[str, Any]:
    return max(
        (row for row, _, _ in candidates),
        key=lambda row: (
            int(row["visible_target_vertices"]),
            float(row["bbox_short_side_fraction"]),
            float(row["depth_visible_ratio"]),
            -int(row["frame"]),
        ),
    )


def _select(
    data_root: Path,
    scan_id: str,
    queue: list[dict[str, Any]],
    rules: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    opened = {"pose_members": 0, "depth_members": 0, "rgb_members": 0, "model_calls": 0}
    for queue_index, item in enumerate(queue):
        instance_id = int(item["instance_id"])
        points, candidates, receipt = views._candidates(data_root, scan_id, instance_id, rules)
        for key in opened:
            opened[key] += int(receipt[key])
        if not candidates:
            continue
        return (
            {
                "queue_index": queue_index,
                "instance_id": instance_id,
                "label": str(item["label"]),
                "target_vertices": int(len(points)),
                "candidate_view_count": len(candidates),
                "selected": _best(candidates),
            },
            opened,
        )
    return None, opened


def _source_receipt(scan_root: Path, artifact_root: Path) -> list[dict[str, Any]]:
    rows = []
    for name in ("semseg.v2.json", "labels.instances.annotated.v2.ply", "sequence.zip"):
        path = scan_root / name
        pixel.require(path.is_file(), f"MISSING_SOURCE:{path}")
        rows.append(
            {
                "path": path.resolve().relative_to(artifact_root.resolve()).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": pixel.sha256(path),
            }
        )
    return rows


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    manifest_row = protocol["manifest"]
    manifest_path = HERE / manifest_row["path"]
    pixel.require(pixel.sha256(manifest_path) == manifest_row["sha256"], "MANIFEST_HASH")
    manifest = pixel.load_json(manifest_path)
    pixel.require(manifest["schema"] == manifest_row["required_schema"], "MANIFEST_SCHEMA")

    artifact_root = ROOT / protocol["source"]["artifact_root"]
    data_root = artifact_root / "datasets/3rscan"
    rules = protocol["candidate_view_rules"]
    admitted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    source_receipts: dict[str, list[dict[str, Any]]] = {}
    totals = {"pose_members": 0, "depth_members": 0, "rgb_members": 0, "model_calls": 0}
    for identity in manifest["training"]["identities"]:
        negative_source = identity["negative_source"]
        scan_id = str(negative_source["scan_id"])
        scan_root = data_root / scan_id
        if scan_id not in source_receipts:
            source_receipts[scan_id] = _source_receipt(scan_root, artifact_root)
        hard_queue = [
            {"instance_id": int(instance_id), "label": "door_or_doorframe"}
            for instance_id in negative_source["same_scene_door_instance_queue"]
        ]
        ordinary_queue = list(negative_source["ordinary_different_class_queue"])
        hard, hard_opened = _select(data_root, scan_id, hard_queue, rules)
        ordinary, ordinary_opened = _select(data_root, scan_id, ordinary_queue, rules)
        for key in totals:
            totals[key] += hard_opened[key] + ordinary_opened[key]
        if hard is None or ordinary is None:
            rejected.append(
                {
                    "identity_key": identity["identity_key"],
                    "scan_id": scan_id,
                    "hard_negative_admitted": hard is not None,
                    "ordinary_negative_admitted": ordinary is not None,
                }
            )
            continue
        admitted.append(
            {
                "identity_key": identity["identity_key"],
                "reference_family": identity["reference_family"],
                "target_instance_id": int(identity["target_instance_id"]),
                "positive_images": identity["positive_images"],
                "negative_scan_id": scan_id,
                "same_scene_hard_negative": hard,
                "ordinary_negative": ordinary,
            }
        )

    pixel.require(len(admitted) >= int(protocol["gate"]["minimum_admitted_training_identities"]), "INSUFFICIENT_GEOMETRY_ADMITTED_IDENTITIES")
    pixel.require(totals["rgb_members"] == 0 and totals["model_calls"] == 0, "FORBIDDEN_INFORMATION_ACCESS")
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "PRE_RGB_PRE_FEATURE_TARGET_DISJOINT_GEOMETRY_DEPTH_NEGATIVE_SELECTION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "manifest": manifest_row,
        "source_receipts": source_receipts,
        "candidate_view_rules": rules,
        "admitted_training_identity_count": len(admitted),
        "rejected_training_identity_count": len(rejected),
        "training_identities": admitted,
        "rejected_identities": rejected,
        "opened": totals,
        "heldout_reference_families": manifest["heldout"]["reference_families"],
        "head": manifest["head"],
        "next_action": protocol["next_action"],
        "conclusion": "L10_3RSCAN_INSTANCE_HEAD_NEGATIVE_GEOMETRY_SOURCE_EVALUABLE",
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
