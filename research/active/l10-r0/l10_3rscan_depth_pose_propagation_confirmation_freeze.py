#!/usr/bin/env python3
"""Freeze a fresh three-reference/two-adjacent-query propagation cohort."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = (
    "blindassist-l10-3rscan-depth-pose-propagation-confirmation-freeze-protocol-v1"
)


def _image(
    episode_id: str,
    role: str,
    scan_id: str,
    candidate: dict[str, Any],
    view: dict[str, Any],
) -> dict[str, Any]:
    return {
        "episode_id": episode_id,
        "role": role,
        "scan_id": scan_id,
        "target_instance_id": int(candidate["target_instance_id"]),
        "target_label": candidate["target_label"],
        "frame": int(view["frame"]),
        "color_size": view["color_size"],
        "bbox_xyxy": view["bbox_xyxy"],
        "zip_member": view["zip_member"],
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(
        pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    row = protocol["source_result"]
    pixel.require(
        pixel.sha256(HERE / row["path"]) == row["sha256"], "SOURCE_RESULT_HASH"
    )
    source = pixel.load_json(HERE / row["path"])
    pixel.require(source["source_evaluable"] is True, "SOURCE_NOT_EVALUABLE")
    pixel.require(source["rgb_members_opened"] == 0, "SOURCE_RGB_OPENED")
    pixel.require(source["model_calls"] == 0, "SOURCE_MODEL_CALLED")
    references = source["reference_memory"]["selected"]
    queries = source["query_pair"]["selected"]
    pixel.require(len(references) == 3 and len(queries) == 2, "MEMORY_CARDINALITY")
    candidate = source["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for manifest in protocol["source"]["sequence_zips"].values():
        path = artifact_root / manifest["path"]
        pixel.require(
            path.stat().st_size == int(manifest["bytes"]),
            f"SOURCE_BYTES:{manifest['path']}",
        )
        pixel.require(
            pixel.sha256(path) == manifest["sha256"],
            f"SOURCE_HASH:{manifest['path']}",
        )

    ids = [str(value) for value in protocol["selection"]["query_episode_ids"]]
    pixel.require(len(ids) == 2, "QUERY_ID_CARDINALITY")
    images: dict[str, dict[str, Any]] = {}
    for episode_id, reference, query in zip(ids, references[:2], queries):
        images[f"{episode_id}_reference"] = _image(
            episode_id,
            "reference",
            str(candidate["reference_scan_id"]),
            candidate,
            reference,
        )
        images[f"{episode_id}_query"] = _image(
            episode_id,
            "query",
            str(candidate["rescan_id"]),
            candidate,
            query,
        )
    extra_key = str(protocol["selection"]["extra_reference_image_key"])
    pixel.require(
        extra_key.startswith(ids[0] + "_") and extra_key.endswith("_reference"),
        "EXTRA_KEY_PREFIX",
    )
    images[extra_key] = _image(
        ids[0],
        "reference",
        str(candidate["reference_scan_id"]),
        candidate,
        references[2],
    )
    output = {
        "schema": "blindassist-l10-3rscan-depth-pose-propagation-confirmation-cohort-v1",
        "authority": "FROZEN_FRESH_FAMILY_PRE_RGB_PRE_MODEL_DEPTH_POSE_PROPAGATION_CONFIRMATION_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": pixel.sha256(Path(__file__)),
        },
        "artifact_root": str(artifact_root.resolve()),
        "source_manifest": protocol["source"]["sequence_zips"],
        "selection": {
            **protocol["selection"],
            "reference_frames": [int(value["frame"]) for value in references],
            "query_frames": [int(value["frame"]) for value in queries],
            "query_frame_gap": abs(
                int(queries[0]["frame"]) - int(queries[1]["frame"])
            ),
            "opened_rgb_members_before_freeze": 0,
            "model_calls_before_freeze": 0,
        },
        "images": images,
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.require(len(images) == 5, "IMAGE_CARDINALITY")
    pixel.atomic_write_json(output_path, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
