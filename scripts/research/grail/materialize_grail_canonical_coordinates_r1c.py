#!/usr/bin/env python3
"""Materialize AI2-THOR native owner frames for the frozen GRAIL-R1C-O cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from collect_grail_m1 import start_controller
from grail_canonical_coordinates_r1c import canonicalize_scene
from grail_procthor_native_m0 import sha256_file
from grail_relational_r0 import load_houses


def materialize(dataset: Path, collection_path: Path, docker_image_id: str,
                dockerfile_sha256: str) -> dict[str, Any]:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != collection["dataset_sha256"]:
        raise ValueError("R1C-O dataset/collection identity mismatch")
    rows = collection["rows"]
    if len(rows) != 78:
        raise ValueError(f"R1C-O requires frozen 78-case Development cohort, got {len(rows)}")
    house_indices = {int(row["house_index"]) for row in rows}
    houses = load_houses(dataset, house_indices)
    scenes: dict[int, dict[str, dict[str, Any]]] = {}
    scene_receipts = []
    controller = None
    try:
        for house_index in sorted(house_indices):
            event = controller.reset(scene=houses[house_index]) if controller else None
            if controller is None:
                controller = start_controller(houses[house_index])
                event = controller.last_event
            if not event.metadata.get("lastActionSuccess"):
                raise RuntimeError(f"scene reset failed {house_index}")
            objects = event.metadata.get("objects", [])
            canonical = canonicalize_scene(objects)
            scenes[house_index] = canonical
            scene_receipts.append({
                "house_index": house_index,
                "runtime_objects": len(objects),
                "canonical_evaluable": sum(value["evaluable"] for value in canonical.values()),
                "canonical_not_evaluable": sum(not value["evaluable"] for value in canonical.values()),
            })
    finally:
        if controller is not None:
            controller.stop()

    output_rows = []
    for row in rows:
        canonical = scenes[int(row["house_index"])]
        candidate_ids = [candidate["object_id"] for candidate in row["candidates"]]
        missing = [object_id for object_id in candidate_ids if object_id not in canonical]
        if missing:
            raise ValueError(f"runtime metadata lacks frozen candidates for {row['sample_id']}: {missing}")
        output_rows.append({
            "sample_id": row["sample_id"],
            "house_index": int(row["house_index"]),
            "target_object_id": row["target_object_id"],
            "candidates": {object_id: canonical[object_id] for object_id in candidate_ids},
        })
    return {
        "schema": "blindassist_grail_r1c_privileged_owner_local_coordinates_v1",
        "mode": "PROJECT_CONSUMED_DEVELOPMENT_PRIVILEGED_NATIVE_METADATA",
        "coordinate_contract": {
            "owner": "native component prefix, else native parentReceptacles, else standalone self",
            "frame": "inverse owner native yaw; axes=(right, up, front); no camera pose or pixels",
            "slot": "rank over all runtime siblings sharing native owner and semantic type",
            "horizontal": "local right -> LEFT/CENTER/RIGHT",
            "vertical": "negative local up -> TOP/MIDDLE/BOTTOM",
        },
        "runtime": {
            "docker_image_id": docker_image_id,
            "dockerfile_sha256": dockerfile_sha256,
            "ai2thor_release": "f0825767cd50d69f666c7f282e54abfe58f1e917",
            "platform": "Linux64/Xvfb/Mesa software GL/FIFO",
        },
        "dataset_sha256": sha256_file(dataset),
        "collection_sha256": sha256_file(collection_path),
        "scene_receipts": scene_receipts,
        "rows": output_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--dockerfile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.dataset, args.collection, args.docker_image_id, args.dockerfile_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"scene_receipts": result["scene_receipts"], "rows": len(result["rows"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

