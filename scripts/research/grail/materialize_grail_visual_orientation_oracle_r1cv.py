#!/usr/bin/env python3
"""Materialize evaluator-only native coordinates for GRAIL-R1C-V."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from collect_grail_m1 import start_controller
from grail_canonical_coordinates_r1c import canonicalize_scene
from grail_procthor_native_m0 import sha256_file
from grail_relational_r0 import load_houses


def materialize(dataset: Path, collection_path: Path, reference_supplement_path: Path,
                docker_image_id: str, dockerfile_sha256: str) -> dict[str, Any]:
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_supplement_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != collection["dataset_sha256"]:
        raise ValueError("R1C-V dataset/collection identity mismatch")
    if reference["dataset_sha256"] != sha256_file(dataset):
        raise ValueError("R1C-V reference/dataset identity mismatch")
    if reference["collection_sha256"] != sha256_file(collection_path):
        raise ValueError("R1C-V reference/collection identity mismatch")
    rows = collection["rows"]
    reference_rows = reference["rows"]
    if len(rows) != 78 or [row["sample_id"] for row in rows] != [row["sample_id"] for row in reference_rows]:
        raise ValueError("R1C-V requires the frozen aligned 78-case cohort")
    house_indices = {int(row["house_index"]) for row in rows}
    houses = load_houses(dataset, house_indices)
    required: dict[int, set[str]] = {house_index: set() for house_index in house_indices}
    for query_row, reference_row in zip(rows, reference_rows):
        house_index = int(query_row["house_index"])
        required[house_index].update(candidate["object_id"] for candidate in query_row["candidates"])
        required[house_index].update(candidate["object_id"] for candidate in reference_row["candidates"])

    scenes: dict[str, dict[str, dict[str, Any]]] = {}
    receipts = []
    controller = None
    try:
        for house_index in sorted(house_indices):
            event = controller.reset(scene=houses[house_index]) if controller else None
            if controller is None:
                controller = start_controller(houses[house_index])
                event = controller.last_event
            if not event.metadata.get("lastActionSuccess"):
                raise RuntimeError(f"scene reset failed {house_index}")
            canonical = canonicalize_scene(event.metadata.get("objects", []))
            missing = sorted(required[house_index] - set(canonical))
            if missing:
                raise ValueError(f"runtime metadata lacks R1C-V candidates in house {house_index}: {missing}")
            scenes[str(house_index)] = {object_id: canonical[object_id] for object_id in sorted(required[house_index])}
            receipts.append({
                "house_index": house_index,
                "required_objects": len(required[house_index]),
                "evaluable_objects": sum(canonical[object_id]["evaluable"] for object_id in required[house_index]),
            })
    finally:
        if controller is not None:
            controller.stop()
    return {
        "schema": "blindassist_grail_r1c_v_evaluator_native_coordinate_oracle_v1",
        "role": "EVALUATOR_ONLY_FORBIDDEN_TO_PREDICTOR",
        "runtime": {
            "docker_image_id": docker_image_id,
            "dockerfile_sha256": dockerfile_sha256,
            "ai2thor_release": "f0825767cd50d69f666c7f282e54abfe58f1e917",
        },
        "dataset_sha256": sha256_file(dataset),
        "collection_sha256": sha256_file(collection_path),
        "reference_supplement_sha256": sha256_file(reference_supplement_path),
        "scene_receipts": receipts,
        "scenes": scenes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--reference-supplement", type=Path, required=True)
    parser.add_argument("--docker-image-id", required=True)
    parser.add_argument("--dockerfile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize(args.dataset, args.collection, args.reference_supplement,
                         args.docker_image_id, args.dockerfile_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"scene_receipts": result["scene_receipts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

