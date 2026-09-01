#!/usr/bin/env python3
"""Freeze a three-reference/two-query Development cohort without RGB access."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_joint_covisibility_selector_posthoc as views  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = (
    "blindassist-l10-3rscan-asymmetric-reference-vacancy-freeze-protocol-v1"
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


def _unit_view(pose: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    direction = centroid - pose[:3, 3]
    norm = float(np.linalg.norm(direction))
    pixel.require(norm > 0.0, "ZERO_VIEW_DIRECTION")
    return direction / norm


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(
        pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for key in ("source_result", "source_protocol"):
        row = protocol[key]
        pixel.require(
            pixel.sha256(HERE / row["path"]) == row["sha256"],
            f"{key.upper()}_HASH",
        )
    source = pixel.load_json(HERE / protocol["source_result"]["path"])
    source_protocol = pixel.load_json(HERE / protocol["source_protocol"]["path"])
    pixel.require(source["source_evaluable"] is True, "SOURCE_NOT_EVALUABLE")
    pixel.require(source["rgb_members_opened"] == 0, "SOURCE_RGB_OPENED")
    pixel.require(source["model_calls"] == 0, "SOURCE_MODEL_CALLED")
    candidate = source["candidate"]
    references = deepcopy(source["reference_memory"]["selected"])
    queries = deepcopy(source["query_memory"]["selected"])
    pixel.require(len(references) == len(queries) == 2, "BASE_MEMORY_CARDINALITY")

    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["sequence_zips"].values():
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")

    scan_id = str(candidate["reference_scan_id"])
    target_id = int(candidate["target_instance_id"])
    points, candidates, opened = views._candidates(
        artifact_root / "datasets/3rscan",
        scan_id,
        target_id,
        source_protocol["candidate_view_rules"],
    )
    by_frame = {int(row["frame"]): (row, mask, pose) for row, mask, pose in candidates}
    selected_frames = {int(row["frame"]) for row in references}
    pixel.require(selected_frames <= set(by_frame), "SELECTED_REFERENCE_NOT_ADMITTED")
    centroid = np.mean(points, axis=0)
    selected_vectors = [
        _unit_view(by_frame[frame][2], centroid) for frame in sorted(selected_frames)
    ]
    remaining = [item for item in candidates if int(item[0]["frame"]) not in selected_frames]
    pixel.require(bool(remaining), "NO_REFERENCE_VACANCY_CANDIDATE")

    def rank(item: tuple[dict[str, Any], np.ndarray, np.ndarray]) -> tuple[float, ...]:
        row, _, pose = item
        vector = _unit_view(pose, centroid)
        maximum_cosine = max(float(np.dot(vector, selected)) for selected in selected_vectors)
        return (
            -maximum_cosine,
            float(row["visible_target_vertices"]),
            float(row["bbox_short_side_fraction"]),
            float(row["depth_visible_ratio"]),
            -float(row["frame"]),
        )

    row, mask, pose = max(remaining, key=rank)
    vector = _unit_view(pose, centroid)
    maximum_cosine = max(float(np.dot(vector, selected)) for selected in selected_vectors)
    minimum_angle = math.degrees(math.acos(float(np.clip(maximum_cosine, -1.0, 1.0))))
    covered = np.zeros(len(points), dtype=bool)
    for frame in selected_frames:
        covered |= by_frame[frame][1]
    marginal = int(np.count_nonzero(mask & ~covered))
    covered |= mask
    vacancy_reference = {
        **deepcopy(row),
        "marginal_visible_target_vertices": marginal,
        "cumulative_visible_target_vertices": int(np.count_nonzero(covered)),
        "cumulative_visible_target_fraction": float(np.count_nonzero(covered) / len(points)),
        "selection_mode": "MAXIMUM_MINIMUM_VIEW_ANGLE_REFERENCE_VACANCY_FILL",
        "minimum_angular_separation_degrees": minimum_angle,
    }

    ids = [str(value) for value in protocol["selection"]["query_episode_ids"]]
    pixel.require(len(ids) == 2, "QUERY_ID_CARDINALITY")
    images: dict[str, dict[str, Any]] = {}
    for episode_id, reference, query in zip(ids, references, queries):
        images[f"{episode_id}_reference"] = _image(
            episode_id, "reference", scan_id, candidate, reference
        )
        images[f"{episode_id}_query"] = _image(
            episode_id, "query", str(candidate["rescan_id"]), candidate, query
        )
    extra_key = str(protocol["selection"]["extra_reference_image_key"])
    pixel.require(extra_key.startswith(ids[0] + "_") and extra_key.endswith("_reference"), "EXTRA_KEY_PREFIX")
    images[extra_key] = _image(ids[0], "reference", scan_id, candidate, vacancy_reference)

    output = {
        "schema": "blindassist-l10-3rscan-asymmetric-reference-vacancy-cohort-v1",
        "authority": "CONSUMED_PRE_RGB_PRE_MODEL_ASYMMETRIC_REFERENCE_VACANCY_DEVELOPMENT_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "artifact_root": str(artifact_root.resolve()),
        "source_manifest": protocol["source"]["sequence_zips"],
        "selection": {
            **protocol["selection"],
            "reference_memory_count": 3,
            "unique_query_count": 2,
            "base_reference_frames": sorted(selected_frames),
            "vacancy_reference_frame": int(vacancy_reference["frame"]),
            "vacancy_reference_marginal_vertices": marginal,
            "vacancy_reference_minimum_angular_separation_degrees": minimum_angle,
            "opened": opened,
            "opened_rgb_members_before_freeze": 0,
            "model_calls_before_freeze": 0,
        },
        "images": images,
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.require(len(images) == 5, "IMAGE_CARDINALITY")
    pixel.atomic_write_json(output_path, output)
    print(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
