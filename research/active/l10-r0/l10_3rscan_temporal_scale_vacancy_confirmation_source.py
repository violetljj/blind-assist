#!/usr/bin/env python3
"""Admit a fresh family with three references and one adjacent query pair."""

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
import l10_3rscan_multiview_observation_portfolio_posthoc as portfolio  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = (
    "blindassist-l10-3rscan-temporal-scale-vacancy-confirmation-source-protocol-v1"
)


def _unit_view(pose: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    direction = centroid - pose[:3, 3]
    norm = float(np.linalg.norm(direction))
    pixel.require(norm > 0.0, "ZERO_VIEW_DIRECTION")
    return direction / norm


def _fill_reference(
    points: np.ndarray,
    candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray]],
    base_selected: list[dict[str, Any]],
    required: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = deepcopy(base_selected)
    fills: list[dict[str, Any]] = []
    by_frame = {int(row["frame"]): (row, mask, pose) for row, mask, pose in candidates}
    selected_frames = {int(row["frame"]) for row in selected}
    if not selected_frames <= set(by_frame):
        return selected, fills
    centroid = np.mean(points, axis=0)
    covered = np.zeros(len(points), dtype=bool)
    for frame in selected_frames:
        covered |= by_frame[frame][1]
    while len(selected) < required:
        selected_vectors = [
            _unit_view(by_frame[int(row["frame"])][2], centroid) for row in selected
        ]
        remaining = [item for item in candidates if int(item[0]["frame"]) not in selected_frames]
        if not remaining or not selected_vectors:
            break

        def rank(item: tuple[dict[str, Any], np.ndarray, np.ndarray]) -> tuple[float, ...]:
            row, _, pose = item
            vector = _unit_view(pose, centroid)
            maximum_cosine = max(float(np.dot(vector, value)) for value in selected_vectors)
            return (
                -maximum_cosine,
                float(row["visible_target_vertices"]),
                float(row["bbox_short_side_fraction"]),
                float(row["depth_visible_ratio"]),
                -float(row["frame"]),
            )

        row, mask, pose = max(remaining, key=rank)
        vector = _unit_view(pose, centroid)
        maximum_cosine = max(float(np.dot(vector, value)) for value in selected_vectors)
        minimum_angle = math.degrees(math.acos(float(np.clip(maximum_cosine, -1.0, 1.0))))
        marginal = int(np.count_nonzero(mask & ~covered))
        covered |= mask
        filled = {
            **deepcopy(row),
            "marginal_visible_target_vertices": marginal,
            "cumulative_visible_target_vertices": int(np.count_nonzero(covered)),
            "cumulative_visible_target_fraction": float(np.count_nonzero(covered) / len(points)),
            "selection_mode": "MAXIMUM_MINIMUM_VIEW_ANGLE_REFERENCE_VACANCY_FILL",
            "minimum_angular_separation_degrees": minimum_angle,
        }
        selected.append(filled)
        fills.append(
            {
                "frame": int(row["frame"]),
                "marginal_visible_target_vertices": marginal,
                "minimum_angular_separation_degrees": minimum_angle,
            }
        )
        selected_frames.add(int(row["frame"]))
    return selected, fills


def _adjacent_pair(
    candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray]], maximum_gap: int
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    pairs: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    for first_index, (first, _, _) in enumerate(candidates):
        for second_index, (second, _, _) in enumerate(candidates[first_index + 1 :], start=first_index + 1):
            gap = abs(int(first["frame"]) - int(second["frame"]))
            if gap == 0 or gap > maximum_gap:
                continue
            ordered = sorted((first, second), key=lambda row: int(row["frame"]))
            score = (
                float(min(row["visible_target_vertices"] for row in ordered)),
                float(sum(row["visible_target_vertices"] for row in ordered)),
                float(min(row["bbox_short_side_fraction"] for row in ordered)),
                float(min(row["depth_visible_ratio"] for row in ordered)),
                -float(gap),
                -float(ordered[0]["frame"]),
                -float(ordered[1]["frame"]),
            )
            pairs.append((score, ordered[0], ordered[1]))
    if not pairs:
        return [], None
    _, first, second = max(pairs, key=lambda item: item[0])
    return [deepcopy(first), deepcopy(second)], {
        "frames": [int(first["frame"]), int(second["frame"])],
        "frame_gap": abs(int(first["frame"]) - int(second["frame"])),
        "minimum_visible_target_vertices": min(
            int(first["visible_target_vertices"]), int(second["visible_target_vertices"])
        ),
        "sum_visible_target_vertices": int(first["visible_target_vertices"])
        + int(second["visible_target_vertices"]),
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(
        pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    candidate_row = protocol["candidate"]
    pixel.require(
        pixel.sha256(HERE / candidate_row["path"]) == candidate_row["sha256"],
        "CANDIDATE_HASH",
    )
    candidate = pixel.load_json(HERE / candidate_row["path"])["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    for row in protocol["source"]["files"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")
    data_root = artifact_root / "datasets/3rscan"
    target_id = int(candidate["target_instance_id"])
    rules = protocol["candidate_view_rules"]
    reference_base = portfolio._portfolio(
        data_root,
        str(candidate["reference_scan_id"]),
        target_id,
        rules,
        int(protocol["reference_memory"]["base_positive_marginal_budget"]),
    )
    reference_points, reference_candidates, reference_opened = views._candidates(
        data_root, str(candidate["reference_scan_id"]), target_id, rules
    )
    reference_selected, vacancy_fills = _fill_reference(
        reference_points,
        reference_candidates,
        reference_base["selected"],
        int(protocol["reference_memory"]["required_views"]),
    )
    _, query_candidates, query_opened = views._candidates(
        data_root, str(candidate["rescan_id"]), target_id, rules
    )
    query_selected, query_pair = _adjacent_pair(
        query_candidates, int(protocol["query_pair"]["maximum_frame_gap"])
    )
    sibling = None
    for sibling_id in candidate["rescan_door_instance_ids"]:
        sibling_id = int(sibling_id)
        if sibling_id == target_id:
            continue
        points, candidates, opened = views._candidates(
            data_root, str(candidate["rescan_id"]), sibling_id, rules
        )
        if not candidates:
            continue
        selected = max(
            (row for row, _, _ in candidates),
            key=lambda row: (
                int(row["visible_target_vertices"]),
                float(row["bbox_short_side_fraction"]),
                float(row["depth_visible_ratio"]),
                -int(row["frame"]),
            ),
        )
        sibling = {
            "instance_id": sibling_id,
            "label": "door_or_doorframe",
            "target_vertices": int(len(points)),
            "candidate_views": len(candidates),
            "selected": selected,
            "opened": opened,
        }
        break
    evaluable = (
        len(reference_base["selected"])
        >= int(protocol["reference_memory"]["minimum_positive_marginal_views"])
        and len(reference_selected) == int(protocol["reference_memory"]["required_views"])
        and len(query_selected) == 2
        and sibling is not None
    )
    result = {
        "schema": "blindassist-l10-3rscan-temporal-scale-vacancy-confirmation-source-result-v1",
        "authority": "FRESH_FAMILY_PRE_RGB_PRE_MODEL_TEMPORAL_SCALE_VACANCY_SOURCE",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "candidate": candidate,
        "conclusion": (
            "L10_3RSCAN_TEMPORAL_SCALE_VACANCY_CONFIRMATION_SOURCE_EVALUABLE"
            if evaluable
            else "L10_3RSCAN_TEMPORAL_SCALE_VACANCY_CONFIRMATION_SOURCE_NOT_EVALUABLE"
        ),
        "source_evaluable": evaluable,
        "reference_memory": {
            **reference_base,
            "eligible_admitted_views": len(reference_candidates),
            "required_views": int(protocol["reference_memory"]["required_views"]),
            "selected": reference_selected,
            "vacancy_fills": vacancy_fills,
            "opened": reference_opened,
        },
        "query_pair": {
            "scan_id": str(candidate["rescan_id"]),
            "target_instance_id": target_id,
            "eligible_admitted_views": len(query_candidates),
            "selected": query_selected,
            "pair_receipt": query_pair,
            "opened": query_opened,
        },
        "same_scene_sibling": sibling,
        "rgb_members_opened": 0,
        "model_calls": 0,
        "next_action": protocol["next_action"]["evaluable" if evaluable else "not_evaluable"],
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
