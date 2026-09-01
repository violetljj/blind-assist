#!/usr/bin/env python3
"""Admit a fresh family with three references and one consecutive query triplet."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402
import l10_3rscan_temporal_scale_vacancy_confirmation_source as old  # noqa: E402


PROTOCOL_SCHEMA = (
    "blindassist-l10-3rscan-discriminative-view-confirmation-source-protocol-v1"
)


def _consecutive_triplet(
    candidates: list[tuple[dict[str, Any], Any, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    by_frame = {int(row["frame"]): row for row, _, _ in candidates}
    ranked: list[tuple[tuple[float, ...], list[dict[str, Any]]]] = []
    for first_frame in sorted(by_frame):
        frames = [first_frame, first_frame + 1, first_frame + 2]
        if not all(frame in by_frame for frame in frames):
            continue
        rows = [by_frame[frame] for frame in frames]
        score = (
            float(min(row["visible_target_vertices"] for row in rows)),
            float(sum(row["visible_target_vertices"] for row in rows)),
            float(min(row["bbox_short_side_fraction"] for row in rows)),
            float(min(row["depth_visible_ratio"] for row in rows)),
            -float(first_frame),
        )
        ranked.append((score, rows))
    if not ranked:
        return [], None
    _, selected = max(ranked, key=lambda item: item[0])
    return [deepcopy(row) for row in selected], {
        "frames": [int(row["frame"]) for row in selected],
        "minimum_visible_target_vertices": min(
            int(row["visible_target_vertices"]) for row in selected
        ),
        "sum_visible_target_vertices": sum(
            int(row["visible_target_vertices"]) for row in selected
        ),
        "minimum_bbox_short_side_fraction": min(
            float(row["bbox_short_side_fraction"]) for row in selected
        ),
        "minimum_depth_visible_ratio": min(
            float(row["depth_visible_ratio"]) for row in selected
        ),
        "initial_frames": [int(selected[0]["frame"]), int(selected[1]["frame"])],
        "fixed_action": "NEXT_FRAME_FORWARD",
        "action_frame": int(selected[2]["frame"]),
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(
        pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for dependency in protocol["dependencies"]:
        pixel.require(
            pixel.sha256(HERE / dependency["path"]) == dependency["sha256"],
            f"DEPENDENCY_HASH:{dependency['path']}",
        )
    candidate_row = protocol["candidate"]
    pixel.require(
        pixel.sha256(HERE / candidate_row["path"]) == candidate_row["sha256"],
        "CANDIDATE_HASH",
    )
    candidate = pixel.load_json(HERE / candidate_row["path"])["candidate"]
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    source_manifest: dict[str, Any] = {}
    for row in protocol["source"]["files"]:
        path = artifact_root / row["path"]
        pixel.require(path.stat().st_size == int(row["bytes"]), f"SOURCE_BYTES:{row['path']}")
        pixel.require(pixel.sha256(path) == row["sha256"], f"SOURCE_HASH:{row['path']}")
        if row["path"].endswith("sequence.zip"):
            scan_id = Path(row["path"]).parent.name
            source_manifest[f"{scan_id}/sequence.zip"] = {
                "path": row["path"],
                "bytes": int(row["bytes"]),
                "sha256": row["sha256"],
            }

    data_root = artifact_root / "datasets/3rscan"
    target_id = int(candidate["target_instance_id"])
    rules = protocol["candidate_view_rules"]
    reference_base = old.portfolio._portfolio(
        data_root,
        str(candidate["reference_scan_id"]),
        target_id,
        rules,
        int(protocol["reference_memory"]["base_positive_marginal_budget"]),
    )
    reference_points, reference_candidates, reference_opened = old.views._candidates(
        data_root, str(candidate["reference_scan_id"]), target_id, rules
    )
    reference_selected, vacancy_fills = old._fill_reference(
        reference_points,
        reference_candidates,
        reference_base["selected"],
        int(protocol["reference_memory"]["required_views"]),
    )
    _, query_candidates, query_opened = old.views._candidates(
        data_root, str(candidate["rescan_id"]), target_id, rules
    )
    query_selected, query_triplet = _consecutive_triplet(query_candidates)

    sibling = None
    for sibling_id in candidate["rescan_door_instance_ids"]:
        sibling_id = int(sibling_id)
        if sibling_id == target_id:
            continue
        points, candidates, opened = old.views._candidates(
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
        and len(query_selected) == 3
        and sibling is not None
    )
    result = {
        "schema": "blindassist-l10-3rscan-discriminative-view-confirmation-source-result-v1",
        "authority": "FRESH_TWELFTH_FAMILY_PRE_RGB_PRE_MODEL_QUERY_TRIPLET_SOURCE",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": pixel.sha256(Path(__file__)),
        },
        "candidate": candidate,
        "source_manifest": source_manifest,
        "conclusion": (
            "L10_3RSCAN_DISCRIMINATIVE_VIEW_CONFIRMATION_SOURCE_EVALUABLE"
            if evaluable
            else "L10_3RSCAN_DISCRIMINATIVE_VIEW_CONFIRMATION_SOURCE_NOT_EVALUABLE"
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
        "query_triplet": {
            "scan_id": str(candidate["rescan_id"]),
            "target_instance_id": target_id,
            "eligible_admitted_views": len(query_candidates),
            "selected": query_selected,
            "triplet_receipt": query_triplet,
            "opened": query_opened,
        },
        "same_scene_sibling": sibling,
        "rgb_members_opened": 0,
        "model_calls": 0,
        "next_action": protocol["next_action"]["evaluable" if evaluable else "not_evaluable"],
        "literature_motivation": protocol["literature_motivation"],
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
