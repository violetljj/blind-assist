#!/usr/bin/env python3
"""Audit an 80-percent containment extent-completion rule on two consumed families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cross-scale-extent-completion-preservation-audit-protocol-v1"


def _area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )


def _containment(inner: list[float], outer: list[float]) -> float:
    intersection_width = max(
        0.0, min(float(inner[2]), float(outer[2])) - max(float(inner[0]), float(outer[0]))
    )
    intersection_height = max(
        0.0, min(float(inner[3]), float(outer[3])) - max(float(inner[1]), float(outer[1]))
    )
    return intersection_width * intersection_height / _area(inner)


def _rank(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda row: (
            -row["layer18_local_appearance_score"],
            -row["target_memory_score"],
            -row["objectness_score"],
            *row["box_xyxy"],
            row["postprocess_index"],
        ),
    )


def _select(
    candidates: list[dict[str, Any]], minimum_containment: float
) -> tuple[dict[str, Any], bool, float]:
    ranked = _rank(candidates)
    anchor = ranked[0]
    if anchor["tile_id"] == "full":
        return anchor, False, 1.0
    completions = [
        (row, _containment(anchor["box_xyxy"], row["box_xyxy"]))
        for row in ranked
        if row["tile_id"] == "full"
        and row["winning_target_reference"] == anchor["winning_target_reference"]
    ]
    completions = [row for row in completions if row[1] >= minimum_containment]
    if not completions:
        return anchor, False, 0.0
    return completions[0][0], True, float(completions[0][1])


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    loaded: dict[str, Any] = {}
    for key, row in protocol["inputs"].items():
        path = HERE / row["path"]
        pixel.require(pixel.sha256(path) == row["sha256"], f"INPUT_HASH:{key}")
        loaded[key] = pixel.load_json(path)
    threshold = float(protocol["selection"]["minimum_anchor_containment"])
    family_specs = {
        "fresh_door8": (loaded["fresh_confirmation"], loaded["fresh_extent_result"]),
        "consumed_door15": (loaded["d15_candidates"], loaded["d15_local_only_result"]),
    }
    family_receipts: dict[str, Any] = {}
    all_preserved = True
    for family, (candidates_result, expected_result) in family_specs.items():
        receipts: dict[str, Any] = {}
        for query, expected in expected_result["query_receipts"].items():
            candidate_receipt = candidates_result["query_receipts"][query]
            candidates = candidate_receipt.get("ranked_candidates")
            pixel.require(candidates is not None, f"RANKED_CANDIDATES:{family}:{query}")
            selected, applied, containment = _select(candidates, threshold)
            preserved = (
                selected["box_xyxy"] == expected["selected"]["box_xyxy"]
                and int(selected["postprocess_index"]) == int(expected["selected"]["postprocess_index"])
            )
            all_preserved = all_preserved and preserved
            receipts[query] = {
                "extent_completion_applied": applied,
                "anchor_containment": containment,
                "selected": selected,
                "matches_sealed_selected_candidate": preserved,
                "sealed_refined_iou": expected["refined_mask_bbox_target_metrics_evaluation_only"]["iou"],
            }
        family_receipts[family] = {
            "all_selected_candidates_preserved": all(row["matches_sealed_selected_candidate"] for row in receipts.values()),
            "sealed_metrics": expected_result["metrics"],
            "query_receipts": receipts,
        }
    result = {
        "schema": "blindassist-l10-3rscan-cross-scale-extent-completion-preservation-audit-result-v1",
        "authority": "CONSUMED_TWO_FAMILY_DETERMINISTIC_PRESERVATION_AUDIT_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_CROSS_SCALE_EXTENT_COMPLETION_TWO_FAMILY_PRESERVATION_GATE_MET"
            if all_preserved
            else "L10_3RSCAN_CROSS_SCALE_EXTENT_COMPLETION_TWO_FAMILY_PRESERVATION_GATE_NOT_MET"
        ),
        "gate_met": all_preserved,
        "selection": protocol["selection"],
        "families": family_receipts,
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
