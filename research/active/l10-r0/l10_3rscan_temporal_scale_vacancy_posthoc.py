#!/usr/bin/env python3
"""Fill one set vacancy with a scale-aware adjacent-frame proposal link."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cross_view_reference_consensus_posthoc as consensus  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-temporal-scale-vacancy-posthoc-protocol-v1"


def _iou(first: list[float], second: list[float]) -> float:
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _scale_distance(
    box: list[float], width: int, height: int, reference_areas: list[float]
) -> float:
    area = consensus.base._area_fraction(box, width, height)
    return min(abs(math.log(area / reference)) for reference in reference_areas)


def _candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    return (
        candidate.get("tile_id"),
        int(candidate.get("postprocess_index", -1)),
        tuple(candidate["box_xyxy"]),
    )


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = consensus.base.pixel.load_json(protocol_path)
    consensus.base.pixel.require(
        protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA"
    )
    consensus.base.pixel.require(
        consensus.base.pixel.sha256(Path(__file__))
        == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for dependency in protocol["dependencies"]:
        consensus.base.pixel.require(
            consensus.base.pixel.sha256(HERE / dependency["path"])
            == dependency["sha256"],
            f"DEPENDENCY_HASH:{dependency['path']}",
        )
    for key in ("predecessor", "candidate_result", "cohort"):
        row = protocol[key]
        consensus.base.pixel.require(
            consensus.base.pixel.sha256(HERE / row["path"]) == row["sha256"],
            f"{key.upper()}_HASH",
        )
    predecessor = consensus.base.pixel.load_json(HERE / protocol["predecessor"]["path"])
    consensus.base.pixel.require(
        predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    candidates_result = consensus.base.pixel.load_json(
        HERE / protocol["candidate_result"]["path"]
    )
    cohort = consensus.base.pixel.load_json(HERE / protocol["cohort"]["path"])
    query_names = [str(value) for value in protocol["temporal_link"]["query_images"]]
    consensus.base.pixel.require(len(query_names) == 2, "QUERY_CARDINALITY")
    first_image, second_image = (cohort["images"][name] for name in query_names)
    frame_gap = abs(int(first_image["frame"]) - int(second_image["frame"]))
    consensus.base.pixel.require(
        frame_gap <= int(protocol["temporal_link"]["maximum_frame_gap"]),
        "FRAME_GAP",
    )
    width, height = map(int, first_image["color_size"])
    consensus.base.pixel.require(
        [width, height] == second_image["color_size"], "IMAGE_SIZE_MISMATCH"
    )
    prefixes = {name.split("_")[0] for name in query_names}
    reference_areas = [
        consensus.base._area_fraction(row["bbox_xyxy"], *map(int, row["color_size"]))
        for key, row in cohort["images"].items()
        if key.endswith("_reference") and key.split("_")[0] in prefixes
    ]
    consensus.base.pixel.require(len(reference_areas) == 3, "REFERENCE_SCALE_COUNT")

    first_candidates = candidates_result["query_receipts"][query_names[0]]["ranked_candidates"]
    second_candidates = candidates_result["query_receipts"][query_names[1]]["ranked_candidates"]
    minimum_link_iou = float(protocol["temporal_link"]["minimum_adjacent_box_iou"])
    penalty = float(protocol["temporal_link"]["scale_log_area_penalty"])
    links: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    for first_index, first in enumerate(first_candidates):
        for second_index, second in enumerate(second_candidates):
            overlap = _iou(first["box_xyxy"], second["box_xyxy"])
            if overlap <= minimum_link_iou:
                continue
            mean_fused = (
                float(first["layer18_nids_fused_score"])
                + float(second["layer18_nids_fused_score"])
            ) / 2.0
            mean_scale_distance = (
                _scale_distance(first["box_xyxy"], width, height, reference_areas)
                + _scale_distance(second["box_xyxy"], width, height, reference_areas)
            ) / 2.0
            score = mean_fused - penalty * mean_scale_distance
            links.append(
                (
                    (score, overlap, mean_fused, -first_index, -second_index),
                    {
                        "first_index": first_index,
                        "second_index": second_index,
                        "adjacent_box_iou": overlap,
                        "mean_fused_score": mean_fused,
                        "mean_reference_scale_log_distance": mean_scale_distance,
                        "temporal_reference_scale_score": score,
                    },
                )
            )
    consensus.base.pixel.require(bool(links), "NO_ADJACENT_LINK")
    _, selected_link = max(links, key=lambda item: item[0])
    selected = {
        query_names[0]: first_candidates[selected_link["first_index"]],
        query_names[1]: second_candidates[selected_link["second_index"]],
    }

    family = protocol["predecessor"]["family_key"]
    previous_receipts = predecessor["families"][family]["query_receipts"]
    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    maximum_size = int(protocol["decision_gate"]["maximum_set_size"])
    receipts: dict[str, Any] = {}
    covered_queries = 0
    best_ious: list[float] = []
    temporal_fills = 0
    temporal_hits = 0
    axis_fallbacks = 0
    for query in query_names:
        prior = previous_receipts[query]
        preserved = [
            deepcopy(row)
            for row in prior["hypotheses"]
            if "reference_scale_axis_vacancy" not in row["mechanisms"]
        ]
        axis_rows = [
            deepcopy(row)
            for row in prior["hypotheses"]
            if "reference_scale_axis_vacancy" in row["mechanisms"]
        ]
        consensus.base.pixel.require(len(preserved) < maximum_size, f"NO_VACANCY:{query}")
        temporal = deepcopy(selected[query])
        temporal_key = _candidate_key(temporal)
        if not any(_candidate_key(row["candidate"]) == temporal_key for row in preserved):
            preserved.append(
                {"mechanisms": ["temporal_reference_scale_vacancy"], "candidate": temporal}
            )
            temporal_fills += 1
            temporal_hits += int(
                float(temporal["target_metrics_evaluation_only"]["iou"]) >= minimum_iou
            )
        elif axis_rows:
            preserved.append(axis_rows[0])
            axis_fallbacks += 1
        consensus.base.pixel.require(len(preserved) <= maximum_size, f"SET_SIZE:{query}")
        best_iou = max(
            float(row["candidate"]["target_metrics_evaluation_only"]["iou"])
            for row in preserved
        )
        covered = best_iou >= minimum_iou
        covered_queries += int(covered)
        best_ious.append(best_iou)
        receipts[query] = {
            "query_truth_used_for_hypothesis_generation": False,
            "frame_gap": frame_gap,
            "selected_temporal_link": selected_link,
            "hypothesis_count": len(preserved),
            "hypotheses": preserved,
            "best_hypothesis_iou_evaluation_only": best_iou,
            "target_covered_at_iou_gate": covered,
        }

    gate_met = covered_queries == int(protocol["decision_gate"]["required_covered_queries"])
    result = {
        "schema": "blindassist-l10-3rscan-temporal-scale-vacancy-posthoc-result-v1",
        "authority": "CONSUMED_NINTH_FAMILY_TEMPORAL_SCALE_VACANCY_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": consensus.base.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": consensus.base.pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_TEMPORAL_SCALE_VACANCY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_TEMPORAL_SCALE_VACANCY_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "query_count": len(query_names),
            "covered_queries": covered_queries,
            "minimum_best_hypothesis_iou": min(best_ious),
            "mean_best_hypothesis_iou": sum(best_ious) / len(best_ious),
            "maximum_hypothesis_count": max(row["hypothesis_count"] for row in receipts.values()),
            "temporal_vacancy_fills": temporal_fills,
            "temporal_vacancy_iou_gate_queries": temporal_hits,
            "axis_fallback_queries": axis_fallbacks,
        },
        "temporal_link": {**protocol["temporal_link"], "selected": selected_link},
        "query_receipts": receipts,
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    consensus.base.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
