#!/usr/bin/env python3
"""Confirm the frozen adjacent-frame scale-vacancy set rule on one fresh family."""

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


PROTOCOL_SCHEMA = (
    "blindassist-l10-3rscan-temporal-scale-vacancy-confirmation-protocol-v1"
)


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
    for key in (
        "development_predecessor",
        "source_result",
        "cohort",
        "candidate_result",
    ):
        row = protocol[key]
        consensus.base.pixel.require(
            consensus.base.pixel.sha256(HERE / row["path"]) == row["sha256"],
            f"{key.upper()}_HASH",
        )
    predecessor = consensus.base.pixel.load_json(
        HERE / protocol["development_predecessor"]["path"]
    )
    consensus.base.pixel.require(
        predecessor["conclusion"]
        == protocol["development_predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    source = consensus.base.pixel.load_json(HERE / protocol["source_result"]["path"])
    consensus.base.pixel.require(source["source_evaluable"] is True, "SOURCE_NOT_EVALUABLE")
    cohort = consensus.base.pixel.load_json(HERE / protocol["cohort"]["path"])
    candidates_result = consensus.base.pixel.load_json(
        HERE / protocol["candidate_result"]["path"]
    )

    query_names = [str(value) for value in protocol["temporal_link"]["query_images"]]
    consensus.base.pixel.require(len(query_names) == 2, "QUERY_CARDINALITY")
    consensus.base.pixel.require(
        set(query_names) == set(candidates_result["query_receipts"]),
        "QUERY_ROSTER",
    )
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

    penalty = float(protocol["hypothesis_generation"]["scale_log_area_penalty"])
    fused = {
        query: consensus.base._best(
            receipt["ranked_candidates"],
            lambda row: float(row["layer18_nids_fused_score"]),
        )
        for query, receipt in candidates_result["query_receipts"].items()
    }
    reference_order = sorted(
        {
            reference
            for receipt in candidates_result["query_receipts"].values()
            for reference in receipt["ranked_candidates"][0]["per_reference_scores"]
        }
    )
    votes = [fused[query]["winning_target_reference"] for query in query_names]
    dominant_reference = sorted(
        set(votes),
        key=lambda reference: (-votes.count(reference), reference_order.index(reference)),
    )[0]

    first_candidates = candidates_result["query_receipts"][query_names[0]][
        "ranked_candidates"
    ]
    second_candidates = candidates_result["query_receipts"][query_names[1]][
        "ranked_candidates"
    ]
    minimum_link_iou = float(protocol["temporal_link"]["minimum_adjacent_box_iou"])
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
    temporal_candidates = {
        query_names[0]: first_candidates[selected_link["first_index"]],
        query_names[1]: second_candidates[selected_link["second_index"]],
    }

    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    maximum_size = int(protocol["decision_gate"]["maximum_set_size"])
    receipts: dict[str, Any] = {}
    proposal_opportunity = 0
    covered_queries = 0
    best_ious: list[float] = []
    set_sizes: list[int] = []
    temporal_fills = 0
    temporal_hits = 0
    temporal_incremental_hits = 0
    for query in query_names:
        source_receipt = candidates_result["query_receipts"][query]
        candidates = source_receipt["ranked_candidates"]
        proposal_opportunity += int(source_receipt["target_opportunity_at_iou_gate"])
        scale = consensus.base._best(
            candidates,
            lambda row: float(row["layer18_local_appearance_score"])
            - penalty * _scale_distance(row["box_xyxy"], width, height, reference_areas),
        )
        contributors = [
            ("semantic_local_fusion", fused[query]),
            ("reference_scale_mixture", scale),
        ]
        constrained = [
            row
            for row in candidates
            if row["winning_target_reference"] == dominant_reference
        ]
        if constrained:
            contributors.append(
                (
                    "cross_view_reference_consensus",
                    consensus.base._best(
                        constrained,
                        lambda row: float(row["layer18_nids_fused_score"]),
                    ),
                )
            )
        hypotheses: list[dict[str, Any]] = []
        for mechanism, candidate in contributors:
            key = _candidate_key(candidate)
            existing = next(
                (row for row in hypotheses if row["candidate_key"] == key), None
            )
            if existing is None:
                hypotheses.append(
                    {
                        "candidate_key": key,
                        "mechanisms": [mechanism],
                        "candidate": candidate,
                    }
                )
            else:
                existing["mechanisms"].append(mechanism)
        preserved_best_iou = max(
            _iou(row["candidate"]["box_xyxy"], cohort["images"][query]["bbox_xyxy"])
            for row in hypotheses
        )
        temporal = temporal_candidates[query]
        temporal_added = False
        if len(hypotheses) < maximum_size and not any(
            row["candidate_key"] == _candidate_key(temporal) for row in hypotheses
        ):
            hypotheses.append(
                {
                    "candidate_key": _candidate_key(temporal),
                    "mechanisms": ["temporal_reference_scale_vacancy"],
                    "candidate": temporal,
                }
            )
            temporal_added = True
            temporal_fills += 1
        consensus.base.pixel.require(len(hypotheses) <= maximum_size, f"SET_SIZE:{query}")

        truth = cohort["images"][query]["bbox_xyxy"]
        evaluated: list[dict[str, Any]] = []
        for hypothesis in hypotheses:
            candidate = deepcopy(hypothesis["candidate"])
            candidate["target_metrics_evaluation_only"] = {
                "iou": _iou(candidate["box_xyxy"], truth)
            }
            evaluated.append(
                {"mechanisms": hypothesis["mechanisms"], "candidate": candidate}
            )
        temporal_iou = _iou(temporal["box_xyxy"], truth)
        temporal_hits += int(temporal_added and temporal_iou >= minimum_iou)
        temporal_incremental_hits += int(
            temporal_added
            and preserved_best_iou < minimum_iou
            and temporal_iou >= minimum_iou
        )
        best_iou = max(
            float(row["candidate"]["target_metrics_evaluation_only"]["iou"])
            for row in evaluated
        )
        covered = best_iou >= minimum_iou
        covered_queries += int(covered)
        best_ious.append(best_iou)
        set_sizes.append(len(evaluated))
        receipts[query] = {
            "query_truth_used_for_hypothesis_generation": False,
            "dominant_reference": dominant_reference,
            "reference_area_fractions": sorted(reference_areas),
            "frame_gap": frame_gap,
            "selected_temporal_link": selected_link,
            "temporal_candidate_added": temporal_added,
            "preserved_best_iou_evaluation_only": preserved_best_iou,
            "temporal_candidate_iou_evaluation_only": temporal_iou,
            "hypothesis_count": len(evaluated),
            "hypotheses": evaluated,
            "best_hypothesis_iou_evaluation_only": best_iou,
            "target_covered_at_iou_gate": covered,
        }

    gate_met = covered_queries == int(
        protocol["decision_gate"]["required_covered_queries"]
    )
    result = {
        "schema": "blindassist-l10-3rscan-temporal-scale-vacancy-confirmation-result-v1",
        "authority": "FRESH_TENTH_FAMILY_TEMPORAL_SCALE_VACANCY_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": consensus.base.pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": consensus.base.pixel.sha256(Path(__file__)),
        },
        "conclusion": (
            "L10_3RSCAN_TEMPORAL_SCALE_VACANCY_FRESH_CONFIRMATION_GATE_MET"
            if gate_met
            else "L10_3RSCAN_TEMPORAL_SCALE_VACANCY_FRESH_CONFIRMATION_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "dominant_reference": dominant_reference,
        "semantic_local_fusion_reference_votes": votes,
        "hypothesis_generation": protocol["hypothesis_generation"],
        "temporal_link": {**protocol["temporal_link"], "selected": selected_link},
        "metrics": {
            "query_count": len(query_names),
            "proposal_opportunity_queries": proposal_opportunity,
            "covered_queries": covered_queries,
            "minimum_best_hypothesis_iou": min(best_ious),
            "mean_best_hypothesis_iou": sum(best_ious) / len(best_ious),
            "maximum_hypothesis_count": max(set_sizes),
            "mean_hypothesis_count": sum(set_sizes) / len(set_sizes),
            "temporal_vacancy_fills": temporal_fills,
            "temporal_vacancy_iou_gate_queries": temporal_hits,
            "temporal_vacancy_incremental_iou_gate_queries": temporal_incremental_hits,
        },
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
