#!/usr/bin/env python3
"""Fill unused consensus-set capacity with one reference-scale axis completion."""

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
import l10_3rscan_reference_scale_axis_completion_posthoc as axis  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-axis-completion-vacancy-posthoc-protocol-v1"


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
    predecessor = protocol["predecessor"]
    predecessor_path = HERE / predecessor["path"]
    consensus.base.pixel.require(
        consensus.base.pixel.sha256(predecessor_path) == predecessor["sha256"],
        "PREDECESSOR_HASH",
    )
    consensus.base.pixel.require(
        consensus.base.pixel.load_json(predecessor_path)["conclusion"]
        == predecessor["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )

    penalty = float(protocol["hypothesis_generation"]["scale_log_area_penalty"])
    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    maximum_size = int(protocol["decision_gate"]["maximum_set_size"])
    family_receipts: dict[str, Any] = {}
    query_count = 0
    covered_queries = 0
    set_sizes: list[int] = []
    best_ious: list[float] = []
    vacancy_available = 0
    vacancy_filled = 0
    axis_hits = 0
    completion_modes = {"identity": 0, "horizontal_band": 0, "vertical_band": 0}

    for family, spec in protocol["families"].items():
        candidate_path = HERE / spec["candidate_result"]["path"]
        cohort_path = HERE / spec["cohort"]["path"]
        consensus.base.pixel.require(
            consensus.base.pixel.sha256(candidate_path)
            == spec["candidate_result"]["sha256"],
            f"RESULT_HASH:{family}",
        )
        consensus.base.pixel.require(
            consensus.base.pixel.sha256(cohort_path) == spec["cohort"]["sha256"],
            f"COHORT_HASH:{family}",
        )
        candidates_result = consensus.base.pixel.load_json(candidate_path)
        cohort = consensus.base.pixel.load_json(cohort_path)
        query_prefixes = {
            key.split("_")[0] for key in candidates_result["query_receipts"]
        }
        reference_areas: list[float] = []
        for key, row in cohort["images"].items():
            if key.endswith("_reference") and key.split("_")[0] in query_prefixes:
                width, height = map(int, row["color_size"])
                reference_areas.append(
                    consensus.base._area_fraction(row["bbox_xyxy"], width, height)
                )
        consensus.base.pixel.require(
            len(reference_areas) == 3, f"REFERENCE_SCALE_COUNT:{family}"
        )

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
        votes = [row["winning_target_reference"] for row in fused.values()]
        dominant_reference = sorted(
            set(votes),
            key=lambda reference: (
                -votes.count(reference),
                reference_order.index(reference),
            ),
        )[0]

        receipts: dict[str, Any] = {}
        family_covered = 0
        for query, source_receipt in candidates_result["query_receipts"].items():
            candidates = source_receipt["ranked_candidates"]
            image_row = cohort["images"][query]
            width, height = map(int, image_row["color_size"])
            scale = consensus.base._best(
                candidates,
                lambda row: float(row["layer18_local_appearance_score"])
                - penalty
                * min(
                    abs(
                        math.log(
                            consensus.base._area_fraction(
                                row["box_xyxy"], width, height
                            )
                            / reference_area
                        )
                    )
                    for reference_area in reference_areas
                ),
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
                key = (
                    candidate["tile_id"],
                    int(candidate["postprocess_index"]),
                    tuple(candidate["box_xyxy"]),
                )
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

            axis_candidate = None
            had_vacancy = len(hypotheses) < maximum_size
            if had_vacancy:
                vacancy_available += 1
                proposed_axis = axis._axis_best(
                    candidates, width, height, reference_areas, penalty
                )
                if not any(
                    tuple(row["candidate"]["box_xyxy"])
                    == tuple(proposed_axis["box_xyxy"])
                    for row in hypotheses
                ):
                    axis_candidate = proposed_axis
                    hypotheses.append(
                        {
                            "candidate_key": (
                                "axis_completion",
                                tuple(proposed_axis["box_xyxy"]),
                            ),
                            "mechanisms": ["reference_scale_axis_vacancy"],
                            "candidate": proposed_axis,
                        }
                    )
                    vacancy_filled += 1
                    completion_modes[proposed_axis["completion_mode"]] += 1
            consensus.base.pixel.require(
                len(hypotheses) <= maximum_size,
                f"SET_SIZE:{family}:{query}",
            )

            # Truth enters only after the preserved set and optional vacancy fill are fixed.
            truth = image_row["bbox_xyxy"]
            evaluated: list[dict[str, Any]] = []
            for hypothesis in hypotheses:
                candidate = deepcopy(hypothesis["candidate"])
                candidate["target_metrics_evaluation_only"] = axis._metrics(
                    candidate["box_xyxy"], truth
                )
                evaluated.append(
                    {"mechanisms": hypothesis["mechanisms"], "candidate": candidate}
                )
            if axis_candidate is not None:
                axis_hits += int(
                    axis._metrics(axis_candidate["box_xyxy"], truth)["iou"]
                    >= minimum_iou
                )
            best_iou = max(
                float(row["candidate"]["target_metrics_evaluation_only"]["iou"])
                for row in evaluated
            )
            covered = best_iou >= minimum_iou
            query_count += 1
            covered_queries += int(covered)
            family_covered += int(covered)
            set_sizes.append(len(evaluated))
            best_ious.append(best_iou)
            receipts[query] = {
                "query_truth_used_for_hypothesis_generation": False,
                "dominant_reference": dominant_reference,
                "reference_area_fractions": sorted(reference_areas),
                "vacancy_available": had_vacancy,
                "vacancy_filled": axis_candidate is not None,
                "hypothesis_count": len(evaluated),
                "hypotheses": evaluated,
                "best_hypothesis_iou_evaluation_only": best_iou,
                "target_covered_at_iou_gate": covered,
            }
        family_receipts[family] = {
            "dominant_reference": dominant_reference,
            "semantic_local_fusion_reference_votes": votes,
            "query_count": len(receipts),
            "covered_queries": family_covered,
            "query_receipts": receipts,
        }

    gate_met = covered_queries == int(
        protocol["decision_gate"]["required_covered_queries"]
    )
    result = {
        "schema": "blindassist-l10-3rscan-axis-completion-vacancy-posthoc-result-v1",
        "authority": "CONSUMED_SIX_FAMILY_AXIS_COMPLETION_VACANCY_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": consensus.base.pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": consensus.base.pixel.sha256(Path(__file__)),
        },
        "conclusion": (
            "L10_3RSCAN_AXIS_COMPLETION_VACANCY_SIX_FAMILY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_AXIS_COMPLETION_VACANCY_SIX_FAMILY_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "hypothesis_generation": protocol["hypothesis_generation"],
        "metrics": {
            "family_count": len(family_receipts),
            "query_count": query_count,
            "covered_queries": covered_queries,
            "maximum_hypothesis_count": max(set_sizes),
            "mean_hypothesis_count": sum(set_sizes) / len(set_sizes),
            "minimum_best_hypothesis_iou": min(best_ious),
            "mean_best_hypothesis_iou": sum(best_ious) / len(best_ious),
            "vacancy_available_queries": vacancy_available,
            "vacancy_filled_queries": vacancy_filled,
            "axis_vacancy_iou_gate_queries": axis_hits,
            "axis_completion_modes": completion_modes,
        },
        "families": family_receipts,
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
