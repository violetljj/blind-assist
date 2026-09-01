#!/usr/bin/env python3
"""Add cross-view reference consensus to a bounded proposal hypothesis set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_tri_evidence_hypothesis_set_posthoc as base  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-cross-view-reference-consensus-posthoc-protocol-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = base.pixel.load_json(protocol_path)
    base.pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    base.pixel.require(
        base.pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    predecessor = protocol["predecessor"]
    predecessor_path = HERE / predecessor["path"]
    base.pixel.require(
        base.pixel.sha256(predecessor_path) == predecessor["sha256"],
        "PREDECESSOR_HASH",
    )
    base.pixel.require(
        base.pixel.load_json(predecessor_path)["conclusion"]
        == predecessor["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )

    penalty = float(protocol["hypothesis_generation"]["scale_mixture_log_area_penalty"])
    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    family_receipts: dict[str, Any] = {}
    query_count = 0
    covered_queries = 0
    set_sizes: list[int] = []
    best_ious: list[float] = []
    mechanism_available = {
        "semantic_local_fusion": 0,
        "reference_scale_mixture": 0,
        "cross_view_reference_consensus": 0,
    }
    mechanism_hits = {key: 0 for key in mechanism_available}

    for family, spec in protocol["families"].items():
        candidate_path = HERE / spec["candidate_result"]["path"]
        cohort_path = HERE / spec["cohort"]["path"]
        base.pixel.require(
            base.pixel.sha256(candidate_path) == spec["candidate_result"]["sha256"],
            f"RESULT_HASH:{family}",
        )
        base.pixel.require(
            base.pixel.sha256(cohort_path) == spec["cohort"]["sha256"],
            f"COHORT_HASH:{family}",
        )
        candidates_result = base.pixel.load_json(candidate_path)
        cohort = base.pixel.load_json(cohort_path)
        query_prefixes = {key.split("_")[0] for key in candidates_result["query_receipts"]}
        reference_areas: list[float] = []
        for key, row in cohort["images"].items():
            if key.endswith("_reference") and key.split("_")[0] in query_prefixes:
                width, height = map(int, row["color_size"])
                reference_areas.append(base._area_fraction(row["bbox_xyxy"], width, height))
        base.pixel.require(len(reference_areas) == 3, f"REFERENCE_SCALE_COUNT:{family}")

        fused = {
            query: base._best(
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
        reference_votes = [row["winning_target_reference"] for row in fused.values()]
        dominant_reference = sorted(
            set(reference_votes),
            key=lambda reference: (
                -reference_votes.count(reference),
                reference_order.index(reference),
            ),
        )[0]

        receipts: dict[str, Any] = {}
        family_covered = 0
        for query, source_receipt in candidates_result["query_receipts"].items():
            candidates = source_receipt["ranked_candidates"]
            width, height = map(int, cohort["images"][query]["color_size"])
            scale = base._best(
                candidates,
                lambda row: float(row["layer18_local_appearance_score"])
                - penalty
                * min(
                    abs(
                        math.log(
                            base._area_fraction(row["box_xyxy"], width, height) / area
                        )
                    )
                    for area in reference_areas
                ),
            )
            contributors = [
                ("semantic_local_fusion", fused[query]),
                ("reference_scale_mixture", scale),
            ]
            consensus_candidates = [
                row
                for row in candidates
                if row["winning_target_reference"] == dominant_reference
            ]
            if consensus_candidates:
                contributors.append(
                    (
                        "cross_view_reference_consensus",
                        base._best(
                            consensus_candidates,
                            lambda row: float(row["layer18_nids_fused_score"]),
                        ),
                    )
                )

            hypotheses: list[dict[str, Any]] = []
            for mechanism, candidate in contributors:
                mechanism_available[mechanism] += 1
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
            base.pixel.require(
                len(hypotheses) <= int(protocol["decision_gate"]["maximum_set_size"]),
                f"SET_SIZE:{family}:{query}",
            )

            # Truth enters only after the family vote, contributors, and deduplication.
            for mechanism, candidate in contributors:
                mechanism_hits[mechanism] += int(
                    float(candidate["target_metrics_evaluation_only"]["iou"]) >= minimum_iou
                )
            best_iou = max(
                float(row["candidate"]["target_metrics_evaluation_only"]["iou"])
                for row in hypotheses
            )
            covered = best_iou >= minimum_iou
            query_count += 1
            covered_queries += int(covered)
            family_covered += int(covered)
            set_sizes.append(len(hypotheses))
            best_ious.append(best_iou)
            receipts[query] = {
                "query_truth_used_for_hypothesis_generation": False,
                "reference_area_fractions": sorted(reference_areas),
                "hypothesis_count": len(hypotheses),
                "hypotheses": [
                    {"mechanisms": row["mechanisms"], "candidate": row["candidate"]}
                    for row in hypotheses
                ],
                "best_hypothesis_iou_evaluation_only": best_iou,
                "target_covered_at_iou_gate": covered,
            }
        family_receipts[family] = {
            "dominant_reference": dominant_reference,
            "semantic_local_fusion_reference_votes": reference_votes,
            "query_count": len(receipts),
            "covered_queries": family_covered,
            "query_receipts": receipts,
        }

    gate_met = covered_queries == int(protocol["decision_gate"]["required_covered_queries"])
    result = {
        "schema": "blindassist-l10-3rscan-cross-view-reference-consensus-posthoc-result-v1",
        "authority": "CONSUMED_FIVE_FAMILY_CROSS_VIEW_REFERENCE_CONSENSUS_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": base.pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_CROSS_VIEW_REFERENCE_CONSENSUS_FIVE_FAMILY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_CROSS_VIEW_REFERENCE_CONSENSUS_FIVE_FAMILY_DEVELOPMENT_GATE_NOT_MET"
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
            "mechanism_available_queries": mechanism_available,
            "individual_mechanism_iou_gate_queries": mechanism_hits,
        },
        "families": family_receipts,
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    base.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
