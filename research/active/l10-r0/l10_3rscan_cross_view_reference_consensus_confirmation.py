#!/usr/bin/env python3
"""Confirm the frozen cross-view reference-consensus proposal set on one family."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_cross_view_reference_consensus_posthoc as consensus  # noqa: E402


PROTOCOL_SCHEMA = (
    "blindassist-l10-3rscan-cross-view-reference-consensus-confirmation-protocol-v1"
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

    penalty = float(protocol["hypothesis_generation"]["scale_mixture_log_area_penalty"])
    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
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
    consensus.base.pixel.require(len(reference_areas) == 3, "REFERENCE_SCALE_COUNT")

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
    reference_votes = [row["winning_target_reference"] for row in fused.values()]
    dominant_reference = sorted(
        set(reference_votes),
        key=lambda reference: (
            -reference_votes.count(reference),
            reference_order.index(reference),
        ),
    )[0]

    receipts: dict[str, Any] = {}
    set_sizes: list[int] = []
    best_ious: list[float] = []
    opportunity_queries = 0
    mechanism_available = {
        "semantic_local_fusion": 0,
        "reference_scale_mixture": 0,
        "cross_view_reference_consensus": 0,
    }
    mechanism_hits = {key: 0 for key in mechanism_available}
    for query, source_receipt in candidates_result["query_receipts"].items():
        candidates = source_receipt["ranked_candidates"]
        width, height = map(int, cohort["images"][query]["color_size"])
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
                        / area
                    )
                )
                for area in reference_areas
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
        consensus.base.pixel.require(
            len(hypotheses) <= int(protocol["decision_gate"]["maximum_set_size"]),
            f"SET_SIZE:{query}",
        )

        # Truth enters only after the cross-view vote, contributors and deduplication.
        for mechanism, candidate in contributors:
            mechanism_hits[mechanism] += int(
                float(candidate["target_metrics_evaluation_only"]["iou"])
                >= minimum_iou
            )
        best_iou = max(
            float(row["candidate"]["target_metrics_evaluation_only"]["iou"])
            for row in hypotheses
        )
        covered = best_iou >= minimum_iou
        opportunity = bool(source_receipt["target_opportunity_at_iou_gate"])
        opportunity_queries += int(opportunity)
        set_sizes.append(len(hypotheses))
        best_ious.append(best_iou)
        receipts[query] = {
            "query_truth_used_for_hypothesis_generation": False,
            "proposal_opportunity_at_iou_gate_evaluation_only": opportunity,
            "best_reachable_proposal_iou_evaluation_only": source_receipt[
                "best_reachable_target_iou_evaluation_only"
            ],
            "reference_area_fractions": sorted(reference_areas),
            "hypothesis_count": len(hypotheses),
            "hypotheses": [
                {"mechanisms": row["mechanisms"], "candidate": row["candidate"]}
                for row in hypotheses
            ],
            "best_hypothesis_iou_evaluation_only": best_iou,
            "target_covered_at_iou_gate": covered,
        }

    covered_queries = sum(
        int(row["target_covered_at_iou_gate"]) for row in receipts.values()
    )
    gate_met = covered_queries == int(
        protocol["decision_gate"]["required_covered_queries"]
    )
    result = {
        "schema": "blindassist-l10-3rscan-cross-view-reference-consensus-confirmation-result-v1",
        "authority": "FRESH_SIXTH_FAMILY_BOUNDED_CROSS_VIEW_REFERENCE_CONSENSUS_PROPOSAL_SET_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": consensus.base.pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": consensus.base.pixel.sha256(Path(__file__)),
        },
        "conclusion": (
            "L10_3RSCAN_CROSS_VIEW_REFERENCE_CONSENSUS_FRESH_CONFIRMATION_GATE_MET"
            if gate_met
            else "L10_3RSCAN_CROSS_VIEW_REFERENCE_CONSENSUS_FRESH_CONFIRMATION_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "dominant_reference": dominant_reference,
        "semantic_local_fusion_reference_votes": reference_votes,
        "hypothesis_generation": protocol["hypothesis_generation"],
        "metrics": {
            "query_count": len(receipts),
            "proposal_opportunity_queries": opportunity_queries,
            "covered_queries": covered_queries,
            "maximum_hypothesis_count": max(set_sizes),
            "mean_hypothesis_count": sum(set_sizes) / len(set_sizes),
            "minimum_best_hypothesis_iou": min(best_ious),
            "mean_best_hypothesis_iou": sum(best_ious) / len(best_ious),
            "mechanism_available_queries": mechanism_available,
            "individual_mechanism_iou_gate_queries": mechanism_hits,
        },
        "query_receipts": receipts,
        "intermediate_runtime": candidates_result["runtime"],
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
