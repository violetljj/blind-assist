#!/usr/bin/env python3
"""Confirm the frozen bounded tri-evidence proposal set on one fresh family."""

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


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-tri-evidence-hypothesis-set-confirmation-protocol-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = base.pixel.load_json(protocol_path)
    base.pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    base.pixel.require(
        base.pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for key in ("development_predecessor", "source_result", "cohort", "candidate_result"):
        row = protocol[key]
        path = HERE / row["path"]
        base.pixel.require(base.pixel.sha256(path) == row["sha256"], f"{key.upper()}_HASH")
    predecessor = base.pixel.load_json(HERE / protocol["development_predecessor"]["path"])
    base.pixel.require(
        predecessor["conclusion"] == protocol["development_predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    source = base.pixel.load_json(HERE / protocol["source_result"]["path"])
    base.pixel.require(source["source_evaluable"] is True, "SOURCE_NOT_EVALUABLE")
    cohort = base.pixel.load_json(HERE / protocol["cohort"]["path"])
    candidates_result = base.pixel.load_json(HERE / protocol["candidate_result"]["path"])

    penalty = float(protocol["hypothesis_generation"]["scale_mixture_log_area_penalty"])
    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    query_prefixes = {key.split("_")[0] for key in candidates_result["query_receipts"]}
    reference_areas: list[float] = []
    for key, row in cohort["images"].items():
        if key.endswith("_reference") and key.split("_")[0] in query_prefixes:
            width, height = map(int, row["color_size"])
            reference_areas.append(base._area_fraction(row["bbox_xyxy"], width, height))
    base.pixel.require(len(reference_areas) == 3, "REFERENCE_SCALE_COUNT")

    receipts: dict[str, Any] = {}
    set_sizes: list[int] = []
    best_ious: list[float] = []
    mechanism_hits = {
        "layer18_local": 0,
        "semantic_local_fusion": 0,
        "reference_scale_mixture": 0,
    }
    for query, source_receipt in candidates_result["query_receipts"].items():
        candidates = source_receipt["ranked_candidates"]
        image_row = cohort["images"][query]
        width, height = map(int, image_row["color_size"])
        mechanisms = {
            "layer18_local": base._best(
                candidates, lambda row: float(row["layer18_local_appearance_score"])
            ),
            "semantic_local_fusion": base._best(
                candidates, lambda row: float(row["layer18_nids_fused_score"])
            ),
            "reference_scale_mixture": base._best(
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
            ),
        }
        hypotheses: list[dict[str, Any]] = []
        for mechanism, candidate in mechanisms.items():
            key = (
                candidate["tile_id"],
                int(candidate["postprocess_index"]),
                tuple(candidate["box_xyxy"]),
            )
            existing = next((row for row in hypotheses if row["candidate_key"] == key), None)
            if existing is None:
                hypotheses.append(
                    {"candidate_key": key, "mechanisms": [mechanism], "candidate": candidate}
                )
            else:
                existing["mechanisms"].append(mechanism)
        base.pixel.require(len(hypotheses) <= int(protocol["decision_gate"]["maximum_set_size"]), "SET_SIZE")

        # Query truth is accessed only below, after all contributors and deduplication are fixed.
        for mechanism, candidate in mechanisms.items():
            mechanism_hits[mechanism] += int(
                float(candidate["target_metrics_evaluation_only"]["iou"]) >= minimum_iou
            )
        best_iou = max(
            float(row["candidate"]["target_metrics_evaluation_only"]["iou"])
            for row in hypotheses
        )
        covered = best_iou >= minimum_iou
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

    covered_queries = sum(int(row["target_covered_at_iou_gate"]) for row in receipts.values())
    gate_met = covered_queries == int(protocol["decision_gate"]["required_covered_queries"])
    result = {
        "schema": "blindassist-l10-3rscan-tri-evidence-hypothesis-set-confirmation-result-v1",
        "authority": "FRESH_FIFTH_FAMILY_BOUNDED_TRI_EVIDENCE_PROPOSAL_SET_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": base.pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": base.pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_TRI_EVIDENCE_HYPOTHESIS_SET_FRESH_CONFIRMATION_GATE_MET"
            if gate_met
            else "L10_3RSCAN_TRI_EVIDENCE_HYPOTHESIS_SET_FRESH_CONFIRMATION_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "hypothesis_generation": protocol["hypothesis_generation"],
        "metrics": {
            "query_count": len(receipts),
            "covered_queries": covered_queries,
            "maximum_hypothesis_count": max(set_sizes),
            "mean_hypothesis_count": sum(set_sizes) / len(set_sizes),
            "minimum_best_hypothesis_iou": min(best_ious),
            "mean_best_hypothesis_iou": sum(best_ious) / len(best_ious),
            "individual_mechanism_iou_gate_queries": mechanism_hits,
        },
        "query_receipts": receipts,
        "intermediate_runtime": candidates_result["runtime"],
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
