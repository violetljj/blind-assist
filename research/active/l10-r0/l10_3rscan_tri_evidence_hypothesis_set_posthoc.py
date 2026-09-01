#!/usr/bin/env python3
"""Build a bounded tri-evidence localization hypothesis set across consumed families."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-tri-evidence-hypothesis-set-posthoc-protocol-v1"


def _area_fraction(box: list[float], width: int, height: int) -> float:
    return max(1.0, (float(box[2]) - float(box[0])) * (float(box[3]) - float(box[1]))) / float(width * height)


def _best(candidates: list[dict[str, Any]], score: Callable[[dict[str, Any]], float]) -> dict[str, Any]:
    return sorted(
        candidates,
        key=lambda row: (
            -score(row),
            -row["layer18_local_appearance_score"],
            -row["target_memory_score"],
            -row["objectness_score"],
            *row["box_xyxy"],
            row["postprocess_index"],
        ),
    )[0]


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    penalty = float(protocol["hypothesis_generation"]["scale_mixture_log_area_penalty"])
    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    family_receipts: dict[str, Any] = {}
    covered = 0
    query_count = 0
    set_sizes: list[int] = []
    best_ious: list[float] = []
    mechanism_hits = {"layer18_local": 0, "semantic_local_fusion": 0, "reference_scale_mixture": 0}
    for family, spec in protocol["families"].items():
        result_path = HERE / spec["candidate_result"]["path"]
        cohort_path = HERE / spec["cohort"]["path"]
        pixel.require(pixel.sha256(result_path) == spec["candidate_result"]["sha256"], f"RESULT_HASH:{family}")
        pixel.require(pixel.sha256(cohort_path) == spec["cohort"]["sha256"], f"COHORT_HASH:{family}")
        result, cohort = pixel.load_json(result_path), pixel.load_json(cohort_path)
        query_prefixes = {key.split("_")[0] for key in result["query_receipts"]}
        reference_areas = []
        for key, row in cohort["images"].items():
            if key.endswith("_reference") and key.split("_")[0] in query_prefixes:
                width, height = map(int, row["color_size"])
                reference_areas.append(_area_fraction(row["bbox_xyxy"], width, height))
        pixel.require(len(reference_areas) == 3, f"REFERENCE_SCALE_COUNT:{family}")
        receipts: dict[str, Any] = {}
        family_covered = 0
        for query, source in result["query_receipts"].items():
            candidates = source["ranked_candidates"]
            image_row = cohort["images"][query]
            width, height = map(int, image_row["color_size"])
            mechanisms = {
                "layer18_local": _best(candidates, lambda row: float(row["layer18_local_appearance_score"])),
                "semantic_local_fusion": _best(candidates, lambda row: float(row["layer18_nids_fused_score"])),
                "reference_scale_mixture": _best(
                    candidates,
                    lambda row: float(row["layer18_local_appearance_score"])
                    - penalty * min(
                        abs(math.log(_area_fraction(row["box_xyxy"], width, height) / area))
                        for area in reference_areas
                    ),
                ),
            }
            hypotheses: list[dict[str, Any]] = []
            for mechanism, candidate in mechanisms.items():
                key = (candidate["tile_id"], int(candidate["postprocess_index"]), tuple(candidate["box_xyxy"]))
                existing = next((row for row in hypotheses if row["candidate_key"] == key), None)
                if existing is None:
                    hypotheses.append({"candidate_key": key, "mechanisms": [mechanism], "candidate": candidate})
                else:
                    existing["mechanisms"].append(mechanism)
                mechanism_hits[mechanism] += int(float(candidate["target_metrics_evaluation_only"]["iou"]) >= minimum_iou)
            best_iou = max(float(row["candidate"]["target_metrics_evaluation_only"]["iou"]) for row in hypotheses)
            query_covered = best_iou >= minimum_iou
            covered += int(query_covered)
            family_covered += int(query_covered)
            query_count += 1
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
                "target_covered_at_iou_gate": query_covered,
            }
        family_receipts[family] = {
            "query_count": len(receipts),
            "covered_queries": family_covered,
            "query_receipts": receipts,
        }
    gate_met = covered == int(protocol["decision_gate"]["required_covered_queries"])
    result = {
        "schema": "blindassist-l10-3rscan-tri-evidence-hypothesis-set-posthoc-result-v1",
        "authority": "CONSUMED_FOUR_FAMILY_TRI_EVIDENCE_HYPOTHESIS_SET_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_TRI_EVIDENCE_HYPOTHESIS_SET_FOUR_FAMILY_DEVELOPMENT_GATE_MET"
            if gate_met else
            "L10_3RSCAN_TRI_EVIDENCE_HYPOTHESIS_SET_FOUR_FAMILY_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "hypothesis_generation": protocol["hypothesis_generation"],
        "metrics": {
            "family_count": len(family_receipts),
            "query_count": query_count,
            "covered_queries": covered,
            "maximum_hypothesis_count": max(set_sizes),
            "mean_hypothesis_count": sum(set_sizes) / len(set_sizes),
            "minimum_best_hypothesis_iou": min(best_ious),
            "mean_best_hypothesis_iou": sum(best_ious) / len(best_ious),
            "individual_mechanism_iou_gate_queries": mechanism_hits,
        },
        "families": family_receipts,
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
