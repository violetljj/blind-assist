#!/usr/bin/env python3
"""Confirm the frozen depth-and-pose proposal propagation rule on a fresh family."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import l10_3rscan_depth_pose_propagation_posthoc as dev


HERE = Path(__file__).resolve().parent
PROTOCOL_SCHEMA = (
    "blindassist-l10-3rscan-depth-pose-propagation-confirmation-protocol-v1"
)


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = dev.confirm.consensus.base.pixel.load_json(protocol_path)
    dev.confirm.consensus.base.pixel.require(
        protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA"
    )
    dev.confirm.consensus.base.pixel.require(
        dev.confirm.consensus.base.pixel.sha256(Path(__file__))
        == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for dependency in protocol["dependencies"]:
        dev.confirm.consensus.base.pixel.require(
            dev.confirm.consensus.base.pixel.sha256(HERE / dependency["path"])
            == dependency["sha256"],
            f"DEPENDENCY_HASH:{dependency['path']}",
        )
    for key in ("predecessor", "source_result", "cohort", "candidate_result"):
        row = protocol[key]
        dev.confirm.consensus.base.pixel.require(
            dev.confirm.consensus.base.pixel.sha256(HERE / row["path"])
            == row["sha256"],
            f"{key.upper()}_HASH",
        )
    predecessor = dev.confirm.consensus.base.pixel.load_json(
        HERE / protocol["predecessor"]["path"]
    )
    dev.confirm.consensus.base.pixel.require(
        predecessor["conclusion"]
        == protocol["predecessor"]["required_conclusion"],
        "PREDECESSOR_CONCLUSION",
    )
    source_result = dev.confirm.consensus.base.pixel.load_json(
        HERE / protocol["source_result"]["path"]
    )
    dev.confirm.consensus.base.pixel.require(
        source_result["conclusion"]
        == protocol["source_result"]["required_conclusion"],
        "SOURCE_CONCLUSION",
    )
    cohort = dev.confirm.consensus.base.pixel.load_json(
        HERE / protocol["cohort"]["path"]
    )
    candidates_result = dev.confirm.consensus.base.pixel.load_json(
        HERE / protocol["candidate_result"]["path"]
    )
    sequence_zip = Path(protocol["geometry"]["sequence_zip"])
    dev.confirm.consensus.base.pixel.require(
        dev.confirm.consensus.base.pixel.sha256(sequence_zip)
        == protocol["geometry"]["sequence_sha256"],
        "SEQUENCE_HASH",
    )

    query_names = [str(value) for value in protocol["geometry"]["query_images"]]
    preserved, dominant_reference, votes, reference_areas = dev._preserved_sets(
        candidates_result,
        cohort,
        query_names,
        float(protocol["hypothesis_generation"]["scale_log_area_penalty"]),
    )
    anchor_query, anchor_hypothesis, anchor_receipt = dev._anchor(
        preserved, query_names
    )
    target_query = next(query for query in query_names if query != anchor_query)
    propagated_box, geometry_receipt = dev._propagate_box(
        sequence_zip,
        int(cohort["images"][anchor_query]["frame"]),
        int(cohort["images"][target_query]["frame"]),
        anchor_hypothesis["candidate"]["box_xyxy"],
    )

    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])
    maximum_size = int(protocol["decision_gate"]["maximum_set_size"])
    query_receipts: dict[str, Any] = {}
    preserved_covered = 0
    final_covered = 0
    propagated_hits = 0
    incremental_hits = 0
    best_ious: list[float] = []
    for query in query_names:
        hypotheses = deepcopy(preserved[query])
        truth = cohort["images"][query]["bbox_xyxy"]
        preserved_best = max(
            dev.confirm._iou(row["candidate"]["box_xyxy"], truth)
            for row in hypotheses
        )
        preserved_covered += int(preserved_best >= minimum_iou)
        propagated_added = False
        if query == target_query and len(hypotheses) < maximum_size:
            hypotheses.append(
                {
                    "candidate_key": [
                        "depth_pose_propagation",
                        anchor_query,
                        propagated_box,
                    ],
                    "mechanisms": ["depth_pose_rectangle_propagation"],
                    "candidate": {
                        "box_xyxy": propagated_box,
                        "anchor_query": anchor_query,
                        "anchor_candidate_key": list(
                            dev.confirm._candidate_key(anchor_hypothesis["candidate"])
                        ),
                        "anchor_mechanisms": anchor_hypothesis["mechanisms"],
                    },
                }
            )
            propagated_added = True
        dev.confirm.consensus.base.pixel.require(
            len(hypotheses) <= maximum_size, f"SET_SIZE:{query}"
        )
        evaluated: list[dict[str, Any]] = []
        propagated_iou = None
        for hypothesis in hypotheses:
            candidate = deepcopy(hypothesis["candidate"])
            iou = dev.confirm._iou(candidate["box_xyxy"], truth)
            candidate["target_metrics_evaluation_only"] = {"iou": iou}
            evaluated.append(
                {"mechanisms": hypothesis["mechanisms"], "candidate": candidate}
            )
            if "depth_pose_rectangle_propagation" in hypothesis["mechanisms"]:
                propagated_iou = iou
        propagated_hits += int(
            propagated_iou is not None and propagated_iou >= minimum_iou
        )
        incremental_hits += int(
            propagated_iou is not None
            and preserved_best < minimum_iou
            and propagated_iou >= minimum_iou
        )
        best_iou = max(
            float(row["candidate"]["target_metrics_evaluation_only"]["iou"])
            for row in evaluated
        )
        covered = best_iou >= minimum_iou
        final_covered += int(covered)
        best_ious.append(best_iou)
        query_receipts[query] = {
            "query_truth_used_for_hypothesis_generation": False,
            "is_anchor_query": query == anchor_query,
            "propagated_candidate_added": propagated_added,
            "preserved_best_iou_evaluation_only": preserved_best,
            "propagated_iou_evaluation_only": propagated_iou,
            "hypothesis_count": len(evaluated),
            "hypotheses": evaluated,
            "best_hypothesis_iou_evaluation_only": best_iou,
            "target_covered_at_iou_gate": covered,
        }

    gate_met = final_covered == int(
        protocol["decision_gate"]["required_covered_queries"]
    )
    result = {
        "schema": "blindassist-l10-3rscan-depth-pose-propagation-confirmation-result-v1",
        "authority": "FRESH_ELEVENTH_FAMILY_DEPTH_POSE_PROPAGATION_CONFIRMATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": dev.confirm.consensus.base.pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": dev.confirm.consensus.base.pixel.sha256(Path(__file__)),
        },
        "conclusion": (
            "L10_3RSCAN_DEPTH_POSE_PROPAGATION_FRESH_CONFIRMATION_GATE_MET"
            if gate_met
            else "L10_3RSCAN_DEPTH_POSE_PROPAGATION_FRESH_CONFIRMATION_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "anchor_selection": {
            "query": anchor_query,
            "candidate_key": list(
                dev.confirm._candidate_key(anchor_hypothesis["candidate"])
            ),
            "mechanisms": anchor_hypothesis["mechanisms"],
            **anchor_receipt,
        },
        "target_query": target_query,
        "dominant_reference": dominant_reference,
        "semantic_local_fusion_reference_votes": votes,
        "reference_area_fractions": sorted(reference_areas),
        "geometry_receipt": geometry_receipt,
        "metrics": {
            "query_count": len(query_names),
            "preserved_covered_queries": preserved_covered,
            "final_covered_queries": final_covered,
            "minimum_best_hypothesis_iou": min(best_ious),
            "mean_best_hypothesis_iou": sum(best_ious) / len(best_ious),
            "propagated_iou_gate_queries": propagated_hits,
            "propagated_incremental_iou_gate_queries": incremental_hits,
        },
        "query_receipts": query_receipts,
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    dev.confirm.consensus.base.pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
