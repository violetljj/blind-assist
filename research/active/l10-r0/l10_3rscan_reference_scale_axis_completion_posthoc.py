#!/usr/bin/env python3
"""Add reference-scale axis completion to the bounded cross-view proposal set."""

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
    "blindassist-l10-3rscan-reference-scale-axis-completion-posthoc-protocol-v1"
)


def _metrics(proposal: list[float], target: list[float]) -> dict[str, float]:
    left = max(float(proposal[0]), float(target[0]))
    top = max(float(proposal[1]), float(target[1]))
    right = min(float(proposal[2]), float(target[2]))
    bottom = min(float(proposal[3]), float(target[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    proposal_area = max(0.0, float(proposal[2]) - float(proposal[0])) * max(
        0.0, float(proposal[3]) - float(proposal[1])
    )
    target_area = max(0.0, float(target[2]) - float(target[0])) * max(
        0.0, float(target[3]) - float(target[1])
    )
    union = proposal_area + target_area - intersection
    return {
        "iou": intersection / union if union > 0.0 else 0.0,
        "target_recall": intersection / target_area if target_area > 0.0 else 0.0,
        "proposal_precision": intersection / proposal_area if proposal_area > 0.0 else 0.0,
    }


def _axis_options(
    candidate: dict[str, Any], width: int, height: int
) -> list[dict[str, Any]]:
    left, top, right, bottom = map(float, candidate["box_xyxy"])
    return [
        {"mode": "identity", "box_xyxy": [left, top, right, bottom]},
        {"mode": "horizontal_band", "box_xyxy": [0.0, top, float(width), bottom]},
        {"mode": "vertical_band", "box_xyxy": [left, 0.0, right, float(height)]},
    ]


def _axis_best(
    candidates: list[dict[str, Any]],
    width: int,
    height: int,
    reference_areas: list[float],
    penalty: float,
) -> dict[str, Any]:
    options: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
    mode_order = {"identity": 0, "horizontal_band": 1, "vertical_band": 2}
    for candidate in candidates:
        for option in _axis_options(candidate, width, height):
            area = consensus.base._area_fraction(option["box_xyxy"], width, height)
            score = float(candidate["layer18_local_appearance_score"]) - penalty * min(
                abs(math.log(area / reference_area))
                for reference_area in reference_areas
            )
            options.append((score, mode_order[option["mode"]], candidate, option))
    selected_score, _selected_mode_order, parent, selected = sorted(
        options,
        key=lambda row: (
            -row[0],
            -float(row[2]["layer18_local_appearance_score"]),
            -float(row[2]["target_memory_score"]),
            -float(row[2]["objectness_score"]),
            row[1],
            *row[3]["box_xyxy"],
            int(row[2]["postprocess_index"]),
        ),
    )[0]
    derived = deepcopy(parent)
    derived["source_box_xyxy"] = list(parent["box_xyxy"])
    derived["source_tile_id"] = parent["tile_id"]
    derived["source_postprocess_index"] = int(parent["postprocess_index"])
    derived["completion_mode"] = selected["mode"]
    derived["box_xyxy"] = selected["box_xyxy"]
    derived["reference_scale_axis_score"] = float(selected_score)
    derived.pop("target_metrics_evaluation_only", None)
    return derived


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
    family_receipts: dict[str, Any] = {}
    query_count = 0
    covered_queries = 0
    set_sizes: list[int] = []
    best_ious: list[float] = []
    mechanism_available = {
        "semantic_local_fusion": 0,
        "cross_view_reference_consensus": 0,
        "reference_scale_axis_completion": 0,
    }
    mechanism_hits = {key: 0 for key in mechanism_available}
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
            axis = _axis_best(candidates, width, height, reference_areas, penalty)
            constrained = [
                row
                for row in candidates
                if row["winning_target_reference"] == dominant_reference
            ]
            contributors = [("semantic_local_fusion", fused[query])]
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
            contributors.append(("reference_scale_axis_completion", axis))

            hypotheses: list[dict[str, Any]] = []
            for mechanism, candidate in contributors:
                mechanism_available[mechanism] += 1
                key = tuple(candidate["box_xyxy"])
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
                f"SET_SIZE:{family}:{query}",
            )

            # Truth enters only after vote, axis completion, contributors and deduplication.
            truth = image_row["bbox_xyxy"]
            evaluated_hypotheses: list[dict[str, Any]] = []
            for hypothesis in hypotheses:
                candidate = deepcopy(hypothesis["candidate"])
                candidate["target_metrics_evaluation_only"] = _metrics(
                    candidate["box_xyxy"], truth
                )
                evaluated_hypotheses.append(
                    {"mechanisms": hypothesis["mechanisms"], "candidate": candidate}
                )
            for mechanism, candidate in contributors:
                mechanism_hits[mechanism] += int(
                    _metrics(candidate["box_xyxy"], truth)["iou"] >= minimum_iou
                )
            completion_modes[axis["completion_mode"]] += 1
            best_iou = max(
                float(row["candidate"]["target_metrics_evaluation_only"]["iou"])
                for row in evaluated_hypotheses
            )
            covered = best_iou >= minimum_iou
            query_count += 1
            covered_queries += int(covered)
            family_covered += int(covered)
            set_sizes.append(len(evaluated_hypotheses))
            best_ious.append(best_iou)
            receipts[query] = {
                "query_truth_used_for_hypothesis_generation": False,
                "dominant_reference": dominant_reference,
                "reference_area_fractions": sorted(reference_areas),
                "hypothesis_count": len(evaluated_hypotheses),
                "hypotheses": evaluated_hypotheses,
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
        "schema": "blindassist-l10-3rscan-reference-scale-axis-completion-posthoc-result-v1",
        "authority": "CONSUMED_SIX_FAMILY_REFERENCE_SCALE_AXIS_COMPLETION_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": consensus.base.pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": consensus.base.pixel.sha256(Path(__file__)),
        },
        "conclusion": (
            "L10_3RSCAN_REFERENCE_SCALE_AXIS_COMPLETION_SIX_FAMILY_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_REFERENCE_SCALE_AXIS_COMPLETION_SIX_FAMILY_DEVELOPMENT_GATE_NOT_MET"
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
