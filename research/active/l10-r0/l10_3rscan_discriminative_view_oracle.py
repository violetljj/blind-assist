#!/usr/bin/env python3
"""Evaluate a one-extra-query-view discrimination oracle on a consumed family."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import l10_3rscan_reference_pixel_field as pixel


HERE = Path(__file__).resolve().parent
PROTOCOL_SCHEMA = "blindassist-l10-3rscan-discriminative-view-oracle-protocol-v1"


def _candidate_iou(candidate: dict[str, Any]) -> float:
    return float(candidate["target_metrics_evaluation_only"]["iou"])


def _view_receipt(
    key: str,
    query: dict[str, Any],
    cohort: dict[str, Any],
    minimum_iou: float,
    initial_frames: list[int],
) -> dict[str, Any]:
    candidates = query["ranked_candidates"]
    ious = [_candidate_iou(candidate) for candidate in candidates]
    scores = [float(candidate["layer18_nids_fused_score"]) for candidate in candidates]
    true_rows = [
        (index, score, iou)
        for index, (score, iou) in enumerate(zip(scores, ious), start=1)
        if iou >= minimum_iou
    ]
    wrong_scores = [score for score, iou in zip(scores, ious) if iou < minimum_iou]
    best_true = max(true_rows, key=lambda row: row[1]) if true_rows else None
    margin = (
        float(best_true[1] - max(wrong_scores))
        if best_true is not None and wrong_scores
        else None
    )
    frame = int(cohort["images"][key]["frame"])
    admitted = next(
        row
        for row in cohort["source_receipt"]["admitted_rows"]
        if int(row["frame"]) == frame
    )
    return {
        "frame": frame,
        "minimum_frame_distance_from_initial": min(
            abs(frame - value) for value in initial_frames
        ),
        "proposal_opportunity": bool(query["target_opportunity_at_iou_gate"]),
        "target_top1": ious[0] >= minimum_iou,
        "top1_iou_evaluation_only": ious[0],
        "target_recall_at_3": max(ious[:3]) >= minimum_iou,
        "best_reachable_iou_evaluation_only": max(ious),
        "first_target_rank_evaluation_only": (
            min(row[0] for row in true_rows) if true_rows else None
        ),
        "best_target_score_evaluation_only": (
            float(best_true[1]) if best_true is not None else None
        ),
        "best_wrong_score_evaluation_only": (
            max(wrong_scores) if wrong_scores else None
        ),
        "target_minus_wrong_score_margin_evaluation_only": margin,
        "target_visibility_receipt": {
            "visible_target_vertices": int(admitted["visible_target_vertices"]),
            "depth_visible_ratio": float(admitted["depth_visible_ratio"]),
            "bbox_short_side_fraction": float(
                admitted["bbox_short_side_fraction"]
            ),
            "bbox_aspect_ratio": float(admitted["bbox_aspect_ratio"]),
            "bbox_xyxy": admitted["bbox_xyxy"],
        },
    }


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(
        pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"],
        "IMPLEMENTATION_HASH",
    )
    for key in ("baseline_roster", "cohort", "extra_view_roster"):
        row = protocol[key]
        pixel.require(
            pixel.sha256(HERE / row["path"]) == row["sha256"],
            f"{key.upper()}_HASH",
        )
    baseline = pixel.load_json(HERE / protocol["baseline_roster"]["path"])
    cohort = pixel.load_json(HERE / protocol["cohort"]["path"])
    extra = pixel.load_json(HERE / protocol["extra_view_roster"]["path"])
    minimum_iou = float(protocol["decision_gate"]["minimum_iou"])

    initial_receipts: dict[str, Any] = {}
    for key in protocol["panel"]["initial_query_keys"]:
        query = baseline["query_receipts"][key]
        candidates = query["ranked_candidates"]
        ious = [_candidate_iou(candidate) for candidate in candidates]
        initial_receipts[key] = {
            "frame": int(protocol["panel"]["initial_frames"][key]),
            "target_top1": ious[0] >= minimum_iou,
            "top1_iou_evaluation_only": ious[0],
            "target_recall_at_3": max(ious[:3]) >= minimum_iou,
            "best_top3_iou_evaluation_only": max(ious[:3]),
            "first_target_rank_evaluation_only": next(
                (index for index, iou in enumerate(ious, start=1) if iou >= minimum_iou),
                None,
            ),
        }

    initial_frames = list(protocol["panel"]["initial_frames"].values())
    view_receipts = {
        key: _view_receipt(
            key,
            extra["query_receipts"][key],
            cohort,
            minimum_iou,
            initial_frames,
        )
        for key in cohort["panel"]["query_keys"]
    }
    local_keys = cohort["panel"]["local_one_step_query_keys"]
    wide_keys = cohort["panel"]["wide_baseline_query_keys"]
    informative = [
        key for key, row in view_receipts.items() if row["target_top1"]
    ]
    oracle_key = max(
        informative,
        key=lambda key: (
            float(
                view_receipts[key][
                    "target_minus_wrong_score_margin_evaluation_only"
                ]
            ),
            float(view_receipts[key]["top1_iou_evaluation_only"]),
            -int(view_receipts[key]["frame"]),
        ),
    ) if informative else None
    local_oracle_key = max(
        (key for key in local_keys if view_receipts[key]["target_top1"]),
        key=lambda key: (
            float(
                view_receipts[key][
                    "target_minus_wrong_score_margin_evaluation_only"
                ]
            ),
            float(view_receipts[key]["top1_iou_evaluation_only"]),
            -int(view_receipts[key]["frame"]),
        ),
        default=None,
    )

    initial_top1 = sum(row["target_top1"] for row in initial_receipts.values())
    initial_recall3 = sum(
        row["target_recall_at_3"] for row in initial_receipts.values()
    )
    local_top1 = sum(view_receipts[key]["target_top1"] for key in local_keys)
    all_top1 = len(informative)
    gate_met = initial_top1 == 0 and initial_recall3 == 2 and local_top1 > 0
    result = {
        "schema": "blindassist-l10-3rscan-discriminative-view-oracle-result-v1",
        "authority": "CONSUMED_ELEVENTH_FAMILY_QUERY_SIDE_VIEW_ORACLE_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {
            "path": Path(__file__).name,
            "sha256": pixel.sha256(Path(__file__)),
        },
        "conclusion": (
            "L10_3RSCAN_QUERY_SIDE_DISCRIMINATIVE_VIEW_ORACLE_DEVELOPMENT_GATE_MET"
            if gate_met
            else "L10_3RSCAN_QUERY_SIDE_DISCRIMINATIVE_VIEW_ORACLE_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "metrics": {
            "initial_query_count": len(initial_receipts),
            "initial_target_top1_queries": initial_top1,
            "initial_target_recall_at_3_queries": initial_recall3,
            "extra_view_count": len(view_receipts),
            "extra_view_proposal_opportunities": sum(
                row["proposal_opportunity"] for row in view_receipts.values()
            ),
            "all_extra_view_target_top1_views": all_top1,
            "all_extra_view_wrong_top1_exclusion_fraction": all_top1
            / len(view_receipts),
            "local_one_step_view_count": len(local_keys),
            "local_one_step_target_top1_views": local_top1,
            "local_one_step_wrong_top1_exclusion_fraction": local_top1
            / len(local_keys),
            "wide_baseline_view_count": len(wide_keys),
            "wide_baseline_target_top1_views": sum(
                view_receipts[key]["target_top1"] for key in wide_keys
            ),
            "oracle_one_extra_view_target_top1": oracle_key is not None,
        },
        "oracle_view": (
            {"query_key": oracle_key, **deepcopy(view_receipts[oracle_key])}
            if oracle_key is not None
            else None
        ),
        "local_one_step_oracle_view": (
            {
                "query_key": local_oracle_key,
                **deepcopy(view_receipts[local_oracle_key]),
            }
            if local_oracle_key is not None
            else None
        ),
        "initial_query_receipts": initial_receipts,
        "extra_view_receipts": view_receipts,
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
