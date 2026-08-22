#!/usr/bin/env python3
"""Adjudicate P1-PA1 against its frozen manifest and sealed PA0 baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from evaluate import evaluate, iou, validated_box
import run_yoloe_tiled_rescue as provider


SCHEMA = "blindassist_p1_pa1_tiled_evaluation_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--pa0-evaluation", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    pa0 = json.loads(args.pa0_evaluation.read_text(encoding="utf-8"))
    private = json.loads(args.private.read_text(encoding="utf-8"))
    if manifest["protocol_id"] != provider.PROTOCOL_ID or prediction["protocol_id"] != provider.PROTOCOL_ID:
        raise ValueError("PA1 protocol identity mismatch")
    if prediction["manifest_sha256"] != sha256(args.manifest):
        raise ValueError("PA1 prediction is not bound to manifest")
    if manifest["inputs"]["private_eval_input_sha256"] != sha256(args.private):
        raise ValueError("PA1 private evaluator hash mismatch")
    if manifest["inputs"]["sealed_pa0_evaluation_sha256"] != sha256(args.pa0_evaluation):
        raise ValueError("sealed PA0 evaluation hash mismatch")
    if not prediction.get("formal_run") or prediction.get("case_limit") is not None:
        raise ValueError("PA1 evaluator requires the formal complete prediction")

    result = evaluate(args.public, args.private, args.prediction)
    truth = {case["case_id"]: case for case in private["cases"]}
    predicted = {case["case_id"]: case for case in prediction["cases"]}
    pa0_rows = {row["case_id"]: row for row in pa0["rows"]}
    thresholds = (0.10, 0.30, 0.50)
    full_rank = {}
    for threshold in thresholds:
        key = str(threshold)
        rows = []
        for case_id, output in predicted.items():
            target_box = validated_box(truth[case_id]["target_bbox_xyxy"], f"{case_id} target")
            overlaps = [iou(validated_box(candidate["bbox_xyxy"], f"{case_id} raw"), target_box) for candidate in output["raw_candidates"]]
            rank = next((index for index, overlap in enumerate(overlaps, start=1) if overlap >= threshold), None)
            rows.append((case_id, rank, max(overlaps, default=0.0)))
        full_rank[key] = {
            "recall": sum(rank is not None for _, rank, _ in rows) / len(rows),
            "first_correct_rank": {case_id: rank for case_id, rank, _ in rows},
            "best_iou": {case_id: best for case_id, _, best in rows},
            "present_full_rank_but_absent_k10": sum(rank is not None and rank > 10 for _, rank, _ in rows),
        }

    rescued = {}
    for threshold in thresholds:
        key = str(threshold)
        baseline_absent = {
            case_id for case_id, row in pa0_rows.items()
            if row["first_correct_rank"][key] is None
        }
        current_rows = {row["case_id"]: row for row in result["rows"]}
        rescued[key] = {
            "sealed_pa0_absent_cases": len(baseline_absent),
            "rescued_into_k10": sorted(
                case_id for case_id in baseline_absent
                if current_rows[case_id]["first_correct_rank"][key] is not None
            ),
        }
    primary = result["recall"]["0.3"]["recall_at_10"]
    result.update({
        "schema_version": SCHEMA,
        "protocol_id": provider.PROTOCOL_ID,
        "manifest_sha256": sha256(args.manifest),
        "sealed_pa0_evaluation_sha256": sha256(args.pa0_evaluation),
        "full_rank_postprocessed": full_rank,
        "sealed_pa0_absent_rescue": rescued,
        "pre_nms_attribution": "NOT_EVALUABLE_PROVIDER_INTERFACE",
        "terminal": (
            manifest["adjudication"]["primary_nonzero"]
            if primary > 0.0
            else manifest["adjudication"]["primary_zero"]
        ),
        "claim_ceiling": manifest["claim_ceiling"],
    })
    atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
