#!/usr/bin/env python3
"""Private evaluator for ordered P1-PA0 candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
from typing import Any


RESULT_SCHEMA = "blindassist_p1_pa0_evaluation_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def validated_box(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{label} must be a four-value XYXY box")
    box = [float(item) for item in value]
    if not all(math.isfinite(item) for item in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"{label} is invalid")
    return box


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def size_bin(shortest: float) -> str:
    if shortest < 16:
        return "LT16"
    if shortest < 32:
        return "16_TO_31"
    if shortest < 64:
        return "32_TO_63"
    return "GE64"


def evaluate(public_path: Path, private_path: Path, prediction_path: Path) -> dict[str, Any]:
    public = json.loads(public_path.read_text(encoding="utf-8"))
    private = json.loads(private_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    if private["public_input_sha256"] != sha256(public_path):
        raise ValueError("private evaluator is not bound to public input")
    if prediction["public_input_sha256"] != sha256(public_path):
        raise ValueError("prediction is not bound to public input")
    if prediction.get("private_truth_access") is not False:
        raise ValueError("provider must explicitly declare zero private truth access")
    public_ids = [case["case_id"] for case in public["cases"]]
    predicted = {case["case_id"]: case for case in prediction["cases"]}
    truth = {case["case_id"]: case for case in private["cases"]}
    if set(predicted) != set(public_ids) or set(truth) != set(public_ids):
        raise ValueError("case identity mismatch")

    thresholds = [float(private["primary_correct_iou_threshold"]), *map(float, private["diagnostic_correct_iou_thresholds"])]
    ks = [int(value) for value in private["recall_at_k"]]
    rows = []
    for case_id in public_ids:
        output = predicted[case_id]
        target = truth[case_id]
        candidates = output["candidates"]
        if len(candidates) > 10:
            raise ValueError(f"candidate cap exceeded: {case_id}")
        if [candidate["rank"] for candidate in candidates] != list(range(1, len(candidates) + 1)):
            raise ValueError(f"candidate ranks are not contiguous: {case_id}")
        target_box = validated_box(target["target_bbox_xyxy"], f"{case_id} target")
        candidate_boxes = [validated_box(candidate["bbox_xyxy"], f"{case_id} candidate") for candidate in candidates]
        overlaps = [iou(box, target_box) for box in candidate_boxes]
        first_ranks = {
            str(threshold): next((rank for rank, overlap in enumerate(overlaps, start=1) if overlap >= threshold), None)
            for threshold in thresholds
        }
        raw = output.get("raw_candidates")
        if first_ranks[str(thresholds[0])] is not None:
            failure = "CORRECT_CANDIDATE_AVAILABLE"
        elif raw is None:
            failure = "GENERATION_OR_PROVIDER_POSTPROCESS_NOT_SEPARABLE"
        elif any(iou(validated_box(candidate["bbox_xyxy"], f"{case_id} raw candidate"), target_box) >= thresholds[0] for candidate in raw):
            failure = "RETENTION_OR_RANKING_BOTTLENECK"
        else:
            failure = "PROPOSAL_GENERATION_BOTTLENECK"
        rows.append({
            "case_id": case_id,
            "candidate_count": len(candidates),
            "best_iou": max(overlaps, default=0.0),
            "first_correct_rank": first_ranks,
            "failure_bucket": failure,
            "latency_ms": float(output["latency_ms"]),
            "target_shortest_side_px": target["target_shortest_side_px"],
            "target_visibility_ratio": target["target_visibility_ratio"],
            "size_bin": size_bin(float(target["target_shortest_side_px"])),
            "diagnostic_target_metadata": target["diagnostic_target_metadata"],
        })

    recall = {}
    for threshold in thresholds:
        key = str(threshold)
        recall[key] = {
            f"recall_at_{k}": sum(row["first_correct_rank"][key] is not None and row["first_correct_rank"][key] <= k for row in rows) / len(rows)
            for k in ks
        }
    primary_key = str(thresholds[0])
    by_size = {}
    for bucket in ("LT16", "16_TO_31", "32_TO_63", "GE64"):
        subset = [row for row in rows if row["size_bin"] == bucket]
        if subset:
            by_size[bucket] = {
                "cases": len(subset),
                **{f"recall_at_{k}": sum(row["first_correct_rank"][primary_key] is not None and row["first_correct_rank"][primary_key] <= k for row in subset) / len(subset) for k in ks},
            }
    r1, r10 = recall[primary_key]["recall_at_1"], recall[primary_key]["recall_at_10"]
    if r10 == 0.0:
        terminal = "P1_PA0_PROPOSAL_GENERATION_FAIL_ON_FAILURE_COHORT"
    elif r10 > r1:
        terminal = "P1_PA0_TOP1_COLLAPSE_SIGNAL_ON_FAILURE_COHORT"
    elif r10 == 1.0:
        terminal = "P1_PA0_BOUNDED_POOL_AVAILABILITY_MECHANICS_PASS"
    else:
        terminal = "P1_PA0_PARTIAL_AVAILABILITY_NO_TOPK_GAIN"
    latencies = [row["latency_ms"] for row in rows]
    counts = [row["candidate_count"] for row in rows]
    return {
        "schema_version": RESULT_SCHEMA,
        "protocol_id": public["protocol_id"],
        "provider": prediction["provider"],
        "inputs": {
            "public_input_sha256": sha256(public_path),
            "private_eval_input_sha256": sha256(private_path),
            "prediction_sha256": sha256(prediction_path),
        },
        "cases": len(rows),
        "recall": recall,
        "background_only_frame_rate": sum(row["first_correct_rank"][primary_key] is None for row in rows) / len(rows),
        "first_correct_rank_primary": [row["first_correct_rank"][primary_key] for row in rows],
        "recall_by_target_shortest_side": by_size,
        "proposal_count": {"mean": mean(counts), "p95": percentile([float(value) for value in counts], 0.95), "cap_hits": sum(value == 10 for value in counts)},
        "latency_ms": {"median": median(latencies), "p95": percentile(latencies, 0.95)},
        "failure_bucket_counts": {bucket: sum(row["failure_bucket"] == bucket for row in rows) for bucket in sorted({row["failure_bucket"] for row in rows})},
        "raw_retention_attribution": "AVAILABLE" if all(predicted[case_id].get("raw_candidates") is not None for case_id in public_ids) else "NOT_EVALUABLE_PROVIDER_INTERFACE",
        "rows": rows,
        "terminal": terminal,
        "claim_ceiling": private["claim_ceiling"],
        "contrastive_verifier": "NOT_EVALUATED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, evaluate(args.public, args.private, args.prediction))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
