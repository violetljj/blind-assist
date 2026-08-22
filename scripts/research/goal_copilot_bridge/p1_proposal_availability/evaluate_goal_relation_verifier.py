#!/usr/bin/env python3
"""Evaluate the frozen leftmost-entrance relation reranker against private truth."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import PRIVATE_SCHEMA, PUBLIC_SCHEMA, PREDICTION_SCHEMA, iou, sha256, validated_box
from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_goal_relation_verifier import MANIFEST_SCHEMA, OUTPUT_SCHEMA


IOU_THRESHOLD = 0.30


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _hit(box: list[float] | None, targets: list[list[float]]) -> bool:
    return box is not None and any(iou(box, target) >= IOU_THRESHOLD for target in targets)


def evaluate(public: Mapping[str, Any], private: Mapping[str, Any], prediction: Mapping[str, Any], relation: Mapping[str, Any], manifest: Mapping[str, Any], *, public_sha256: str, private_sha256: str, prediction_sha256: str, relation_sha256: str, manifest_sha256: str) -> dict[str, Any]:
    _require(public.get("schema_version") == PUBLIC_SCHEMA, "public schema mismatch")
    _require(private.get("schema_version") == PRIVATE_SCHEMA, "private schema mismatch")
    _require(prediction.get("schema_version") == PREDICTION_SCHEMA, "prediction schema mismatch")
    _require(relation.get("schema_version") == OUTPUT_SCHEMA, "relation output schema mismatch")
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema mismatch")
    _require(private.get("public_input_sha256") == public_sha256, "private/public binding mismatch")
    _require(prediction.get("public_input_sha256") == public_sha256, "prediction/public binding mismatch")
    _require(relation.get("public_input_sha256") == public_sha256, "relation/public binding mismatch")
    _require(relation.get("prediction_sha256") == prediction_sha256, "relation/prediction binding mismatch")
    _require(relation.get("verifier_manifest_sha256") == manifest_sha256, "relation/manifest binding mismatch")
    _require(relation.get("private_truth_access") is False, "relation output accessed private truth")

    private_cases = {case["case_id"]: case for case in private.get("cases", [])}
    prediction_cases = {case["case_id"]: case for case in prediction.get("cases", [])}
    relation_cases = {case["case_id"]: case for case in relation.get("cases", [])}
    public_ids = {case["case_id"] for case in public.get("cases", [])}
    _require(public_ids == set(private_cases) == set(prediction_cases) == set(relation_cases), "case roster mismatch")
    rows = []
    baseline_hits = relation_hits = available = 0
    contrastive_count = contrastive_baseline_hits = contrastive_relation_hits = 0
    for case_id in sorted(public_ids):
        truth = private_cases[case_id]
        if truth.get("target_visibility") != "VISIBLE":
            continue
        targets = [validated_box(value, f"{case_id} target") for value in truth.get("legal_target_bboxes_xyxy", [])]
        _require(len(targets) == 1 and truth.get("reference_mode") == "UNIQUE", "visible relation case must have one UNIQUE target")
        distractors = [validated_box(value, f"{case_id} distractor") for value in truth.get("same_class_distractor_bboxes_xyxy", [])]
        baseline_candidates = prediction_cases[case_id].get("candidates", [])
        relation_candidates = relation_cases[case_id].get("candidates", [])
        baseline_identity = sorted(
            (int(candidate["rank"]), tuple(validated_box(candidate["bbox_xyxy"], f"{case_id} baseline candidate")), float(candidate["proposal_score"]))
            for candidate in baseline_candidates
        )
        relation_identity = sorted(
            (int(candidate["original_rank"]), tuple(validated_box(candidate["bbox_xyxy"], f"{case_id} relation candidate")), float(candidate["proposal_score"]))
            for candidate in relation_candidates
        )
        _require(relation_identity == baseline_identity, "relation output is not a permutation of the bounded candidate pool")
        _require([int(candidate["relation_rank"]) for candidate in relation_candidates] == list(range(1, len(relation_candidates) + 1)), "relation ranks are not contiguous")
        expected_relation_order = sorted(
            relation_candidates,
            key=lambda candidate: ((candidate["bbox_xyxy"][0] + candidate["bbox_xyxy"][2]) / 2.0, int(candidate["original_rank"])),
        )
        _require(relation_candidates == expected_relation_order, "relation ordering rule drift")
        baseline_top = validated_box(baseline_candidates[0]["bbox_xyxy"], f"{case_id} baseline top") if baseline_candidates else None
        relation_top = validated_box(relation_candidates[0]["bbox_xyxy"], f"{case_id} relation top") if relation_candidates else None
        baseline_hit = _hit(baseline_top, targets)
        relation_hit = _hit(relation_top, targets)
        target_available = any(_hit(validated_box(candidate["bbox_xyxy"], f"{case_id} pool"), targets) for candidate in baseline_candidates)
        background_count = sum(
            not _hit(validated_box(candidate["bbox_xyxy"], f"{case_id} background"), targets + distractors)
            for candidate in baseline_candidates
        )
        is_contrastive = bool(distractors) and background_count > 0 and target_available
        baseline_hits += int(baseline_hit)
        relation_hits += int(relation_hit)
        available += int(target_available)
        if is_contrastive:
            contrastive_count += 1
            contrastive_baseline_hits += int(baseline_hit)
            contrastive_relation_hits += int(relation_hit)
        rows.append({
            "case_id": case_id,
            "candidate_count": len(baseline_candidates),
            "same_class_distractor_count": len(distractors),
            "background_candidate_count": background_count,
            "target_available_at_10": target_available,
            "baseline_top1_hit": baseline_hit,
            "relation_top1_hit": relation_hit,
            "contrastive_abc_evaluable": is_contrastive,
        })
    denominator = len(rows)
    _require(denominator > 0, "no visible UNIQUE cases")
    delta = (relation_hits - baseline_hits) / denominator
    terminal = "P1_GOAL_RELATION_VERIFIER_IMPROVED_TOP1" if delta > 0 else ("P1_GOAL_RELATION_VERIFIER_NO_TOP1_GAIN" if delta == 0 else "P1_GOAL_RELATION_VERIFIER_DEGRADED_TOP1")
    return {
        "schema_version": "blindassist_p1_goal_relation_verifier_evaluation_v1",
        "protocol_id": manifest.get("protocol_id"),
        "primary_iou_threshold": IOU_THRESHOLD,
        "evaluable_case_count": denominator,
        "proposal_target_availability_at_10": available / denominator,
        "baseline_top1_accuracy": baseline_hits / denominator,
        "relation_top1_accuracy": relation_hits / denominator,
        "absolute_top1_delta": delta,
        "contrastive_abc_case_count": contrastive_count,
        "contrastive_baseline_top1_accuracy": contrastive_baseline_hits / contrastive_count if contrastive_count else None,
        "contrastive_relation_top1_accuracy": contrastive_relation_hits / contrastive_count if contrastive_count else None,
        "rows": rows,
        "inputs": {"public_sha256": public_sha256, "private_sha256": private_sha256, "prediction_sha256": prediction_sha256, "relation_sha256": relation_sha256, "manifest_sha256": manifest_sha256},
        "claim_ceiling": manifest.get("claim_ceiling"),
        "terminal": terminal,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--private", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--relation", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    _require(not args.output.exists(), "evaluation output is immutable and already exists")
    values = [json.loads(path.read_text(encoding="utf-8")) for path in (args.public, args.private, args.prediction, args.relation, args.manifest)]
    payload = evaluate(*values, public_sha256=sha256(args.public), private_sha256=sha256(args.private), prediction_sha256=sha256(args.prediction), relation_sha256=sha256(args.relation), manifest_sha256=sha256(args.manifest))
    _atomic_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
