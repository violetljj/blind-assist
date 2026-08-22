#!/usr/bin/env python3
"""Apply a frozen, GT-blind current-frame goal-relation reranker."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    PREDICTION_SCHEMA,
    PUBLIC_SCHEMA,
    sha256,
    validated_box,
)


MANIFEST_SCHEMA = "blindassist_p1_goal_relation_verifier_manifest_v1"
OUTPUT_SCHEMA = "blindassist_p1_goal_relation_verifier_output_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rerank(public: Mapping[str, Any], prediction: Mapping[str, Any], manifest: Mapping[str, Any], *, public_sha256: str, prediction_sha256: str, manifest_sha256: str) -> dict[str, Any]:
    _require(public.get("schema_version") == PUBLIC_SCHEMA, "public schema mismatch")
    _require(prediction.get("schema_version") == PREDICTION_SCHEMA, "prediction schema mismatch")
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema mismatch")
    _require(prediction.get("public_input_sha256") == public_sha256, "prediction/public binding mismatch")
    _require(manifest.get("goal_type") == "LEFTMOST_BUILDING_ENTRANCE", "unsupported goal relation")
    _require(manifest.get("reference_mode") == "UNIQUE", "relation verifier requires UNIQUE goal")
    verifier = manifest.get("verifier", {})
    _require(verifier.get("kind") == "DETERMINISTIC_PUBLIC_GOAL_RELATION", "verifier kind drift")
    _require(verifier.get("private_truth_access") is False, "verifier permits private truth")
    _require(verifier.get("threshold_sweep") is False and verifier.get("configuration_sweep") is False, "verifier permits a sweep")

    public_cases = {case["case_id"]: case for case in public.get("cases", [])}
    prediction_cases = {case["case_id"]: case for case in prediction.get("cases", [])}
    _require(len(public_cases) == len(public.get("cases", [])), "duplicate public case id")
    _require(set(public_cases) == set(prediction_cases), "prediction/public case mismatch")
    outputs = []
    for case_id, public_case in public_cases.items():
        goal = public_case.get("goal_contract", {})
        _require(goal.get("goal_type") == manifest["goal_type"], "public goal type drift")
        _require(goal.get("reference_mode") == "UNIQUE", "public reference mode drift")
        candidates = []
        for candidate in prediction_cases[case_id].get("candidates", []):
            box = validated_box(candidate.get("bbox_xyxy"), f"{case_id} candidate")
            original_rank = int(candidate.get("rank"))
            candidates.append({
                "bbox_xyxy": box,
                "proposal_score": float(candidate.get("proposal_score")),
                "original_rank": original_rank,
                "relation_key": [(box[0] + box[2]) / 2.0, original_rank],
            })
        ranked = sorted(candidates, key=lambda item: (item["relation_key"][0], item["original_rank"]))
        outputs.append({
            "case_id": case_id,
            "candidates": [dict(candidate, relation_rank=index) for index, candidate in enumerate(ranked, start=1)],
        })
    return {
        "schema_version": OUTPUT_SCHEMA,
        "protocol_id": manifest.get("protocol_id"),
        "public_input_sha256": public_sha256,
        "prediction_sha256": prediction_sha256,
        "verifier_manifest_sha256": manifest_sha256,
        "private_truth_access": False,
        "goal_relation": "LEFTMOST_BBOX_X_CENTER",
        "cases": outputs,
        "claim_ceiling": manifest.get("claim_ceiling"),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--prediction", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    _require(not args.output.exists(), "relation output is immutable and already exists")
    public = json.loads(args.public.read_text(encoding="utf-8"))
    prediction = json.loads(args.prediction.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _atomic_json(args.output, rerank(public, prediction, manifest, public_sha256=sha256(args.public), prediction_sha256=sha256(args.prediction), manifest_sha256=sha256(args.manifest)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
